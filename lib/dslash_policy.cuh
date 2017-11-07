static cudaColorSpinorField *inSpinor;

// hooks into tune.cpp variables for policy tuning
typedef std::map<TuneKey, TuneParam> map;
const map& getTuneCache();

void disableProfileCount();
void enableProfileCount();

void setPolicyTuning(bool);

// these variables are used for benchmarking the dslash components in isolation
static bool dslash_pack_compute = true;
static bool dslash_interior_compute = true;
static bool dslash_exterior_compute = true;
static bool dslash_comms = true;
static bool dslash_copy = true;

/**
 * Arrays used for the dynamic scheduling.
 */
struct DslashCommsPattern {
  int gatherCompleted[Nstream];
  int previousDir[Nstream];
  int commsCompleted[Nstream];
  int dslashCompleted[Nstream];
  int commDimTotal;
  int completeSum;

  DslashCommsPattern(const int commDim[], bool gdr_send=false)
    : commsCompleted{ }, dslashCompleted{ }, completeSum(0) {

    for (int i=0; i<Nstream-1; i++) gatherCompleted[i] = gdr_send ? 1 : 0;
    gatherCompleted[Nstream-1] = 1;
    commsCompleted[Nstream-1] = 1;
    dslashCompleted[Nstream-1] = 1;

    //   We need to know which was the previous direction in which
    //   communication was issued, since we only query a given event /
    //   comms call after the previous the one has successfully
    //   completed.
    for (int i=3; i>=0; i--) {
      if (commDim[i]) {
	int prev = Nstream-1;
	for (int j=3; j>i; j--) if (commDim[j]) prev = 2*j;
	previousDir[2*i + 1] = prev;
	previousDir[2*i + 0] = 2*i + 1; // always valid
      }
    }
    
    // this tells us how many events / comms occurances there are in
    // total.  Used for exiting the while loop
    commDimTotal = 0;
    for (int i=3; i>=0; i--) {
      commDimTotal += commDim[i];
    }
    commDimTotal *= gdr_send ? 2 : 4; // 2 from pipe length, 2 from direction
  }
};


inline void setFusedParam(DslashParam& param, DslashCuda &dslash, const int* faceVolumeCB){
  int prev = -1;

  param.threads = 0;
  for (int i=0; i<4; ++i) {
    param.threadDimMapLower[i] = 0;
    param.threadDimMapUpper[i] = 0;
    if (!dslashParam.commDim[i]) continue;
    param.threadDimMapLower[i] = (prev >= 0 ? param.threadDimMapUpper[prev] : 0);
    param.threadDimMapUpper[i] = param.threadDimMapLower[i] + dslash.Nface()*faceVolumeCB[i];
    param.threads = param.threadDimMapUpper[i];
    prev=i;
  }

  param.kernel_type = EXTERIOR_KERNEL_ALL;
}

#undef DSLASH_PROFILE
#ifdef DSLASH_PROFILE
#define PROFILE(f, profile, idx)		\
  profile.TPSTART(idx);				\
  f;						\
  profile.TPSTOP(idx); 
#else
#define PROFILE(f, profile, idx) f;
#endif



#ifdef PTHREADS
#include <pthread.h>


namespace {

  struct ReceiveParam 
  {
    TimeProfile* profile;
    int nFace;
    int dagger;
  };

  void *issueMPIReceive(void* receiveParam)
  {
    ReceiveParam* param = static_cast<ReceiveParam*>(receiveParam);
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inSpinor->recvStart(param->nFace, 2*i+dir, param->dagger), (*(param->profile)), QUDA_PROFILE_COMMS_START);
      }
    }
    return nullptr;
  }

  struct InteriorParam 
  {
    TimeProfile* profile;
    DslashCuda* dslash;
    int current_device;
  };


 void* launchInteriorKernel(void* interiorParam)
  {
    InteriorParam* param = static_cast<InteriorParam*>(interiorParam);
    cudaSetDevice(param->current_device); // set device in the new thread
    PROFILE(param->dslash->apply(streams[Nstream-1]), (*(param->profile)), QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);
    return nullptr;
  }

} // anonymous namespace
#endif


namespace {

  /**
     @brief This helper function simply posts all receives in all directions
     @param[out] input Field that we are doing halo exchange
     @param[in] dslash The dslash object
     @param[in] stream Stream were the receive is being posted (effectively ignored)
     @param[in] gdr Whether we are using GPU Direct RDMA or not
  */
  inline void issueRecv(cudaColorSpinorField &input, const DslashCuda &dslash, cudaStream_t *stream, bool gdr) {
    for(int i=3; i>=0; i--){
      if (!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--) {
        PROFILE(if (dslash_comms) input.recvStart(dslash.Nface()/2, 2*i+dir, dslash.Dagger(), stream, gdr), profile, QUDA_PROFILE_COMMS_START);
      }
    }
  }

  /**
     @brief This helper function simply posts the packing kernel needed for halo exchange
     @param[out] input Field that we are packing
     @param[in] dslash The dslash object
     @param[in] parity Field parity
     @param[in] location Memory location where we are packing to - if
     Host is requested, the only non-p2p halos will be sent to host
     with p2p halos kept on the device
     @param[in] packIndex Stream index where the packing kernel will run
  */
  inline void issuePack(cudaColorSpinorField &input, const DslashCuda &dslash, int parity, MemoryLocation location, int packIndex) {

    using namespace dslash;

    bool pack = false;
    for (int i=3; i>=0; i--)
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }

    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    // always packing to local device for p2p directions or if requested, else pack to host (zero copy)
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	pack_dest[2*dim+dir] = (location == Device || comm_peer2peer_enabled(dir,dim)) ? Device : Host;

    if (pack) {
      PROFILE(if (dslash_pack_compute) input.pack(dslash.Nface()/2, parity, dslash.Dagger(), packIndex,
						  pack_dest, dslashParam.twist_a, dslashParam.twist_b),
	      profile, QUDA_PROFILE_PACK_KERNEL);

      // Record the end of the packing
      PROFILE(if (location != Host) cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

  }

  /**
     @brief This helper function simply posts the device-host memory
     copies of all halos in all dimensions and directions
     @param[out] input Field that whose halos we are communicating
     @param[in] dslash The dslash object
  */
  inline void issueGather(cudaColorSpinorField &input, const DslashCuda &dslash) {

    using namespace dslash;

    for (int i = 3; i >=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) { // forwards gather
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) input.gather(dslash.Nface()/2, dslash.Dagger(), 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

  }

  /**
     @brief Returns a stream index for posting the pack/scatters to.
     We desire a stream index that is not being used for peer-to-peer
     communication.  This is used by the fused halo dslash kernels
     where we post all scatters to the same stream so we only have a
     single event to wait on before the exterior kernel is applied,
     and by the zero-copy dslash kernels where we want to post the
     packing kernel to an unused stream.
     @return stream index
  */
  inline int getStreamIndex() {
    // set index to a stream index not being used for p2p
    int index = -1;
    for (int i = 3; i >=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      if (!comm_peer2peer_enabled(0,i)) index = 2*i+0;
      else if (!comm_peer2peer_enabled(1,i)) index = 2*i+1;
    }
    // make sure we pick a valid index, in case we are fully p2p connected
    if (index == -1) index = 0;
    return index;
  }

  /**
     @brief Wrapper for querying if communication is finished in the
     dslash, and if it is take the appropriate action:

     - if peer-to-peer then we now know that the peer-to-peer copy is
       now in flight and we are safe to post the cudaStreamWaitEvent
       in our GPU context

     - if gdr or zero-copy for the receive buffer then we have nothing
       else to do, it is now safe to post halo kernel

     - if staging with async, we release the scatter by setting the approriate commsEnd_h flag

     - if basic staging, we post the scatter (host to device memory copy)

     @param[in,out] input
     @param[in] dslash The dslash object
     @param[in] dim Dimension we are working on
     @param[in] dir Direction we are working on
     @param[in] gdr_send Whether GPU Direct RDMA is being used for sending
     @param[in] gdr_recv Whether GPU Direct RDMA is being used for receiving
     @param[in] zero_copy_recv Whether we are using zero-copy on the
     receive end (and hence do not need to do CPU->GPU copy)
     @param[in] async Whether GPU Direct Async is being used
     @param[in] scatterIndex The stream index used for posting the host-to-device memory copy in
   */
  inline bool commsComplete(cudaColorSpinorField &input, const DslashCuda &dslash, int dim, int dir,
			    bool gdr_send, bool gdr_recv, bool zero_copy_recv, bool async, int scatterIndex=-1) {

    using namespace dslash;

    cudaStream_t *stream = nullptr;

    PROFILE(int comms_test = dslash_comms ? input.commsQuery(dslash.Nface()/2, 2*dim+dir, dslash.Dagger(), stream, gdr_send, gdr_recv) : 1, profile, QUDA_PROFILE_COMMS_QUERY);
    if (comms_test) {
      // now we are receive centric
      int dir2 = 1-dir;

      // if peer-2-peer in a given direction then we need to insert a wait on that copy event
      if (comm_peer2peer_enabled(dir2,dim)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], input.getIPCRemoteCopyEvent(dir2,dim), 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      } else {

	if (!gdr_recv && !zero_copy_recv) { // Issue CPU->GPU copy if not GDR

	  if (async) {
#if (CUDA_VERSION >= 8000)
	    // this will trigger the copy asynchronously
	    *((volatile cuuint32_t*)(commsEnd_h+2*dim+dir2)) = 1;
#else
	    errorQuda("Async dslash policy variants require CUDA 8.0 and above");
#endif
	  } else {
	    // note the ColorSpinorField::scatter transforms from
	    // scatter centric to gather centric (e.g., flips
	    // direction) so here just use dir not dir2
	    if (scatterIndex == -1) scatterIndex = 2*dim+dir;
	    PROFILE(if (dslash_copy) input.scatter(dslash.Nface()/2, dslash.Dagger(), 2*dim+dir, streams+scatterIndex), profile, QUDA_PROFILE_SCATTER);
	  }

	}

      }
    }
    return comms_test;
  }

  /**
     @brief Set the ghosts to the mapped CPU ghost buffer, or unsets
     if already set.
     @param[in,out] dslash The dslash object
     @param[in,out] input The ColorSpinorField source
     @param[in] to_mapped Whether we are switching to mapped ghosts or not
   */
  void setGhost(DslashCuda &dslash, cudaColorSpinorField &input, bool to_mapped) {

    char aux_copy[TuneKey::aux_n];
    static bool set_mapped = false;

    if (to_mapped) {
      if (set_mapped) errorQuda("set_mapped already set");
      // in the below we switch to the mapped ghost buffer and update the tuneKey to reflect this
      input.bufferIndex += 2;
#ifdef USE_TEXTURE_OBJECTS
      dslashParam.ghostTex = input.GhostTex();
      dslashParam.ghostTexNorm = input.GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
      strcpy(aux_copy,dslash.getAux(dslashParam.kernel_type));
      dslash.augmentAux(dslashParam.kernel_type, ",zero_copy");
      set_mapped = true;
    } else {
      if (!set_mapped) errorQuda("set_mapped not set");
      // reset to default
      dslash.setAux(dslashParam.kernel_type, aux_copy);
      input.bufferIndex -= 2;
#ifdef USE_TEXTURE_OBJECTS
      dslashParam.ghostTex = input.GhostTex();
      dslashParam.ghostTexNorm = input.GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
      set_mapped = false;
    }
  }

  struct DslashPolicyImp {

  virtual void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor,
                        const size_t regSize, const int parity, const int dagger,
                        const int volume, const int *faceVolumeCB, TimeProfile &profile) = 0;

  virtual ~DslashPolicyImp(){}
};

/**
   Standard dslash parallelization with host staging for send and receive
 */
struct DslashBasic : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {

	  for (int dir=1; dir>=0; dir--) {
	    if (!comm_peer2peer_enabled(1-dir,i)) { // if not peer-to-peer we post an event in the scatter stream and wait on that
	      // Record the end of the scattering
	      PROFILE(cudaEventRecord(scatterEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
	      // wait for scattering to finish and then launch dslash
	      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+dir], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	    }
	  }

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

struct DslashPthreads : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		    const int volume, const int *faceVolumeCB, TimeProfile &profile) {
#ifdef PTHREADS
    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
  // Record the start of the dslash if doing communication in T and not kernel packing
    {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]),
              profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    inputSpinor->streamInit(streams);
    // and launch the interior dslash kernel

    const int packIndex = Nstream-2;
    //const int packIndex = Nstream-1;
    pthread_t receiveThread, interiorThread;
    ReceiveParam receiveParam;
    receiveParam.profile = &profile;
    receiveParam.nFace   = (dslash.Nface() >> 1);
    receiveParam.dagger  = dagger;

    if(pthread_create(&receiveThread, NULL, issueMPIReceive, &receiveParam)){
      errorQuda("pthread_create failed");
    }

    InteriorParam interiorParam;
    interiorParam.dslash   = &dslash;
    interiorParam.profile  = &profile; 

    cudaGetDevice(&(interiorParam.current_device)); // get the current device number
    if(pthread_create(&interiorThread, NULL, launchInteriorKernel, &interiorParam)){
      errorQuda("pthread_create failed");
    }

    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }

    if (pack){
      PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0),
              profile, QUDA_PROFILE_STREAM_WAIT_EVENT); 
    }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]),
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }
    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0),
	        profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]),
	        profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

#if (!defined MULTI_GPU)
    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);
#endif

#ifdef MULTI_GPU 
    if(pthread_join(receiveThread, NULL)) errorQuda("pthread_join failed");
    bool interiorLaunched = false;
    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]),
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if(!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	     pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;

	      // Scatter into the end zone
	      PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_SCATTER);
	    }
	  }

        } // dir=0,1

        // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
        if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	// Record the end of the scattering
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]),
		  profile, QUDA_PROFILE_EVENT_RECORD);

	  if(!interiorLaunched){
	    if(pthread_join(interiorThread, NULL)) errorQuda("pthread_join failed");
	    interiorLaunched = true;
          }

	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0),
		  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }

      }

    }
    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
#else // !PTHREADS
    errorQuda("Pthreads has not been built\n"); 
#endif
  }
};

/**
   Standard dslash parallelization with host staging for send and receive, and fused halo update kernel
 */
struct DslashFusedExterior : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    const int scatterIndex = getStreamIndex();
    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, false, scatterIndex) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }
        } // dir=0,1
      } // i
    } // while(pattern.completeSum < commDimTotal)

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    for (int i=3; i>=0; i--) {
      if (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i))) { // if not peer-to-peer we post an event in the scatter stream and wait on that
	PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	break;
      }
    }

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

/**
   Dslash parallelization with GDR for send and receive
 */
struct DslashGDR : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    bool pack_event = false;
    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	if (!pack_event) {
	  cudaEventSynchronize(packEnd[0]);
	  pack_event = true;
	}

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger, 0, true), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, true, true, false, false) ) {;
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};


/**
   Dslash parallelization with GDR for send and receive with fused halo update kernel
 */
struct DslashFusedGDR : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    bool pack_event = false;
    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	if (!pack_event) {
	  cudaEventSynchronize(packEnd[0]);
	  pack_event = true;
	}

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger, 0, true), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	}
      }
    } // p2p

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, true, true, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }
	} // dir=0,1
      } // i
    } // pattern.completeSum < pattern.CommDimTotal

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};

/**
   Dslash parallelization with host staging for send and GDR for receive
 */
struct DslashGDRRecv : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, false, true);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, true, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

/**
   Dslash parallelization with host staging for send and GDR for receive, with fused halo update kernel
 */
struct DslashFusedGDRRecv : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, false, true);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, true, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }
        } // dir=0,1
      } // i
    } // while(pattern.completeSum < commDimTotal)

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

#ifdef HOST_DEBUG
#define CUDA_CALL( call )						\
  {									\
    CUresult cudaStatus = call;						\
    if ( CUDA_SUCCESS != cudaStatus ) {					\
      const char *err_str = nullptr;					\
      cuGetErrorString(cudaStatus, &err_str);				\
      fprintf(stderr, "ERROR: CUDA call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, err_str, cudaStatus); \
    }									\
}
#else
#define CUDA_CALL( call ) call
#endif

/**
   Experimental Dslash parallelization with host staging for send and receive, with GPU Direct Async
 */
struct DslashAsync : DslashPolicyImp {

#if (CUDA_VERSION >= 8000)

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) {
		*((volatile cuuint32_t*)(commsEnd_h+2*i+1-dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[2*i+dir], commsEnd_d[2*i+1-dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, &streams[2*i+dir]), profile, QUDA_PROFILE_SCATTER);
		printfQuda("dim = %d dir=%d scheduled \n", i, dir);
	      }
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, true) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {

	  for (int dir=1; dir>=0; dir--) {
	    if (!comm_peer2peer_enabled(1-dir,i)) { // if not peer-to-peer we post an event in the scatter stream and wait on that
	      // Record the end of the scattering
	      PROFILE(cudaEventRecord(scatterEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
	      // wait for scattering to finish and then launch dslash
	      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+dir], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	    }
	  }

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }

      }

    }
    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
#else

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    errorQuda("Async dslash policy variants require CUDA 8.0 and above");
  }

#endif // CUDA_VERSION >= 8000

};


/**
   Experimental Dslash parallelization with host staging for send and
   receive, with GPU Direct Async, and fused hao update kernel
 */
struct DslashFusedExteriorAsync : DslashPolicyImp {

#if (CUDA_VERSION >= 8000)

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = Nstream-1;
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueGather(*inputSpinor, dslash);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    const int scatterIndex = getStreamIndex();
    DslashCommsPattern pattern(dslashParam.commDim);
    while (pattern.completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) { // gather centric
		*((volatile cuuint32_t*)(commsEnd_h+2*i+1-dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[scatterIndex], commsEnd_d[2*i+1-dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex), profile, QUDA_PROFILE_SCATTER);
	      }
	    }

	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, true, scatterIndex) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1
      } // i
    } // while(pattern.completeSum < commDimTotal)

    setFusedParam(dslashParam,dslash,faceVolumeCB);

    for (int i=3; i>=0; i--) {
      if (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i))) {
	// if not peer-to-peer we post an event in the scatter stream and wait on that
	PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	break;
      }
    }

    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

#else

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    errorQuda("Async dslash policy variants require CUDA 8.0 and above");
  }

#endif // CUDA_VERSION >= 8000

};


/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory
*/
struct DslashZeroCopyPack : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField *inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Host, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

	}

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  for (int dir=1; dir>=0; dir--) {
	    if (!comm_peer2peer_enabled(1-dir,i)) { // if not peer-to-peer we post an event in the scatter stream and wait on that
	      // Record the end of the scattering
	      PROFILE(cudaEventRecord(scatterEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
	      // wait for scattering to finish and then launch dslash
	      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+dir], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	    }
	  }

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
	}
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};


/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory with fused halo update kernel
*/
struct DslashFusedZeroCopyPack : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField *inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    const int packScatterIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packScatterIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Host, packScatterIndex);

    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packScatterIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, false, false, packScatterIndex) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

	} // dir=0,1
      } // i
    } // pattern.completeSum

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    for (int i=3; i>=0; i--) {
      if (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i))) {
	// if not peer-to-peer we post an event in the scatter stream and wait on that
	PROFILE(cudaEventRecord(scatterEnd[0], streams[packScatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	break;
      }
    }

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};

/**
   Multi-GPU Dslash zero-copy for the send and GDR for the receive
 */
struct DslashZeroCopyPackGDRRecv : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    const int packIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, false, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, true, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

        } // dir=0,1

        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // all faces use this stream
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

/**
   Multi-GPU Dslash zero-copy for the send and GDR for the receive,
   with fused halo update kernel
 */
struct DslashFusedZeroCopyPackGDRRecv : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    const int packIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Device, packIndex);

    issueRecv(*inputSpinor, dslash, 0, true); // Prepost receives

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, false, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, true, false, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }
        } // dir=0,1
      } // i
    } // while(pattern.completeSum < commDimTotal)

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory
*/
struct DslashZeroCopy : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField *inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    comm_enable_peer2peer(false);
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Host, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, true, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

	}

	// FIXME - will not work with P2P until we can split where the halos originate
	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
        if ( !pattern.dslashCompleted[2*i] && pattern.dslashCompleted[pattern.previousDir[2*i+1]] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  setGhost(dslash, *inputSpinor, true);
	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
	  setGhost(dslash, *inputSpinor, false);

	  pattern.dslashCompleted[2*i] = 1;
	}
      }
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
    comm_enable_peer2peer(true);
  }
};


/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory with fused halo update kernel
*/
struct DslashFusedZeroCopy : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField *inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    comm_enable_peer2peer(false);
    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    inputSpinor->streamInit(streams);
    issueRecv(*inputSpinor, dslash, 0, false); // Prepost receives

    const int packIndex = getStreamIndex();
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    issuePack(*inputSpinor, dslash, 1-parity, Host, packIndex);

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

    cudaStreamSynchronize(streams[packIndex]);

    for (int p2p=0; p2p<2; p2p++) { // schedule non-p2p traffic first, then do p2p
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {
	  if ( (comm_peer2peer_enabled(dir,i) + p2p) % 2 == 0 ) {
	    PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    DslashCommsPattern pattern(dslashParam.commDim, true);
    while (pattern.completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    if ( commsComplete(*inputSpinor, dslash, i, dir, false, false, true, false) ) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      pattern.completeSum++;
	    }
	  }

	}

      }

    }

    // FIXME - will not work with P2P until we can split where the halos originate
    if (pattern.commDimTotal) {
      // setup for exterior kernel
      setFusedParam(dslashParam,dslash,faceVolumeCB);

      setGhost(dslash, *inputSpinor, true);
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
      setGhost(dslash, *inputSpinor, false);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
    comm_enable_peer2peer(true);
  }
};


struct DslashNC : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		    const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    profile.TPSTART(QUDA_PROFILE_TOTAL);
    
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

  enum QudaDslashPolicy {
    QUDA_DSLASH,
    QUDA_FUSED_DSLASH,
    QUDA_GDR_DSLASH,
    QUDA_FUSED_GDR_DSLASH,
    QUDA_GDR_RECV_DSLASH,
    QUDA_FUSED_GDR_RECV_DSLASH,
    QUDA_ZERO_COPY_PACK_DSLASH,
    QUDA_FUSED_ZERO_COPY_PACK_DSLASH,
    QUDA_ZERO_COPY_DSLASH,
    QUDA_FUSED_ZERO_COPY_DSLASH,
    QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH,
    QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH,
    QUDA_DSLASH_ASYNC,
    QUDA_FUSED_DSLASH_ASYNC,
    QUDA_PTHREADS_DSLASH,
    QUDA_DSLASH_NC
  };

struct DslashFactory {

  static DslashPolicyImp* create(const QudaDslashPolicy &dslashPolicy)
  {
    DslashPolicyImp* result = nullptr;

    switch(dslashPolicy){
    case QUDA_DSLASH:
      result = new DslashBasic;
      break;
    case QUDA_DSLASH_ASYNC:
      result = new DslashAsync;
      break;
    case QUDA_PTHREADS_DSLASH:
      result = new DslashPthreads;
      break;
    case QUDA_FUSED_DSLASH:
      result = new DslashFusedExterior;
      break;
    case QUDA_FUSED_DSLASH_ASYNC:
      result = new DslashFusedExteriorAsync;
      break;
    case QUDA_GDR_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashGDR;
      else result = new DslashBasic;
      break;
    case QUDA_FUSED_GDR_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashFusedGDR;
      else result = new DslashFusedExterior;
      break;
    case QUDA_GDR_RECV_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashGDRRecv;
      else result = new DslashBasic;
      break;
    case QUDA_FUSED_GDR_RECV_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashFusedGDRRecv;
      else result = new DslashFusedExterior;
      break;
    case QUDA_ZERO_COPY_PACK_DSLASH:
      result = new DslashZeroCopyPack;
      break;
    case QUDA_FUSED_ZERO_COPY_PACK_DSLASH:
      result = new DslashFusedZeroCopyPack;
      break;
    case QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashZeroCopyPackGDRRecv;
      else result = new DslashZeroCopyPack;
      break;
    case QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH:
      if (!comm_gdr_blacklist()) result = new DslashFusedZeroCopyPackGDRRecv;
      else result = new DslashFusedZeroCopyPack;
      break;
    case QUDA_ZERO_COPY_DSLASH:
      result = new DslashZeroCopy;
      break;
    case QUDA_FUSED_ZERO_COPY_DSLASH:
      result = new DslashFusedZeroCopy;
      break;
    case QUDA_DSLASH_NC:
      result = new DslashNC;
      break;
    default:
      errorQuda("Dslash policy %d not recognized",dslashPolicy);
      break;
    }
    return result; // default 
  }
};

 static bool dslash_init = false;
 static std::vector<QudaDslashPolicy> policy;
 static int config = 0; // 2-bit number used to record the machine config (p2p / gdr) and if this changes we will force a retune

 class DslashPolicyTune : public Tunable {

   DslashCuda &dslash;
   cudaColorSpinorField *in;
   const size_t regSize;
   const int parity;
   const int dagger;
   const int volume;
   const int *ghostFace;
   TimeProfile &profile;

   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

 public:
   DslashPolicyTune(DslashCuda &dslash, cudaColorSpinorField *in, const size_t regSize, const int parity,
		    const int dagger, const int volume, const int *ghostFace, TimeProfile &profile)
     : dslash(dslash), in(in), regSize(regSize), parity(parity), dagger(dagger),
       volume(volume), ghostFace(ghostFace), profile(profile)
   {

     if (!dslash_init) {
       policy.reserve(10);
       static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_POLICY");
       if (dslash_policy_env) { // set the policies to tune for explicitly
	 std::stringstream policy_list(dslash_policy_env);

	 int policy_;
	 while (policy_list >> policy_) {
	   QudaDslashPolicy dslash_policy = static_cast<QudaDslashPolicy>(policy_);

	   // check this is a valid policy choice
	   if ( (dslash_policy == QUDA_GDR_DSLASH || dslash_policy == QUDA_FUSED_GDR_DSLASH ||
		 dslash_policy == QUDA_GDR_RECV_DSLASH || dslash_policy == QUDA_FUSED_GDR_RECV_DSLASH)
		&& !comm_gdr_enabled() ) {
	     errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", dslash_policy);
	   }

	   policy.push_back(static_cast<QudaDslashPolicy>(policy_));
	   if (policy_list.peek() == ',') policy_list.ignore();
	 }
       } else {
	 policy.push_back(QUDA_DSLASH);
	 policy.push_back(QUDA_FUSED_DSLASH);

	 // if we have gdr then enable tuning these policies
	 if (comm_gdr_enabled()) {
	   policy.push_back(QUDA_GDR_DSLASH);
	   policy.push_back(QUDA_FUSED_GDR_DSLASH);
	   policy.push_back(QUDA_GDR_RECV_DSLASH);
	   policy.push_back(QUDA_FUSED_GDR_RECV_DSLASH);
	   config+=2;
	 }

	 policy.push_back(QUDA_ZERO_COPY_PACK_DSLASH);
	 policy.push_back(QUDA_FUSED_ZERO_COPY_PACK_DSLASH);

	 if (comm_gdr_enabled()) {
	   policy.push_back(QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH);
	   policy.push_back(QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH);
	 }

	 // if we have p2p for now exclude zero-copy dslash halo reads
	 // since we can't mix these two until we can source halos
	 // from separate memory spaces (requires ghost texture per dim/dir?)
	 bool p2p = comm_peer2peer_enabled_global();
	 config+=p2p;

	 // note these policies do not presently use p2p
	 policy.push_back(QUDA_ZERO_COPY_DSLASH);
	 policy.push_back(QUDA_FUSED_ZERO_COPY_DSLASH);

	 // Async variants are only supported on CUDA 8.0 and up
#if (CUDA_VERSION >= 8000) && 0
#if (CUDA_VERSION >= 9000)
	 CUdevice device;
	 cuDeviceGet(&device, comm_gpuid());
	 int can_use_stream_mem_ops;
	 cuDeviceGetAttribute(&can_use_stream_mem_ops, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, device);
#else
	 int can_use_stream_mem_ops = 1;
#endif
	 if (can_use_stream_mem_ops) {
	   policy.push_back(QUDA_DSLASH_ASYNC);
	   policy.push_back(QUDA_FUSED_DSLASH_ASYNC);
	 }
#endif
       }

       static char *dslash_pack_env = getenv("QUDA_ENABLE_DSLASH_PACK");
       if (dslash_pack_env && strcmp(dslash_pack_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash halo packing");
	 dslash_pack_compute = false;
       }

       static char *dslash_interior_env = getenv("QUDA_ENABLE_DSLASH_INTERIOR");
       if (dslash_interior_env && strcmp(dslash_interior_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash interior computation");
	 dslash_interior_compute = false;
       }

       static char *dslash_exterior_env = getenv("QUDA_ENABLE_DSLASH_EXTERIOR");
       if (dslash_exterior_env && strcmp(dslash_exterior_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash exterior computation");
	 dslash_exterior_compute = false;
       }

       static char *dslash_copy_env = getenv("QUDA_ENABLE_DSLASH_COPY");
       if (dslash_copy_env && strcmp(dslash_copy_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash host-device copying");
	 dslash_copy = false;
       }

       static char *dslash_comms_env = getenv("QUDA_ENABLE_DSLASH_COMMS");
       if (dslash_comms_env && strcmp(dslash_comms_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash communication");
	 dslash_comms = false;
       }
     }

     // before we do policy tuning we must ensure the kernel
     // constituents have been tuned since we can't do nested tuning
     if (getTuning() && getTuneCache().find(tuneKey()) == getTuneCache().end()) {
       disableProfileCount();

       for (auto &i : policy) {

	 if (i == QUDA_DSLASH || i == QUDA_FUSED_DSLASH ||
	     i == QUDA_DSLASH_ASYNC || i == QUDA_FUSED_DSLASH_ASYNC) {

	   DslashPolicyImp* dslashImp = DslashFactory::create(i);
	   (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	   delete dslashImp;

	 } else if (i == QUDA_GDR_DSLASH || i == QUDA_FUSED_GDR_DSLASH ||
		    i == QUDA_GDR_RECV_DSLASH || i == QUDA_FUSED_GDR_RECV_DSLASH ||
		    i == QUDA_ZERO_COPY_PACK_DSLASH || i == QUDA_FUSED_ZERO_COPY_PACK_DSLASH ||
		    i == QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH || i == QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH ||
		    i == QUDA_ZERO_COPY_DSLASH || i == QUDA_FUSED_ZERO_COPY_DSLASH) {
	   // these dslash policies all must have kernel packing enabled

	   bool kernel_pack_old = getKernelPackT();

	   // if we are using GDR policies then we must tune the
	   // non-GDR variants as well with and without kernel packing
	   // enabled - this ensures that all GPUs will have the
	   // required tune cache entries prior to potential process
	   // divergence regardless of which GPUs are blacklisted
	   if (i == QUDA_GDR_DSLASH || i == QUDA_FUSED_GDR_DSLASH ||
	       i == QUDA_GDR_RECV_DSLASH || i == QUDA_FUSED_GDR_RECV_DSLASH) {
	     QudaDslashPolicy policy = (i==QUDA_GDR_DSLASH || i==QUDA_GDR_RECV_DSLASH) ? QUDA_DSLASH : QUDA_FUSED_DSLASH;
	     DslashPolicyImp* dslashImp = DslashFactory::create(policy);
	     setKernelPackT(false);
	     (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	     setKernelPackT(true);
	     (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	     delete dslashImp;
	   }

	   setKernelPackT(true);

	   DslashPolicyImp* dslashImp = DslashFactory::create(i);
	   (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	   delete dslashImp;

	   // restore default kernel packing
	   setKernelPackT(kernel_pack_old);

	 } else {
	   errorQuda("Unsupported dslash policy %d\n", i);
	 }
       }

       enableProfileCount();
       setPolicyTuning(true);
     }
     dslash_init = true;
   }

   virtual ~DslashPolicyTune() { setPolicyTuning(false); }

   void apply(const cudaStream_t &stream) {
     TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_DEBUG_VERBOSE /*getVerbosity()*/);

     if (config != tp.aux.y && comm_size() > 1) {
       errorQuda("Machine configuration (P2P/GDR=%d) changed since tunecache was created (P2P/GDR=%d).  Please delete "
		 "this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.",
		 config, tp.aux.y);
     }

     if (tp.aux.x >= (int)policy.size()) errorQuda("Requested policy that is outside of range");

     // switch on kernel packing for the policies that need it
     bool kernel_pack_old = getKernelPackT();
     if (policy[tp.aux.x] == QUDA_GDR_DSLASH || policy[tp.aux.x] == QUDA_FUSED_GDR_DSLASH ||
	 policy[tp.aux.x] == QUDA_ZERO_COPY_PACK_DSLASH || policy[tp.aux.x] == QUDA_FUSED_ZERO_COPY_PACK_DSLASH ||
	 policy[tp.aux.x] == QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH || policy[tp.aux.x] == QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH ||
	 policy[tp.aux.x] == QUDA_ZERO_COPY_DSLASH || policy[tp.aux.x] == QUDA_FUSED_ZERO_COPY_DSLASH) {
       setKernelPackT(true);
     }

     DslashPolicyImp* dslashImp = DslashFactory::create(policy[tp.aux.x]);
     (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
     delete dslashImp;

	 // restore default kernel packing
     setKernelPackT(kernel_pack_old);
   }

   int tuningIter() const { return 10; }

   // Find the best dslash policy
   bool advanceAux(TuneParam &param) const
   {
     if ((unsigned)param.aux.x < policy.size()-1) {
       param.aux.x++;
       return true;
     } else {
       param.aux.x = 0;
       return false;
     }
   }

   bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

   void initTuneParam(TuneParam &param) const  {
     Tunable::initTuneParam(param);
     param.aux.x = 0; param.aux.y = config; param.aux.z = 0; param.aux.w = 0;
   }

   void defaultTuneParam(TuneParam &param) const  {
     Tunable::defaultTuneParam(param);
     param.aux.x = 0; param.aux.y = config; param.aux.z = 0; param.aux.w = 0;
   }

   TuneKey tuneKey() const {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     TuneKey key = dslash.tuneKey();
     strcat(key.aux,comm_dim_topology_string());
     dslashParam.kernel_type = kernel_type;
     return key;
   }

   long long flops() const {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     long long flops_ = dslash.flops();
     dslashParam.kernel_type = kernel_type;
     return flops_;
   }

   long long bytes() const {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     long long bytes_ = dslash.bytes();
     dslashParam.kernel_type = kernel_type;
     return bytes_;
   }

   void preTune() { dslash.preTune(); }

   void postTune() { dslash.postTune(); }

 };

} // anonymous namespace
