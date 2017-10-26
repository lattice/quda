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

  DslashCommsPattern(const int commDim[], bool gdr=false) {

    for (int i=0; i<Nstream-1; i++) {
      gatherCompleted[i] = gdr ? 1 : 0;
      commsCompleted[i] = 0;
      dslashCompleted[i] = 0;
    }
    gatherCompleted[Nstream-1] = 1;
    commsCompleted[Nstream-1] = 1;

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
    commDimTotal *= gdr ? 2 : 4; // 2 from pipe length, 2 from direction
  }
};


#ifdef MULTI_GPU
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
#endif

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
    return NULL;
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
    return NULL;
  }

} // anonymous namespace
#endif


namespace{

struct DslashPolicyImp {

  virtual void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, 
                        const size_t regSize, const int parity, const int dagger,
                        const int volume, const int *faceVolumeCB, TimeProfile &profile) = 0;

  virtual ~DslashPolicyImp(){}
};


struct DslashBasic : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    DslashCommsPattern pattern(dslashParam.commDim);

    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--) {
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;
	for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_SCATTER);
	    }
	  }

        } // dir=0,1

	// if peer-2-peer in a given direction then we need only wait on that copy event to finish
	// else we post an event in the scatter stream and wait on that
        if ( (i==3 || pattern.dslashCompleted[2*(i+1)]) && !pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  if (comm_peer2peer_enabled(0,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+1], streams[2*i+1]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+1], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  }

	  if (comm_peer2peer_enabled(1,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+0], streams[2*i+0]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
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
#endif // MULTI_GPU
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
		
    DslashCommsPattern pattern(dslashParam.commDim);
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
    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) { 
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]),
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
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
	      completeSum++;

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

struct DslashGPUComms : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    DslashCommsPattern pattern(dslashParam.commDim, true);

    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--) {
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger, 0, true), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

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
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	} // dir
      } // i
    } // p2p

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] && pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // if peer-2-peer in a given direction then we need to insert a wait on that copy event
	      if (comm_peer2peer_enabled(dir,i)) {
		PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(dir,i), 0),
			profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	      }
	    }
	  }

        } // dir=0,1

	if ( (i==3 || pattern.dslashCompleted[2*(i+1)]) && !pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
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
    setKernelPackT(kernel_pack_old);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};


struct DslashFusedGPUComms : DslashPolicyImp {
  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;
    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU

    DslashCommsPattern pattern(dslashParam.commDim, true);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if (!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger, 0, true), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

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
	    if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true); // do a comms query to ensure MPI has begun
	  } // is p2p?
	}
      }
    } // p2p

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger, 0, true) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // if peer-2-peer in a given direction then we need to insert a wait on that copy event
	      if (comm_peer2peer_enabled(dir,i)) {
		PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(dir,i), 0),
			profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	      }
	    }
	  }
	} // dir=0,1
      } // i
    } // completeSum < pattern.CommDimTotal

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU
    setKernelPackT(kernel_pack_old);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};

struct DslashFusedExterior : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    int scatterIndex = -1;
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT())
    {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }


    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }
    
    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      // set scatterIndex to a stream not being used for p2p
      if (!comm_peer2peer_enabled(0,i)) scatterIndex = 2*i+0;
      else if (!comm_peer2peer_enabled(1,i)) scatterIndex = 2*i+1;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) { 
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // All directions use the same stream (streams[scatterIndex])
	      PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex),
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }
        } // dir=0,1
      } // i
    } // while(completeSum < commDimTotal) 

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // if peer-2-peer in a given direction then we need to wait on that copy event
    // if any comms is not peer-2-peer then we need to post a scatter event and wait on that
    bool post_scatter_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      if (comm_peer2peer_enabled(0,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (comm_peer2peer_enabled(1,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) post_scatter_event = true;
    }

    if (post_scatter_event) {
      PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    }

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
};


#ifdef HOST_DEBUG
#define CUDA_CALL( call )						\
  {									\
    CUresult cudaStatus = call;						\
    if ( CUDA_SUCCESS != cudaStatus ) {					\
      const char *err_str = NULL;					\
      cuGetErrorString(cudaStatus, &err_str);				\
      fprintf(stderr, "ERROR: CUDA call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, err_str, cudaStatus); \
    }									\
}
#else
#define CUDA_CALL( call ) call
#endif

struct DslashAsync : DslashPolicyImp {

#if (CUDA_VERSION >= 8000)

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    bool pack = false;
    for (int i=3; i>=0; i--)
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) {
		*((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[2*i+dir], commsEnd_d[2*i+dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, &streams[2*i+dir]), profile, QUDA_PROFILE_SCATTER);
	      }
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	      // this will trigger scatter
	      if (!comm_peer2peer_enabled(1-dir,i)) *((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 1;
	    }
	  }

        } // dir=0,1

	// if peer-2-peer in a given direction then we need only wait on that copy event to finish
	// else we post an event in the scatter stream and wait on that
	if ( (i==3 || pattern.dslashCompleted[2*(i+1)]) && !pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  if (comm_peer2peer_enabled(0,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+1], streams[2*i+1]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+1], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  }

	  if (comm_peer2peer_enabled(1,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+0], streams[2*i+0]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
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
#endif // MULTI_GPU
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }
#else

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    errorQuda("Async dslash policy variants require CUDA 8.0 and above");
  }

#endif // CUDA_VERSION >= 8000

};


struct DslashFusedExteriorAsync : DslashPolicyImp {

#if (CUDA_VERSION >= 8000)

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    int scatterIndex = -1;
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !getKernelPackT()) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--)
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT()))
        { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int i=0; i<2*QUDA_MAX_DIM; i++) pack_dest[i] = Device;
    PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      // set scatterIndex to a stream not being used for p2p
      if (!comm_peer2peer_enabled(0,i)) scatterIndex = 2*i+0;
      else if (!comm_peer2peer_enabled(1,i)) scatterIndex = 2*i+1;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]),
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(if (dslash_comms) inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	      if (dslash_comms) inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger);  // do a comms query to ensure MPI has begun

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) {
		*((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[scatterIndex], commsEnd_d[2*i+dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex), profile, QUDA_PROFILE_SCATTER);
	      }
	    }

	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	      // this will trigger scatter
	      if (!comm_peer2peer_enabled(1-dir,i)) *((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 1;
	    }
	  }
        } // dir=0,1
      } // i
    } // while(completeSum < commDimTotal)

    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // if peer-2-peer in a given direction then we need to wait on that copy event
    // if any comms is not peer-2-peer then we need to post a scatter event and wait on that
    bool post_scatter_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      if (comm_peer2peer_enabled(0,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (comm_peer2peer_enabled(1,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) post_scatter_event = true;
    }

    if (post_scatter_event) {
      PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    }

    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

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

    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->streamInit(streams);

    DslashCommsPattern pattern(dslashParam.commDim);

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    const int packIndex = 0;
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0),
	    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    bool pack = false;
    for (int i=3; i>=0; i--) if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	pack_dest[2*dim+dir] = comm_peer2peer_enabled(dir,dim) ? Device : Host;
    if (pack) PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

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

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_SCATTER);
	    }
	  }

	}

	// if peer-2-peer in a given direction then we need only wait on that copy event to finish
	// else we post an event in the scatter stream and wait on that
	if ( (i==3 || pattern.dslashCompleted[2*(i+1)]) && !pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  if (comm_peer2peer_enabled(0,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+1], streams[2*i+1]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+1], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  }

	  if (comm_peer2peer_enabled(1,i)) {
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  } else {
	    // Record the end of the scattering
	    PROFILE(cudaEventRecord(scatterEnd[2*i+0], streams[2*i+0]), profile, QUDA_PROFILE_EVENT_RECORD);
	    // wait for scattering to finish and then launch dslash
	    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
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
#endif // MULTI_GPU

    setKernelPackT(kernel_pack_old); // reset kernel packing
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

    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->streamInit(streams);

    DslashCommsPattern pattern(dslashParam.commDim);

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    const int packIndex = 0;
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0),
	    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    bool pack = false;
    for (int i=3; i>=0; i--) if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	pack_dest[2*dim+dir] = comm_peer2peer_enabled(dir,dim) ? Device : Host;
    if (pack) PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

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

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    const int scatterIndex = packIndex;

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone (all use same stream)
	      PROFILE(if (dslash_copy) inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex),
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }
	} // dir=0,1
      } // i
    } // completeSum

    // setup for exterior kernel
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // if peer-2-peer in a given direction then we need to wait on that copy event
    // if any comms is not peer-2-peer then we need to post a scatter event and wait on that
    bool post_scatter_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      if (comm_peer2peer_enabled(0,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(0,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (comm_peer2peer_enabled(1,i)) {
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], inputSpinor->getIPCRemoteCopyEvent(1,i), 0),
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
      }
      if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) post_scatter_event = true;
    }

    if (post_scatter_event) {
      PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
    }

    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    setKernelPackT(kernel_pack_old); // reset kernel packing
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

    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->streamInit(streams);

    DslashCommsPattern pattern(dslashParam.commDim);

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    const int packIndex = 0;
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0),
	    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    bool pack = false;
    for (int i=3; i>=0; i--) if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	pack_dest[2*dim+dir] = comm_peer2peer_enabled(dir,dim) ? Device : Host;
    if (pack) PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

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

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    }
	  }

	}

	// FIXME - will not work with P2P until we can split where the halos originate
	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
	if ( (i==3 || pattern.dslashCompleted[2*(i+1)]) && !pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // in the below we switch to the mapped ghost buffer and update the tuneKey to reflect this
	  inputSpinor->bufferIndex += 2;
#ifdef USE_TEXTURE_OBJECTS
	  dslashParam.ghostTex = inputSpinor->GhostTex();
	  dslashParam.ghostTexNorm = inputSpinor->GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
	  char aux_copy[TuneKey::aux_n];
	  strcpy(aux_copy,dslash.getAux(dslashParam.kernel_type));
	  dslash.augmentAux(dslashParam.kernel_type, ",zero_copy");

	  PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  // reset to default
	  dslash.setAux(dslashParam.kernel_type, aux_copy);
	  inputSpinor->bufferIndex -= 2;
#ifdef USE_TEXTURE_OBJECTS
	  dslashParam.ghostTex = inputSpinor->GhostTex();
	  dslashParam.ghostTexNorm = inputSpinor->GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
	  pattern.dslashCompleted[2*i] = 1;
	}

      }

    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    setKernelPackT(kernel_pack_old); // reset kernel packing
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
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

    bool kernel_pack_old = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->streamInit(streams);

    DslashCommsPattern pattern(dslashParam.commDim);

    // record start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);

    const int packIndex = 0;
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0),
	    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    bool pack = false;
    for (int i=3; i>=0; i--) if (dslashParam.commDim[i]) { pack = true; break; }

    // Initialize pack from source spinor
    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	pack_dest[2*dim+dir] = comm_peer2peer_enabled(dir,dim) ? Device : Host;
    if (pack) PROFILE(if (dslash_pack_compute) inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, pack_dest, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(if (dslash_comms) inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(if (dslash_interior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

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

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir]) {
	    PROFILE(int comms_test = dslash_comms ? inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger) : 1,
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    }
	  }

	}

      }

    }

    // FIXME - will not work with P2P until we can split where the halos originate
    if (pattern.commDimTotal) {
      // setup for exterior kernel
      setFusedParam(dslashParam,dslash,faceVolumeCB);

      // in the below we switch to the mapped ghost buffer and update the tuneKey to reflect this
      inputSpinor->bufferIndex += 2;
#ifdef USE_TEXTURE_OBJECTS
      dslashParam.ghostTex = inputSpinor->GhostTex();
      dslashParam.ghostTexNorm = inputSpinor->GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
      char aux_copy[TuneKey::aux_n];
      strcpy(aux_copy,dslash.getAux(dslashParam.kernel_type));
      dslash.augmentAux(dslashParam.kernel_type, ",zero_copy");

      PROFILE(if (dslash_exterior_compute) dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

      // reset to default
      dslash.setAux(dslashParam.kernel_type, aux_copy);
      inputSpinor->bufferIndex -= 2;
#ifdef USE_TEXTURE_OBJECTS
      dslashParam.ghostTex = inputSpinor->GhostTex();
      dslashParam.ghostTexNorm = inputSpinor->GhostTexNorm();
#endif // USE_TEXTURE_OBJECTS
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    setKernelPackT(kernel_pack_old); // reset kernel packing
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
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

struct DslashFactory {

  static DslashPolicyImp* create(const QudaDslashPolicy &dslashPolicy)
  {

    DslashPolicyImp* result = NULL;    

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
    case QUDA_GPU_COMMS_DSLASH:
      result = new DslashGPUComms;
      break;
    case QUDA_FUSED_GPU_COMMS_DSLASH:
      result = new DslashFusedGPUComms;
      break;
    case QUDA_ZERO_COPY_DSLASH_PACK:
      result = new DslashZeroCopyPack;
      break;
    case QUDA_FUSED_ZERO_COPY_DSLASH_PACK:
      result = new DslashFusedZeroCopyPack;
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
	   if ( (dslash_policy == QUDA_GPU_COMMS_DSLASH || dslash_policy == QUDA_FUSED_GPU_COMMS_DSLASH) && !comm_gdr_enabled() ) {
	     errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", dslash_policy);
	   }

	   if ((dslash_policy == QUDA_ZERO_COPY_DSLASH || dslash_policy == QUDA_FUSED_ZERO_COPY_DSLASH) && comm_peer2peer_enabled_global()) {
	     errorQuda("Cannot select a zero-copy dslash policy %d unless no peer-to-peer devices are present or peer-to-peer is disabled", dslash_policy);
	   }

	   policy.push_back(static_cast<QudaDslashPolicy>(policy_));
	   if (policy_list.peek() == ',') policy_list.ignore();
	 }
       } else {
	 policy.push_back(QUDA_DSLASH);
	 policy.push_back(QUDA_FUSED_DSLASH);

	 // if we have gdr then enable tuning these policies
	 if (comm_gdr_enabled()) {
	   policy.push_back(QUDA_GPU_COMMS_DSLASH);
	   policy.push_back(QUDA_FUSED_GPU_COMMS_DSLASH);
	   config+=2;
	 }

	 policy.push_back(QUDA_ZERO_COPY_DSLASH_PACK);
	 policy.push_back(QUDA_FUSED_ZERO_COPY_DSLASH_PACK);

	 // if we have p2p for now exclude zero-copy dslash halo reads
	 // since we can't mix these two until we can source halos
	 // from separate memory spaces (requires ghost texture per dim/dir?)
	 bool p2p = comm_peer2peer_enabled_global();
	 config+=p2p;

	 if (!p2p && 0) {
	   policy.push_back(QUDA_ZERO_COPY_DSLASH);
	   policy.push_back(QUDA_FUSED_ZERO_COPY_DSLASH);
	 }

	 // Async variants are only supported on CUDA 8.0 and are buggy  - so exclude for now
#if (CUDA_VERSION >= 8000) && 0
	 policy.push_back(QUDA_DSLASH_ASYNC);
	 policy.push_back(QUDA_FUSED_DSLASH_ASYNC);
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
	 DslashPolicyImp* dslashImp = DslashFactory::create(i);
	 (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	 delete dslashImp;
       }

       enableProfileCount();
       setPolicyTuning(true);
     }
     dslash_init = true;
   }

   virtual ~DslashPolicyTune() { setPolicyTuning(false); }

   void apply(const cudaStream_t &stream) {
     TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_DEBUG_VERBOSE /*getVerbosity()*/);

     if (config != tp.aux.y) {
       errorQuda("Machine configuration (P2P/GDR=%d) changed since tunecache was created (P2P/GDR=%d).  Please delete "
		 "this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.",
		 config, tp.aux.y);
     }

     if (tp.aux.x >= (int)policy.size()) errorQuda("Requested policy that is outside of range");

     DslashPolicyImp* dslashImp = DslashFactory::create(policy[tp.aux.x]);
     (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
     delete dslashImp;
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
