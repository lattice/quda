static cudaColorSpinorField *inSpinor;

// hooks into tune.cpp variables for policy tuning
typedef std::map<TuneKey, TuneParam> map;
const map& getTuneCache();

void disableProfileCount();
void enableProfileCount();

void setPolicyTuning(bool);

/**
 * Arrays used for the dynamic scheduling.
 */
struct DslashCommsPattern {
  int gatherCompleted[Nstream];
  int previousDir[Nstream];
  int commsCompleted[Nstream];
  int dslashCompleted[Nstream];
  int commDimTotal;

  DslashCommsPattern(const int commDim[]) {

    for (int i=0; i<Nstream-1; i++) {
#ifndef GPU_COMMS
      gatherCompleted[i] = 0;
#else
      gatherCompleted[i] = 1;      
#endif
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
#ifndef GPU_COMMS
    commDimTotal *= 4; // 2 from pipe length, 2 from direction
#else
    commDimTotal *= 2; // 2 from pipe length, 2 from direction
#endif
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


struct DslashCuda2 : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);
  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
              profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);	

    DslashCommsPattern pattern(dslashParam.commDim);

    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }

    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
	      PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger),
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_SCATTER);
	    }
	  }

        } // dir=0,1

	// if peer-2-peer in a given direction then we need only wait on that copy event to finish
	// else we post an event in the scatter stream and wait on that
        if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
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
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

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
		
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);	
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
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }

    if (pack){
      PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), 
              profile, QUDA_PROFILE_STREAM_WAIT_EVENT); 
    }

    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }
    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
	        profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
	        profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

#if (!defined MULTI_GPU)
    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
	      PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }

	  // Query if comms has finished
	  if(!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // Both directions use the same stream // FIXME
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), 
		      profile, QUDA_PROFILE_SCATTER);
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
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

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

#ifdef GPU_COMMS
    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);

  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) 
      {
        PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
                profile, QUDA_PROFILE_EVENT_RECORD);
      }
		
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);	
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }

    // Initialize pack from source spinor
      PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	      profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

    bool pack_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      if ((i!=3 || getKernelPackT() || getTwistPack()) && !pack_event) {
        cudaEventSynchronize(packEnd[0]);
        pack_event = true;
      } else {
        cudaEventSynchronize(dslashStart);
      }

      for (int dir=1; dir>=0; dir--) {	
        PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
        inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
      }
    }
    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {


	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // Both directions use the same stream
	    }
	  }

        } // dir=0,1

        // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
        if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  // Record the end of the scattering

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }
      }
    }
    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
#else 
    errorQuda("GPU_COMMS has not been built\n");
#endif // GPU_COMMS
  }
};


struct DslashFusedGPUComms : DslashPolicyImp {
  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

#ifdef GPU_COMMS
    using namespace dslash;

    profile.TPSTART(QUDA_PROFILE_TOTAL);

  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) 
      {
        PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
                profile, QUDA_PROFILE_EVENT_RECORD);
      }
		
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);	
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }

    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU 

    bool pack_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      if ((i!=3 || getKernelPackT() || getTwistPack()) && !pack_event) {
        cudaEventSynchronize(packEnd[0]);
        pack_event = true;
      } else {
        cudaEventSynchronize(dslashStart);
      }

      for (int dir=1; dir>=0; dir--) {	
        PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
        inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
      }
    }

   
    // setup for exterior kernel 
    setFusedParam(dslashParam,dslash,faceVolumeCB);

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    }
	  }
        } // dir=0,1
      } // i
    } // completeSum < pattern.CommDimTotal


    // Launch exterior kernel
    if (pattern.commDimTotal) {
      PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }


    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
#else 
    errorQuda("GPU_COMMS has not been built\n");
#endif // GPU_COMMS
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
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) 
    {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }


    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
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
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
	      PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // Both directions use the same stream (streams[scatterIndex])
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex), 
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }
        } // dir=0,1
      } // i
    } // while(completeSum < commDimTotal) 

    setFusedParam(dslashParam,dslash,faceVolumeCB);

    // if peer-2-peer in a given direction then we need to to wait on that copy event
    // if any comms is not peer-2-peer then we need to post a scatter event and wait on that
    bool post_scatter_event = false;
    for (int i=3; i>=0; i--) {
      if (!commDimPartitioned(i)) continue;
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
      PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    bool pack = false;
    for (int i=3; i>=0; i--)
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack()))
        { pack = true; break; }

    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
	      PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) {
		*((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[2*i+dir], commsEnd_d[2*i+dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, &streams[2*i+dir]), profile, QUDA_PROFILE_SCATTER);
	      }
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger),
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
        if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
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
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

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
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), profile, QUDA_PROFILE_EVENT_RECORD);
    }

    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);
    DslashCommsPattern pattern(dslashParam.commDim);
    inputSpinor->streamInit(streams);
    const int packIndex = Nstream-1;
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--)
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack()))
        { pack = true; break; }

    // Initialize pack from source spinor
    PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
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
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(inputSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
	      PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);

	      // schedule post comms work (scatter into the end zone)
	      if (!comm_peer2peer_enabled(1-dir,i)) {
		*((volatile cuuint32_t*)(commsEnd_h+2*i+dir)) = 0;
		CUDA_CALL(cuStreamWaitValue32( streams[scatterIndex], commsEnd_d[2*i+dir], 1, CU_STREAM_WAIT_VALUE_EQ ));
		PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex), profile, QUDA_PROFILE_SCATTER);
	      }
	    }

	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger),
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

    // if peer-2-peer in a given direction then we need to to wait on that copy event
    // if any comms is not peer-2-peer then we need to post a scatter event and wait on that
    bool post_scatter_event = false;
    for (int i=3; i>=0; i--) {
      if (!commDimPartitioned(i)) continue;
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
      PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
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
struct DslashZeroCopy : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField *inputSpinor, const size_t regSize, const int parity, const int dagger,
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    using namespace dslash;

    bool old_pack = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);
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
    if (pack) PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, true, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

    cudaStreamSynchronize(streams[packIndex]);

    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      for (int dir=1; dir>=0; dir--) {
        PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger),
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // Both directions use the same stream
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_SCATTER);
	    }
	  }

	}

	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
	if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  // Record the end of the scattering
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), profile, QUDA_PROFILE_EVENT_RECORD);

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
	}

      }

    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    setKernelPackT(old_pack); // reset kernel packing
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

    bool old_pack = getKernelPackT();
    setKernelPackT(true);

    profile.TPSTART(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    inputSpinor->allocateGhostBuffer(dslash.Nface()/2);
    inputSpinor->createComms(dslash.Nface()/2);
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
    if (pack) PROFILE(inputSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, true, twist_a, twist_b),
		      profile, QUDA_PROFILE_PACK_KERNEL);

    // Prepost receives
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inputSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
#endif

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    if (aux_worker) aux_worker->apply(streams[Nstream-1]);

#ifdef MULTI_GPU

    cudaStreamSynchronize(streams[packIndex]);

    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      for (int dir=1; dir>=0; dir--) {
        PROFILE(inputSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }

    int completeSum = 0;
    pattern.commDimTotal /= 2; // pipe is shorter for zero-copy variant

    const int scatterIndex = packIndex;

    while (completeSum < pattern.commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;

	for (int dir=1; dir>=0; dir--) {

	  // Query if comms have finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]]) {
	    PROFILE(int comms_test = inputSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger),
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) {
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone (all use same stream)
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex),
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }

	}

      }

    }

    if (pattern.commDimTotal) {
      // setup for exterior kernel
      setFusedParam(dslashParam,dslash,faceVolumeCB);

      PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), profile, QUDA_PROFILE_EVENT_RECORD);
      PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

      PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU

    setKernelPackT(old_pack); // reset kernel packing
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

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

    profile.TPSTOP(QUDA_PROFILE_TOTAL);
  }

};

struct DslashFactory {

  static DslashPolicyImp* create(const QudaDslashPolicy &dslashPolicy)
  {

    DslashPolicyImp* result = NULL;    

    switch(dslashPolicy){
    case QUDA_DSLASH2:
      result = new DslashCuda2;
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


// which policies are we going to tune over?
#if (CUDA_VERSION >= 8000) && 0
// Async variants are only supported on CUDA 8.0
 static constexpr int n_policy = 6;
#else
 static constexpr int n_policy = 4;
#endif

 // list of policies to tune over
 static constexpr QudaDslashPolicy policy[] = { QUDA_DSLASH2,
						QUDA_FUSED_DSLASH,
						QUDA_DSLASH_ASYNC,
						QUDA_FUSED_DSLASH_ASYNC,
						QUDA_ZERO_COPY_DSLASH,
						QUDA_FUSED_ZERO_COPY_DSLASH };

 class DslashPolicyTune : public Tunable {

   DslashCuda &dslash;
   cudaColorSpinorField *in;
   const size_t regSize;
   const int parity;
   const int dagger;
   const int volume;
   const int *ghostFace;
   TimeProfile &profile;
   int n_policy_;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

 public:
   DslashPolicyTune(DslashCuda &dslash, cudaColorSpinorField *in, const size_t regSize, const int parity,
		    const int dagger, const int volume, const int *ghostFace, TimeProfile &profile)
     : dslash(dslash), in(in), regSize(regSize), parity(parity), dagger(dagger),
       volume(volume), ghostFace(ghostFace), profile(profile)
   {
     // if we have p2p for now exclude zero-copy dslash since we can't mix these two
     bool p2p = false;
     for (int dim=0; dim<4; dim++) for (int dir=0; dir<2; dir++)
       p2p = p2p || comm_peer2peer_enabled(dir,dim);
     n_policy_ = p2p ? n_policy - 2 : n_policy;
     n_policy_ = n_policy-2; // FIXME: don't enable enable zero-copy variants (broken for multi-dimensional partitioning

     // before we do policy tuning we must ensure the kernel
     // constituents have been tuned since we can't do nested tuning
     if (getTuning() && getTuneCache().find(tuneKey()) == getTuneCache().end()) {
       disableProfileCount();

       for (int i=0; i<n_policy_; i++) {
	 DslashPolicyImp* dslashImp = DslashFactory::create(policy[i]);
	 (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
	 delete dslashImp;
       }

       enableProfileCount();
       setPolicyTuning(true);
     }
   }

   virtual ~DslashPolicyTune() { setPolicyTuning(false); }

   void apply(const cudaStream_t &stream) {
     TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

     DslashPolicyImp* dslashImp = DslashFactory::create(policy[tp.aux.x]);
     (*dslashImp)(dslash, in, regSize, parity, dagger, volume, ghostFace, profile);
     delete dslashImp;
   }

   int tuningIter() const { return 10; }

   // Find the best dslash policy
   bool advanceAux(TuneParam &param) const
   {
     if (param.aux.x < n_policy_-1) {
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
     param.aux.x = 0; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
   }

   void defaultTuneParam(TuneParam &param) const  {
     Tunable::defaultTuneParam(param);
     param.aux.x = 0; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
   }

   TuneKey tuneKey() const {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     TuneKey key = dslash.tuneKey();
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
