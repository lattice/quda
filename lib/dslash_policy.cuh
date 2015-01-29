static cudaColorSpinorField *inSpinor;

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
      void setThreadDimMap(DslashParam& param, DslashCuda &dslash, const int* faceVolumeCB){
        int prev = -1;

        for(int i=0; i<4; ++i){
          param.threadDimMapLower[i] = 0;
          param.threadDimMapUpper[i] = 0;
          if (!dslashParam.commDim[i]) continue;
          param.threadDimMapLower[i] = (prev >= 0 ? param.threadDimMapUpper[prev] : 0);
          param.threadDimMapUpper[i] = param.threadDimMapLower[i] + dslash.Nface()*faceVolumeCB[i];
          prev=i;
        }
      }
#endif



#define PROFILE(f, profile, idx)		\
  profile.Start(idx);				\
  f;						\
  profile.Stop(idx); 




void dslashCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		const int volume, const int *faceVolumeCB, TimeProfile &profile) {
  using namespace dslash;
  profile.Start(QUDA_PROFILE_TOTAL);

  dslashParam.parity = parity;
  dslashParam.kernel_type = INTERIOR_KERNEL;
  dslashParam.threads = volume;

#ifdef MULTI_GPU
  DslashCommsPattern pattern(dslashParam.commDim);
  // Record the start of the dslash
  PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	  profile, QUDA_PROFILE_EVENT_RECORD);

  for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
    for(int dir=1; dir>=0; dir--){
      PROFILE(face[it]->recvStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
    } 
  }


  bool pack = false;
  for (int i=3; i>=0; i--) 
    if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
      { pack = true; break; }

  // Initialize pack from source spinor

  PROFILE(face[it]->pack(*inSpinor, 1-parity, dagger, streams, false, twist_a, twist_b), 
	  profile, QUDA_PROFILE_PACK_KERNEL);

  if (pack) {
    // Record the end of the packing
    PROFILE(cudaEventRecord(packEnd[0], streams[Nstream-1]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);
  }

  for(int i = 3; i >=0; i--){
    if (!dslashParam.commDim[i]) continue;

    for (int dir=1; dir>=0; dir--) {
      cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

      PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
	      profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

      // Initialize host transfer from source spinor
      PROFILE(face[it]->gather(*inSpinor, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

      // Record the end of the gathering
      PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }
  }
#endif

  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

  int completeSum = 0;
  while (completeSum < pattern.commDimTotal) {
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {

	// Query if gather has completed
	if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) { 
	  //CUresult event_test;
	  //event_test = cuEventQuery(gatherEnd[2*i+dir]);
	  PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), 
		  profile, QUDA_PROFILE_EVENT_QUERY);

	  if (cudaSuccess == event_test) {
	    pattern.gatherCompleted[2*i+dir] = 1;
	    completeSum++;
	    PROFILE(face[it]->sendStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
	  }
	}

	// Query if comms has finished
	if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	    pattern.gatherCompleted[2*i+dir]) {
	  PROFILE(int comms_test = face[it]->commsQuery(2*i+dir), 
		  profile, QUDA_PROFILE_COMMS_QUERY);
	  if (comms_test) { 
	    pattern.commsCompleted[2*i+dir] = 1;
	    completeSum++;

	    // Scatter into the end zone
	    // Both directions use the same stream
	    PROFILE(face[it]->scatter(*inSpinor, dagger, 2*i+dir), 
		    profile, QUDA_PROFILE_SCATTER);
	  }
	}

      }

      // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
      if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	// Record the end of the scattering
	PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		profile, QUDA_PROFILE_EVENT_RECORD);

	dslashParam.kernel_type = static_cast<KernelType>(i);
	dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	// wait for scattering to finish and then launch dslash
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	// all faces use this stream
	PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	pattern.dslashCompleted[2*i] = 1;
      }

    }

  }
  it = (it^1);
#endif // MULTI_GPU

  profile.Stop(QUDA_PROFILE_TOTAL);
}

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

    profile.Start(QUDA_PROFILE_TOTAL);

  
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

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

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
	      // Both directions use the same stream
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
    profile.Stop(QUDA_PROFILE_TOTAL);
  }

};

struct DslashPthreads : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger, 
		    const int volume, const int *faceVolumeCB, TimeProfile &profile) {
#ifdef PTHREADS
    using namespace dslash;

    profile.Start(QUDA_PROFILE_TOTAL);

  
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
	      // Both directions use the same stream
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
    profile.Stop(QUDA_PROFILE_TOTAL);
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

    profile.Start(QUDA_PROFILE_TOTAL);

  
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
    profile.Stop(QUDA_PROFILE_TOTAL);
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

    profile.Start(QUDA_PROFILE_TOTAL);

  
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
    setThreadDimMap(dslashParam,dslash,faceVolumeCB);
    dslashParam.kernel_type = EXTERIOR_KERNEL_ALL;
    dslashParam.threads = 0;

    for(int i=0; i<4; ++i){
      if(!dslashParam.commDim[i]) continue;
      dslashParam.threads = dslashParam.threadDimMapUpper[i];
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
    profile.Stop(QUDA_PROFILE_TOTAL);
#else 
    errorQuda("GPU_COMMS has not been built\n");
#endif // GPU_COMMS
  }
};

struct DslashFaceBuffer : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger, 
		const int volume, const int *faceVolumeCB, TimeProfile &profile) {
  
    using namespace dslash;
    profile.Start(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    DslashCommsPattern pattern(dslashParam.commDim);
    // Record the start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);

    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(face[it]->recvStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
      } 
    }


    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
        { pack = true; break; }

    // Initialize pack from source spinor

    PROFILE(face[it]->pack(*inputSpinor, 1-parity, dagger, streams, false, twist_a, twist_b), 
	    profile, QUDA_PROFILE_PACK_KERNEL);

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[Nstream-1]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
        cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

        PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
	        profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(face[it]->gather(*inputSpinor, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering
        PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
	        profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }
#endif

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

    int completeSum = 0;
    while (completeSum < pattern.commDimTotal) {
      for (int i=3; i>=0; i--) {
        if (!dslashParam.commDim[i]) continue;

        for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) { 
	    //CUresult event_test;
	    //event_test = cuEventQuery(gatherEnd[2*i+dir]);
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), 
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      pattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(face[it]->sendStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }

	  // Query if comms has finished
	  if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	      pattern.gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = face[it]->commsQuery(2*i+dir), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      pattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;

	      // Scatter into the end zone
	      // Both directions use the same stream
	      PROFILE(face[it]->scatter(*inputSpinor, dagger, 2*i+dir), 
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }

        }

        // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
        if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	  // Record the end of the scattering
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		  profile, QUDA_PROFILE_EVENT_RECORD);

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  pattern.dslashCompleted[2*i] = 1;
        }

      }

    }
    it = (it^1);
#endif // MULTI_GPU
    profile.Stop(QUDA_PROFILE_TOTAL);
  }

};

struct DslashFusedExterior : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger, 
		   const int volume, const int *faceVolumeCB, TimeProfile &profile) {


    using namespace dslash;

    profile.Start(QUDA_PROFILE_TOTAL);

  
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    int scatterIndex = 0;
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
    
    setThreadDimMap(dslashParam,dslash,faceVolumeCB);


    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      if(!scatterIndex) scatterIndex = 2*i+1;

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

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

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
	      // Both directions use the same stream
	      PROFILE(inputSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir, streams+scatterIndex), 
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }
        } // dir=0,1
      } // i
    } // while(completeSum < commDimTotal) 


    dslashParam.kernel_type = EXTERIOR_KERNEL_ALL;
    dslashParam.threads = 0;

    for(int i=0; i<4; ++i){
      if(!dslashParam.commDim[i]) continue;
      dslashParam.threads = dslashParam.threadDimMapUpper[i];
    }

    PROFILE(cudaEventRecord(scatterEnd[0], streams[scatterIndex]), 
      profile, QUDA_PROFILE_EVENT_RECORD);

    PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[0], 0),
      profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    if (pattern.commDimTotal) {
      PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
    }

    inputSpinor->bufferIndex = (1 - inputSpinor->bufferIndex);
#endif // MULTI_GPU


    profile.Stop(QUDA_PROFILE_TOTAL);
  }
};

struct DslashNC : DslashPolicyImp {

  void operator()(DslashCuda &dslash, cudaColorSpinorField* inputSpinor, const size_t regSize, const int parity, const int dagger, 
		    const int volume, const int *faceVolumeCB, TimeProfile &profile) {

    profile.Start(QUDA_PROFILE_TOTAL);
    
    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

    profile.Stop(QUDA_PROFILE_TOTAL);
  }

};

struct DslashFactory {

  static DslashPolicyImp* create(const QudaDslashPolicy &dslashPolicy)
  {

    DslashPolicyImp* result = NULL;    

    switch(dslashPolicy){
    case QUDA_DSLASH:
      result = new DslashFaceBuffer;
      break;
    case QUDA_DSLASH2:
      result = new DslashCuda2;
      break;
    case QUDA_PTHREADS_DSLASH:
      result = new DslashPthreads;
      break;
    case QUDA_FUSED_DSLASH:
      result = new DslashFusedExterior;
      break;
    case QUDA_GPU_COMMS_DSLASH:
      result = new DslashGPUComms;
      break;
    case QUDA_DSLASH_NC:
      result = new DslashNC;
      break;
    default:
      errorQuda("Dslash policy %d not recognized",dslashPolicy);
      result = new DslashFaceBuffer;
      break;
    }
    return result; // default 
  }
};

} // anonymous namespace

void dslashCuda2(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		 const int volume, const int *faceVolumeCB, TimeProfile &profile) {

  using namespace dslash;

  profile.Start(QUDA_PROFILE_TOTAL);

  
  dslashParam.parity = parity;
  dslashParam.kernel_type = INTERIOR_KERNEL;
  dslashParam.threads = volume;

#ifdef MULTI_GPU
  // Record the start of the dslash if doing communication in T and not kernel packing
#ifndef PTHREADS
  if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) 
#endif
    {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
              profile, QUDA_PROFILE_EVENT_RECORD);
    }
		
  inSpinor->allocateGhostBuffer(dslash.Nface()/2);
  inSpinor->createComms(dslash.Nface()/2);	
  DslashCommsPattern pattern(dslashParam.commDim);
  inSpinor->streamInit(streams);
#ifdef PTHREADS // create two new threads to issue MPI receives 
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
#else // single CPU thread per MPI process
  const int packIndex = Nstream-1;
  for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
    for(int dir=1; dir>=0; dir--){
      PROFILE(inSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
    }
  }
#endif
  bool pack = false;
  for (int i=3; i>=0; i--) 
    if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
      { pack = true; break; }

#ifdef PTHREADS
  if (pack){
    PROFILE(cudaStreamWaitEvent(streams[packIndex], dslashStart, 0), 
            profile, QUDA_PROFILE_STREAM_WAIT_EVENT); 
  }
#endif

  // Initialize pack from source spinor
  PROFILE(inSpinor->pack(dslash.Nface()/2, 1-parity, dagger, packIndex, false, twist_a, twist_b),
	  profile, QUDA_PROFILE_PACK_KERNEL);

  if (pack) {
    // Record the end of the packing
    PROFILE(cudaEventRecord(packEnd[0], streams[packIndex]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);
  }
#ifndef GPU_COMMS
  for(int i = 3; i >=0; i--){
    if (!dslashParam.commDim[i]) continue;

    for (int dir=1; dir>=0; dir--) {
      cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

      PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
	      profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

      // Initialize host transfer from source spinor
      PROFILE(inSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

      // Record the end of the gathering
      PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }
  }
#endif // GPU_COMMS

#endif // MULTI_GPU

#if (!defined MULTI_GPU) || (!defined PTHREADS)
  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);
#endif

#ifdef MULTI_GPU 

#ifdef PTHREADS
  if(pthread_join(receiveThread, NULL)) errorQuda("pthread_join failed");
#endif
#ifdef GPU_COMMS
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
      PROFILE(inSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      inSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
    }
  }
#endif
#ifdef PTHREADS
  bool interiorLaunched = false;
#endif
  int completeSum = 0;
  while (completeSum < pattern.commDimTotal) {
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {

#ifndef GPU_COMMS
	// Query if gather has completed
	if (!pattern.gatherCompleted[2*i+dir] && pattern.gatherCompleted[pattern.previousDir[2*i+dir]]) { 
	  PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), 
		  profile, QUDA_PROFILE_EVENT_QUERY);

	  if (cudaSuccess == event_test) {
	    pattern.gatherCompleted[2*i+dir] = 1;
	    completeSum++;
	    PROFILE(inSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	  }
	}
#endif

	// Query if comms has finished
	if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]] &&
	    pattern.gatherCompleted[2*i+dir]) {
	  PROFILE(int comms_test = inSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		  profile, QUDA_PROFILE_COMMS_QUERY);
	  if (comms_test) { 
	    pattern.commsCompleted[2*i+dir] = 1;
	    completeSum++;

	    // Scatter into the end zone
	    // Both directions use the same stream
#ifndef GPU_COMMS
	    PROFILE(inSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), 
		    profile, QUDA_PROFILE_SCATTER);
#endif
	  }
	}

      } // dir=0,1

      // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
      if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	// Record the end of the scattering
#ifndef GPU_COMMS
	PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		profile, QUDA_PROFILE_EVENT_RECORD);
#ifdef PTHREADS  
	if(!interiorLaunched){
	  if(pthread_join(interiorThread, NULL)) errorQuda("pthread_join failed");
	  interiorLaunched = true;
	}
#endif

	// wait for scattering to finish and then launch dslash
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
#endif

	dslashParam.kernel_type = static_cast<KernelType>(i);
	dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	// all faces use this stream
	PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	pattern.dslashCompleted[2*i] = 1;
      }

    }

  }
  inSpinor->bufferIndex = (1 - inSpinor->bufferIndex);
  //	inSpinor->switchBufferPinned(); // Use a different pinned memory buffer for the next application
#endif // MULTI_GPU
  profile.Stop(QUDA_PROFILE_TOTAL);
}

/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory
*/
void dslashZeroCopyCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
			const int volume, const int *faceVolumeCB, TimeProfile &profile) {
  using namespace dslash;

  profile.Start(QUDA_PROFILE_TOTAL);

  dslashParam.parity = parity;
  dslashParam.kernel_type = INTERIOR_KERNEL;
  dslashParam.threads = volume;

#ifdef MULTI_GPU
  DslashCommsPattern pattern(dslashParam.commDim);
  for(int i=3; i>=0; i--){
    if(!dslashParam.commDim[i]) continue;
    for(int dir=1; dir>=0; dir--){
      PROFILE(face[it]->recvStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);    
    }
  }


  setKernelPackT(true);

  // Record the end of the packing
  PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	  profile, QUDA_PROFILE_EVENT_RECORD);

  PROFILE(cudaStreamWaitEvent(streams[0], dslashStart, 0), 
	  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

  // Initialize pack from source spinor
  PROFILE(face[it]->pack(*inSpinor, 1-parity, dagger, streams, true, twist_a, twist_b), 
	  profile, QUDA_PROFILE_PACK_KERNEL);

  // Record the end of the packing
  PROFILE(cudaEventRecord(packEnd[0], streams[0]), 
	  profile, QUDA_PROFILE_EVENT_RECORD);
#endif

  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

  int doda=0;
  while (doda++>=0) {
    PROFILE(cudaError_t event_test = cudaEventQuery(packEnd[0]), 
	    profile, QUDA_PROFILE_EVENT_QUERY);
    if (event_test == cudaSuccess) doda=-1;
  }

  for (int i=3; i>=0; i--) {
    if (!dslashParam.commDim[i]) continue;
    for (int dir=1; dir>=0; dir--) {
      PROFILE(face[it]->sendStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);    
    }
  }


  int completeSum = 0;
  pattern.commDimTotal /= 2; // pipe is shorter for zero-variant

  while (completeSum < pattern.commDimTotal) {

    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {

	// Query if comms have finished
	if (!pattern.commsCompleted[2*i+dir] && pattern.commsCompleted[pattern.previousDir[2*i+dir]]) {
	  PROFILE(int comms_test = face[it]->commsQuery(2*i+dir), 
		  profile, QUDA_PROFILE_COMMS_QUERY);
	  if (comms_test) { 
	    pattern.commsCompleted[2*i+dir] = 1;
	    completeSum++;

	    // Scatter into the end zone
	    // Both directions use the same stream
	    PROFILE(face[it]->scatter(*inSpinor, dagger, 2*i+dir), 
		    profile, QUDA_PROFILE_SCATTER);
	  }
	}

      }

      // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
      if (!pattern.dslashCompleted[2*i] && pattern.commsCompleted[2*i] && pattern.commsCompleted[2*i+1] ) {
	// Record the end of the scattering
	PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		profile, QUDA_PROFILE_EVENT_RECORD);

	dslashParam.kernel_type = static_cast<KernelType>(i);
	dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

	// wait for scattering to finish and then launch dslash
	PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	// all faces use this stream
	PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	pattern.dslashCompleted[2*i] = 1;
      }

    }

  }
  it = (it^1);
#endif // MULTI_GPU

  profile.Stop(QUDA_PROFILE_TOTAL);
}

void dslashCudaNC(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
  profile.Start(QUDA_PROFILE_TOTAL);

  dslashParam.parity = parity;
  dslashParam.kernel_type = INTERIOR_KERNEL;
  dslashParam.threads = volume;

  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

  profile.Stop(QUDA_PROFILE_TOTAL);
}
