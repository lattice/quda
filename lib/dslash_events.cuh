namespace dslash {
  extern int it;

  extern cudaEvent_t packEnd[Nstream];
  extern cudaEvent_t gatherStart[Nstream];
  extern cudaEvent_t gatherEnd[Nstream];
  extern cudaEvent_t scatterStart[Nstream];
  extern cudaEvent_t scatterEnd[Nstream];
  extern cudaEvent_t dslashStart;
  extern cudaEvent_t dslashEnd;

#ifdef PTHREADS
  extern volatile CommsParam *commsParam;
  extern volatile bool sleepCommsThread;
  extern volatile bool killCommsThread;

#if 0
#include <pthread.h>

  /*
    The interior kernel launch thread isn't used anywhere
  */
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

  { // example code for how to use it
    // set the parameters for the InteriorParam
    InteriorParam interiorParam;
    interiorParam.dslash   = &dslash;
    interiorParam.profile  = &profile;   
    cudaGetDevice(&(interiorParam.current_device)); // get the current device number
    
    // create thread
    bool interiorLaunched = false;
    if(pthread_create(&interiorThread, NULL, launchInteriorKernel, &interiorParam)){
      errorQuda("pthread_create failed");
    }
    
    // reconverge
    if(!interiorLaunched){
      if(pthread_join(interiorThread, NULL)) errorQuda("pthread_join failed");
      interiorLaunched = true;
    }
  }
#endif // 0

#endif // PTHREADS

}
