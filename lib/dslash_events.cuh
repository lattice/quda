namespace dslash {
  extern int it;

#ifdef PTHREADS
  extern cudaEvent_t interiorDslashEnd;
#endif
  extern cudaEvent_t packEnd[2]; // double buffered
  extern cudaEvent_t gatherStart[Nstream];
  extern cudaEvent_t gatherEnd[Nstream];
  extern cudaEvent_t scatterStart[Nstream];
  extern cudaEvent_t scatterEnd[Nstream];
  extern cudaEvent_t dslashStart[2]; // double buffered

  // FIX this is a hack from hell
  // Auxiliary work that can be done while waiting on comms to finish
  extern Worker *aux_worker;
  // Any auxiliary communication op that can be done during computational stage
  // Note that this object can be an alias of aux_worker
  extern  Worker *aux_communicator;

  extern int *commsEnd_h;
  extern CUdeviceptr commsEnd_d[Nstream];
}
