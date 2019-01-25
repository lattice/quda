namespace dslash {
  extern int it;

  extern cudaEvent_t packEnd[2]; // double buffered
  extern cudaEvent_t gatherStart[Nstream];
  extern cudaEvent_t gatherEnd[Nstream];
  extern cudaEvent_t scatterStart[Nstream];
  extern cudaEvent_t scatterEnd[Nstream];
  extern cudaEvent_t dslashStart[2]; // double buffered

  // FIX this is a hack from hell
  // Auxiliary work that can be done while waiting on comms to finis
  extern Worker *aux_worker;

  extern int *commsEnd_h;
  extern CUdeviceptr commsEnd_d[Nstream];
}
