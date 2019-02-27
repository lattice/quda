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

  // these variables are used for benchmarking the dslash components in isolation
  extern bool dslash_pack_compute;
  extern bool dslash_interior_compute;
  extern bool dslash_exterior_compute;
  extern bool dslash_comms;
  extern bool dslash_copy;

  // whether we have initialized the dslash policy tuner
  extern bool dslash_policy_init;

  // used to keep track of which policy to start the autotuning
  extern int first_active_policy;
  extern int first_active_p2p_policy;
}
