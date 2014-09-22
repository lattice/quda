namespace dslash {
  extern int it;

#ifdef PTHREADS
  extern cudaEvent_t interiorDslashEnd;
#endif
  extern cudaEvent_t packEnd[Nstream];
  extern cudaEvent_t gatherStart[Nstream];
  extern cudaEvent_t gatherEnd[Nstream];
  extern cudaEvent_t scatterStart[Nstream];
  extern cudaEvent_t scatterEnd[Nstream];
  extern cudaEvent_t dslashStart;
  extern cudaEvent_t dslashEnd;
}
