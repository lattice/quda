// This file redirects the compiler to the correct target file for
// tune.cpp
#ifdef CUDA_TARGET
#include "targets/cuda/cuda_tune.cpp"
#endif
