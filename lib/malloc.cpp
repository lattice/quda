// This file redirects the compiler to the correct target file for
// quda_malloc.cpp
#ifdef CUDA_TARGET
#include "targets/cuda/cuda_malloc.cpp"
#endif
