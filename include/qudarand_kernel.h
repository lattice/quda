#pragma once

#ifdef CUDA_BACKEND


#include <curand_kernel.h>
  
#define qurandStateXORWOW curandStateXORWOW
#define qurandStateMRG32k3a curandStateMRG32k3a
  
#if defined(XORWOW)
  typedef struct qurandStateXORWOW quRNGState;
#elif defined(MRG32k3a)
  typedef struct qurandStateMRG32k3a quRNGState;
#else
  typedef struct qurandStateMRG32k3a quRNGState;
#endif
  
inline  __device__ float qurand_uniform(quRNGState state) {
  return curand_uniform(&state);
}

inline  __device__ double qurand_uniform_double(quRNGState state) {
  return curand_uniform_double(&state);
}

inline  __device__ float qurand_normal(quRNGState state) {
  return curand_normal(&state);
}

inline  __device__ double qurand_normal_double(quRNGState state) {
  return curand_normal_double(&state);
}

#endif

#ifdef HIP_BACKEND
#include <hiprand_kernel.h>
#endif
