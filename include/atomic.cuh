#pragma once

/**
   @file atomic.cuh

   @section Description
 
   Provides definitions of atomic functions that are not native to
   CUDA.  These are intentionally not declared in the namespace to
   avoid confusion when resolving the native atomicAdd functions.
 */

#if defined(__CUDA_ARCH__) 

#if __COMPUTE_CAPABILITY__ < 600
/**
   Implementation of double-precision atomic addition using compare
   and swap.
   
   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ double atomicAdd(double *addr, double val){
  double old = *addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
					  __double_as_longlong(assumed),
					  __double_as_longlong(val + assumed)));
  } while ( __double_as_longlong(assumed) != __double_as_longlong(old) );
  
  return old;
}
#endif

/**
   Implementation of double2 atomic addition using two
   double-precision additions.
   
   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ double2 atomicAdd(double2 *addr, double2 val){
  double2 old = *addr;
  old.x = atomicAdd((double*)addr, val.x);
  old.y = atomicAdd((double*)addr + 1, val.y);
  return old;
}

#endif
