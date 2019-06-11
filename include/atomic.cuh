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
   @brief Implementation of double-precision atomic addition using compare
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
   @brief Implementation of double2 atomic addition using two
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

/**
   @brief Implementation of float2 atomic addition using two
   single-precision additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ float2 atomicAdd(float2 *addr, float2 val){
  float2 old = *addr;
  old.x = atomicAdd((float*)addr, val.x);
  old.y = atomicAdd((float*)addr + 1, val.y);
  return old;
}

/**
   @brief Implementation of int2 atomic addition using two
   int additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ int2 atomicAdd(int2 *addr, int2 val){
  int2 old = *addr;
  old.x = atomicAdd((int*)addr, val.x);
  old.y = atomicAdd((int*)addr + 1, val.y);
  return old;
}

union uint32_short2 { unsigned int i; short2 s; };

/**
   @brief Implementation of short2 atomic addition using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ short2 atomicAdd(short2 *addr, short2 val){
  uint32_short2 old, assumed, incremented;
  old.s = *addr;
  do {
    assumed.s = old.s;
    incremented.s = make_short2(val.x + assumed.s.x, val.y + assumed.s.y);
    old.i =  atomicCAS((unsigned int*)addr, assumed.i, incremented.i);
  } while ( assumed.i != old.i );

  return old.s;
}

union uint32_char2 { unsigned short i; char2 s; };

/**
   @brief Implementation of char2 atomic addition using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ char2 atomicAdd(char2 *addr, char2 val){
  uint32_char2 old, assumed, incremented;
  old.s = *addr;
  do {
    assumed.s = old.s;
    incremented.s = make_char2(val.x + assumed.s.x, val.y + assumed.s.y);
    old.i =  atomicCAS((unsigned int*)addr, assumed.i, incremented.i);
  } while ( assumed.i != old.i );

  return old.s;
}

/**
   @brief Implementation of double-precision atomic max using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ double atomicMax(double *addr, double val){
  double old = *addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
            __double_as_longlong(assumed),
            __double_as_longlong(val > assumed ? val : assumed)));
  } while ( __double_as_longlong(assumed) != __double_as_longlong(old) );
  
  return old;
}

/**
   @brief Implementation of single-precision atomic max using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ double atomicMax(float *addr, float val){
  float old = *addr, assumed;
  do {
    assumed = old;
    old = __int_as_float( atomicCAS((unsigned long long int*)addr,
            __float_as_int(assumed),
            __float_as_int(val > assumed ? val : assumed)));
  } while ( __float_as_int(assumed) != __float_as_int(old) );
  
  return old;
}

#endif
