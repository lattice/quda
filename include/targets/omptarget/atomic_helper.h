#pragma once

#include <array.h>

/**
   @file atomic_helper.h

   @section Provides definitions of atomic functions that are used in QUDA.
 */

namespace quda
{

  /**
     @brief atomic_fetch_add function performs similarly as atomic_ref::fetch_add
     @param[in,out] addr The memory address of the variable we are
     updating atomically
     @param[in] val The value we summing to the value at addr
  */
  template <typename T> __device__ __host__ inline void atomic_fetch_add(T *addr, T val)
  {
#pragma omp atomic update
      *addr += val;
  }

  template <typename T> __device__ __host__ inline void atomic_fetch_add(complex<T> *addr, complex<T> val)
  {
    atomic_fetch_add(reinterpret_cast<T *>(addr) + 0, val.real());
    atomic_fetch_add(reinterpret_cast<T *>(addr) + 1, val.imag());
  }

  template <typename T, int n> __device__ __host__ inline void atomic_fetch_add(array<T, n> *addr, array<T, n> val)
  {
    for (int i = 0; i < n; i++) atomic_fetch_add(&(*addr)[i], val[i]);
  }

  /**
     @brief atomic_fetch_max function that does an atomic max.
     @param[in,out] addr The memory address of the variable we are
     updating atomically
     @param[in] val The value we are comparing against.  Must be
     positive valued else result is undefined.
  */
  __device__ __host__ inline void atomic_fetch_abs_max(float *addr, float val)
  {
#pragma omp atomic compare
    if(*addr<val){*addr=val;}
  }
  __device__ __host__ inline void atomic_fetch_abs_max(double *addr, double val)
  {
#pragma omp atomic compare
    if(*addr<val){*addr=val;}
  }

} // namespace quda
