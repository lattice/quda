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

  template <typename T>
  inline T atomic_read(T &x)
  {
    T v;
    #pragma omp atomic read
    v = x;
    return v;
  }
  template <typename T, int N>
  inline array<T,N> atomic_read(array<T,N> &x)
  {
    array<T,N> v;
    for (int i = 0; i < N; ++i)
      v[i] = atomic_read(x[i]);
    return v;
  }
  template <typename T>
  inline complex<T> atomic_read(complex<T> &x)
  {
    complex<T> v (atomic_read(x.x), atomic_read(x.y));
    return v;
  }
  template <typename T>
  inline deviation_t<T> atomic_read(deviation_t<T> &x)
  {
    deviation_t<T> v;
    v.diff = atomic_read(x.diff);
    v.ref = atomic_read(x.ref);
    return v;
  }
} // namespace quda
