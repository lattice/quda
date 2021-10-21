#pragma once

#include <array.h>

/**
   @file atomic_helper.h

   @section Provides definitions of atomic functions that are used in QUDA.
 */

namespace quda
{

  template <bool is_device> struct atomic_fetch_add_impl {
    template <typename T> inline void operator()(T *addr, T val)
    {
#pragma omp atomic update
      *addr += val;
    }
  };

  template <> struct atomic_fetch_add_impl<true> {
    template <typename T> __device__ inline void operator()(T *addr, T val) { atomicAdd(addr, val); }
  };

  /**
     @brief atomic_fetch_add function performs similarly as atomic_ref::fetch_add
     @param[in,out] addr The memory address of the variable we are
     updating atomically
     @param[in] val The value we summing to the value at addr
  */
  template <typename T> __device__ __host__ inline void atomic_fetch_add(T *addr, T val)
  {
    target::dispatch<atomic_fetch_add_impl>(addr, val);
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

  template <bool is_device> struct atomic_fetch_abs_max_impl {
    template <typename T> inline void operator()(T *addr, T val)
    {
#pragma omp atomic update
      *addr = std::max(*addr, val);
    }
  };

  template <> struct atomic_fetch_abs_max_impl<true> {
    /**
       @brief Implementation of single-precision atomic max specialized
       for positive-definite numbers.  Here we take advantage of the
       property that when positive floating point numbers are
       reinterpretted as unsigned integers, they have the same unique
       sorted order.
       @param addr Address that stores the atomic variable to be updated
       @param val Value to be added to the atomic
    */
    __device__ inline void operator()(float *addr, float val)
    {
      uint32_t val_ = __float_as_uint(val);
      uint32_t *addr_ = reinterpret_cast<uint32_t *>(addr);
      atomicMax(addr_, val_);
    }
  };

  /**
     @brief atomic_fetch_max function that does an atomic max.
     @param[in,out] addr The memory address of the variable we are
     updating atomically
     @param[in] val The value we are comparing against.  Must be
     positive valued else result is undefined.
  */
  template <typename T> __device__ __host__ inline void atomic_fetch_abs_max(T *addr, T val)
  {
    target::dispatch<atomic_fetch_abs_max_impl>(addr, val);
  }

} // namespace quda
