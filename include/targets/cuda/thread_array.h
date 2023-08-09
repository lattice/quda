#pragma once

#ifndef _NVHPC_CUDA

//#include "../generic/thread_array.h"
#include <array.h>
//#include <shared_memory_cache_helper.h>
#include <shared_memory_helper.h>
#include <helpers.h>

namespace quda
{
  template <typename T, int n, typename O = void>
  struct thread_array : SharedMemory<array<T,n>, SizePerThread<1>, O> {
    int offset;
    array<T, n> host_array;
    array<T, n> &array_;
    using Smem = SharedMemory<array<T,n>, SizePerThread<1>, O>;
    constexpr Smem smem() const { return *dynamic_cast<const Smem*>(this); }

    __device__ __host__ constexpr thread_array() :
      offset(target::thread_idx_linear<3>()),
      array_(*(&smem()[0] + offset))
      //array_(target::is_device() ? *(&smem()[0] + offset) : host_array)
    {
      array_ = array<T, n>(); // call default constructor
    }

    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other) :
      offset(target::thread_idx_linear<3>()),
      array_(*(&smem()[0] + offset))
      //array_(target::is_device() ? *(&smem()[0] + offset) : host_array)
    {
      array_ = array<T, n> {first, other...};
    }

    __device__ __host__ T &operator[](int i) { return array_[i]; }
    __device__ __host__ const T &operator[](int i) const { return array_[i]; }
  };
}

#else

#include <array.h>
namespace quda
{
  template <typename T, int n> struct thread_array : array<T, n> {};
}

#endif
