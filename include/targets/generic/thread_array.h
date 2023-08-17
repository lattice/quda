#pragma once

#include <helpers.h>
#include <shared_memory_helper.h>
#include <array.h>

namespace quda
{

  /**
     @brief Class that provides indexable per-thread storage.  On CUDA
     this maps to using assigning each thread a unique window of
     shared memory using the SharedMemoryCache object.
   */
  template <typename T, int n, typename O = void>
  class thread_array : SharedMemory<array<T,n>, SizePerThread<1>, O>
  {
    using Smem = SharedMemory<array<T,n>, SizePerThread<1>, O>;
    //constexpr Smem smem() const { return *dynamic_cast<const Smem*>(this); }
    //constexpr Smem smem() const { return *static_cast<const Smem*>(this); }
    using Smem::smem;
    array<T, n> &array_;

  public:
    __device__ __host__ constexpr thread_array() :
      array_(smem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n>(); // call default constructor
    }

    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other) :
      array_(smem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n> {first, other...};
    }

    __device__ __host__ T &operator[](int i) { return array_[i]; }
    __device__ __host__ const T &operator[](int i) const { return array_[i]; }
  };

} // namespace quda
