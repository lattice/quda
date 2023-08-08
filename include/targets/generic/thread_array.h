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
    int offset;
    using Smem = SharedMemory<array<T,n>, SizePerThread<1>, O>;
    constexpr Smem smem() const { return *dynamic_cast<const Smem*>(this); }
    array<T,n> &data() const { return smem()[offset]; }

  public:
    __device__ __host__ constexpr thread_array()
    {
      offset = target::thread_idx_linear<3>();
      data() = array<T, n>(); // call default constructor
    }

    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other)
    {
      offset = target::thread_idx_linear<3>();
      data() = array<T, n> {first, other...};
    }

    __device__ __host__ T &operator[](int i) { return data()[i]; }
    __device__ __host__ const T &operator[](int i) const { return data()[i]; }
  };

} // namespace quda
