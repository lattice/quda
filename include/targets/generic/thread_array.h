#pragma once

#include <helpers.h>
#include <shared_memory_helper.h>
#include <array.h>

namespace quda
{

  /**
     @brief Class that provides indexable per-thread storage for n
     elements of type T.  This version uses shared memory for storage.
     The offset into the shared memory region is determined from the
     type O.
   */
  template <typename T, int n, typename O = void>
  class thread_array : SharedMemory<array<T,n>, SizePerThread<1>, O>
  {
    using Smem = SharedMemory<array<T,n>, SizePerThread<1>, O>;
    using Smem::sharedMem;
    array<T, n> &array_;

  public:
    __device__ __host__ constexpr thread_array() :
      array_(sharedMem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n>(); // call default constructor
    }

    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other) :
      array_(sharedMem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n> {first, other...};
    }

    __device__ __host__ T &operator[](int i) { return array_[i]; }
    __device__ __host__ const T &operator[](int i) const { return array_[i]; }
  };

} // namespace quda
