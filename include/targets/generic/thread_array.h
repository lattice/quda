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
  public:
    using SharedMemoryT = SharedMemory<array<T,n>, SizePerThread<1>, O>;

  private:
    using SharedMemoryT::smem;
    array<T, n> &array_;

  public:
    using SharedMemoryT::shared_mem_size;

    __device__ __host__ constexpr thread_array() :
      array_(smem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n>(); // call default constructor
    }

#if 0
    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other) :
      array_(smem()[target::thread_idx_linear<3>()])
    {
      array_ = array<T, n> {first, other...};
    }
#endif

    template <typename... U>
    constexpr thread_array(const SpecialOps<U...> &ops) :
      SharedMemoryT(ops),
      array_(smem()[target::thread_idx_linear<3>()])
    {
      checkSpecialOp<thread_array<T,n,O>,U...>();
      array_ = array<T, n>(); // call default constructor
    }

    __device__ __host__ T &operator[](int i) { return array_[i]; }
    __device__ __host__ const T &operator[](int i) const { return array_[i]; }
  };

} // namespace quda
