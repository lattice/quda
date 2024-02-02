#pragma once

#include <kernel_ops.h>
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
  template <typename T, int n, typename O = void> class thread_array : SharedMemory<array<T, n>, SizePerThread<1>, O>
  {
    using Smem = SharedMemory<array<T, n>, SizePerThread<1>, O>;
    using Smem::sharedMem;
    array<T, n> &array_;

  public:
    using Smem::shared_mem_size;

    template <typename... U>
    __device__ __host__ constexpr thread_array(const KernelOps<U...> &ops) :
      Smem(ops), array_(sharedMem()[target::thread_idx_linear<3>()])
    {
      checkKernelOps<thread_array<T, n, O>>(ops);
      array_ = array<T, n> {}; // call default constructor
    }

    constexpr thread_array(const thread_array<T, n, O> &) = delete;

    __device__ __host__ inline T &operator[](const int i) { return array_[i]; }
    __device__ __host__ inline const T &operator[](const int i) const { return array_[i]; }
  };

} // namespace quda
