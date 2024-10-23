#pragma once

#include <target_device.h>

/**
   @file shared_memory_helper.h

   Target specific helper for allocating and accessing shared memory.
 */

namespace quda
{

  /**
     @brief Class which is used to allocate and access shared memory.
     The shared memory is treated as an array of type T, with the
     number of elements given by the call to the static member
     S::size(target::block_dim()).  The offset from the beginning of
     the total shared memory block is given by the static member
     O::shared_mem_size(target::block_dim()), or 0 if O is void.
   */
  template <typename T, typename S, typename O = void> class SharedMemory
  {
    sycl::local_ptr<T> data;
    const unsigned int size;  // number of elements of type T

  public:
    using value_type = T;

    static constexpr unsigned int get_offset(dim3 block)
    {
      unsigned int o = 0;
      if constexpr (!std::is_same_v<O, void>) { o = O::shared_mem_size(block); }
      return o;
    }

    static constexpr unsigned int shared_mem_size(dim3 block)
    {
      return get_offset(block) + S::size(block)*sizeof(T);
    }

    /**
       @brief Constructor for SharedMemory object.
    */
#if 0
    SharedMemory() : size(S::size(target::block_dim()))
    {
      auto grp = getGroup();
      using atype = T[512]; // FIXME
      auto mem0 = sycl::ext::oneapi::group_local_memory_for_overwrite<atype>(grp);
      auto offset = get_offset(target::block_dim());
      data = *mem0.get() + offset;
    }
#endif

    template <typename ...U>
    SharedMemory(const KernelOps<U...> &ops) : size(S::size(target::block_dim()))
    {
      //auto op = getDependentOps<op_SharedMemory<T,SizeSmem<SharedMemory<T,S,O>>>>(ops);
      auto op = ops;
      auto offset = get_offset(target::block_dim());
      sycl::local_ptr<void> v(op.smem + offset);
      sycl::local_ptr<T> p(v);
      data = p;
    }

    constexpr auto sharedMem() const { return *this; }

    /**
       @brief Subscripting operator returning a reference to element.
       @param[in] i The index to use.
       @return Reference to value stored at that index.
     */
    __device__ __host__ T &operator[](const int i) { return data[i]; }
  };

} // namespace quda
