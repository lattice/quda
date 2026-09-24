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
    const unsigned int size; // number of elements of type T

  public:
    using value_type = T;

    /**
       @brief Byte offset for this shared memory object.
    */
    template <typename... Arg> static constexpr unsigned int get_offset(const dim3 block, const Arg &...arg)
    {
      unsigned int o = 0;
      if constexpr (!std::is_same_v<O, void>) { o = O::shared_mem_size(block, arg...); }
      return o;
    }

    /**
       @brief Shared memory size in bytes.
    */
    template <typename... Arg> static constexpr unsigned int shared_mem_size(const dim3 block, const Arg &...arg)
    {
      return get_offset(block, arg...) + S::size(block, arg...) * sizeof(T);
    }

    /**
       @brief Constructor for SharedMemory object.
    */
    template <typename... U, typename... Arg>
    SharedMemory(const KernelOps<U...> &ops, const Arg &...arg) : size(S::size(target::block_dim(), arg...))
    {
      auto op = ops;
      auto offset = get_offset(target::block_dim(), arg...);
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
    T &operator[](const int i) { return data[i]; }
  };

  template <typename T, typename S, typename O> static constexpr bool needsSharedMemImpl<SharedMemory<T, S, O>> = true;

} // namespace quda
