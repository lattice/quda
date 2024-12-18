#pragma once

#include <target_device.h>
#include <kernel_ops.h>

/**
   @file shared_memory_helper.h

   Target specific helper for allocating and accessing shared memory.
 */

namespace quda
{

  /**
     @brief Class which is used to allocate and access shared memory.
     The shared memory is treated as an array of type T, with the
     number of elements given by a call to the static member
     S::size(target::block_dim()).  The byte offset from the beginning
     of the total shared memory block is given by the static member
     O::shared_mem_size(target::block_dim()), or 0 if O is void.
   */
  template <typename T, typename S, typename O = void> class SharedMemory
  {
  public:
    using value_type = T;

  private:
    T *data;

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      T *operator()(unsigned int)
      {
        static T *cache_;
        return cache_;
      }
    };

    /**
       @brief This is the handle to the dynamic shared memory
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      __device__ inline T *operator()(unsigned int offset)
      {
        extern __shared__ int cache_[];
        return reinterpret_cast<T *>(reinterpret_cast<char *>(cache_) + offset);
      }
    };

    __device__ __host__ inline T *cache(unsigned int offset) const { return target::dispatch<cache_dynamic>(offset); }

  public:
    /**
       @brief Byte offset for this shared memory object.
    */
    static constexpr unsigned int get_offset(dim3 block)
    {
      unsigned int o = 0;
      if constexpr (!std::is_same_v<O, void>) { o = O::shared_mem_size(block); }
      return o;
    }

    /**
       @brief Shared memory size in bytes.
    */
    static constexpr unsigned int shared_mem_size(dim3 block) { return get_offset(block) + S::size(block) * sizeof(T); }

    /**
       @brief Constructor for SharedMemory object.
    */
    __device__ __host__ constexpr SharedMemory() : data(cache(get_offset(target::block_dim()))) { }

    /**
       @brief Constructor for SharedMemory object.
    */
    template <typename... U>
    constexpr SharedMemory(const KernelOps<U...> &) : data(cache(get_offset(target::block_dim())))
    {
    }

    /**
       @brief Return this SharedMemory object.
    */
    __device__ __host__ constexpr auto sharedMem() const { return *this; }

    /**
       @brief Subscripting operator returning a reference to element.
       @param[in] i The index to use.
       @return Reference to value stored at that index.
     */
    __device__ __host__ T &operator[](int i) const { return data[i]; }
  };

} // namespace quda
