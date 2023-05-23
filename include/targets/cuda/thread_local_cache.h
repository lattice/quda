#pragma once

#include <target_device.h>

/**
   @file thread_local_cache.h

   Thread local cache object which may use shared memory for optimization.
 */

namespace quda
{

  /**
     @brief Class for threads to store a unique value which can use
     shared memory for optimization purposes.
   */
  template <typename T, int N_ = 0, typename O = void> class ThreadLocalCache
  {
  public:
    using value_type = T;
    using offset_type = O; // type of object that may also use shared memory at the same which is created before this one
    static constexpr int N = N_;
    static constexpr int len = std::max(1,N);

  private:
    using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    const int stride;
    const unsigned int offset = 0; // dynamic offset in bytes

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      atom_t *operator()(unsigned)
      {
        static atom_t *cache_;
        return cache_;
      }
    };

    /**
       @brief This is the handle to the shared memory, dynamic specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      __device__ inline atom_t *operator()(unsigned int offset)
      {
        extern __shared__ int cache_[];
        return reinterpret_cast<atom_t *>(reinterpret_cast<char *>(cache_) + offset);
      }
    };

    __device__ __host__ inline atom_t *cache() const { return target::dispatch<cache_dynamic>(offset); }

    __device__ __host__ inline void save_detail(const T &a, const int k) const
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
      int j = target::thread_idx_linear<3>();
#pragma unroll
      for (int i = 0; i < n_element; i++) cache()[(k*n_element + i) * stride + j] = tmp[i];
    }

    __device__ __host__ inline T load_detail(const int k) const
    {
      atom_t tmp[n_element];
      int j = target::thread_idx_linear<3>();
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache()[(k*n_element + i) * stride + j];
      T a;
      memcpy((void *)&a, tmp, sizeof(T));
      return a;
    }

    static constexpr unsigned int get_offset(dim3 block)
    {
      unsigned int o = 0;
      if constexpr (!std::is_same_v<O, void>) { o = O::shared_mem_size(block); }
      return o;
    }

  public:
    static constexpr unsigned int shared_mem_size(dim3 block)
    {
      return get_offset(block) + len * block.x * block.y * block.z * sizeof(T);
    }

    /**
       @brief Constructor for ThreadLocalCache.
    */
    constexpr ThreadLocalCache() : stride(target::block_size<3>()), offset(get_offset(target::block_dim())) { }

    /**
       @brief Grab the raw base address to this cache.
    */
    __device__ __host__ inline auto data() const { return reinterpret_cast<T *>(cache()); }

    /**
       @brief Save the value into the thread local cache.
       @param[in] a The value to store in the thread local cache
     */
    __device__ __host__ inline void save(const T &a) const {
      static_assert(N == 0);
      save_detail(a, 0);
    }

    /**
       @brief Save the value into the thread local cache.
       @param[in] a The value to store in the thread local cache
     */
    __device__ __host__ inline void save(const T &a, const int k) const { save_detail(a, k); }

    /**
       @brief Load a value from the thread local cache
       @return The value at the linear thread index
     */
    __device__ __host__ inline T load() const {
      static_assert(N == 0);
      return load_detail(0);
    }

    /**
       @brief Load a value from the thread local cache
       @return The value at the linear thread index
     */
    __device__ __host__ inline T load(const int k) const { return load_detail(k); }

    /**
       @brief Cast operator to allow cache objects to be used where T
       is expected
     */
    __device__ __host__ operator T() const {
      static_assert(N == 0);
      return load(0);
    }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected.
     */
    __device__ __host__ void operator=(const T &src) const {
      static_assert(N == 0);
      save(src, 0);
    }

    /**
       @brief Subscripting operator returning reference to allow cache objects
       to assign to a subscripted element.
     */
    __device__ __host__ auto operator[](int i) { return load(i); }
  };

} // namespace quda

// include overloads
#include "../generic/thread_local_cache.h"
