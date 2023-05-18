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
  template <typename T, typename O = void> class ThreadLocalCache
  {
  public:
    using value_type = T;
    using offset_type = O; // type of object that may also use shared memory at the same which is created before this one

  private:
    const unsigned int offset = 0; // dynamic offset in bytes

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      T *operator()(unsigned)
      {
        static T *cache_;
        return cache_;
      }
    };

    /**
       @brief This is the handle to the shared memory, dynamic specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      __device__ inline T *operator()(unsigned int offset)
      {
        extern __shared__ int cache_[];
        return reinterpret_cast<T *>(reinterpret_cast<char *>(cache_) + offset);
      }
    };

    __device__ __host__ inline T * cache() const
    {
      return target::dispatch<cache_dynamic>(offset);
    }

    __device__ __host__ inline void save_detail(const T &a) const
    {
      int j = target::thread_idx_linear<3>();
      cache()[j] = a;
    }

    __device__ __host__ inline T &load_detail() const
    {
      int j = target::thread_idx_linear<3>();
      return cache()[j];
    }

  public:
    static constexpr unsigned int get_offset() {
      if constexpr(std::is_same_v<O,void>) {
	return 0;
      } else {
	return O::size();
      }
    }

    static constexpr unsigned int size() {
      return get_offset() + target::block_size<3>() * sizeof(T);
    }

    /**
       @brief Constructor for ThreadLocalCache.
    */
    constexpr ThreadLocalCache() : offset(get_offset()) {}

    /**
       @brief Grab the raw base address to this cache.
    */
    __device__ __host__ inline auto data() const { return reinterpret_cast<T *>(cache()); }

    /**
       @brief Save the value into the thread local cache.
       @param[in] a The value to store in the thread local cache
     */
    __device__ __host__ inline void save(const T &a) const
    {
      save_detail(a);
    }

    /**
       @brief Load a value from the thread local cache
       @return The value at the linear thread index
     */
    __device__ __host__ inline T &load() const
    {
      return load_detail();
    }

    /**
       @brief Cast operator to allow cache objects to be used where T
       is expected
     */
    __device__ __host__ operator T() const { return load(); }
    //__device__ __host__ operator T() const { T a; return a; }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected.
     */
    __device__ __host__ void operator=(const T &src) const { save(src); }
    //__device__ __host__ void operator=(const T &src) const { ; }

    /**
       @brief Subscripting operator returning reference to allow cache objects
       to assign to a subscripted element.
     */
    __device__ __host__ auto& operator[](int i) { return load()[i]; }
  };

} // namespace quda

// include overloads
#include "../generic/thread_local_cache.h"
