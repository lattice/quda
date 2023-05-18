#include <shared_memory_cache_helper.h>

/**
   @file thread_local_cache_helper.h
   @brief Convenience overloads to allow ThreadLocalCache objects to
   appear in simple expressions.  The actual implementation of
   ThreadLocalCache is target specific, and located in e.g.,
   include/targets/cuda/thread_local_cache.h, etc.
 */

namespace quda
{

  template <typename T, typename O>
  __device__ __host__ inline T operator+(const ThreadLocalCache<T, O> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, typename O>
  __device__ __host__ inline T operator+(const T &a, const ThreadLocalCache<T, O> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T, typename O>
  __device__ __host__ inline T operator-(const ThreadLocalCache<T, O> &a, const T &b)
  {
    return static_cast<const T &>(a) - b;
  }

  template <typename T, typename O>
  __device__ __host__ inline T operator-(const T &a, const ThreadLocalCache<T, O> &b)
  {
    return a - static_cast<const T &>(b);
  }

  template <typename T, typename O>
  __device__ __host__ inline auto operator+=(ThreadLocalCache<T, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) + b);
    return a;
  }

  template <typename T, typename O>
  __device__ __host__ inline auto operator-=(ThreadLocalCache<T, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) - b);
    return a;
  }

  template <typename T, typename O>
  __device__ __host__ inline auto conj(const ThreadLocalCache<T, O> &a)
  {
    return conj(static_cast<const T &>(a));
  }

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or ThreadLocalCache<T,O>
   */
  template <class T>
  struct get_type<
    T, std::enable_if_t<std::is_same_v<T, ThreadLocalCache<typename T::value_type, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };

} // namespace quda
