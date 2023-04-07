/**
   @file shared_memory_cache_helper.h
   @brief Convenience overloads to allow SharedMemoryCache objects to
   appear in simple expressions.  The actual implementation of
   SharedMemoryCache is target specific, and located in e.g.,
   include/targets/cuda/shared_memory_cache_helper.h, etc.
 */

namespace quda
{

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline T operator+(const SharedMemoryCache<T, by, bz, dynamic> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline T operator+(const T &a, const SharedMemoryCache<T, by, bz, dynamic> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline T operator-(const SharedMemoryCache<T, by, bz, dynamic> &a, const T &b)
  {
    return static_cast<const T &>(a) - b;
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline T operator-(const T &a, const SharedMemoryCache<T, by, bz, dynamic> &b)
  {
    return a - static_cast<const T &>(b);
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline auto operator+=(SharedMemoryCache<T, by, bz, dynamic> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) + b);
    return a;
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline auto operator-=(SharedMemoryCache<T, by, bz, dynamic> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) - b);
    return a;
  }

  template <typename T, int by, int bz, bool dynamic>
  __device__ __host__ inline auto conj(const SharedMemoryCache<T, by, bz, dynamic> &a)
  {
    return conj(static_cast<const T &>(a));
  }

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or SharedMemoryCache<T>
   */
  template <class T, class enable = void> struct get_type {
    using type = T;
  };
  template <class T>
  struct get_type<
    T, std::enable_if_t<std::is_same_v<T, SharedMemoryCache<typename T::value_type, T::block_size_y, T::block_size_z, T::dynamic>>>> {
    using type = typename T::value_type;
  };

} // namespace quda
