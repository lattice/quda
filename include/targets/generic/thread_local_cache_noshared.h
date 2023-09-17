#pragma once

#include <helpers.h>

/**
   @file thread_local_cache.h

   Thread local cache object which may use shared memory for optimization.
   The storage can be a single object or an array of objects.
 */

namespace quda
{

  /**
     @brief Class for threads to store a unique value, or array of values, which can use
     shared memory for optimization purposes.
   */
  template <typename T, int N_ = 0, typename O = void> class ThreadLocalCache
  {
  public:
    using value_type = T;
    static constexpr int N = N_; // size of array, 0 means to behave like T instead of array<T, 1>
    using offset_type = O; // type of object that may also use shared memory at the same time and is located before this one
    static constexpr int len = std::max(1,N); // actual number of elements to store

  private:
    array<T,len> data;

  public:
    /**
       @brief Constructor for ThreadLocalCache.
    */
    constexpr ThreadLocalCache() {}

    template <typename ...U>
    constexpr ThreadLocalCache(const SpecialOps<U...> &ops) {
      checkSpecialOp<ThreadLocalCache<T,N,O>,U...>();
    }

    static constexpr unsigned int shared_mem_size(dim3) { return 0; }

    /**
       @brief Save the value into the thread local cache.  Used when N==0 so cache acts like single object.
       @param[in] a The value to store in the thread local cache
     */
    __device__ __host__ inline void save(const T &a) {
      static_assert(N == 0);
      data[0] = a;
    }

    /**
       @brief Save the value into an element of the thread local cache.
       @param[in] a The value to store in the thread local cache
       @param[in] k The index to use
     */
    __device__ __host__ inline void save(const T &a, const int k) { data[k] = a; }

    /**
       @brief Load a value from the thread local cache.  Used when N==0 so cache acts like single object.
       @return The value stored in the thread local cache
     */
    __device__ __host__ inline T load() const {
      static_assert(N == 0);
      return data[0];
    }

    /**
       @brief Load a value from an element of the thread local cache
       @param[in] k The index to use
       @return The value stored in the thread local cache at that index
     */
    __device__ __host__ inline T load(const int k) const { return data[k]; }

    /**
       @brief Cast operator to allow cache objects to be used where T is expected (when N==0).
     */
    __device__ __host__ operator T() const {
      static_assert(N == 0);
      return data[0];
    }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected (when N==0).
     */
    __device__ __host__ void operator=(const T &src) {
      static_assert(N == 0);
      data[0] = src;
    }

    /**
       @brief Subscripting operator returning value at index for convenience.
       @param[in] i The index to use
       @return The value stored in the thread local cache at that index
     */
    __device__ __host__ T operator[](int i) { return data[i]; }
  };

  template <typename T, int N, typename O> __device__ __host__ inline T operator+(const ThreadLocalCache<T, N, O> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator+(const T &a, const ThreadLocalCache<T, N, O> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator-(const ThreadLocalCache<T, N, O> &a, const T &b)
  {
    return static_cast<const T &>(a) - b;
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator-(const T &a, const ThreadLocalCache<T, N, O> &b)
  {
    return a - static_cast<const T &>(b);
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto operator+=(ThreadLocalCache<T, N, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) + b);
    return a;
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto operator-=(ThreadLocalCache<T, N, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) - b);
    return a;
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto conj(const ThreadLocalCache<T, N, O> &a)
  {
    return conj(static_cast<const T &>(a));
  }

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or ThreadLocalCache<T,O>
   */
  template <class T>
  struct get_type<T, std::enable_if_t<std::is_same_v<T, ThreadLocalCache<typename T::value_type, T::N, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };

} // namespace quda
