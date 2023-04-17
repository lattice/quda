#pragma once

#include <shared_memory_cache_helper.h>

namespace quda
{

  template <typename T, typename O = void>
    class LocalMemory
    {
      T data;

    public:
      using value_type = T;
      using offset_type = O;
      using LocalMemory_t = LocalMemory<T, O>;

      __device__ __host__ inline LocalMemory() { }

      template <typename Ops>
      __device__ __host__ inline LocalMemory(const Ops &ops) {
	// check type
      }

      __device__ __host__ inline void save(const T &a) { data = a; }

      __device__ __host__ inline T &load() { return data; }

      __device__ __host__ operator const T&() const { return data; }

      __device__ __host__ auto& operator[](int i) { return data[i]; }

    };

  template <typename T, typename O>
    __device__ __host__ inline T operator+(const LocalMemory<T, O> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, typename O>
    __device__ __host__ inline T operator+(const T &a, const LocalMemory<T, O> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T>
    struct get_type<T, std::enable_if_t<std::is_same_v<T, LocalMemory<typename T::value_type, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };
}
