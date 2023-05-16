#pragma once

#include <shared_memory_cache_helper.h>
#include <special_ops.h>

/**
   @file thread_local_cache.h

   Thread local cache object which may use shared memory for optimization.
*/

namespace quda
{

  template <typename T, typename O>
  class ThreadLocalCacheDefault
  {
    T data;

  public:
    using value_type = T;
    using offset_type = O;
    using self_type = ThreadLocalCacheDefault<T, O>;

    __device__ __host__ inline ThreadLocalCacheDefault() { }
    __device__ __host__ inline ThreadLocalCacheDefault(const self_type&x) : data(x.data) {}
    __device__ __host__ inline ThreadLocalCacheDefault(self_type&&x) : data(x.data) {}
    template <typename Ops> __device__ __host__ inline ThreadLocalCacheDefault(const Ops &ops) {}

    __device__ __host__ inline void save(const T &a) { data = a; }
    __device__ __host__ inline T &load() { return data; }
    __device__ __host__ inline const T&loadconst() const { return data; }
    __device__ __host__ auto& index(int i) { return data[i]; }
  };

  template <typename T, typename O, bool override = true>
  class ThreadLocalCacheBase : ThreadLocalCacheDefault<T,O> {};

  template <typename T, typename O = void>
  class ThreadLocalCache : public ThreadLocalCacheBase<T,O>
  {
  public:
    using value_type = T;
    using offset_type = O;
    using self_type = ThreadLocalCache<T, O>;
    using base_type = ThreadLocalCacheBase<T, O>;

    __device__ __host__ inline ThreadLocalCache() : base_type() { }
    __device__ __host__ inline ThreadLocalCache(const self_type&x) : base_type(x) {}
    __device__ __host__ inline ThreadLocalCache(self_type&&x) : base_type(x) {}
    template <typename Ops>
    __device__ __host__ inline ThreadLocalCache(const Ops &ops) : base_type(ops) {
      checkSpecialOps<self_type>(ops);
    }

    __device__ __host__ inline ThreadLocalCache& operator=(const self_type&x) {
      base_type::save(x);
      return *this;
    }
    __device__ __host__ inline ThreadLocalCache& operator=(self_type&&x) {
      base_type::save(x);
      return *this;
    }
    __device__ __host__ inline ThreadLocalCache& operator=(const T&x) {
      base_type::save(x);
      return *this;
    }

    __device__ __host__ inline void save(const T &a) { base_type::save(a); }
    __device__ __host__ inline T &load() { return base_type::load(); }
    __device__ __host__ operator const T&() const { return base_type::loadconst(); }
    __device__ __host__ auto& operator[](int i) { return base_type::index(i); }
  };

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

  template <typename T> struct get_type<
    T, std::enable_if_t<std::is_same_v<T, ThreadLocalCache<typename T::value_type, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };
}
