#pragma once

#include <target_device.h>

namespace quda
{

  template <size_t prefetch> struct prefetch_t {
    static constexpr int size = prefetch;
  };

  /**
     @brief Non-specialized load operation
  */
  template <bool is_device> struct vector_load_impl {
    template <typename T, size_t prefetch_size>
    __device__ __host__ inline void operator()(T &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      value = reinterpret_cast<const T *>(ptr)[idx];
    }
  };

  template <typename vector_t, size_t prefetch = 0>
  __device__ __host__ inline vector_t vector_load_internal(const void *ptr, int idx)
  {
    vector_t value;
    target::dispatch<vector_load_impl>(value, ptr, idx, prefetch_t<prefetch>());
    return value;
  }

  template <typename scalar_t, int N, size_t prefetch = 0>
  __device__ __host__ inline array<scalar_t, N> vector_load(const void *ptr, int idx)
  {
    using vector_t = typename VectorType<scalar_t, N>::type;
    auto value_v = vector_load_internal<vector_t, prefetch>(ptr, idx);
    array<scalar_t, N> value_a;
    static_assert(sizeof(value_a) == sizeof(value_v), "array type and vector type are different sizes");
    memcpy(&value_a, &value_v, sizeof(vector_t));
    return value_a;
  }

  /**
     @brief Non-specialized store operation
  */
  template <bool is_device> struct vector_store_impl {
    template <typename T> __device__ __host__ inline void operator()(void *ptr, int idx, const T &value)
    {
      reinterpret_cast<T *>(ptr)[idx] = value;
    }
  };

  template <typename vector_t> __device__ __host__ inline void vector_store(void *ptr, int idx, const vector_t &value)
  {
    target::dispatch<vector_store_impl>(ptr, idx, value);
  }

  template <typename scalar_t, int N>
  __device__ __host__ inline void vector_store(void *ptr, int idx, const array<scalar_t, N> &value_a)
  {
    using vector_t = typename VectorType<scalar_t, N>::type;
    vector_t value_v;
    static_assert(sizeof(value_a) == sizeof(value_v), "array type and vector type are different sizes");
    memcpy(&value_v, &value_a, sizeof(vector_t));
    vector_store<vector_t>(ptr, idx, value_v);
  }

  template <bool is_device> struct prefetch_cache_line_imp {
    __device__ __host__ inline void operator()(const void *) { }
  };

  __device__ __host__ inline void prefetch_cache_line(const void *p) { target::dispatch<prefetch_cache_line_imp>(p); }

  template <bool is_device> struct prefetch_cache_bulk_imp {
    __device__ __host__ inline void operator()(const void *, size_t) { }
  };

  __device__ __host__ inline void prefetch_cache_bulk(const void *p, size_t bytes)
  {
    target::dispatch<prefetch_cache_bulk_imp>(p, bytes);
  }

} // namespace quda
