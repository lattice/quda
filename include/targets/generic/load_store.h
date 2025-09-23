#pragma once

#include <register_traits.h>
#include <target_device.h>

namespace quda
{

  /**
     @brief Element type used for coalesced storage.
   */
  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

  /**
     @brief Non-specialized load operation
  */
  template <bool is_device> struct vector_load_impl {
    template <typename T> __device__ __host__ inline void operator()(T &value, const void *ptr, int idx)
    {
      // value = reinterpret_cast<const T *>(ptr)[idx];
      memcpy(&value, static_cast<const T *>(ptr) + idx, sizeof(value));
    }
  };

  template <typename vector_t> __device__ __host__ inline vector_t vector_load(const void *ptr, int idx)
  {
    vector_t value;
    target::dispatch<vector_load_impl>(value, ptr, idx);
    return value;
  }

  template <typename scalar_t, int N>
  __device__ __host__ inline array<scalar_t, N> vector_load(const void *ptr, int idx)
  {
    using vector_t = typename VectorType<scalar_t, N>::type;
    auto value_v = vector_load<vector_t>(ptr, idx);
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
      // reinterpret_cast<T *>(ptr)[idx] = value;
      memcpy(static_cast<T *>(ptr) + idx, &value, sizeof(value));
    }
  };

  template <typename vector_t> __device__ __host__ inline void vector_storeV(void *ptr, int idx, const vector_t &value)
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
    //vector_storeV<vector_t>(ptr, idx, value_v);
    scalar_t *a = static_cast<scalar_t *>(ptr) + N*idx;
    memcpy(a, &value_v, sizeof(value_v));
  }

} // namespace quda
