#pragma once

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
      //value = reinterpret_cast<const T *>(ptr)[idx];
      memcpy(&value, static_cast<const T *>(ptr)+idx, sizeof(value));
    }
  };

  template <typename VectorType> __device__ __host__ inline VectorType vector_load(const void *ptr, int idx)
  {
    VectorType value;
    target::dispatch<vector_load_impl>(value, ptr, idx);
    return value;
  }

  /**
     @brief Non-specialized store operation
  */
  template <bool is_device> struct vector_store_impl {
    template <typename T> __device__ __host__ inline void operator()(void *ptr, int idx, const T &value)
    {
      //reinterpret_cast<T *>(ptr)[idx] = value;
      memcpy(static_cast<T *>(ptr)+idx, &value, sizeof(value));
    }
  };

  template <typename VectorType>
  __device__ __host__ inline void vector_store(void *ptr, int idx, const VectorType &value)
  {
    target::dispatch<vector_store_impl>(ptr, idx, value);
  }

} // namespace quda
