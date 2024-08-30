#pragma once

#include <register_traits.h>

namespace quda
{

  /**
     @brief Element type used for coalesced storage.
   */
  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

  // pre-declaration of vector_load that we wish to specialize
  template <bool> struct vector_load_impl;

  // CUDA specializations of the vector_load
  template <> struct vector_load_impl<true> {
    template <typename T> __device__ inline void operator()(T &value, const void *ptr, int idx)
    {
      memcpy(&value, reinterpret_cast<const T *>(ptr) + idx, sizeof(T));
    }

    __device__ inline void operator()(short8 &value, const void *ptr, int idx)
    {
      float4 tmp;
      operator()(tmp, ptr, idx);
      memcpy(&value, &tmp, sizeof(float4));
    }

    __device__ inline void operator()(char8 &value, const void *ptr, int idx)
    {
      float2 tmp;
      operator()(tmp, ptr, idx);
      memcpy(&value, &tmp, sizeof(float2));
    }
  };

  // pre-declaration of vector_store that we wish to specialize
  template <bool> struct vector_store_impl;

  // CUDA specializations of the vector_store using inline ptx
  template <> struct vector_store_impl<true> {
    template <typename T> __device__ inline void operator()(void *ptr, int idx, const T &value)
    {
      memcpy(reinterpret_cast<T *>(ptr) + idx, &value, sizeof(T));
    }

    __device__ inline void operator()(void *ptr, int idx, const short8 &value)
    {
      memcpy(reinterpret_cast<float4 *>(ptr) + idx, &value, sizeof(float4));
    }

    __device__ inline void operator()(void *ptr, int idx, const char8 &value)
    {
      memcpy(reinterpret_cast<float2 *>(ptr) + idx, &value, sizeof(float2));
    }

    __device__ inline void operator()(void *ptr, int idx, const char4 &value)
    {
      memcpy(reinterpret_cast<short2 *>(ptr) + idx, &value, sizeof(short2));
    }
  };

} // namespace quda

#include "../generic/load_store.h"
