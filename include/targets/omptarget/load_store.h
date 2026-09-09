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

  template <size_t prefetch> struct prefetch_t;

  // OpenMP specializations of vector_load
  template <> struct vector_load_impl<true> {
    template <typename T, size_t prefetch_size>
    __device__ inline void operator()(T &value, const void *ptr, index_t idx, const prefetch_t<prefetch_size> &)
    {
      memcpy(&value, reinterpret_cast<const T *>(ptr) + idx, sizeof(T));
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(short8 &value, const void *ptr, index_t idx,
                                      const prefetch_t<prefetch_size> &prefetch)
    {
      float4 tmp;
      operator()(tmp, ptr, idx, prefetch);
      memcpy(&value, &tmp, sizeof(float4));
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(char8 &value, const void *ptr, index_t idx,
                                      const prefetch_t<prefetch_size> &prefetch)
    {
      float2 tmp;
      operator()(tmp, ptr, idx, prefetch);
      memcpy(&value, &tmp, sizeof(float2));
    }
  };

  // pre-declaration of vector_store that we wish to specialize
  template <bool> struct vector_store_impl;

  // OpenMP specializations of vector_store
  template <> struct vector_store_impl<true> {
    template <typename T> __device__ inline void operator()(void *ptr, index_t idx, const T &value)
    {
      memcpy(reinterpret_cast<T *>(ptr) + idx, &value, sizeof(T));
    }

    __device__ inline void operator()(void *ptr, index_t idx, const short8 &value)
    {
      memcpy(reinterpret_cast<float4 *>(ptr) + idx, &value, sizeof(float4));
    }

    __device__ inline void operator()(void *ptr, index_t idx, const char8 &value)
    {
      memcpy(reinterpret_cast<float2 *>(ptr) + idx, &value, sizeof(float2));
    }

    __device__ inline void operator()(void *ptr, index_t idx, const char4 &value)
    {
      memcpy(reinterpret_cast<short2 *>(ptr) + idx, &value, sizeof(short2));
    }
  };

} // namespace quda

#include "../generic/load_store.h"
