#pragma once

#include <register_traits.h>

namespace quda
{

  // pre-declaration of vector_load that we wish to specialize
  template <bool> struct vector_load_impl;

  // CUDA specializations of the vector_load
  template <> struct vector_load_impl<true> {
    template <typename T> __device__ inline void operator()(T &value, const void *ptr, int idx)
    {
      value = reinterpret_cast<const T *>(ptr)[idx];
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
      reinterpret_cast<T *>(ptr)[idx] = value;
    }

    __device__ inline void operator()(void *ptr, int idx, const short8 &value)
    {
      this->operator()(ptr, idx, *reinterpret_cast<const float4 *>(&value));
    }

    __device__ inline void operator()(void *ptr, int idx, const char8 &value)
    {
      this->operator()(ptr, idx, *reinterpret_cast<const float2 *>(&value));
    }

    __device__ inline void operator()(void *ptr, int idx, const char4 &value)
    {
      this->operator()(ptr, idx, *reinterpret_cast<const short2 *>(&value)); // A char4 is the same as a short2
    }
  };

} // namespace quda

#include "../generic/load_store.h"
