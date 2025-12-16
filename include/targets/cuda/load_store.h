#pragma once

#include <register_traits.h>
#include <inline_ptx.h>
#include <tma_helper.hpp>

namespace quda
{

  /**
     @brief Element type used for coalesced storage.
   */
  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

  // pre-declaration of vector_load that we wish to specialize
  template <bool> struct vector_load_impl;

  // pre-declaration of the prefetch type
  template <size_t prefetch> struct prefetch_t;

  // CUDA specializations of the vector_load
  template <> struct vector_load_impl<true> {
    template <typename T, size_t prefetch_size>
    __device__ inline void operator()(T &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &) {
      value = reinterpret_cast<const T *>(ptr)[idx];
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(float4 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_float4<prefetch_size>(value, reinterpret_cast<const float4 *>(ptr) + idx);
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(float2 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_float2<prefetch_size>(value, reinterpret_cast<const float2 *>(ptr) + idx);
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(float &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_float<prefetch_size>(value, reinterpret_cast<const float *>(ptr) + idx);
    }

#if __COMPUTE_CAPABILITY__ >= 1000
    template <typename T, size_t prefetch_size>
    __device__ inline void operator()(double4 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_double4<prefetch_size>(value, reinterpret_cast<const double4 *>(ptr) + idx);
    }

    template <typename T, size_t prefetch_size>
    __device__ inline void operator()(float8 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_float8<prefetch_size>(value, reinterpret_cast<const float8 *>(ptr) + idx);
    }
#endif

    template <size_t prefetch_size>
    __device__ inline void operator()(double2 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_double2<prefetch_size>(value, reinterpret_cast<const double2 *>(ptr) + idx);
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(short2 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_short2<prefetch_size>(value, reinterpret_cast<const short2 *>(ptr) + idx);      
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(short4 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &)
    {
      load_cached_short4<prefetch_size>(value, reinterpret_cast<const short4 *>(ptr) + idx);      
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(short8 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &prefetch)
    {
      float4 tmp;
      operator()(tmp, ptr, idx, prefetch);
      memcpy(&value, &tmp, sizeof(float4));
    }

    template <size_t prefetch_size>
    __device__ inline void operator()(char8 &value, const void *ptr, int idx, const prefetch_t<prefetch_size> &prefetch)
    {
      float2 tmp;
      operator()(tmp, ptr, idx, prefetch);
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

#if __COMPUTE_CAPABILITY__ >= 1000
    __device__ inline void operator()(void *ptr, int idx, const double4 &value)
    {
      store_streaming_double4(reinterpret_cast<double4 *>(ptr) + idx, value.x, value.y, value.z, value.w);
    }

    __device__ inline void operator()(void *ptr, int idx, const float8 &value)
    {
      store_streaming_float8(reinterpret_cast<float8 *>(ptr) + idx, value);
    }
#endif

    __device__ inline void operator()(void *ptr, int idx, const double2 &value)
    {
      store_streaming_double2(reinterpret_cast<double2 *>(ptr) + idx, value.x, value.y);
    }

    __device__ inline void operator()(void *ptr, int idx, const float4 &value)
    {
      store_streaming_float4(reinterpret_cast<float4 *>(ptr) + idx, value.x, value.y, value.z, value.w);
    }

    __device__ inline void operator()(void *ptr, int idx, const float2 &value)
    {
      store_streaming_float2(reinterpret_cast<float2 *>(ptr) + idx, value.x, value.y);
    }

    __device__ inline void operator()(void *ptr, int idx, const short4 &value)
    {
      store_streaming_short4(reinterpret_cast<short4 *>(ptr) + idx, value.x, value.y, value.z, value.w);
    }

    __device__ inline void operator()(void *ptr, int idx, const short8 &value)
    {
      this->operator()(ptr, idx, *reinterpret_cast<const float4 *>(&value));
    }

    __device__ inline void operator()(void *ptr, int idx, const short2 &value)
    {
      store_streaming_short2(reinterpret_cast<short2 *>(ptr) + idx, value.x, value.y);
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

  // pre-declaration of the prefetch_cache that we wish to specialize
  template <bool> struct prefetch_cache_line_imp;

  // CUDA specialization of the prefetch_cache that uses inline ptx
  template <> struct prefetch_cache_line_imp<true> {
    __device__ inline void operator()(const void *p) { prefetch_L2(p); }
  };

  // pre-declaration of the prefetch_cache that we wish to specialize
  template <bool> struct prefetch_L1_cache_line_imp;

  template <> struct prefetch_L1_cache_line_imp<true> {
    __device__ inline void operator()(const void *p)
    {
      static __shared__ float smem[32]; // dummy shared memory allocation
      auto tid = target::thread_idx_linear<3>();
      auto lane_id = tid & 31;
      prefetch_L1(smem + lane_id, p);
    }
  };

  // pre-declaration of the prefetch_cache that we wish to specialize
  template <bool> struct prefetch_cache_bulk_imp;
  template <bool> struct prefetch_cache_tensor_3d_imp;
  template <bool> struct prefetch_cache_tensor_4d_imp;
  template <bool> struct prefetch_cache_tensor_5d_imp;

#if __COMPUTE_CAPABILITY__ >= 900
  // CUDA specialization of the prefetch_cache_bulk that uses TMA (requires Hopper+)
  template <> struct prefetch_cache_bulk_imp<true> {
    __device__ inline void operator()(const void *p, size_t bytes) { prefetch_tma(p, bytes); }
  };

  // CUDA specialization of the prefetch_cache_tensor_3d that uses TMA (requires Hopper+)
  template <> struct prefetch_cache_tensor_3d_imp<true> {
    __device__ inline void operator()(const tma_descriptor_t &desc, int x, int y, int z)
    {
      prefetch_tma_3d(desc.map, x, y, z);
    }
  };

  // CUDA specialization of the prefetch_cache_tensor_4d that uses TMA (requires Hopper+)
  template <> struct prefetch_cache_tensor_4d_imp<true> {
    __device__ inline void operator()(const tma_descriptor_t &desc, int x, int y, int z, int w)
    {
      prefetch_tma_4d(desc.map, x, y, z, w);
    }
  };

  // CUDA specialization of the prefetch_cache_tensor_5d that uses TMA (requires Hopper+)
  template <> struct prefetch_cache_tensor_5d_imp<true> {
    __device__ inline void operator()(const tma_descriptor_t &desc, int x, int y, int z, int w, int u)
    {
      prefetch_tma_5d(desc.map, x, y, z, w, u);
    }
  };
#endif

} // namespace quda

#include "../generic/load_store.h"
