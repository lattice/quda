#pragma once

// trove requires CUDA and has issues with device debug
#if !defined(DEVICE_DEBUG)
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif

namespace quda {

  /**
     @brief This is just a dummy structure we use for trove to define the
     required structure size
     @param real Real number type
     @param length Number of elements in the structure
  */
  template <typename T, int n> struct S {
    T v[n];
    __host__ __device__ const T &operator[](int i) const { return v[i]; }
    __host__ __device__ T &operator[](int i) { return v[i]; }
  };

  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
    using struct_t = S<T, n>;
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    struct_t v = *(trove::coalesced_ptr<const struct_t>(reinterpret_cast<const struct_t*>(in)));
#else
    struct_t v = *(reinterpret_cast<const struct_t*>(in));
#endif

#pragma unroll
    for (int i = 0; i < n; i++) out[i] = v[i];
  }

  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
    using struct_t = S<T, n>;
    struct_t v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = in[i];

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    *(trove::coalesced_ptr<struct_t>(reinterpret_cast<struct_t *>(out))) = v;
#else
    *(reinterpret_cast<struct_t*>(out)) = v;
#endif
  }

  template <typename T> __host__ __device__ void block_load(T &out, const T *in)
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    out = *(trove::coalesced_ptr<const T>(in));
#else
    out = *(in);
#endif
  }

  template <typename T> __host__ __device__ void block_store(T *out, const T &in)
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    *(trove::coalesced_ptr<T>(out)) = in;
#else
    *out = in;
#endif
  }

}
