#pragma once

#include <target_device.h>
#include <trove/ptr.h>

namespace quda {

  /**
     @brief This is just a dummy structure we use for trove to define the
     required structure size
     @param real Real number type
     @param length Number of elements in the structure
  */
  template <typename T, int n> struct S {
    T v[n];
    __host__ __device__ inline const T &operator[](int i) const { return v[i]; }
    __host__ __device__ inline T &operator[](int i) { return v[i]; }
  };

  /**
     @brief block_store for the host or when DEVICE_DEBUG is enabled
  */
  template <bool use_trove> struct block_load_impl {
    template <typename T> __host__ __device__ inline T operator()(const T *in) { return *(in); }
  };

#ifndef DEVICE_DEBUG
  /**
     @brief Device block_store that uses trove (except when DEVICE_DEBUG is enabled)
  */
  template <> struct block_load_impl<true> {
    template <typename T> __device__ inline T operator()(const T *in) { return *(trove::coalesced_ptr<const T>(in)); }
  };
#endif

  template <typename T, int n> __device__ __host__ inline void block_load(T out[n], const T *in)
  {
    using struct_t = S<T, n>;
    struct_t v = target::dispatch<block_load_impl>(reinterpret_cast<const struct_t*>(in));

#pragma unroll
    for (int i = 0; i < n; i++) out[i] = v[i];
  }

  template <typename T> __host__ __device__ inline void block_load(T &out, const T *in)
  {
    out = target::dispatch<block_load_impl>(in);
  }

  /**
     @brief block_store for the host or when DEVICE_DEBUG is enabled
  */
  template <bool use_trove> struct block_store_impl {
    template <typename T> __host__ __device__ inline void operator()(T *out, const T &in) { *out = in; }
  };

#ifndef DEVICE_DEBUG
  /**
     @brief Device block_store that uses trove (except when DEVICE_DEBUG is enabled)
  */
  template <> struct block_store_impl<true> {
    template <typename T> __device__ inline void operator()(T *out, const T &in) { *(trove::coalesced_ptr<T>(out)) = in; }
  };
#endif

  template <typename T, int n> __host__ __device__ inline void block_store(T *out, const T in[n])
  {
    using struct_t = S<T, n>;
    struct_t v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = in[i];

    target::dispatch<block_store_impl>(reinterpret_cast<struct_t*>(out), v);
  }

  template <typename T> __host__ __device__ inline void block_store(T *out, const T &in)
  {
    target::dispatch<block_store_impl>(out, in);
  }

}
