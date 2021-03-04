#pragma once

#include <target_device.h>
#include <trove/ptr.h>

namespace quda {

  constexpr bool enable_trove()
  {
#ifndef DEVICE_DEBUG
    return device::is_device();
#else
    return false; // trove has issues with device debug
#endif
  }

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

  template <typename T, bool use_trove> __device__ std::enable_if_t<use_trove, T> block_load(const T *in)
  {
    return *(trove::coalesced_ptr<const T>(in));
  }

  template <typename T, bool use_trove> __device__ __host__ std::enable_if_t<!use_trove, T> block_load(const T *in)
  {
    return *(in);
  }

  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
    using struct_t = S<T, n>;
    struct_t v = block_load<struct_t, enable_trove()>(reinterpret_cast<const struct_t*>(in));

#pragma unroll
    for (int i = 0; i < n; i++) out[i] = v[i];
  }

  template <typename T> __host__ __device__ void block_load(T &out, const T *in)
  {
    out = block_load<T, enable_trove()>(in);
  }

  template <typename T, bool use_trove> __device__ std::enable_if_t<use_trove, void> block_store(T *out, const T &in)
  {
    *(trove::coalesced_ptr<T>(out)) = in;
  }

  template <typename T, bool use_trove> std::enable_if_t<!use_trove, void> block_store(T *out, const T &in)
  {
    *out = in;
  }

  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
    using struct_t = S<T, n>;
    struct_t v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = in[i];

    block_store<struct_t, enable_trove()>(reinterpret_cast<struct_t*>(out), v);
  }

  template <typename T> __host__ __device__ void block_store(T *out, const T &in)
  {
    block_store<T, enable_trove()>(out, in);
  }

}
