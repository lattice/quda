#pragma once

#include <target_device.h>
#include <trove/ptr.h>

namespace quda
{

  /**
     @brief This is just a dummy structure we use for trove to define the
     required structure size
     @tparam T Array element type
     @tparam n Number of elements in the structure
  */
  template <typename T, int n> struct S {
    T v[n];
    __host__ __device__ inline const T &operator[](int i) const { return v[i]; }
    __host__ __device__ inline T &operator[](int i) { return v[i]; }
  };

  /**
     @brief block_load for the host
  */
  template <bool use_trove> struct block_load_impl {
    template <typename T> __host__ __device__ inline T operator()(const T *in) { return *(in); }
  };

  /**
     @brief Device block_load that uses trove
  */
  template <> struct block_load_impl<true> {
    template <typename T> __device__ inline T operator()(const T *in) { return *(trove::coalesced_ptr<const T>(in)); }
  };

  /**
     @brief Load n-length block of memory of type T and return in local array
     @tparam T Array element type
     @tparam n Number of elements in the structure
     @param[out] out Output array
     @param[in] in Input memory pointer we are block loading from
   */
  template <typename T, int n> __device__ __host__ inline void block_load(T out[n], const T *in)
  {
    using struct_t = S<T, n>;
    struct_t v = target::dispatch<block_load_impl>(reinterpret_cast<const struct_t *>(in));

#pragma unroll
    for (int i = 0; i < n; i++) out[i] = v[i];
  }

  /**
     @brief Load type T from contiguous memory
     @tparam T Element type
     @param[out] out Output value
     @param[in] in Input memory pointer we are loading from
  */
  template <typename T> __host__ __device__ inline void block_load(T &out, const T *in)
  {
    out = target::dispatch<block_load_impl>(in);
  }

  /**
     @brief block_store for the host
  */
  template <bool use_trove> struct block_store_impl {
    template <typename T> __host__ __device__ inline void operator()(T *out, const T &in) { *out = in; }
  };

  /**
     @brief Device block_store that uses trove
  */
  template <> struct block_store_impl<true> {
    template <typename T> __device__ inline void operator()(T *out, const T &in)
    {
      *(trove::coalesced_ptr<T>(out)) = in;
    }
  };

  /**
     @brief Store n-length array of type T in block of memory
     @tparam T Array element type
     @tparam n Number of elements in the array
     @param[out] out Output memory pointer we are block storing to
     @param[in] in Input array
   */
  template <typename T, int n> __host__ __device__ inline void block_store(T *out, const T in[n])
  {
    using struct_t = S<T, n>;
    struct_t v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = in[i];

    target::dispatch<block_store_impl>(reinterpret_cast<struct_t *>(out), v);
  }

  /**
     @brief Store type T in contiguous memory
     @tparam Element type
     @param[out] out Output memory pointer we are storing to
     @param[in] in Input value
  */
  template <typename T> __host__ __device__ inline void block_store(T *out, const T &in)
  {
    target::dispatch<block_store_impl>(out, in);
  }

} // namespace quda
