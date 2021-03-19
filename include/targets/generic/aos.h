#pragma once

namespace quda {

  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
  }

  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
  }

  template <typename T> __host__ __device__ void block_load(T &out, const T *in)
  {
    out = *in;
  }

  template <typename T> __host__ __device__ void block_store(T *out, const T &in)
  {
    *out = in;
  }

}
