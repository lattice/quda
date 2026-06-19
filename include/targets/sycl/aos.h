#pragma once

//#ifdef __SYCL_DEVICE_ONLY__
#if 0

#include <quda_sycl.h>

namespace quda
{

  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
    auto g = getGroup();
    sycl::span<T,n> outs(&out[0],n);
    sycl::ext::oneapi::experimental::group_load(g, in - n*g.get_local_linear_id(), outs);
  }

  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
    auto g = getGroup();
    sycl::span<const T,n> ins(&in[0],n);
    sycl::ext::oneapi::experimental::group_store(g, ins, out - n*g.get_local_linear_id());
  }

  template <typename T> __host__ __device__ void block_load(T &out, const T *in)
  {
    auto g = getGroup();
    sycl::ext::oneapi::experimental::group_load(g, in - g.get_local_linear_id(), out);
  }

  template <typename T> __host__ __device__ void block_store(T *out, const T &in)
  {
    auto g = getGroup();
    sycl::ext::oneapi::experimental::group_store(g, in, out - g.get_local_linear_id());
  }

} // namespace quda

#else

#include "../generic/aos.h"

#endif
