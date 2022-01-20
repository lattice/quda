#pragma once

#include <target_device.h>

namespace quda
{

  template <bool is_device> struct warp_combine_impl {
    template <typename T> T operator()(T &x, int) { return x; }
  };

  template <> struct warp_combine_impl<true> {
    template <typename T> __device__ inline T operator()(T &x, int warp_split)
    {
      constexpr int warp_size = device::warp_size();
      if (warp_split > 1) {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          // reduce down to the first group of column-split threads
#pragma unroll
          for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
            // TODO - add support for non-converged warps
            x[i].real(x[i].real() + __shfl_down_sync(device::warp_converged_mask(), x[i].real(), offset));
            x[i].imag(x[i].imag() + __shfl_down_sync(device::warp_converged_mask(), x[i].imag(), offset));
          }
        }
      }
      return x;
    }
  };

  template <int warp_split, typename T> __device__ __host__ inline T warp_combine(T &x)
  {
    return target::dispatch<warp_combine_impl>(x, warp_split);
  }

} // namespace quda
