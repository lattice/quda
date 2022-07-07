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
      using R = typename T::value_type;
      constexpr auto max_nthr = device::max_block_size();
      static_assert(max_nthr*sizeof(R) <= device::max_shared_memory_size()-sizeof(target::omptarget::get_shared_cache()[0])*128, "Shared cache not large enough for tempStorage");  // FIXME arbitrary, the number is arbitrary, offset 128 below
      R *storage = (R*)&target::omptarget::get_shared_cache()[128];  // FIXME arbitrary
      const int tid = omp_get_thread_num();

      constexpr int warp_size = device::warp_size();
      const int wid = tid / warp_size;
      const int warpend = (wid+1)*warp_size;
      const int split_size = warp_size / warp_split;
      if (warp_split > 1) {
QUDA_UNROLL
        for (int i = 0; i < x.size(); i++) {
          // reduce down to the first group of column-split threads
QUDA_UNROLL
          for (int offset = warp_size / 2; offset >= split_size; offset /= 2) {
            #pragma omp barrier
            storage[tid] = x[i];
            const auto tid_offset = tid + offset;
            const auto load_id = tid_offset >= warpend ? tid_offset-warp_size : tid_offset;
            #pragma omp barrier
            const auto& y = storage[load_id];
            x[i].real(x[i].real() + y.real());
            x[i].imag(x[i].imag() + y.imag());
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
