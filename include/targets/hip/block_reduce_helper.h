#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specialziations for
   warp- and block-level reductions, using the CUB library
 */

using namespace quda;

#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <hipcub/block/block_scan.hpp>

namespace quda
{

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief HIP specialization of warp_reduce, utilizing hipcub::WarpReduce
  */
  template <> struct warp_reduce<true> {

    /**
       @brief Perform a warp-wide reduction
       @param[in] value_ thread-local value to be reduced
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       warp (all = false)
       @param[in] r The reduction operation we want to apply
       @return The warp-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    __device__ inline T operator()(const T &value_, bool all, const reducer_t &r, const param_t &)
    {
      using warp_reduce_t = hipcub::WarpReduce<T, param_t::width>;
      typename warp_reduce_t::TempStorage dummy_storage;
      warp_reduce_t warp_reduce(dummy_storage);
      T value = {};
      if constexpr (reducer_t::do_sum) {
        value = warp_reduce.Sum(value_);
      } else {
        value = warp_reduce.Reduce(value_, r);
      }

      if (all) {
        using warp_scan_t = hipcub::WarpScan<T, param_t::width>;
        typename warp_scan_t::TempStorage dummy_storage;
        warp_scan_t warp_scan(dummy_storage);
        value = warp_scan.Broadcast(value, 0);
      }

      return value;
    }
  };

  // pre-declaration of block_reduce that we wish to specialize
  template <bool> struct block_reduce;

  /**
     @brief HIP specialization of block_reduce, building on the warp reduce
  */
  template <> struct block_reduce<true> {

    template <int width_> struct warp_reduce_param {
      static constexpr int width = width_;
    };

    /**
       @brief Perform a block-wide reduction
       @param[in] value_ thread-local value to be reduced
       @param[in] async Whether this reduction will be performed
       asynchronously with respect to the calling threads
       @param[in] batch The batch index of the reduction
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       block (all = false)
       @param[in] r The reduction operation we want to apply
       @return The block-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    __device__ inline T operator()(const T &value_, bool async, int batch, bool all, const reducer_t &r, const param_t &)
    {
      constexpr auto max_items = device::max_block_size() / device::warp_size();
      const auto thread_idx = target::thread_idx_linear<param_t::block_dim>();
      const auto block_size = target::block_size<param_t::block_dim>();
      const auto warp_idx = thread_idx / device::warp_size();
      const auto warp_items = (block_size + device::warp_size() - 1) / device::warp_size();

      // first do warp reduce
      T value = warp_reduce<true>()(value_, false, r, warp_reduce_param<device::warp_size()>());

      // now do reduction between warps
      if (!async) __syncthreads(); // only synchronize if we are not pipelining

      __shared__ T storage[max_items];

      // if first thread in warp, write result to shared memory
      if (thread_idx % device::warp_size() == 0) storage[batch * warp_items + warp_idx] = value;
      __syncthreads();

      // whether to use the first warp or first thread for the final reduction
      constexpr bool final_warp_reduction = true;

      if constexpr (final_warp_reduction) { // first warp completes the reduction (requires first warp is full)
        if (warp_idx == 0) {
          if constexpr (max_items > device::warp_size()) { // never true for max block size 1024, warp = 32
            value = r.init();
            for (auto i = thread_idx; i < warp_items; i += device::warp_size())
              value = r(storage[batch * warp_items + i], value);
          } else { // optimized path where we know the final reduction will fit in a warp
            value = thread_idx < warp_items ? storage[batch * warp_items + thread_idx] : r.init();
          }
          value = warp_reduce<true>()(value, false, r, warp_reduce_param<device::warp_size()>());
        }
      } else { // first thread completes the reduction
        if (thread_idx == 0) {
          for (unsigned int i = 1; i < warp_items; i++) value = r(storage[batch * warp_items + i], value);
        }
      }

      if (all) {
        if (thread_idx == 0) storage[batch * warp_items + 0] = value;
        __syncthreads();
        value = storage[batch * warp_items + 0];
      }

      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
