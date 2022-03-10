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
      T value = reducer_t::do_sum ? warp_reduce.Sum(value_) : warp_reduce.Reduce(value_, r);

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
     @brief HIP specialization of block_reduce, utilizing hipcub::BlockReduce
  */
  template <> struct block_reduce<true> {

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
      using block_reduce_t = hipcub::BlockReduce<T, param_t::block_size_x, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                                 param_t::block_size_y, param_t::block_size_z>;
      __shared__ typename block_reduce_t::TempStorage storage[param_t::batch_size];
      block_reduce_t block_reduce(storage[batch]);
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      T value = reducer_t::do_sum ? block_reduce.Sum(value_) : block_reduce.Reduce(value_, r);

      if (all) {
        T &value_shared = reinterpret_cast<T &>(storage[batch]);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) value_shared = value;
        __syncthreads();
        value = value_shared;
      }
      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
