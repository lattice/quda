#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specialziations for
   warp- and block-level reductions, using the CUB library
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

using namespace quda;

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace quda
{

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief CUDA specialization of warp_reduce, utilizing cub::WarpReduce
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
      using warp_reduce_t = cub::WarpReduce<T, param_t::width, __COMPUTE_CAPABILITY__>;
      typename warp_reduce_t::TempStorage dummy_storage;
      warp_reduce_t warp_reduce(dummy_storage);
      T value = reducer_t::do_sum ? warp_reduce.Sum(value_) : warp_reduce.Reduce(value_, r);

      if (all) {
        using warp_scan_t = cub::WarpScan<T, param_t::width, __COMPUTE_CAPABILITY__>;
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
     @brief CUDA specialization of block_reduce, utilizing cub::BlockReduce
  */
  template <> struct block_reduce<true> {

    template <int width_>  struct warp_reduce_param {
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
    __device__ inline T operator()(const T &value_, bool async, int batch, bool all, const reducer_t &r,
                                   const param_t &)
    {
      constexpr auto max_block_size_x = device::max_reduce_block_size<param_t::block_size_y, param_t::block_size_z>(); 
      constexpr auto max_items = (max_block_size_x  / device::warp_size()) * param_t::block_size_y * param_t::block_size_z;
      const auto warp_items = blockDim.x * param_t::block_size_y * param_t::block_size_z / device::warp_size();
      __shared__ T storage[param_t::batch_size][max_items];

      // first do warp reduce
      T value = warp_reduce<true>()(value_, false, r, warp_reduce_param<device::warp_size()>());

      // now do reduction between warps
      auto warp_idx = threadIdx.x / device::warp_size();
      auto n_warp = blockDim.x / device::warp_size();
      auto item_idx = ((param_t::batch_size == 1 ? threadIdx.z * blockDim.y : 0) + threadIdx.y) * n_warp + warp_idx;

      if (!async) __syncthreads(); // only synchronize if we are not pipelining

      if (threadIdx.x % device::warp_size() == 0) {
        storage[batch][item_idx] = value;
      }
      __syncthreads();

      bool thread_zero = (threadIdx.x && threadIdx.y == 0 && (param_t::batch_size == 1 ? threadIdx.z == 0 : true));
      if (thread_zero == 0) {
        for (int i = 1; i < warp_items; i++) // start from 1 to avoid rereading
          value = r(storage[batch][i], value);
      }

      if (all) {
        if (item_idx == 0) storage[batch][0] = value;
        __syncthreads();
        value = storage[batch][0];
      }

      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
