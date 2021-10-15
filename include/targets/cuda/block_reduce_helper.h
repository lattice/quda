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

namespace quda {

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief CUDA specialization of warp_reduce
  */
  template <> struct warp_reduce<true> {
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
     @brief CUDA specialization of block_reduce
  */
  template <> struct block_reduce<true> {
    template <typename T, typename reducer_t, typename param_t>
    __device__ inline T operator()(const T &value_, bool pipeline, int batch, bool all, const reducer_t &r, const param_t &)
    {
      using block_reduce_t =
        cub::BlockReduce<T, param_t::block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, param_t::block_size_y, param_t::block_size_z, __COMPUTE_CAPABILITY__>;
      static __shared__ typename block_reduce_t::TempStorage storage[param_t::batch_size];
      block_reduce_t block_reduce(storage[batch]);
      if (!pipeline) __syncthreads(); // only synchronize if we are not pipelining
      T value = reducer_t::do_sum ? block_reduce.Sum(value_) : block_reduce.Reduce(value_, r);

      if (all) {
        T &value_shared = reinterpret_cast<T&>(storage[batch]);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) value_shared = value;
        __syncthreads();
        value = value_shared;
      }
      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
