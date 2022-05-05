#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specialziations for
   warp- and block-level reductions, using the CUB library
 */

#define USE_CG
#ifndef USE_CG

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

using namespace quda;

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#else

#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

#endif

namespace quda
{

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;


#ifndef USE_CG

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

#else

  template <typename T> struct atomic_type;

  template <typename T, typename U> constexpr auto get_reducer(const plus<U> &r) { return plus<T>(); }
  template <typename T, typename U> constexpr auto get_reducer(const maximum<U> &r) { return maximum<T>(); }
  template <typename T, typename U> constexpr auto get_reducer(const minimum<U> &r) { return minimum<T>(); }

  template <typename T, typename U> constexpr auto get_cg_reducer(const plus<U> &r) { return cg::plus<T>(); }
  template <typename T, typename U> constexpr auto get_cg_reducer(const maximum<U> &r) { return cg::greater<T>(); }
  template <typename T, typename U> constexpr auto get_cg_reducer(const minimum<U> &r) { return cg::less<T>(); }

  /**
     @brief CUDA specialization of warp_reduce, utilizing cooperative groups
  */
  template <> struct warp_reduce<true> {

    /**
       @brief Perform a warp-wide reduction using cooperative groups
       @param[in] value_ thread-local value to be reduced
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       warp (all = false).
       @param[in] r The reduction operation we want to apply
       @return The warp-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    std::enable_if_t<sizeof(T) <= 32, T> __device__ inline operator()(const T &value_, bool all, const reducer_t &r, const param_t &)
    {
      cg::thread_block cta = cg::this_thread_block();
      cg::thread_block_tile<device::warp_size()> tile = cg::tiled_partition<device::warp_size()>(cta);

      constexpr bool use_cg_reduce = false;

      if constexpr (!use_cg_reduce) {
        T value = value_;
#pragma unroll
        for (int offset = device::warp_size() / 2; offset >= 1; offset /= 2) value = get_reducer<T>(r)(value, tile.shfl_down(value, offset));
        if (all) value = tile.shfl(value, 0);
        return value;
      } else {
        return cg::reduce(tile, value_, get_cg_reducer<T>(r));
      }
    }

    /**
       @brief Perform a warp-wide reduction using cooperative groups.
       This is a wrapper that split up large values into atomic-size
       pieces, since cg doens't support > 32 byte types.
       @param[in] value_ thread-local value to be reduced
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       warp (all = false).  Unused since cg alway assumes all = true.
       @param[in] r The reduction operation we want to apply
       @return The warp-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    std::enable_if_t<(sizeof(T) > 32), T> __device__ inline operator()(const T &value_, bool all, const reducer_t &r, const param_t &)
    {
      using atomic_t = typename atomic_type<T>::type;
      constexpr size_t n = sizeof(T) / sizeof(atomic_t);
      static_assert(sizeof(T) == n * sizeof(atomic_t));
      atomic_t sum_tmp[n];
      memcpy(sum_tmp, &value_, sizeof(T));
#pragma unroll
      for (int i = 0; i < n; i++) sum_tmp[i] = operator()(sum_tmp[i], all, r, param_t());
      T value;
      memcpy(&value, sum_tmp, sizeof(T));
      return value;
    }

  };


#endif


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
      constexpr auto max_items = device::max_reduce_block_size() / (device::warp_size() * param_t::batch_size);
      const auto thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
      const auto warp_idx = thread_idx / device::warp_size();
      const auto warp_items = blockDim.x * blockDim.y / device::warp_size();

      // first do warp reduce
      T value = warp_reduce<true>()(value_, false, r, warp_reduce_param<device::warp_size()>());

      if (!all && warp_items == 1) return value; // short circuit for single warp CTA

      // now do reduction between warps
      if (!async) __syncthreads(); // only synchronize if we are not pipelining

      __shared__ T storage[param_t::batch_size][max_items];

      // if first thread in warp, write result to shared memory
      if (thread_idx % device::warp_size() == 0) storage[batch][warp_idx] = value;
      __syncthreads();

      constexpr bool final_warp_reduction = true;

      if constexpr (final_warp_reduction) { // first warp completes the reduction
        if (warp_idx == 0) {
          if constexpr (max_items > device::warp_size()) { // never true for max block size 1024, warp = 32
            value = r.init();
            for (int i = threadIdx.x; i < warp_items; i += device::warp_size()) value = r(storage[batch][i], value);
          } else { // optimized path where we know the final reduction will fit in a warp
            value = threadIdx.x < warp_items ? storage[batch][threadIdx.x] : r.init();
            value = warp_reduce<true>()(value, false, r, warp_reduce_param<device::warp_size()>());
          }
        }
      } else { // first thread completes the reduction
        if (thread_idx == 0) {
          for (int i = 1; i < warp_items; i++) value = r(storage[batch][i], value);
        }
      }

      if (all) {
        if (thread_idx == 0) storage[batch][0] = value;
        __syncthreads();
        value = storage[batch][0];
      }

      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
