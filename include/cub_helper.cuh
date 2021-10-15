#pragma once

#include <target_device.h>

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

using namespace quda;

#include <cub/block/block_reduce.cuh>

namespace quda {

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the warp or sub-warp level
  */
  template <typename T, int width> struct WarpReduce
  {
    static_assert(width <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    using warp_reduce_t = cub::WarpReduce<T, width, __COMPUTE_CAPABILITY__>;

    __device__ __host__ inline WarpReduce() {}

    template <bool is_device, typename dummy = void> struct sum { T operator()(const T &value) { return value; } };

    template <typename dummy> struct sum<true, dummy> {
      __device__ inline T operator()(const T &value) {
        typename warp_reduce_t::TempStorage dummy_storage;
        warp_reduce_t warp_reduce(dummy_storage);
        return warp_reduce.Sum(value);
      }
    };

    __device__ __host__ inline T Sum(const T &value) { return target::dispatch<sum>(value); }
  };

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the block level
  */
  template <typename T, int block_size_x, int batch_size = 1, int block_size_y = 1, int block_size_z = 1> struct BlockReduce
  {
    static_assert(!((block_size_y > 1 || block_size_z > 1) && batch_size > 1), "Cannot use batching and multi-dimensional BlockReduce");

    using block_reduce_t =
      cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y, block_size_z, __COMPUTE_CAPABILITY__>;
    const int batch;

    __device__ __host__ inline BlockReduce(int batch = 0) : batch(batch) {}

    template <bool is_device, typename dummy = void> struct reduce {
      template <typename reducer_t> T operator()(const T &value, bool, int, bool, bool, reducer_t) { return value; }
    };

    template <typename dummy> struct reduce<true, dummy> {
      template <typename reducer_t>
      __device__ inline T operator()(const T &value_, bool pipeline, int batch, bool all, bool is_sum, const reducer_t &r)
      {
        static __shared__ typename block_reduce_t::TempStorage storage[batch_size];
        block_reduce_t block_reduce(storage[batch]);
        if (!pipeline) __syncthreads(); // only synchronize if we are not pipelining
        T value = is_sum ? block_reduce.Sum(value_) : block_reduce.Reduce(value_, r);

        if (all) {
          T &value_shared = reinterpret_cast<T&>(storage[batch]);
          if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) value_shared = value;
          __syncthreads();
          value = value_shared;
        }
        return value;
      }
    };

    template <bool pipeline = false> __device__ __host__ inline T Sum(const T &value)
    {
      return target::dispatch<reduce>(value, pipeline, batch, false, true, cub::Sum());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllSum(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllSum with batch_size > 1");
      return target::dispatch<reduce>(value, pipeline, batch, true, true, cub::Sum());
    }

    template <bool pipeline = false> __device__ __host__ inline T Max(const T &value)
    {
      return target::dispatch<reduce>(value, pipeline, batch, false, false, cub::Max());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllMax(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllMax with batch_size > 1");
      return target::dispatch<reduce>(value, pipeline, batch, true, false, cub::Max());
    }

    template <bool pipeline = false> __device__ __host__ inline T Min(const T &value)
    {
      return target::dispatch<reduce>(value, pipeline, batch, false, false, cub::Min());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllMin(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllMin with batch_size > 1");
      return target::dispatch<reduce>(value, pipeline, batch, true, false, cub::Min());
    }

  };

} // namespace quda
