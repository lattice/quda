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
    using warp_reduce_t = cub::WarpReduce<T, width>;

    __device__ inline auto& shared_state()
    {
      typename warp_reduce_t::TempStorage dummy;
      return dummy;
    }

    __device__ __host__ inline WarpReduce() {}

    __device__ __host__ inline T Sum(const T &value)
    {
#ifdef __CUDA_ARCH__
      warp_reduce_t warp_reduce(shared_state());
      return warp_reduce.Sum(value);
#else
      return value;
#endif
    }
  };

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the block level
  */
  template <typename T, int block_size_x, int block_size_y = 1> struct BlockReduce
  {
    using block_reduce_t = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y>;

    __device__ inline auto& shared_state()
    {
      static __shared__ typename block_reduce_t::TempStorage storage;
      return storage;
    }

    __device__ __host__ inline BlockReduce() {}

    __device__ __host__ inline T Sum(const T &value_)
    {
#ifdef __CUDA_ARCH__
      block_reduce_t block_reduce(shared_state());
      T &value_shared = (T&)shared_state();
      __syncthreads();
      T value = block_reduce.Sum(value_);
      if (threadIdx.x == 0 && threadIdx.y == 0) value_shared = value;
      __syncthreads();
      value = value_shared;
#else
      T value = value_;
#endif
      return value;
    }
  };

} // namespace quda
