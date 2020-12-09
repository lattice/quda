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

#ifndef QUDA_BACKEND_OMPTARGET
#include <cub/block/block_reduce.cuh>
#endif

namespace quda {

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the warp or sub-warp level
  */
  template <typename T, int width> struct WarpReduce
  {
    static_assert(width <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
#ifdef QUDA_BACKEND_OMPTARGET
    struct warp_reduce_t {  // FIXME
      using TempStorage = int;
    };
#else
    using warp_reduce_t = cub::WarpReduce<T, width>;
#endif

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
  template <typename T, int block_size_x, int batch_size = 1> struct BlockReduce
  {
    const int batch;

#ifdef QUDA_BACKEND_OMPTARGET
    struct block_reduce_t {  // FIXME
      using TempStorage = int;
    };
#else
    using block_reduce_t = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
#endif

    __device__ inline auto& shared_state()
    {
      static __shared__ typename block_reduce_t::TempStorage storage[batch_size];
      return storage[batch];
    }

    __device__ __host__ inline BlockReduce(int batch = 0) : batch(batch) {}

    template <bool pipeline = false>
    __device__ __host__ inline T Sum(const T &value_)
    {
#ifdef __CUDA_ARCH__
      block_reduce_t block_reduce(shared_state());
      if (!pipeline) __syncthreads(); // only need to synchronize if we are not pipelining
      T value = block_reduce.Sum(value_);
#else
      T value = value_;
#endif
      return value;
    }

    template <bool pipeline = false>
    __device__ __host__ inline T AllSum(const T &value_)
    {
      T value = Sum<pipeline>(value_);
#ifdef __CUDA_ARCH__
      T &value_shared = reinterpret_cast<T&>(shared_state());
      if (threadIdx.x == 0 && threadIdx.y == 0) value_shared = value;
      __syncthreads();
      value = value_shared;
#endif
      return value;
    }
  };

} // namespace quda
