#pragma once

#include <target_device.h>
#include <reducer.h>

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
#include <cub/block/block_scan.cuh>

namespace quda {

  template <int width_ = device::warp_size()>
  struct warp_reduce_param {
    static_assert(width_ <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    static constexpr int width = width_;
  };

  template <int block_size_x_, int block_size_y_, int block_size_z_, bool batched>
  struct block_reduce_param {
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    static constexpr int block_size_z = !batched ? block_size_z_ : 1;
    static constexpr int batch_size = !batched ? 1 : block_size_z;
  };

  template <bool is_device> struct warp_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, reducer_t, param_t) { return value; }
  };

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

  template <bool is_device> struct block_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, int, bool, reducer_t, param_t) { return value; }
  };

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

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the warp or sub-warp level
  */
  template <typename T, int width> class WarpReduce
  {
    using param_t = warp_reduce_param<width>;

  public:
    constexpr WarpReduce() {}

    __device__ __host__ inline T Sum(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::plus<T>(), param_t());
    }

    __device__ __host__ inline T AllSum(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::plus<T>(), param_t());
    }

    __device__ __host__ inline T Max(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::maximum<T>(), param_t());
    }

    __device__ __host__ inline T AllMax(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::maximum<T>(), param_t());
    }

    __device__ __host__ inline T Min(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::minimum<T>(), param_t());
    }

    __device__ __host__ inline T AllMin(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::minimum<T>(), param_t());
    }
  };

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the block level
  */
  template <typename T, int block_size_x, int block_size_y = 1, int block_size_z = 1, bool batched = false>
  class BlockReduce {
    using param_t = block_reduce_param<block_size_x, block_size_y, block_size_z, batched>;
    const int batch;

  public:
    constexpr BlockReduce(int batch = 0) : batch(batch) {}

    template <bool pipeline = false> __device__ __host__ inline T Sum(const T &value)
    {
      return target::dispatch<block_reduce>(value, pipeline, batch, false, quda::plus<T>(), param_t());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllSum(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllSum with batch_size > 1");
      return target::dispatch<block_reduce>(value, pipeline, batch, true, quda::plus<T>(), param_t());
    }

    template <bool pipeline = false> __device__ __host__ inline T Max(const T &value)
    {
      return target::dispatch<block_reduce>(value, pipeline, batch, false, quda::maximum<T>(), param_t());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllMax(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllMax with batch_size > 1");
      return target::dispatch<block_reduce>(value, pipeline, batch, true, quda::maximum<T>(), param_t());
    }

    template <bool pipeline = false> __device__ __host__ inline T Min(const T &value)
    {
      return target::dispatch<block_reduce>(value, pipeline, batch, false, quda::minimum<T>(), param_t());
    }

    template <bool pipeline = false> __device__ __host__ inline T AllMin(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllMin with batch_size > 1");
      return target::dispatch<block_reduce>(value, pipeline, batch, true, quda::minimum<T>(), param_t());
    }

    template <bool pipeline = false, typename reducer_t>
    __device__ __host__ inline T Reduce(const T &value, const reducer_t &r)
    {
      return target::dispatch<block_reduce>(value, pipeline, batch, false, r, param_t());
    }

    template <bool pipeline = false, typename reducer_t>
    __device__ __host__ inline T AllReduce(const T &value, const reducer_t &r)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllReduce with batch_size > 1");
      return target::dispatch<block_reduce>(value, pipeline, batch, true, r, param_t());
    }

  };

} // namespace quda
