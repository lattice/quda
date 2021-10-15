#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the generic (dummy) implementations
   for warp- and block-level reductions.
 */

using namespace quda;

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

  /**
     @brief Dummy generic implementation of warp_reduce
  */
  template <bool is_device> struct warp_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, reducer_t, param_t) { return value; }
  };

  /**
     @brief Dummy generic implementation of block_reduce
  */
  template <bool is_device> struct block_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, int, bool, reducer_t, param_t) { return value; }
  };

  /**
     @brief This allows us to perform reductions at the warp or
     sub-warp level
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
     @brief This allows us to perform reductions at the block level
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
