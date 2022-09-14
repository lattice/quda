#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the generic (dummy) implementations
   for warp- and block-level reductions.
 */

using namespace quda;

namespace quda
{

  /**
     @brief warp_reduce_param is used as a container for passing
     non-type parameters to specialize warp_reduce through the
     target::dispatch
     @tparam width The number of logical threads taking part in the warp reduction
   */
  template <int width_ = device::warp_size()> struct warp_reduce_param {
    static_assert(width_ <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    static constexpr int width = width_;
  };

  /**
     @brief block_reduce_param is used as a container for passing
     non-type parameters to specialize block_reduce through the
     target::dispatch
     @tparam block_dim Block dimension of the reduction (1, 2 or 3)
     @tparam batch_size Batch size of the reduction.  Threads will be
     ordered such that batch size is the slowest running index.  Note
     that batch_size > 1 requires block_dim <= 2.
   */
  template <int block_dim_, int batch_size_ = 1> struct block_reduce_param {
    static constexpr int block_dim = block_dim_;
    static constexpr int batch_size = batch_size_;
    static_assert(batch_size == 1 || block_dim <= 2, "Batching not possible with 3-d block reduction");
  };

  /**
     @brief Dummy generic implementation of warp_reduce
  */
  template <bool is_device> struct warp_reduce {
    template <typename T, typename reducer_t, typename param_t> T operator()(const T &value, bool, reducer_t, param_t)
    {
      return value;
    }
  };

  /**
     @brief Dummy generic implementation of block_reduce
  */
  template <bool is_device> struct block_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, int, bool, reducer_t, param_t)
    {
      return value;
    }
  };

  /**
     @brief WarpReduce provides a generic interface for performing
     perform reductions at the warp or sub-warp level
     @tparam T The type of the value that we are reducing
     @tparam width The number of logical threads taking part in the warp reduction
  */
  template <typename T, int width> class WarpReduce
  {
    using param_t = warp_reduce_param<width>;

  public:
    constexpr WarpReduce() { }

    /**
       @brief Perform a warp-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    __device__ __host__ inline T Sum(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    __device__ __host__ inline T AllSum(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    __device__ __host__ inline T Max(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    __device__ __host__ inline T AllMax(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    __device__ __host__ inline T Min(const T &value)
    {
      return target::dispatch<warp_reduce>(value, false, quda::minimum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    __device__ __host__ inline T AllMin(const T &value)
    {
      return target::dispatch<warp_reduce>(value, true, quda::minimum<T>(), param_t());
    }
  };

  /**
     @brief BlockReduce provides a generic interface for performing
     reductions at the block level
     @tparam T The type of the value that we are reducing
     @tparam block_dim The number of thread block dimensions we are reducing
     @tparam batch_size Batch size of the reduction.  Threads will be
     ordered such that batch size is the slowest running index.
  */
  template <typename T, int block_dim, int batch_size = 1> class BlockReduce
  {
    using param_t = block_reduce_param<block_dim, batch_size>;
    const int batch;

  public:
    constexpr BlockReduce(int batch = 0) : batch(batch) { }

    /**
       @brief Perform a block-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> __device__ __host__ inline T Sum(const T &value)
    {
      return target::dispatch<block_reduce>(value, async, batch, false, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a block-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllSum(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllSum with batch_size > 1");
      return target::dispatch<block_reduce>(value, async, batch, true, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a block-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> __device__ __host__ inline T Max(const T &value)
    {
      return target::dispatch<block_reduce>(value, async, batch, false, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a block-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllMax(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllMax with batch_size > 1");
      return target::dispatch<block_reduce>(value, async, batch, true, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a block-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> __device__ __host__ inline T Min(const T &value)
    {
      return target::dispatch<block_reduce>(value, async, batch, false, quda::minimum<T>(), param_t());
    }

    /**
       @brief Perform a block-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllMin(const T &value)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllMin with batch_size > 1");
      return target::dispatch<block_reduce>(value, async, batch, true, quda::minimum<T>(), param_t());
    }

    /**
       @brief Perform a block-wide custom reduction
       @param[in] value Thread-local value to be reduced
       @param[in] r The reduction operation we want to apply
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true, typename reducer_t>
    __device__ __host__ inline T Reduce(const T &value, const reducer_t &r)
    {
      return target::dispatch<block_reduce>(value, async, batch, false, r, param_t());
    }

    /**
       @brief Perform a block-wide custom reduction
       @param[in] value Thread-local value to be reduced
       @param[in] r The reduction operation we want to apply
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true, typename reducer_t>
    __device__ __host__ inline T AllReduce(const T &value, const reducer_t &r)
    {
      static_assert(param_t::batch_size == 1, "Cannot do AllReduce with batch_size > 1");
      return target::dispatch<block_reduce>(value, async, batch, true, r, param_t());
    }
  };

} // namespace quda
