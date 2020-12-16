#pragma once


#include <target_device.h>
namespace quda {

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the warp or sub-warp level
  */
  template <typename T, int width> struct WarpReduce
  {
    static_assert(width <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    using warp_reduce_t = QudaCub::WarpReduce<T, width>;

    __device__ __host__ inline WarpReduce() {}

    __device__ __host__ inline T Sum(const T &value)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_COMPILE_DEVICE__)
      typename warp_reduce_t::TempStorage dummy;
      warp_reduce_t warp_reduce(dummy);
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

    using block_reduce_t = QudaCub::BlockReduce<T, block_size_x, QudaCub::BLOCK_REDUCE_WARP_REDUCTIONS>;

    __device__ inline auto& shared_state()
    {
      static __shared__ typename block_reduce_t::TempStorage storage[batch_size];
      return storage[batch];
    }

    __device__ __host__ inline BlockReduce(int batch = 0) : batch(batch) {}

    template <bool pipeline = false>
    __device__ __host__ inline T Sum(const T &value_)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_COMPILE_DEVICE__)
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
#if defined( __CUDA_ARCH__) || defined(__HIP_COMPILE_DEVICE__)
      T &value_shared = reinterpret_cast<T&>(shared_state());
      if (threadIdx.x == 0 && threadIdx.y == 0) value_shared = value;
      __syncthreads();
      value = value_shared;
#endif
      return value;
    }
  };

} // namespace quda
