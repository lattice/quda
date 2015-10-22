#pragma once
#include <cub/cub.cuh>

/**
   @file cub_helper.cuh

   @section Description
 
   Provides helper functors for custom datatypes for cub algorithms.
 */

namespace quda {

  /**
     Helper functor for generic addition reduction.
  */
  template <typename T>
  struct Summ {
    __host__ __device__ __forceinline__ T operator() (const T &a, const T &b){
      return a + b;
    }
  };

  /**
     Helper functor for double2 addition reduction.
  */
  template <>
  struct Summ<double2>{
    __host__ __device__ __forceinline__ double2 operator() (const double2 &a, const double2 &b){
      return make_double2(a.x + b.x, a.y + b.y);
    }
  };


  template <typename T>
  struct ReduceArg {
    T *partial;
    T *result_d;
    T *result_h;
    ReduceArg() :
      partial(static_cast<T*>(getDeviceReduceBuffer())),
      result_d(static_cast<T*>(getMappedHostReduceBuffer())),
      result_h(static_cast<T*>(getHostReduceBuffer())) 
    { }

  };

  __device__ inline void zero(double &a) { a = 0.0; }
  __device__ inline void zero(double2 &a) { a.x = 0.0; a.y = 0.0; }

  __device__ unsigned int count = 0;
  __shared__ bool isLastBlockDone;

  template <int block_size_x, int block_size_y, typename T>
  __device__ inline void reduce2d(ReduceArg<T> arg, const T &in) {

    typedef cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduce;
    __shared__ typename BlockReduce::TempStorage cub_tmp;

    T aggregate = BlockReduce(cub_tmp).Reduce(in, Summ<T>());

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      arg.partial[blockIdx.x] = aggregate;
      __threadfence(); // flush result

      // increment global block counter
      unsigned int value = atomicInc(&count, gridDim.x);

      // determine if last block
      isLastBlockDone = (value == (gridDim.x-1));
    }

    __syncthreads();

    // finish the reduction if last block
    if (isLastBlockDone) {
      unsigned int i = threadIdx.y*block_size_x + threadIdx.x;
      T sum;
      zero(sum);
      while (i<gridDim.x) {
	sum += arg.partial[i];
	i += block_size_x*block_size_y;
      }

      sum = BlockReduce(cub_tmp).Reduce(sum, Summ<T>());

      // write out the final reduced value
      if (threadIdx.y*block_size_x + threadIdx.x == 0) {
	*arg.result_d = sum;
	count = 0; // set to zero for next time
      }
    }
  }

  template <int block_size, typename T>
  __device__ inline void reduce(ReduceArg<T> arg, const T &in) {
    reduce2d<block_size, 1, T>(arg, in);
  }

} // namespace quda
