#pragma once
//#include <quda_constants.h>
//#include <float_vector.h>
//#include <comm_quda.h>
#include <blas_quda.h>

using namespace quda;

/**
   @file reduce_helper_cpu.h

   @section Description

   Provides helper functors for reduction algorithms.
 */

template <typename T> struct Sum {
  __device__ __host__ T operator()(T a, T b) { return a + b; }
};

template <typename T> struct Maximum {
  __device__ __host__ T operator()(T a, T b) { return a > b ? a : b; }
};

template <typename T> struct Minimum {
  __device__ __host__ T operator()(T a, T b) { return a < b ? a : b; }
};

template <typename T> struct Identity {
  __device__ __host__ T operator()(T a) { return a; }
};

namespace quda {

  template <typename T>
  struct ReduceArg {
    T *partial;
    T *result_h;
    ReduceArg() :
      partial(static_cast<T*>(blas::getDeviceReduceBuffer())),
      result_h(static_cast<T*>(blas::getHostReduceBuffer()))
    {
    }
  };

#ifdef QUAD_SUM
  inline void zero(doubledouble &x) { x.a.x = 0.0; x.a.y = 0.0; }
  inline void zero(doubledouble2 &x) { zero(x.x); zero(x.y); }
  inline void zero(doubledouble3 &x) { zero(x.x); zero(x.y); zero(x.z); }
#endif

  unsigned int count[QUDA_MAX_MULTI_REDUCE] = { };
  bool isLastBlockDone;

  template <int block_size_x, int block_size_y, typename T, bool do_sum=true, typename Reducer=Sum<T>>
  inline void reduce2d(ReduceArg<T> arg, const T &in, const int idx=0) {

    Reducer r;
    #if 0
    T aggregate = (do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r));

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      arg.partial[idx*gridDim.x + blockIdx.x] = aggregate;
      __threadfence(); // flush result

      // increment global block counter
      unsigned int value = atomicInc(&count[idx], gridDim.x);

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
        sum = r(sum, arg.partial[idx*gridDim.x + i]);
	//sum += arg.partial[idx*gridDim.x + i];
	i += block_size_x*block_size_y;
      }

      sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum,r));

      // write out the final reduced value
      if (threadIdx.y*block_size_x + threadIdx.x == 0) {
	arg.result_d[idx] = sum;
	count[idx] = 0; // set to zero for next time
      }
    }
    #endif
    arg.result_h[idx] = in;
  }

  template <int block_size, typename T, bool do_sum=true, typename Reducer=Sum<T>>
  inline void reduce(ReduceArg<T> arg, const T &in, const int idx=0) {
    reduce2d<block_size, 1, T, do_sum, Reducer>(arg, in, idx);
  }

  /**
     functor that defines how to do a row-wise vector reduction
   */
  template <typename T>
  struct reduce_vector {
    inline T operator()(const T &a, const T &b) {
      T sum;
      for (int i=0; i<sum.size(); i++) sum[i] = a[i] + b[i];
      return sum;
    }
  };

} // namespace quda
