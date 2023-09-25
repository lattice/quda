#pragma once

/** @file The wrapper here abstract the cuda::pipeline, but _only_ when
 *     we believe it gives the better performance.
 */

#if (__COMPUTE_CAPABILITY__ >= 800) && (CUDA_VERSION >= 11080)
#define QUDA_USE_CUDA_PIPELINE
#include <cuda/pipeline>
#endif

namespace quda
{

#ifdef QUDA_USE_CUDA_PIPELINE
  struct pipeline_t {
    cuda::pipeline<cuda::thread_scope_thread> pipe;

    __device__ inline void producer_acquire() { pipe.producer_acquire(); }

    __device__ inline void producer_commit() { pipe.producer_commit(); }

    __device__ inline void consumer_wait() { pipe.consumer_wait(); }

    __device__ inline void consumer_release() { pipe.consumer_release(); }
  };

  __device__ inline pipeline_t make_pipeline()
  {
    pipeline_t p = {cuda::make_pipeline()};
    return p;
  }
#else
  struct pipeline_t {
    __device__ inline void producer_acquire() { }

    __device__ inline void producer_commit() { }

    __device__ inline void consumer_wait() { }

    __device__ inline void consumer_release() { }
  };

  __device__ inline pipeline_t make_pipeline()
  {
    pipeline_t p;
    return p;
  }
#endif

  template <class T> __device__ inline void memcpy_async(T *destination, T *source, size_t size, pipeline_t &pipe)
  {
#ifdef QUDA_USE_CUDA_PIPELINE
    cuda::memcpy_async(destination, source, size, pipe.pipe);
#else
    *destination = *source;
#endif
  }

} // namespace quda
