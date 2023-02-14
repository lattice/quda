#pragma once

#include <cuda/pipeline>

namespace quda
{

#if (__COMPUTE_CAPABILITY__ >= 800)
  struct pipeline_t {
    cuda::pipeline<cuda::thread_scope_thread> pipe;

    __device__ inline void producer_acquire() { pipe.producer_acquire(); }

    __device__ inline void producer_commit() { pipe.producer_commit(); }

    __device__ inline void consumer_wait() { pipe.consumer_wait(); }

    __device__ inline void consumer_release() { pipe.consumer_release(); }
  };

  __device__ inline pipeline_t make_pipeline() {
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

  __device__ inline pipeline_t make_pipeline() {
    pipeline_t p;
    return p;
  }
#endif

  template <class T>
  __device__ inline void memcpy_async(T *destination, T *source, size_t size, pipeline_t &pipe) {
#if (__COMPUTE_CAPABILITY__ >= 800)
    cuda::memcpy_async(destination, source, size, pipe.pipe);
#else
    *destination = *source;
#endif
  }

}

