#pragma once

#include <cub_helper.cuh>

#ifdef QUAD_SUM
using device_reduce_t = doubledouble;
#else
using device_reduce_t = double;
#endif

#ifdef HETEROGENEOUS_ATOMIC
#include <cuda/std/atomic>
using count_t = cuda::atomic<unsigned int, cuda::thread_scope_device>;
#else
using count_t = unsigned int;
#endif

namespace quda
{

  namespace reducer
  {
    /** returns the reduce buffer size allocated */
    size_t buffer_size();

    void *get_device_buffer();
    void *get_mapped_buffer();
    void *get_host_buffer();
    count_t *get_count();
    cudaEvent_t &get_event();
  } // namespace reducer

  /**
     @brief The initialization value we used to check for completion
   */
  template <typename T> constexpr T init_value() { return -std::numeric_limits<T>::infinity(); }
  template <typename T> struct atomic_type {
    using type = device_reduce_t;
  };
  template <> struct atomic_type<float> {
    using type = float;
  };

  template <typename T> struct ReduceArg {
    const int n_reduce; // number of reductions of length n_item
#ifdef HETEROGENEOUS_ATOMIC
    using system_atomic_t = typename atomic_type<T>::type;
    static constexpr int n_item = sizeof(T) / sizeof(system_atomic_t);
    cuda::atomic<T, cuda::thread_scope_device> *partial;
    // for heterogeneous atomics we need to use lock-free atomics -> operate on scalars
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_d;
#else
    T *partial;
    T *result_d;
#endif
    T *result_h;
    count_t *count;

    ReduceArg(int n_reduce = 1) :
      n_reduce(n_reduce),
      partial(static_cast<decltype(partial)>(reducer::get_device_buffer())),
      result_d(static_cast<decltype(result_d)>(reducer::get_mapped_buffer())),
      result_h(static_cast<decltype(result_h)>(reducer::get_host_buffer())),
      count {reducer::get_count()}
    {
      // write reduction to GPU memory if asynchronous
      if (commAsyncReduction()) result_d = static_cast<decltype(result_d)>(reducer::get_device_buffer());

      // check reduction buffers are large enough if requested
      auto max_reduce_blocks = 2 * deviceProp.multiProcessorCount;
      auto reduce_size = max_reduce_blocks * n_reduce * sizeof(*partial);
      if (reduce_size > reducer::buffer_size())
        errorQuda("Requested reduction requires a larger buffer %lu than allocated %lu", reduce_size,
                  reducer::buffer_size());

#ifdef HETEROGENEOUS_ATOMIC
      // initialize the result buffer so we can test for completion
      for (int i = 0; i < n_reduce * n_item; i++) {
        new (result_d + i) cuda::atomic<system_atomic_t, cuda::thread_scope_system> {};
        result_d[i].store(init_value<system_atomic_t>(), cuda::std::memory_order_release);
      }
#endif
    }

    template <typename host_t, typename device_t = host_t>
    void complete(host_t *result, const qudaStream_t stream = 0)
    {
#ifdef HETEROGENEOUS_ATOMIC
      for (int i = 0; i < n_reduce * n_item; i++) {
        while (result_d[i].load(cuda::std::memory_order_acquire) == init_value<system_atomic_t>()) {}
      }
#else
      auto event = reducer::get_event();
      qudaEventRecord(event, stream);
      while (cudaSuccess != qudaEventQuery(event)) {}
#endif
      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t*>(result_h)[i];
    }
  };

#ifdef HETEROGENEOUS_ATOMIC
  /**
     @brief Generic reduction function that reduces block-distributed
     data "in" per thread to a single value.  This is the
     heterogeneous-atomic version which uses std::atomic to signal the
     completion of the reduction to the host.
     
     @param arg The argument struct that must be derived from ReduceArg
     @param in The input per-thread data to be reduced
     @param idx In the case of multiple reductions, idx identifies
     which reduction this thread block corresponds to.  Typically idx
     will be constant along constant blockIdx.y and blockIdx.z.
  */
  template <int block_size_x, int block_size_y, typename T, bool do_sum = true, typename Reducer = cub::Sum, typename Arg>
  __device__ inline void reduce2d(Arg &arg, const T &in, const int idx = 0)
  {
    using BlockReduce = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y>;
    __shared__ typename BlockReduce::TempStorage cub_tmp;
    __shared__ bool isLastBlockDone;

    Reducer r;
    T aggregate = do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      // need to call placement new constructor since arg.partial is not necessarily constructed
      new (arg.partial + idx * gridDim.x + blockIdx.x) cuda::atomic<T, cuda::thread_scope_device> {aggregate};

      // increment global block counter for this reduction
      auto value = arg.count[idx].fetch_add(1, cuda::std::memory_order_release);

      // determine if last block
      isLastBlockDone = (value == (gridDim.x - 1));
    }

    __syncthreads();

    // finish the reduction if last block
    if (isLastBlockDone) {
      unsigned int i = threadIdx.y * block_size_x + threadIdx.x;
      T sum;
      zero(sum);
      while (i < gridDim.x) {
        sum = r(sum, arg.partial[idx * gridDim.x + i].load(cuda::std::memory_order_relaxed));
        i += block_size_x * block_size_y;
      }

      sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

      // write out the final reduced value
      if (threadIdx.y * block_size_x + threadIdx.x == 0) {
        using atomic_t = typename atomic_type<T>::type;
        constexpr size_t n = sizeof(T) / sizeof(atomic_t);
        atomic_t sum_tmp[n];
        memcpy(sum_tmp, &sum, sizeof(sum));
#pragma unroll
        for (int i = 0; i < n; i++) { arg.result_d[n * idx + i].store(sum_tmp[i], cuda::std::memory_order_relaxed); }
        arg.count[idx].store(0, cuda::std::memory_order_relaxed); // set to zero for next time
      }
    }
  }

#else
  /**
     @brief Generic reduction function that reduces block-distributed
     data "in" per thread to a single value.  This is the legacy
     variant which require explicit host-device synchronization to
     signal the completion of the reduction to the host.
     
     @param arg The argument struct that must be derived from ReduceArg
     @param in The input per-thread data to be reduced
     @param idx In the case of multiple reductions, idx identifies
     which reduction this thread block corresponds to.  Typically idx
     will be constant along constant blockIdx.y and blockIdx.z.
  */
  template <int block_size_x, int block_size_y, typename T, bool do_sum = true, typename Reducer = cub::Sum, typename Arg>
  __device__ inline void reduce2d(Arg &arg, const T &in, const int idx = 0)
  {
    using BlockReduce = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y>;
    __shared__ typename BlockReduce::TempStorage cub_tmp;
    __shared__ bool isLastBlockDone;

    Reducer r;
    T aggregate = do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      arg.partial[idx * gridDim.x + blockIdx.x] = aggregate;
      __threadfence(); // flush result

      // increment global block counter
      unsigned int value = atomicInc(&arg.count[idx], gridDim.x);

      // determine if last block
      isLastBlockDone = (value == (gridDim.x - 1));
    }

    __syncthreads();

    // finish the reduction if last block
    if (isLastBlockDone) {
      unsigned int i = threadIdx.y * block_size_x + threadIdx.x;
      T sum;
      zero(sum);
      while (i < gridDim.x) {
        auto partial = const_cast<T &>(static_cast<volatile T *>(arg.partial)[idx * gridDim.x + i]);
        sum = r(sum, partial);
        i += block_size_x * block_size_y;
      }

      sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

      // write out the final reduced value
      if (threadIdx.y * block_size_x + threadIdx.x == 0) {
        arg.result_d[idx] = sum;
        arg.count[idx] = 0; // set to zero for next time
      }
    }
  }
#endif

  template <int block_size, typename T, bool do_sum = true, typename Reducer = cub::Sum, typename Arg>
  __device__ inline void reduce(Arg &arg, const T &in, const int idx = 0)
  {
    reduce2d<block_size, 1, T, do_sum, Reducer>(arg, in, idx);
  }

} // namespace quda
