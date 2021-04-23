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

  constexpr int max_n_reduce() { return QUDA_MAX_MULTI_REDUCE; }

  /**
     @brief The initialization value we used to check for completion
   */
  template <typename T> constexpr T init_value() { return -std::numeric_limits<T>::infinity(); }

  /**
     @brief The termination value we use to prevent a possible hang in
     case the computed reduction is equal to the initialization
  */
  template <typename T> constexpr T terminate_value() { return std::numeric_limits<T>::infinity(); }

  /**
     @brief The atomic word size we use for a given reduction type.
     This type should be lock-free to guarantee correct behaviour on
     platforms that are not coherent with respect to the host
   */
  template <typename T> struct atomic_type {
    using type = device_reduce_t;
  };
  template <> struct atomic_type<float> {
    using type = float;
  };

  template <typename T> struct ReduceArg {

    qudaError_t launch_error; // only do complete if no launch error to avoid hang

  private:
    const int n_reduce; // number of reductions of length n_item
#ifdef HETEROGENEOUS_ATOMIC
    using system_atomic_t = typename atomic_type<T>::type;
    static constexpr int n_item = sizeof(T) / sizeof(system_atomic_t);
    cuda::atomic<T, cuda::thread_scope_device> *partial;
    // for heterogeneous atomics we need to use lock-free atomics -> operate on scalars
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_d;
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_h;
#else
    T *partial;
    T *result_d;
    T *result_h;
#endif
    count_t *count;
    bool consumed; // check to ensure that we don't complete more than once unless we explicitly reset

  public:
    ReduceArg(int n_reduce = 1) :
      launch_error(QUDA_ERROR_UNINITIALIZED),
      n_reduce(n_reduce),
      partial(static_cast<decltype(partial)>(reducer::get_device_buffer())),
      result_d(static_cast<decltype(result_d)>(reducer::get_mapped_buffer())),
      result_h(static_cast<decltype(result_h)>(reducer::get_host_buffer())),
      count {reducer::get_count()},
      consumed(false)
    {
      // check reduction buffers are large enough if requested
      auto max_reduce_blocks = 2 * deviceProp.multiProcessorCount;
      auto reduce_size = max_reduce_blocks * n_reduce * sizeof(*partial);
      if (reduce_size > reducer::buffer_size())
        errorQuda("Requested reduction requires a larger buffer %lu than allocated %lu", reduce_size,
                  reducer::buffer_size());

#ifdef HETEROGENEOUS_ATOMIC
      if (!commAsyncReduction()) {
        // initialize the result buffer so we can test for completion
        for (int i = 0; i < n_reduce * n_item; i++) {
          new (result_h + i) cuda::atomic<system_atomic_t, cuda::thread_scope_system> {init_value<system_atomic_t>()};
        }
        std::atomic_thread_fence(std::memory_order_release);
      } else {
        // write reduction to GPU memory if asynchronous (reuse partial)
        result_d = nullptr;
      }
#else
      if (commAsyncReduction()) result_d = partial;
#endif
    }

    /**
       @brief Finalize the reduction, returning the computed reduction
       into result.  With heterogeneous atomics this means we poll the
       atomics until their value differs from the init_value.  The
       alternate legacy path posts an event after the kernel and then
       polls on completion of the event.
       @param[out] result The reduction result is copied here
       @param[in] stream The stream on which we the reduction is being done
       @param[in] reset Whether to reset the atomics after the
       reduction has completed; required if the same aReducearg
       instance will be used for multiple reductions.
     */
    template <typename host_t, typename device_t = host_t>
    void complete(std::vector<host_t> &result, const qudaStream_t stream = 0, bool reset = false)
    {
      if (launch_error == QUDA_ERROR) return; // kernel launch failed so return
      if (launch_error == QUDA_ERROR_UNINITIALIZED) errorQuda("No reduction kernel appears to have been launched");
#ifdef HETEROGENEOUS_ATOMIC
      if (consumed) errorQuda("Cannot call complete more than once for each construction");

      for (int i = 0; i < n_reduce * n_item; i++) {
        while (result_h[i].load(cuda::std::memory_order_relaxed) == init_value<system_atomic_t>()) {}
      }
#else
      auto event = reducer::get_event();
      qudaEventRecord(event, stream);
      while (!qudaEventQuery(event)) {}
#endif
      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        errorQuda("result vector length %lu does not match n_reduce %d", result.size(), n_reduce);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];

#ifdef HETEROGENEOUS_ATOMIC
      if (!reset) {
        consumed = true;
      } else {
        // reset the atomic counter - this allows multiple calls to complete with ReduceArg construction
        for (int i = 0; i < n_reduce * n_item; i++) {
          result_h[i].store(init_value<system_atomic_t>(), cuda::std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_release);
      }
#endif
    }

    /**
       @brief Overload providing a simple reference interface
       @param[out] result The reduction result is copied here
       @param[in] stream The stream on which we the reduction is being done
       @param[in] reset Whether to reset the atomics after the
       reduction has completed; required if the same ReduceArg
       instance will be used for multiple reductions.
     */
    template <typename host_t, typename device_t = host_t>
    void complete(host_t &result, const qudaStream_t stream = 0, bool reset = false)
    {
      std::vector<host_t> result_(1);
      complete(result_, stream, reset);
      result = result_[0];
    }

#ifdef HETEROGENEOUS_ATOMIC
    /**
       @brief Generic reduction function that reduces block-distributed
       data "in" per thread to a single value.  This is the
       heterogeneous-atomic version which uses std::atomic to signal the
       completion of the reduction to the host.

       @param in The input per-thread data to be reduced
       @param idx In the case of multiple reductions, idx identifies
       which reduction this thread block corresponds to.  Typically idx
       will be constant along constant blockIdx.y and blockIdx.z.
    */
    template <int block_size_x, int block_size_y, bool do_sum = true, typename Reducer = cub::Sum>
    __device__ inline void reduce2d(const T &in, const int idx = 0)
    {
      using BlockReduce = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y>;
      __shared__ typename BlockReduce::TempStorage cub_tmp;
      __shared__ bool isLastBlockDone;

      Reducer r;
      T aggregate = do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r);

      if (threadIdx.x == 0 && threadIdx.y == 0) {
        // need to call placement new constructor since partial is not necessarily constructed
        new (partial + idx * gridDim.x + blockIdx.x) cuda::atomic<T, cuda::thread_scope_device> {aggregate};

        // increment global block counter for this reduction
        auto value = count[idx].fetch_add(1, cuda::std::memory_order_release);

        // determine if last block
        isLastBlockDone = (value == (gridDim.x - 1));
      }

      __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        auto i = threadIdx.y * block_size_x + threadIdx.x;
        T sum;
        zero(sum);
        while (i < gridDim.x) {
          sum = r(sum, partial[idx * gridDim.x + i].load(cuda::std::memory_order_relaxed));
          i += block_size_x * block_size_y;
        }

        sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

        // write out the final reduced value
        if (threadIdx.y * block_size_x + threadIdx.x == 0) {
          if (result_d) { // write to host mapped memory
            using atomic_t = typename atomic_type<T>::type;
            constexpr size_t n = sizeof(T) / sizeof(atomic_t);
            atomic_t sum_tmp[n];
            memcpy(sum_tmp, &sum, sizeof(sum));
#pragma unroll
            for (int i = 0; i < n; i++) {
              // catch the case where the computed value is equal to the init_value
              sum_tmp[i] = sum_tmp[i] == init_value<atomic_t>() ? terminate_value<atomic_t>() : sum_tmp[i];
              result_d[n * idx + i].store(sum_tmp[i], cuda::std::memory_order_relaxed);
            }
          } else { // write to device memory
            partial[idx].store(sum, cuda::std::memory_order_relaxed);
          }
          count[idx].store(0, cuda::std::memory_order_relaxed); // set to zero for next time
        }
      }
    }

#else
    /**
       @brief Generic reduction function that reduces block-distributed
       data "in" per thread to a single value.  This is the legacy
       variant which require explicit host-device synchronization to
       signal the completion of the reduction to the host.

       @param in The input per-thread data to be reduced
       @param idx In the case of multiple reductions, idx identifies
       which reduction this thread block corresponds to.  Typically idx
       will be constant along constant blockIdx.y and blockIdx.z.
    */
    template <int block_size_x, int block_size_y, bool do_sum = true, typename Reducer = cub::Sum>
    __device__ inline void reduce2d(const T &in, const int idx = 0)
    {
      using BlockReduce = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y>;
      __shared__ typename BlockReduce::TempStorage cub_tmp;
      __shared__ bool isLastBlockDone;

      Reducer r;
      T aggregate = do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r);

      if (threadIdx.x == 0 && threadIdx.y == 0) {
        partial[idx * gridDim.x + blockIdx.x] = aggregate;
        __threadfence(); // flush result

        // increment global block counter
        auto value = atomicInc(&count[idx], gridDim.x);

        // determine if last block
        isLastBlockDone = (value == (gridDim.x - 1));
      }

      __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        auto i = threadIdx.y * block_size_x + threadIdx.x;
        T sum;
        zero(sum);
        while (i < gridDim.x) {
          sum = r(sum, const_cast<T &>(static_cast<volatile T *>(partial)[idx * gridDim.x + i]));
          i += block_size_x * block_size_y;
        }

        sum = (do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

        // write out the final reduced value
        if (threadIdx.y * block_size_x + threadIdx.x == 0) {
          result_d[idx] = sum;
          count[idx] = 0; // set to zero for next time
        }
      }
    }
#endif

    template <int block_size, bool do_sum = true, typename Reducer = cub::Sum>
    __device__ inline void reduce(const T &in, const int idx = 0)
    {
      reduce2d<block_size, 1, do_sum, Reducer>(in, idx);
    }
  };

} // namespace quda
