#pragma once

#include <quda_internal.h>

#ifdef HETEROGENEOUS_ATOMIC

#include <target_device.h>
#include <block_reduce_helper.h>
#include <kernel_helper.h>

#ifdef QUAD_SUM
using device_reduce_t = doubledouble;
#else
using device_reduce_t = double;
#endif

#include <cuda/std/climits>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/atomic>
using count_t = cuda::atomic<unsigned int, cuda::thread_scope_device>;

namespace quda
{

  /**
     @brief The initialization value we used to check for completion
   */
  template <typename T> constexpr T init_value() { return -cuda::std::numeric_limits<T>::infinity(); }

  /**
     @brief The termination value we use to prevent a possible hang in
     case the computed reduction is equal to the initialization
  */
  template <typename T> constexpr T terminate_value() { return cuda::std::numeric_limits<T>::infinity(); }

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

  // declaration of reduce function
  template <int block_size_x, int block_size_y = 1, typename Reducer, typename Arg, typename T>
  __device__ inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx = 0);

  /**
     @brief ReduceArg is the argument type that all kernel arguments
     shoud inherit from if the kernel is to utilize global reductions.
     @tparam T the type that will be reduced
     @tparam use_kernel_arg Whether the kernel will source the
     parameter struct as an explicit kernel argument or from constant
     memory
   */
  template <typename T, bool use_kernel_arg = true> struct ReduceArg : kernel_param<use_kernel_arg> {

    template <int, int, typename Reducer, typename Arg, typename I>
    friend __device__ void reduce(Arg &, const Reducer &, const I &, const int);
    qudaError_t launch_error; /** only do complete if no launch error to avoid hang */
    static constexpr unsigned int max_n_batch_block
      = 1; /** by default reductions do not support batching withing the block */

  private:
    const int n_reduce; /** number of reductions of length n_item */
    bool reset = false; /** reset the counter post completion (required for multiple calls with the same arg instance */
    using system_atomic_t = typename atomic_type<T>::type; /** heterogeneous atomics must use lock-free atomics -> operate on scalars */
    static constexpr int n_item = sizeof(T) / sizeof(system_atomic_t); /** number of words per reduction variable */
    cuda::atomic<T, cuda::thread_scope_device> *partial; /** device atomic buffer */
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_d; /** device-mapped host atomic buffer */
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_h; /** host atomic buffer */
    count_t *count; /** count array that is used to track the number of completed thread blocks at a given batch index */
    bool consumed; // check to ensure that we don't complete more than once unless we explicitly reset

  public:
    /**
       @brief Constructor for ReduceArg
       @param[in] threads The number threads partaking in the kernel
       @param[in] n_reduce The number of reductions
       @param[in] reset Whether to reset the atomics after the
       reduction has completed; required if the same ReduceArg
       instance will be used for multiple reductions.
    */
    ReduceArg(dim3 threads, int n_reduce = 1, bool reset = false) :
      kernel_param<use_kernel_arg>(threads),
      launch_error(QUDA_ERROR_UNINITIALIZED),
      n_reduce(n_reduce),
      reset(reset),
      consumed(false)
    {
      reducer::init(n_reduce, sizeof(*partial));
      // these buffers may be allocated in init, so we can't set the local copies until now
      partial = static_cast<decltype(partial)>(reducer::get_device_buffer());
      result_d = static_cast<decltype(result_d)>(reducer::get_mapped_buffer());
      result_h = static_cast<decltype(result_h)>(reducer::get_host_buffer());
      count = reducer::get_count<count_t>();

      if (!commAsyncReduction()) {
        // initialize the result buffer so we can test for completion
        for (int i = 0; i < n_reduce * n_item; i++) {
          new (result_h + i) cuda::atomic<system_atomic_t, cuda::thread_scope_system> {init_value<system_atomic_t>()};
        }
        cuda::std::atomic_thread_fence(cuda::std::memory_order_release);
      } else {
        // write reduction to GPU memory if asynchronous (reuse partial)
        result_d = nullptr;
      }
    }

    /**
       @brief Finalize the reduction, returning the computed reduction
       into result.  With heterogeneous atomics this means we poll the
       atomics until their value differs from the init_value.
       @param[out] result The reduction result is copied here
       @param[in] stream The stream on which we the reduction is being done
     */
    template <typename host_t, typename device_t = host_t>
    void complete(std::vector<host_t> &result, const qudaStream_t = device::get_default_stream())
    {
      if (launch_error == QUDA_ERROR) return; // kernel launch failed so return
      if (launch_error == QUDA_ERROR_UNINITIALIZED) errorQuda("No reduction kernel appears to have been launched");
      if (consumed) errorQuda("Cannot call complete more than once for each construction");

      for (int i = 0; i < n_reduce * n_item; i++) {
        while (result_h[i].load(cuda::std::memory_order_relaxed) == init_value<system_atomic_t>()) { }
      }

      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        errorQuda("result vector length %lu does not match n_reduce %d", result.size(), n_element);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];

      if (!reset) {
        consumed = true;
      } else {
        // reset the atomic counter - this allows multiple calls to complete with ReduceArg construction
        for (int i = 0; i < n_reduce * n_item; i++) {
          result_h[i].store(init_value<system_atomic_t>(), cuda::std::memory_order_relaxed);
        }
        cuda::std::atomic_thread_fence(cuda::std::memory_order_release);
      }
    }
  };

  /**
     @brief Generic reduction function that reduces block-distributed
     data "in" per thread to a single value.  This is the
     heterogeneous-atomic version which uses std::atomic to signal the
     completion of the reduction to the host.

     @param arg The kernel argument, this must derive from ReduceArg
     @param r Instance of the reducer to be used in this reduction
     @param in The input per-thread data to be reduced
     @param idx In the case of multiple reductions, idx identifies
     which reduction this thread block corresponds to.  Typically idx
     will be constant along constant block_idx().y and block_idx().z.
  */
  template <int block_size_x, int block_size_y, typename Reducer, typename Arg, typename T>
  __device__ inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx)
  {
    constexpr auto n_batch_block
      = std::min(Arg::max_n_batch_block, device::max_block_size() / (block_size_x * block_size_y));
    using BlockReduce = BlockReduce<T, block_size_x, block_size_y, n_batch_block, true>;
    __shared__ bool isLastBlockDone[n_batch_block];

    T aggregate = BlockReduce(target::thread_idx().z).Reduce(in, r);

    if (target::thread_idx().x == 0 && target::thread_idx().y == 0) {
      // need to call placement new constructor since partial is not necessarily constructed
      new (arg.partial + idx * target::grid_dim().x + target::block_idx().x) cuda::atomic<T, cuda::thread_scope_device> {aggregate};

      // increment global block counter for this reduction
      auto value = arg.count[idx].fetch_add(1, cuda::std::memory_order_release);

      // determine if last block
      isLastBlockDone[target::thread_idx().z] = (value == (target::grid_dim().x - 1));
    }

    __syncthreads();

    // finish the reduction if last block
    if (isLastBlockDone[target::thread_idx().z]) {
      auto i = target::thread_idx().y * block_size_x + target::thread_idx().x;
      T sum = arg.init();
      while (i < target::grid_dim().x) {
        sum = r(sum, arg.partial[idx * target::grid_dim().x + i].load(cuda::std::memory_order_relaxed));
        i += block_size_x * block_size_y;
      }

      sum = BlockReduce(target::thread_idx().z).Reduce(sum, r);

      // write out the final reduced value
      if (target::thread_idx().y * block_size_x + target::thread_idx().x == 0) {
        if (arg.result_d) { // write to host mapped memory
          using atomic_t = typename atomic_type<T>::type;
          constexpr size_t n = sizeof(T) / sizeof(atomic_t);
          atomic_t sum_tmp[n];
          memcpy(sum_tmp, &sum, sizeof(sum));
#pragma unroll
          for (unsigned int i = 0; i < n; i++) {
            // catch the case where the computed value is equal to the init_value
            sum_tmp[i] = sum_tmp[i] == init_value<atomic_t>() ? terminate_value<atomic_t>() : sum_tmp[i];
            arg.result_d[n * idx + i].store(sum_tmp[i], cuda::std::memory_order_relaxed);
          }
        } else { // write to device memory
          arg.partial[idx].store(sum, cuda::std::memory_order_relaxed);
        }
        // TODO in principle we could remove this final atomic store
        // if we use a sense reversal barrier, avoiding the need to
        // reset the count at the end
        arg.count[idx].store(0, cuda::std::memory_order_relaxed); // set to zero for next time
      }
    }
  }

} // namespace quda

#else

// if not using heterogeneous atomics use the generic variant
#include "../generic/reduce_helper.h"

#endif
