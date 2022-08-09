#pragma once

#include <quda_internal.h>

#ifdef HETEROGENEOUS_ATOMIC

#include <target_device.h>
#include <block_reduce_helper.h>
#include <kernel_helper.h>

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

  // declaration of reduce function
  template <typename Reducer, typename Arg, typename T>
  __device__ inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx = 0);

  /**
     @brief ReduceArg is the argument type that all kernel arguments
     shoud inherit from if the kernel is to utilize global reductions.
     @tparam T the type that will be reduced
     @tparam use_kernel_arg Whether the kernel will source the
     parameter struct as an explicit kernel argument or from constant
     memory
   */
  template <typename T, use_kernel_arg_p use_kernel_arg = use_kernel_arg_p::TRUE> struct ReduceArg : kernel_param<use_kernel_arg> {
    using reduce_t = T;

    template <typename Arg, typename I> friend __device__ void write_result(Arg &, const I &, const int);
    template <typename Reducer, typename Arg, typename I>
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
    T *device_output_async_buffer = nullptr; // Optional device output buffer for the reduction result

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
      @brief Set device_output_async_buffer
    */
    void set_output_async_buffer(T *ptr)
    {
      if (!commAsyncReduction()) {
        errorQuda("When setting the asynchronous buffer the commAsyncReduction option must be set.");
      }
      device_output_async_buffer = ptr;
    }

    /**
      @brief Get device_output_async_buffer
    */
    __device__ __host__ T *get_output_async_buffer() const { return device_output_async_buffer; }

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
     @brief Write the reduced result out.  Three different destinations are supported:
       * If arg.result_d is non-null we atomically write out the
         result (to host-mapped memory)
       * Else if arg.get_output_async_buffer returns non-zero we
         non-atomically write out the result to device memory
       * Else we atomically write out the result to device memory
     @param[in,out] arg Kernel argument
     @param[in] sum Reduced value to be written out
     @param[in] idx Memory index we are writing to
   */
  template <typename Arg, typename T> __device__ inline void write_result(Arg &arg, const T &sum, const int idx)
  {
    using atomic_t = typename atomic_type<T>::type;
    constexpr size_t n = sizeof(T) / sizeof(atomic_t);
    auto tid = target::thread_idx_linear<2>();

    if (arg.result_d) { // write to host mapped memory
#ifdef _NVHPC_CUDA      // WAR for nvc++
      constexpr bool coalesced_write = false;
#else
      constexpr bool coalesced_write = true;
#endif
      if constexpr (coalesced_write) {
        static_assert(n <= device::warp_size(), "reduction array is greater than warp size");
        auto mask = __ballot_sync(0xffffffff, tid < n);
        if (tid < n) {
          atomic_t sum_tmp[n];
          memcpy(sum_tmp, &sum, sizeof(sum));

          atomic_t s = sum_tmp[0];
#pragma unroll
          for (int i = 1; i < n; i++) {
            auto si = __shfl_sync(mask, sum_tmp[i], 0);
            if (i == tid) s = si;
          }

          s = (s == init_value<atomic_t>()) ? terminate_value<atomic_t>() : s;
          arg.result_d[n * idx + tid].store(s, cuda::std::memory_order_relaxed);
        }
      } else {
        // write out the final reduced value
        if (tid == 0) {
          atomic_t sum_tmp[n];
          memcpy(sum_tmp, &sum, sizeof(sum));
#pragma unroll
          for (unsigned int i = 0; i < n; i++) {
            // catch the case where the computed value is equal to the init_value
            sum_tmp[i] = sum_tmp[i] == init_value<atomic_t>() ? terminate_value<atomic_t>() : sum_tmp[i];
            arg.result_d[n * idx + i].store(sum_tmp[i], cuda::std::memory_order_relaxed);
          }
        }
      }
    } else {

      if (tid == 0) {
        if (arg.get_output_async_buffer()) {
          arg.get_output_async_buffer()[idx] = sum;
        } else { // write to device memory
          arg.partial[idx].store(sum, cuda::std::memory_order_relaxed);
        }
      }
    }

    if (tid == 0) {
      // TODO in principle we could remove this final atomic store
      // if we use a sense reversal barrier, avoiding the need to
      // reset the count at the end
      arg.count[idx].store(0, cuda::std::memory_order_relaxed); // set to zero for next time
    }
  }

  /**
     @brief Generic reduction function that reduces block-distributed
     data "in" per thread to a single value.  This is the
     heterogeneous-atomic version which uses std::atomic to signal the
     completion of the reduction to the host.

     The reduce function supports:
     - a global reduction across the x thread dimension
     - a local block reduction across the y thread dimension
     - the z thread dimension is a batching dimension in the case of independent reductions

     @param[in,out] arg The kernel argument, this must derive from ReduceArg
     @param[in] r Instance of the reducer to be used in this reduction
     @param[in] in The input per-thread data to be reduced
     @param[in] idx In the case of multiple reductions, idx identifies
     which reduction this thread block corresponds to and should be
     constant along the x and y thread dimensions.
  */
  template <typename Reducer, typename Arg, typename T>
  __device__ inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx)
  {
    constexpr auto n_batch_block = std::min(Arg::max_n_batch_block, device::max_block_size());
    using BlockReduce = BlockReduce<T, Reducer::reduce_block_dim, n_batch_block>;

    T aggregate = BlockReduce(target::thread_idx().z).Reduce(in, r);

    if (target::grid_dim().x == 1) { // short circuit where we have a single CTA - no need to do final reduction
      write_result(arg, aggregate, idx);
    } else {
      __shared__ bool isLastBlockDone[n_batch_block];

      if (target::thread_idx().x == 0 && target::thread_idx().y == 0) {
        // need to call placement new constructor since partial is not necessarily constructed
        new (arg.partial + idx * target::grid_dim().x + target::block_idx().x)
          cuda::atomic<T, cuda::thread_scope_device> {aggregate};

        // increment global block counter for this reduction
        auto value = arg.count[idx].fetch_add(1, cuda::std::memory_order_release);

        // determine if last block
        isLastBlockDone[target::thread_idx().z] = (value == (target::grid_dim().x - 1));
      }

      __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone[target::thread_idx().z]) {
        auto i = target::thread_idx().y * target::block_dim().x + target::thread_idx().x;
        T sum = r.init();
        while (i < target::grid_dim().x) {
          sum = r(sum, arg.partial[idx * target::grid_dim().x + i].load(cuda::std::memory_order_relaxed));
          i += target::block_size<2>();
        }

        sum = BlockReduce(target::thread_idx().z).Reduce(sum, r);

        write_result(arg, sum, idx);
      }
    }
  }

} // namespace quda

#else

// if not using heterogeneous atomics use the generic variant
#include "../generic/reduce_helper.h"

#endif
