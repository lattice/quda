#pragma once

#include <quda_internal.h>
#include <target_device.h>
#include <block_reduce_helper.h>
#include <kernel_helper.h>

#ifdef QUAD_SUM
using device_reduce_t = doubledouble;
#else
using device_reduce_t = double;
#endif

using count_t = unsigned int;

namespace quda
{

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
    using reduce_t = T;

    template <int, int, typename Reducer, typename Arg, typename I>
    friend __device__ void reduce(Arg &, const Reducer &, const I &, const int);
    qudaError_t launch_error; /** only do complete if no launch error to avoid hang */
    static constexpr unsigned int max_n_batch_block
      = 1; /** by default reductions do not support batching withing the block */

  private:
    const int n_reduce; /** number of reductions of length n_item */
    T *partial; /** device buffer */
    T *result_d; /** device-mapped host buffer */
    T *result_h; /** host buffer */
    count_t *count; /** count array that is used to track the number of completed thread blocks at a given batch index */
    T *device_output_async_buffer = nullptr; // Optional device output buffer for the reduction result

  public:
    /**
       @brief Constructor for ReduceArg
       @param[in] threads The number threads partaking in the kernel
       @param[in] n_reduce The number of reductions
    */
    ReduceArg(dim3 threads, int n_reduce = 1, bool = false) :
      kernel_param<use_kernel_arg>(threads), launch_error(QUDA_ERROR_UNINITIALIZED), n_reduce(n_reduce)
    {
      reducer::init(n_reduce, sizeof(*partial));
      // these buffers may be allocated in init, so we can't set the local copies until now
      partial = static_cast<decltype(partial)>(reducer::get_device_buffer());
      result_d = static_cast<decltype(result_d)>(reducer::get_mapped_buffer());
      result_h = static_cast<decltype(result_h)>(reducer::get_host_buffer());
      count = reducer::get_count<count_t>();

      if (commAsyncReduction()) result_d = partial;
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
       into result.  The generic path posts an event after the kernel
       and then polls on completion of the event.
       @param[out] result The reduction result is copied here
       @param[in] stream The stream on which we the reduction is being done
     */
    template <typename host_t, typename device_t = host_t>
    void complete(std::vector<host_t> &result, const qudaStream_t stream = device::get_default_stream())
    {
      if (launch_error == QUDA_ERROR) return; // kernel launch failed so return
      if (launch_error == QUDA_ERROR_UNINITIALIZED) errorQuda("No reduction kernel appears to have been launched");
      auto event = reducer::get_event();
      qudaEventRecord(event, stream);
      while (!qudaEventQuery(event)) { }

      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        errorQuda("result vector length %lu does not match n_reduce %d", result.size(), n_element);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];
    }
  };

  /**
     @brief Generic reduction function that reduces block-distributed
     data "in" per thread to a single value.  This is the generic
     variant which require explicit host-device synchronization to
     signal the completion of the reduction to the host.

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
    // bool isLastBlockDone[n_batch_block];
    static_assert(sizeof(bool)*n_batch_block <= sizeof(target::omptarget::get_shared_cache()[0])*64, "Shared cache not large enough for isLastBlockDone");  // FIXME arbitrary, 128 is used in block_reduce_helper.h:/tempStorage/
    bool *isLastBlockDone = (bool*)target::omptarget::get_shared_cache();
    bool *hasLastBlockDone = (bool*)&target::omptarget::get_shared_cache()[64];
    // printf("team %d thread %d isLastBlockDone %p\n", omp_get_team_num(), omp_get_thread_num(), isLastBlockDone);

    T aggregate = BlockReduce(target::thread_idx().z).Reduce(in, r);
    // printf("team %d thread %d  r %g  aggregate %g\n", omp_get_team_num(), omp_get_thread_num(), *(double*)(&in), *(double*)(&aggregate));

    if (target::thread_idx().x == 0 && target::thread_idx().y == 0) {
      arg.partial[idx * target::grid_dim().x + target::block_idx().x] = aggregate;
      // __threadfence(); // flush result

      // increment global block counter
      // auto value = atomicInc(&arg.count[idx], target::grid_dim().x);
      unsigned int value = 0u;
      unsigned int *c = &arg.count[idx];
      #pragma omp atomic capture
      { value = *c; *c = *c + 1; }

      // determine if last block
      isLastBlockDone[target::thread_idx().z] = (value == (target::grid_dim().x - 1));
    }
    #pragma omp barrier
    #pragma omp single
    {
      *hasLastBlockDone = 0;
      for(int i=0;i<target::block_dim().z;++i)
        if(isLastBlockDone[i]){
          *hasLastBlockDone = 1;
          break;
        }
    }

    // finish the reduction if last block
    if (*hasLastBlockDone) {
      bool thisSubBlock = isLastBlockDone[target::thread_idx().z];
      T sum = r.init();
      if (thisSubBlock) {
        auto i = target::thread_idx().y * block_size_x + target::thread_idx().x;
        while (i < target::grid_dim().x) {
          sum = r(sum, const_cast<T &>(static_cast<volatile T *>(arg.partial)[idx * target::grid_dim().x + i]));
          // printf("team %d thread %d  sum %g\n", omp_get_team_num(), omp_get_thread_num(), *(double*)(&sum));
          i += block_size_x * block_size_y;
        }
      }

      sum = BlockReduce(target::thread_idx().z).Reduce(sum, r);

      // write out the final reduced value
      if (thisSubBlock) {
        if (target::thread_idx().y * block_size_x + target::thread_idx().x == 0) {
          // printf("team %d thread %d  final sum %g\n", omp_get_team_num(), omp_get_thread_num(), *(double*)(&sum));
          if (arg.get_output_async_buffer()) {
            arg.get_output_async_buffer()[idx] = sum;
          } else {
            arg.result_d[idx] = sum;
          }
          arg.count[idx] = 0; // set to zero for next time
        }
      }
      #pragma omp barrier
    }  // hasLastBlockDone
  }

} // namespace quda
