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

/*
  // OMP TARGET TODO: due to limitation of openmp implementation, we inline reduce manually.
  // declaration of reduce function
  template <int block_size_x, int block_size_y = 1, typename Reducer, typename Arg, typename T>
  __device__ __forceinline__ void reduce(Arg &arg, const Reducer &r, const T &in, const int idx = 0);
*/

  /**
     @brief ReduceArg is the argument type that all kernel arguments
     shoud inherit from if the kernel is to utilize global reductions.
     @tparam T the type that will be reduced
     @tparam use_kernel_arg Whether the kernel will source the
     parameter struct as an explicit kernel argument or from constant
     memory
   */
  template <typename T, bool use_kernel_arg = true> struct ReduceArg : kernel_param<use_kernel_arg> {

    // OMP TARGET TODO: due to limitation of openmp implementation, we inline reduce manually.
    // We also remove private restrictions below.
    // template <int, int, typename Reducer, typename Arg, typename I>
    // friend __device__ void reduce(Arg &, const Reducer &, const I &, const int);
    qudaError_t launch_error; /** only do complete if no launch error to avoid hang */

    const int n_reduce; /** number of reductions of length n_item */
    T *partial; /** device buffer */
    T *result_d; /** device-mapped host buffer */
    T *result_h; /** host buffer */
    count_t *count; /** count array that is used to track the number of completed thread blocks at a given batch index */

    /**
       @brief Constructor for ReduceArg
       @param[in] threads The number threads partaking in the kernel
       @param[in] n_reduce The number of reductions
    */
    ReduceArg(dim3 threads, int n_reduce = 1, bool = false) :
      kernel_param<use_kernel_arg>(threads),
      launch_error(QUDA_ERROR_UNINITIALIZED),
      n_reduce(n_reduce),
      partial(static_cast<decltype(partial)>(reducer::get_device_buffer())),
      result_d(static_cast<decltype(result_d)>(reducer::get_mapped_buffer())),
      result_h(static_cast<decltype(result_h)>(reducer::get_host_buffer())),
      count {reducer::get_count<count_t>()}
    {
      // check reduction buffers are large enough if requested
      auto max_reduce_blocks = 2 * device::processor_count();
      auto reduce_size = max_reduce_blocks * n_reduce * sizeof(*partial);
      if (reduce_size > reducer::buffer_size())
        errorQuda("Requested reduction requires a larger buffer %lu than allocated %lu", reduce_size,
                  reducer::buffer_size());

      if (commAsyncReduction()) result_d = partial;
    }

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
      // OMP TARGET TODO: try shared memory for result_d/h
      omp_target_memcpy(result_h, result_d, n_element*sizeof(device_t), 0, 0, omp_get_initial_device(), omp_get_default_device());
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

     OMP TARGET: should be run only as the main thread of each team.
     here, we need the block reduction value, `aggregate`.
     OMP TARGET TODO: not supported for now.  We manually inline this.

  template <int block_size_x, int block_size_y, typename Reducer, typename Arg, typename T>
  __device__ __forceinline__ void reduce(Arg &arg, const Reducer &r, const T &aggregate, const int idx)
  {
    printf("reduce::aggregate team %d %g\n", omp_get_team_num(), *(double*)(&aggregate));
    bool isLastBlockDone;

    arg.partial[idx * target::grid_dim().x + target::block_idx().x] = aggregate;
    __threadfence(); // flush result

    // increment global block counter
    // auto value = atomicInc(&arg.count[idx], gridDim.x);
    unsigned int value = 0;
    unsigned int *c = &arg.count[idx];
    #pragma omp atomic capture
    { value = *c; *c = *c + 1; }

    // determine if last block
    isLastBlockDone = (value == (target::grid_dim().x - 1));

    __syncthreads();

    // finish the reduction if last block
#pragma omp declare reduction(OMPReduce_ : typename Reducer::reduce_t : omp_out=Reducer::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Reducer::init_omp())
    if (isLastBlockDone) {
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      T sum = arg.init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
      {
        auto i = target::thread_idx().y * block_size_x + target::thread_idx().x;
        while (i < target::grid_dim().x) {
          sum = r(sum, const_cast<T &>(static_cast<volatile T *>(arg.partial)[idx * target::grid_dim().x + i]));
          i += block_size_x * block_size_y;
        }
      }

      // write out the final reduced value
      printf("reduce::sum %g\n", *(double*)(&sum));
      arg.result_d[idx] = sum;
      arg.count[idx] = 0; // set to zero for next time
    }
  }
  */

} // namespace quda
