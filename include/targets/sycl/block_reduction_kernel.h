#pragma once

#include <reduce_helper.h>

namespace quda {

  /**
     @brief This helper function swizzles the block index through
     mapping the block index onto a matrix and tranposing it.  This is
     done to potentially increase the cache utilization.  Requires
     that the argument class has a member parameter "swizzle" which
     determines if we are swizzling and a parameter "swizzle_factor"
     which is the effective matrix dimension that we are tranposing in
     this mapping.
   */
  template <typename Arg> constexpr int virtual_block_idx(const Arg &arg)
  {
    int block_idx = blockIdx.x;
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      const int gridp = gridDim.x - gridDim.x % arg.swizzle_factor;

      block_idx = blockIdx.x;
      if (blockIdx.x < gridp) {
        // this is the portion of the block that we are going to transpose
        const int i = blockIdx.x % arg.swizzle_factor;
        const int j = blockIdx.x / arg.swizzle_factor;

        // transpose the coordinates
        block_idx = i * (gridp / arg.swizzle_factor) + j;
      }
    }
    return block_idx;
  }

  /**
     @brief Generic block kernel.  Here, we split the block and thread
     indices in the x and y dimension and pass these indices
     separately to the transform functor.  The x thread dimension is
     templated, e.g., for efficient reductions, and typically the y
     thread dimension is a trivial vectorizable dimension.
  */
  template <int block_size, template <int, typename> class Transformer, typename Arg>
  void BlockKernel2D(Arg arg, sycl::nd_item<3> ndi)
  {
    const dim3 block_idx(virtual_block_idx(arg), blockIdx.y, 0);
    const dim3 thread_idx(threadIdx.x, threadIdx.y, 0);
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (j >= arg.threads.y) return;

    Transformer<block_size, Arg> t(arg);
    t(block_idx, thread_idx);
  }

  template <int block_size, template <int, typename> class Transformer, typename Arg>
  qudaError_t launchBlockKernel2D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    warningQuda("launchBlockKernel2D");
    warningQuda("%s  %s", str(globalSize).c_str(), str(localSize).c_str());
    warningQuda("%s", str(arg.threads).c_str());
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class BlockKernel2D>
		 (ndRange,
		  [=](sycl::nd_item<3> ndi) {
		    quda::BlockKernel2D<block_size, Transformer, Arg>(arg, ndi);
		  });
	     });
    warningQuda("end launchBlockKernel2D");
    return QUDA_SUCCESS;
  }

}
