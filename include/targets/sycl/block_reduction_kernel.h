#pragma once
#include <tune_quda.h>
#include <reduce_helper.h>

namespace quda {

  /**
     @brief This class is derived from the arg class that the functor
     creates and curries in the block size.  This allows the block
     size to be set statically at launch time in the actual argument
     class that is passed to the kernel.
   */
  template <unsigned int block_size_, typename Arg_> struct BlockKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr unsigned int block_size = block_size_;
    BlockKernelArg(const Arg &arg) : Arg(arg) { }
  };

  /**
     @brief This helper function swizzles the block index through
     mapping the block index onto a matrix and tranposing it.  This is
     done to potentially increase the cache utilization.  Requires
     that the argument class has a member parameter "swizzle" which
     determines if we are swizzling and a parameter "swizzle_factor"
     which is the effective matrix dimension that we are tranposing in
     this mapping.
   */
  template <typename Arg>
  int virtual_block_idx(const Arg &arg, sycl::nd_item<3> ndi)
  {
    //int block_idx = blockIdx.x;
    int block_idx = ndi.get_group(0);
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      //const int gridp = gridDim.x - gridDim.x % arg.swizzle_factor;
      const int ngrp = ndi.get_group_range(0);
      const int gridp = ngrp - ngrp % arg.swizzle_factor;

      //block_idx = blockIdx.x;
      //if (blockIdx.x < gridp) {
      if (block_idx < gridp) {
        // this is the portion of the block that we are going to transpose
        //const int i = blockIdx.x % arg.swizzle_factor;
        //const int j = blockIdx.x / arg.swizzle_factor;
        const int i = block_idx % arg.swizzle_factor;
        const int j = block_idx / arg.swizzle_factor;

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
  template <template <typename> class Transformer, typename Arg>
  void BlockKernel2DImpl(Arg arg, sycl::nd_item<3> ndi)
  {
    const dim3 block_idx(virtual_block_idx(arg,ndi), ndi.get_group(1), 0);
    const dim3 thread_idx(ndi.get_local_id(0), ndi.get_local_id(1), 0);
    //const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int j = ndi.get_global_id(1);
    if (j >= arg.threads.y) return;

    Transformer<Arg> t(arg);
    t(block_idx, thread_idx);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride=false>
  qudaError_t
  BlockKernel2D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    auto err = QUDA_SUCCESS;
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, 1};
    sycl::range<3> localSize{tp.block.x, tp.block.y, 1};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("BlockKernel2D sizeof(arg): %lu\n", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Transformer: %s\n", typeid(Transformer<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
    }
    try {
      q.submit([&](sycl::handler& h) {
	//h.parallel_for<class BlockKernel2D>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) {
	     quda::BlockKernel2DImpl<Transformer, Arg>(arg, ndi);
	   });
      });
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end BlockKernel2D\n");
    }
    return err;
  }

}
