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
  template <typename Arg> __device__ constexpr int virtual_block_idx(const Arg &arg)
  {
    using IntType = decltype(arg.swizzle_factor);
    IntType block_idx = static_cast<IntType>(blockIdx.x);
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      const IntType gridp = static_cast<IntType>(gridDim.x) - static_cast<IntType>(gridDim.x) % arg.swizzle_factor;

      block_idx = static_cast<IntType>(blockIdx.x);
      if (static_cast<IntType>(blockIdx.x) < gridp) {
        // this is the portion of the block that we are going to transpose
        const auto i = static_cast<IntType>(blockIdx.x) % arg.swizzle_factor;
        const auto j = static_cast<IntType>(blockIdx.x) / arg.swizzle_factor;

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
  __forceinline__ __device__ void BlockKernel2D_impl(const Arg &arg)
  {
    const dim3 block_idx(virtual_block_idx(arg), blockIdx.y, 0);
    const dim3 thread_idx(threadIdx.x, threadIdx.y, 0);
    auto j = blockDim.y*blockIdx.y + threadIdx.y;
    if (j >= arg.threads.y) return;

    Transformer<block_size, Arg> t(arg);
    t(block_idx, thread_idx);
  }
  
  template <unsigned int block_size, template <int, typename> class Transformer, typename Arg>
    __launch_bounds__(block_size)
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>()
    							  && (Arg::launch_bounds || block_size > 512)	
    							, void> BlockKernel2D(Arg arg)
  {
    BlockKernel2D_impl<block_size, Transformer, Arg>(arg);
  }

  template <unsigned int block_size, template <int, typename> class Transformer, typename Arg>
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>()
    							  && !(Arg::launch_bounds || block_size > 512)	
    							, void> BlockKernel2D(Arg arg)
  {
    BlockKernel2D_impl<block_size, Transformer, Arg>(arg);
  }


  template <unsigned int block_size, template <int, typename> class Transformer, typename Arg>
    __launch_bounds__(block_size)
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>()
                                  && (Arg::launch_bounds || block_size > 512)	
                                , void> BlockKernel2D()
  {
    BlockKernel2D_impl<block_size, Transformer, Arg>(device::get_arg<Arg>());
  }
  
   template <unsigned int block_size, template <int, typename> class Transformer, typename Arg>
   __global__ std::enable_if_t<!device::use_kernel_arg<Arg>()
                                  && !(Arg::launch_bounds || block_size > 512)	
                                , void> BlockKernel2D()
  {
    BlockKernel2D_impl<block_size, Transformer, Arg>(device::get_arg<Arg>());
  }

}
