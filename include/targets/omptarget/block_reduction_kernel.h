#pragma once

#include <target_device.h>
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
    QUDA_RT_CONSTS;
    auto block_idx = blockIdx.x;
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      const auto gridp = gridDim.x - gridDim.x % arg.swizzle_factor;

      block_idx = blockIdx.x;
      if (block_idx < gridp) {
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
     @brief Generic block kernel.  Here, we split the block and thread
     indices in the x and y dimension and pass these indices
     separately to the transform functor.  The x thread dimension is
     templated, e.g., for efficient reductions, and typically the y
     thread dimension is a trivial vectorizable dimension.
  */
  template <template <typename> class Transformer, typename Arg>
  __forceinline__ __device__ void BlockKernel2D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    const dim3 block_idx(virtual_block_idx(arg), blockIdx.y, 0);
    const dim3 thread_idx(threadIdx.x, threadIdx.y, 0);
    auto j = blockDim.y*blockIdx.y + threadIdx.y;
    if (j >= arg.threads.y) return;

    Transformer<Arg> t(arg);
    t(block_idx, thread_idx);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = false>
    __launch_bounds__(Arg::launch_bounds || Arg::block_size > 512 ? Arg::block_size : 0)
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> BlockKernel2D(Arg arg)
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // const int tx = arg.threads.x;
    // const int ty = arg.threads.y;
    // const int tz = arg.threads.z;
    // printf("BlockKernel2D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
      //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
      //          "omp reports: teams %d threads %d\n",
      //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
      //          omp_get_num_teams(), omp_get_num_threads());
      BlockKernel2D_impl<Transformer, Arg>(arg);
    }
    }
    omp_target_free(dparg, omp_get_default_device());
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = false>
    __launch_bounds__(Arg::launch_bounds || Arg::block_size > 512 ? Arg::block_size : 0)
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> BlockKernel2D()
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // printf("Kernel2D: launch parameter: gd %d ld %d\n", gd, ld);
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
      //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
      //          "omp reports: teams %d threads %d\n",
      //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
      //          omp_get_num_teams(), omp_get_num_threads());
      BlockKernel2D_impl<Transformer, Arg>(device::get_arg<Arg>());
    }
    }
  }

}
