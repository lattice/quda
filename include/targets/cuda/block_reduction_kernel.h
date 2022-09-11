#pragma once

#include <target_device.h>
#include <kernel_helper.h>
#include <block_reduce_helper.h>

namespace quda
{

  /**
     @brief This helper function swizzles the block index through
     mapping the block index onto a matrix and tranposing it.  This is
     done to potentially increase the cache utilization.  Requires
     that the argument class has a member parameter "swizzle" which
     determines if we are swizzling and a parameter "swizzle_factor"
     which is the effective matrix dimension that we are tranposing in
     this mapping.

     Specifically, the thread block id is remapped by
     transposing its coordinates: if the original order can be
     parameterized by

     blockIdx.x = j * swizzle + i,

     then the new order is

     block_idx = i * (gridDim.x / swizzle) + j

     We need to factor out any remainder and leave this in original
     ordering.

     @param arg Kernel argument struct
     @return Swizzled block index
   */
  template <typename Arg> __device__ constexpr int virtual_block_idx(const Arg &arg)
  {
    auto block_idx = blockIdx.x;
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      const auto gridp = gridDim.x - gridDim.x % arg.swizzle_factor;

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

     @tparam block_size x-dimension block-size
     @param[in] arg Kernel argument
   */
  template <unsigned int block_size_, typename Arg_> struct BlockKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr unsigned int block_size = block_size_;
    BlockKernelArg(const Arg &arg) : Arg(arg) { }
  };

  /**
     @brief BlockKernel2D_impl is the implementation of the Generic
     block kernel.  Here, we split the block (CTA) and thread indices
     and pass them separately to the transform functor.  The x thread
     dimension is templated (Arg::block_size), e.g., for efficient
     reductions.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @param[in] arg Kernel argument
  */
  template <template <typename> class Functor, typename Arg>
  __forceinline__ __device__ void BlockKernel2D_impl(const Arg &arg)
  {
    const dim3 block_idx(virtual_block_idx(arg), blockIdx.y, blockIdx.z);
    const dim3 thread_idx(threadIdx.x, threadIdx.y, threadIdx.z);
    auto j = blockDim.y * blockIdx.y + threadIdx.y;
    auto k = blockDim.z * blockIdx.z + threadIdx.z;
    if (j >= arg.threads.y) return;
    if (k >= arg.threads.z) return;

    Functor<Arg> t(arg);
    t(block_idx, thread_idx);
  }

  /**
     @brief BlockKernel2D is the entry point of the generic block
     kernel.  This is the specialization where the kernel argument
     struct is passed by value directly to the kernel.  The kernel
     type will impose launch bounds if requested (Arg::launch_bounds)
     or if a block_size > 512 is required.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension).  Not supported at present.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::launch_bounds ?
                      Arg::block_size :
                      0) __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> BlockKernel2D(Arg arg)
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    BlockKernel2D_impl<Functor, Arg>(arg);
  }

  /**
     @brief BlockKernel2D is the entry point of the generic block
     kernel.  This is the specialization where the kernel argument
     struct is copied to the device prior to kernel launch.  The kernel
     type will impose launch bounds if requested (Arg::launch_bounds)
     or if a block_size > 512 is required.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension).  Not supported at present.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::launch_bounds ?
                      Arg::block_size :
                      0) __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> BlockKernel2D()
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    BlockKernel2D_impl<Functor, Arg>(device::get_arg<Arg>());
  }

} // namespace quda
