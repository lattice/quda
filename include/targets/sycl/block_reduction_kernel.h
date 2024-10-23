#pragma once
#include <tunable_kernel.h>
#include <reduce_helper.h>
#include <quda_sycl_api.h>

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
  template <typename Arg>
  int virtual_block_idx(const Arg &arg, const sycl::nd_item<3> &)
  {
    int block_idx = groupIdX;
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      //const int gridp = gridDim.x - gridDim.x % arg.swizzle_factor;
      const int ngrp = groupRangeX;
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
  template <template <typename> class Functor, typename Arg, typename ...S>
  inline std::enable_if_t<!needsFullBlock<Functor<Arg>>, void>
  BlockKernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    const dim3 block_idx(virtual_block_idx(arg,ndi), groupIdY, groupIdZ);
    const dim3 thread_idx(localIdX, localIdY, localIdZ);
    const unsigned int j = globalIdY;
    if (j >= arg.threads.y) return;
    const unsigned int k = globalIdZ;
    if (k >= arg.threads.z) return;

#if 0
    Functor<Arg> f(arg);
    if constexpr (hasKernelOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }
#else
    Ftor<Functor<Arg>> f(arg, ndi, smem...);
#endif

    f(block_idx, thread_idx);
  }
  template <template <typename> class Functor, typename Arg, typename ...S>
  inline std::enable_if_t<needsFullBlock<Functor<Arg>>, void>
  BlockKernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    const dim3 block_idx(virtual_block_idx(arg,ndi), groupIdY, groupIdZ);
    const dim3 thread_idx(localIdX, localIdY, localIdZ);
    bool active = true;
    const unsigned int j = globalIdY;
    if (j >= arg.threads.y) active = false;
    const unsigned int k = globalIdZ;
    if (k >= arg.threads.z) active = false;

#if 0
    Functor<Arg> f(arg);
    if constexpr (hasKernelOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }
#else
    Ftor<Functor<Arg>> f(arg, ndi, smem...);
#endif

    f.template operator()<true>(block_idx, thread_idx, active);
  }
  template <template <typename> class Functor, typename Arg>
  struct BlockKernel2DS {
    using KernelOpsT = getKernelOps<Functor<Arg>>;
    template <typename ...S>
    BlockKernel2DS(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
    {
#ifdef QUDA_THREADS_BLOCKED
      BlockKernel2DImpl<Functor,Arg>(arg, ndi);
#else
      BlockKernel2DImpl<Functor,Arg>(arg, ndi, smem...);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride=false>
  qudaError_t
  BlockKernel2D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    static_assert(!grid_stride, "grid_stride not supported for BlockKernel");
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //return QUDA_ERROR;
    //}
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("BlockKernel2D sizeof(arg): %lu\n", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>, needsSharedMem<Functor<Arg>>);
      printfQuda("  sharedMemSize: %i\n", sharedMemSize<getKernelOps<Functor<Arg>>>(tp.block));
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //if(needsFullBlock<Functor<Arg>>) {
    //std::ostringstream what;
    //what << "localSizeX (" << localSize[RANGE_X] << ") % warp_size (" << device::warp_size() << ") != 0";
    //target::sycl::set_error(what.str(), "pre-launch", __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
    //return QUDA_ERROR;
    //}
    //}
    //if (arg.threads.x%localSize[RANGE_X] != 0) {
      //warningQuda("arg.threads.x (%i) %% localSize X (%lu) != 0", arg.threads.x, localSize[RANGE_X]);
    //  return QUDA_ERROR;
    //}
    //if (globalSize[RANGE_Y] != arg.threads.y) {
      //warningQuda("globalSize Y (%lu) != arg.threads.y (%i)", globalSize[RANGE_Y], arg.threads.y);
    //  return QUDA_ERROR;
    //}
    //if (globalSize[RANGE_Z] != arg.threads.z) {
      //warningQuda("globalSize Z (%lu) != arg.threads.z (%i)", globalSize[RANGE_Z], arg.threads.z);
    //  return QUDA_ERROR;
    //}
    sycl::nd_range<3> ndRange{globalSize, localSize};
    err = launch<BlockKernel2DS<Functor, Arg>>(stream, ndRange, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end BlockKernel2D\n");
    }
    return err;
  }

}
