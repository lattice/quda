#pragma once

#include <target_device.h>
#include <reduce_helper.h>

namespace quda {

  /**
     @brief This class is derived from the arg class that the functor
     creates and curries in the block size.  This allows the block
     size to be set statically at launch time in the actual argument
     class that is passed to the kernel.
  */
  template <int block_size_x_, int block_size_y_, typename Arg_> struct ReduceKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    ReduceKernelArg(const Arg &arg) : Arg(arg) { }
  };

  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void Reduction2D_impl(const Arg &arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      if (grid_stride) idx += blockDim.x * gridDim.x; else break;
    }

    // perform final inter-block reduction and write out result
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Reduction2D(Arg arg)
  {
    Reduction2D_impl<Transformer, Arg, grid_stride>(arg);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D()
  {
    Reduction2D_impl<Transformer, Arg, grid_stride>(device::get_arg<Arg>());
  }


  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void MultiReduction_impl(const Arg &arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z;

    if (j >= arg.threads.y) return;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j, k);
      if (grid_stride) idx += blockDim.x * gridDim.x; else break;
    }

    // perform final inter-block reduction and write out result
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value, j);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> MultiReduction(Arg arg)
  {
    MultiReduction_impl<Transformer, Arg, grid_stride>(arg);
  }

  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction()
  {
    MultiReduction_impl<Transformer, Arg, grid_stride>(device::get_arg<Arg>());
  }

}
