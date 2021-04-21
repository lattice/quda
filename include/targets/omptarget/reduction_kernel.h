#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void Reduction2D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
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
    reduce<block_size_x, block_size_y>(arg, t, value);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Reduction2D(Arg arg)
  {
    Reduction2D_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D()
  {
    Reduction2D_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(device::get_arg<Arg>());
  }


  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void MultiReduction_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
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
    reduce<block_size_x, block_size_y>(arg, t, value, j);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> MultiReduction(Arg arg)
  {
    MultiReduction_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction()
  {
    MultiReduction_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(device::get_arg<Arg>());
  }

}
