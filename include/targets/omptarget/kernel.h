#pragma once

namespace quda {

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ void Kernel1D(Arg arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ void Kernel2D(Arg arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= arg.threads.y) return;

    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ void Kernel3D(Arg arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z + blockIdx.z * blockDim.z;
    if (j >= arg.threads.y) return;
    if (k >= arg.threads.z) return;

    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(Arg arg)
  {
    Functor<Arg> f(arg);
    f();
  }

}
