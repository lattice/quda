#pragma once

#include <kernel_helper.h>

namespace quda {

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel1D_impl(const Arg &arg)
  {
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel1D(Arg arg) { Kernel1D_impl<Functor, Arg, grid_stride>(arg); }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel1D() { Kernel1D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>()); }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel2D_impl(const Arg &arg)
  {
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
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel2D(Arg arg) { Kernel2D_impl<Functor, Arg, grid_stride>(arg); }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel2D() { Kernel2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>()); }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel3D_impl(const Arg &arg)
  {
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
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel3D(Arg arg) { Kernel3D_impl<Functor, Arg, grid_stride>(arg); }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel3D() { Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>()); }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(Arg arg)
  {
    Functor<Arg> f(arg);
    f();
  }

}
