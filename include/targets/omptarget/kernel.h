#pragma once

#include <target_device.h>
#include <kernel_helper.h>

#define OMP_KERNEL(kern) \
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> \
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> kern(Arg arg) \
  { \
    QUDA_OMPTARGET_KERNEL_BEGIN(arg) \
      kern##_impl<Functor, Arg, grid_stride>(arg); \
    QUDA_OMPTARGET_KERNEL_END \
  }

#define OMP_KERNEL_PTR(kern) \
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> \
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> kern(Arg *argp) \
  { \
    QUDA_OMPTARGET_KERNEL_BEGIN_PTR(argp) \
      kern##_impl<Functor, Arg, grid_stride>(*argp); \
    QUDA_OMPTARGET_KERNEL_END \
  }

namespace quda
{

  /**
     @brief Kernel1D_impl is the implementation of the generic 1-d
     kernel.  Functors that utilize this kernel have a
     single parallelization dimension.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel1D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < arg.threads.x) {
      f(i);
      if (grid_stride)
        i += gridDim.x * blockDim.x;
      else
        break;
    }
  }

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  OMP_KERNEL(Kernel1D);

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  OMP_KERNEL_PTR(Kernel1D);

  /**
     @brief Kernel2D_impl is the implementation of the generic 2-d
     kernel.  Functors that utilize this kernel have two
     parallelization dimensions.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel2D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= arg.threads.y) return;

    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride)
        i += gridDim.x * blockDim.x;
      else
        break;
    }
  }

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  OMP_KERNEL(Kernel2D);

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  OMP_KERNEL_PTR(Kernel2D);

  /**
     @brief Kernel3D_impl is the implementation of the generic 3-d
     kernel.  Functors that utilize this kernel have three
     parallelization dimensions.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel3D_impl(const Arg &arg)
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
      if (grid_stride)
        i += gridDim.x * blockDim.x;
      else
        break;
    }
  }

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  OMP_KERNEL(Kernel3D);

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  OMP_KERNEL_PTR(Kernel3D);

}

#undef OMP_KERNEL
#undef OMP_KERNEL_PTR
