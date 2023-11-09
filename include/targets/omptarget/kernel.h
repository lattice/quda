#pragma once

#include <target_device.h>
#include <kernel_helper.h>

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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel1D(Arg arg)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) firstprivate(arg,grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel1D_impl<Functor, Arg, grid_stride>(arg);
      }
    }
  }

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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel1D(Arg *argp)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(argp) firstprivate(grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel1D_impl<Functor, Arg, grid_stride>(*argp);
      }
    }
  }

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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel2D(Arg arg)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) firstprivate(arg,grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel2D_impl<Functor, Arg, grid_stride>(arg);
      }
    }
  }

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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel2D(Arg *argp)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(argp) firstprivate(grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel2D_impl<Functor, Arg, grid_stride>(*argp);
      }
    }
  }

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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel3D(Arg arg)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) firstprivate(arg,grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel3D_impl<Functor, Arg, grid_stride>(arg);
      }
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
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel3D(Arg *argp)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(argp) firstprivate(grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Kernel3D_impl<Functor, Arg, grid_stride>(*argp);
      }
    }
  }

  /**
     @brief raw_kernel is used for CUDA-specific kernels where we want
     to avoid using the generic framework.  For these kernels, we
     delegate the parallelism and bounds checking for the kernel
     functor.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam dummy unused template parameter, present to allow us to
     utilize the generic launching framework

     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool dummy = false>
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(Arg arg)
  {
    const dim3 grid = target::omptarget::launch_param_grid();
    const dim3 block = target::omptarget::launch_param_block();
    const int gd = grid.x*grid.y*grid.z;
    const int ld = block.x*block.y*block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld) firstprivate(arg,grid,block)
    {
      target::omptarget::launch_param_device_set(grid, block);
      #pragma omp parallel num_threads(ld)
      {
        Functor<Arg> f(&arg);
        f();
      }
    }
  }

}
