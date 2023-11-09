#pragma once

#include <target_device.h>
#include <reduce_helper.h>

namespace quda
{

  /**
     @brief Reduction2D_impl is the implementation of the generic 2-d
     reduction kernel.  Functors that utilize this kernel have two
     parallelization dimensions.  The y thread dimenion is constrained
     to remain inside the thread block and this dimension is
     contracted in the reduction.

     @tparam Transformer Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void Reduction2D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = t.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      if (grid_stride)
        idx += blockDim.x * gridDim.x;
      else
        break;
    }

    // perform final inter-block reduction and write out result
    reduce(arg, t, value);
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Reduction2D(Arg arg)
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
        Reduction2D_impl<Functor, Arg, grid_stride>(arg);
      }
    }
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is copied to the device prior to kernel launch.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D(Arg *argp)
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
        Reduction2D_impl<Functor, Arg, grid_stride>(*argp);
      }
    }
  }

  /**
     @brief MultiReduction_impl is the implementation of the generic
     multi-reduction kernel.  Functors that utilize this kernel have
     three parallelization dimensions.  The y thread dimension is
     constrained to remain inside the thread block and this dimension
     is contracted in the reduction.  The z thread dimension is a
     batch dimension that is not contracted in the reduction.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void MultiReduction_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto k = threadIdx.y;
    auto j = threadIdx.z + blockIdx.z * blockDim.z;

    if (j >= arg.threads.z) return;

    reduce_t value = t.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, k, j);
      if (grid_stride)
        idx += blockDim.x * gridDim.x;
      else
        break;
    }

    // perform final inter-block reduction and write out result
    reduce(arg, t, value, j);
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> MultiReduction(Arg arg)
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
        MultiReduction_impl<Functor, Arg, grid_stride>(arg);
      }
    }
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction(Arg *argp)
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
        MultiReduction_impl<Functor, Arg, grid_stride>(*argp);
      }
    }
  }

} // namespace quda
