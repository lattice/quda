#pragma once

#include <target_device.h>
#include <reduce_helper.h>

namespace quda
{

  /**
     @brief This class is derived from the arg class that the functor
     creates and curries in the block size.  This allows the block
     size to be set statically at launch time in the actual argument
     class that is passed to the kernel.

     @tparam block_size_x x-dimension block-size
     @tparam block_size_y y-dimension block-size
     @tparam Arg Kernel argument struct
  */
  template <int block_size_x_, int block_size_y_, typename Arg_> struct ReduceKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    ReduceKernelArg(const Arg &arg) : Arg(arg) { }
  };

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
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value);
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
    const int gd = target::omptarget::launch_param.grid.x*target::omptarget::launch_param.grid.y*target::omptarget::launch_param.grid.z;
    const int ld = target::omptarget::launch_param.block.x*target::omptarget::launch_param.block.y*target::omptarget::launch_param.block.z;
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      target::omptarget::shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      char buffer[sizeof(Arg)];
      memcpy(buffer, (void *)dparg, sizeof(Arg));
      Reduction2D_impl<Functor, Arg, grid_stride>(*(Arg *)buffer);
    }
    }
    omp_target_free(dparg, omp_get_default_device());
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
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D()
  {
    const int gd = target::omptarget::launch_param.grid.x*target::omptarget::launch_param.grid.y*target::omptarget::launch_param.grid.z;
    const int ld = target::omptarget::launch_param.block.x*target::omptarget::launch_param.block.y*target::omptarget::launch_param.block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      target::omptarget::shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      Reduction2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
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
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value, j);
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
    const int gd = target::omptarget::launch_param.grid.x*target::omptarget::launch_param.grid.y*target::omptarget::launch_param.grid.z;
    const int ld = target::omptarget::launch_param.block.x*target::omptarget::launch_param.block.y*target::omptarget::launch_param.block.z;
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      target::omptarget::shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      char buffer[sizeof(Arg)];
      memcpy(buffer, (void *)dparg, sizeof(Arg));
      MultiReduction_impl<Functor, Arg, grid_stride>(*(Arg *)buffer);
    }
    }
    omp_target_free(dparg, omp_get_default_device());
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
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction()
  {
    const int gd = target::omptarget::launch_param.grid.x*target::omptarget::launch_param.grid.y*target::omptarget::launch_param.grid.z;
    const int ld = target::omptarget::launch_param.block.x*target::omptarget::launch_param.block.y*target::omptarget::launch_param.block.z;
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      target::omptarget::shared_cache.addr = cache;
    #pragma omp parallel num_threads(ld)
    {
      MultiReduction_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
    }
    }
  }

} // namespace quda
