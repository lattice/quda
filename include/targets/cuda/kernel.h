#pragma once

#include <cuda/work_stealing>
#include <target_device.h>
#include <constant_kernel_arg.h>
#include <work_steal.h>
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
#ifdef QUDA_SHARED_MEMORY_SPILL
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
#endif
    Functor<Arg> f(arg);

#ifdef QUDA_WORK_STEAL_COOPERATIVE
    __shared__ uint4 work_steal_result;
    __shared__ uint64_t work_steal_bar;
    work_steal<1> robber(&work_steal_result, &work_steal_bar);
#endif

    if constexpr (Arg::work_steal) {
      static_assert(!grid_stride, "grid stride cannot be used with work stealing");

#ifdef QUDA_WORK_STEAL_COOPERATIVE
      dim3 block_idx = {blockIdx.x, blockIdx.y, blockIdx.z};
      if constexpr (Arg::work_steal_functor) {
        while (true) {
          auto i = threadIdx.x + block_idx.x * blockDim.x;
          auto in_bounds = !Arg::check_bounds ? true : i < arg.threads.x;

          bool success = false;
          // Only in_bounds threads run the functor (and thus request/complete); keep barrier scope consistent.
          if (in_bounds) {
            if constexpr (Arg::set_block_idx) {
              f.block_idx = block_idx;
              f.set_robber(robber);
            }
            f(i);
            success = robber.last_success();
            block_idx = robber.next_block_idx();
            if (success) robber.release();
          }
          if (!success) break;
        }
      } else {
        while (true) {
          robber.request();

          auto i = threadIdx.x + block_idx.x * blockDim.x;
          if constexpr (Arg::set_block_idx) f.block_idx = block_idx;

          auto in_bounds = !Arg::check_bounds ? true : i < arg.threads.x;
          if (in_bounds) f(i);

          bool success = robber.complete();
          if (!success) break;

          block_idx = robber.get_block_idx();

          robber.release();
        }
      }
#else
      // When QUDA_WORK_STEAL_COOPERATIVE is not defined, use runtime work stealing; work_steal_functor has no effect
      cuda::device::for_each_canceled_block<1>([&](dim3 block_idx) {
        auto i = threadIdx.x + block_idx.x * blockDim.x;
        if constexpr (Arg::check_bounds)
          if (i >= arg.threads.x) return;
        f(i);
      });
#endif

    } else {

      auto i = threadIdx.x + blockIdx.x * blockDim.x;

      if constexpr (Arg::check_bounds) {
        while (i < arg.threads.x) {
          f(i);
          if constexpr (grid_stride)
            i += gridDim.x * blockDim.x;
          else
            break;
        }
      } else {
        f(i);
      }
    }
  }

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and a max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel1D(const GRID_CONSTANT Arg arg)
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and no max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void>
  Kernel1D(const GRID_CONSTANT Arg arg)
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch and a max register
     count is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel1D()
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  /**
     @brief Kernel1D is the entry point of the generic 1-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch and no max register
     count is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread.
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void> Kernel1D()
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
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
#ifdef QUDA_SHARED_MEMORY_SPILL
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
#endif
    Functor<Arg> f(arg);

#ifdef QUDA_WORK_STEAL_COOPERATIVE
    __shared__ uint4 work_steal_result;
    __shared__ uint64_t work_steal_bar;
    work_steal<2> robber(&work_steal_result, &work_steal_bar);
#endif

    if constexpr (Arg::work_steal) {
      static_assert(!grid_stride, "grid stride cannot be used with work stealing");

#ifdef QUDA_WORK_STEAL_COOPERATIVE
      dim3 block_idx = {blockIdx.x, blockIdx.y, blockIdx.z};
      if constexpr (Arg::work_steal_functor) {
        while (true) {
          auto i = threadIdx.x + block_idx.x * blockDim.x;
          auto j = threadIdx.y + block_idx.y * blockDim.y;
          auto in_bounds = !Arg::check_bounds ? true : (i < arg.threads.x && j < arg.threads.y);

          bool success = false;
          // Only in_bounds threads run the functor (and thus request/complete); keep barrier scope consistent.
          if (in_bounds) {
            if constexpr (Arg::set_block_idx) {
              f.block_idx = block_idx;
              f.set_robber(robber);
            }
            f(i, j);
            success = robber.last_success();
            block_idx = robber.next_block_idx();
            if (success) robber.release();
          }
          if (!success) break;
        }
      } else {
        while (true) {
          robber.request();

          auto i = threadIdx.x + block_idx.x * blockDim.x;
          auto j = threadIdx.y + block_idx.y * blockDim.y;
          if constexpr (Arg::set_block_idx) f.block_idx = block_idx;

          auto in_bounds = !Arg::check_bounds ? true : (i < arg.threads.x && j < arg.threads.y);
          if (in_bounds) f(i, j);

          bool success = robber.complete();
          if (!success) break;

          block_idx = robber.get_block_idx();

          robber.release();
        }
      }
#else
      // When QUDA_WORK_STEAL_COOPERATIVE is not defined, use runtime work stealing; work_steal_functor has no effect
      cuda::device::for_each_canceled_block<2>([&](dim3 block_idx) {
        auto i = threadIdx.x + block_idx.x * blockDim.x;
        auto j = threadIdx.y + block_idx.y * blockDim.y;

        if constexpr (Arg::check_bounds) {
          if (i >= arg.threads.x) return;
          if (j >= arg.threads.y) return;
        }

        f(i, j);
      });
#endif

    } else {

      auto i = threadIdx.x + blockIdx.x * blockDim.x;
      auto j = threadIdx.y + blockIdx.y * blockDim.y;

      if constexpr (Arg::check_bounds) {
        if (j >= arg.threads.y) return;

        while (i < arg.threads.x) {
          f(i, j);
          if constexpr (grid_stride)
            i += gridDim.x * blockDim.x;
          else
            break;
        }
      } else {
        f(i, j);
      }
    }
  }

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and a max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel2D(const GRID_CONSTANT Arg arg)
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and no max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void>
  Kernel2D(const GRID_CONSTANT Arg arg)
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch and a max register
     count is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel2D()
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  /**
     @brief Kernel2D is the entry point of the generic 2-d kernel.
     This is the specialization where the kernel argument struct is
     copied to the device prior to kernel launch and not max register
     count is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void> Kernel2D()
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
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
#ifdef QUDA_SHARED_MEMORY_SPILL
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
#endif
    Functor<Arg> f(arg);

#ifdef QUDA_WORK_STEAL_COOPERATIVE
    __shared__ uint4 work_steal_result;
    __shared__ uint64_t work_steal_bar;
    work_steal<3> robber(&work_steal_result, &work_steal_bar);
#endif

    if constexpr (Arg::work_steal) {
      static_assert(!grid_stride, "grid stride cannot be used with work stealing");

#ifdef QUDA_WORK_STEAL_COOPERATIVE
      dim3 block_idx = {blockIdx.x, blockIdx.y, blockIdx.z};
      if constexpr (Arg::work_steal_functor) {
        while (true) {
          auto i = threadIdx.x + block_idx.x * blockDim.x;
          auto j = threadIdx.y + block_idx.y * blockDim.y;
          auto k = threadIdx.z + block_idx.z * blockDim.z;
          auto in_bounds = !Arg::check_bounds ? true : (i < arg.threads.x && j < arg.threads.y && k < arg.threads.z);

          bool success = false;
          // Only in_bounds threads run the functor (and thus request/complete); keep barrier scope consistent.
          if (in_bounds) {
            if constexpr (Arg::set_block_idx) {
              f.block_idx = block_idx;
              f.set_robber(robber);
            }
            f(i, j, k);
            success = robber.last_success();
            block_idx = robber.next_block_idx();
            if (success) robber.release();
          }
          if (!success) break;
        }
      } else {
        while (true) {
          robber.request();

          auto i = threadIdx.x + block_idx.x * blockDim.x;
          auto j = threadIdx.y + block_idx.y * blockDim.y;
          auto k = threadIdx.z + block_idx.z * blockDim.z;
          if constexpr (Arg::set_block_idx) f.block_idx = block_idx;

          auto in_bounds = !Arg::check_bounds ? true : (i < arg.threads.x && j < arg.threads.y && k < arg.threads.z);
          if (in_bounds) f(i, j, k);

          bool success = robber.complete();
          if (!success) break;

          block_idx = robber.get_block_idx();

          robber.release();
        }
      }
#else
      // When QUDA_WORK_STEAL_COOPERATIVE is not defined, use runtime work stealing; work_steal_functor has no effect
      cuda::device::for_each_canceled_block<3>([&](dim3 block_idx) {
        auto i = threadIdx.x + block_idx.x * blockDim.x;
        auto j = threadIdx.y + block_idx.y * blockDim.y;
        auto k = threadIdx.z + block_idx.z * blockDim.z;

        if constexpr (Arg::check_bounds) {
          if (i >= arg.threads.x) return;
          if (j >= arg.threads.y) return;
          if (k >= arg.threads.z) return;
        }

        if constexpr (Arg::set_block_idx) f.block_idx = block_idx;
        f(i, j, k);
      });
#endif

    } else {

      if constexpr (Arg::set_block_idx) f.block_idx = dim3(blockIdx.x, blockIdx.y, blockIdx.z);
      auto i = threadIdx.x + blockIdx.x * blockDim.x;
      auto j = threadIdx.y + blockIdx.y * blockDim.y;
      auto k = threadIdx.z + blockIdx.z * blockDim.z;

      if constexpr (Arg::check_bounds) {
        if (j >= arg.threads.y) return;
        if (k >= arg.threads.z) return;

        while (i < arg.threads.x) {
          f(i, j, k);
          if constexpr (grid_stride)
            i += gridDim.x * blockDim.x;
          else
            break;
        }
      } else {
        f(i, j, k);
      }
    }
  }

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and a max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel3D(const GRID_CONSTANT Arg arg)
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and no max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void>
  Kernel3D(const GRID_CONSTANT Arg arg)
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and a max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Kernel3D()
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  /**
     @brief Kernel3D is the entry point of the generic 3-d kernel.
     This is the specialization where the kernel argument struct is
     passed by value directly to the kernel and no max register count
     is specified.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs == 0), void> Kernel3D()
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
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
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(const GRID_CONSTANT Arg arg)
  {
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
    Functor<Arg> f(arg);
    f();
  }

} // namespace quda
