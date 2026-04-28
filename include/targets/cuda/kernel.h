#pragma once

#include <target_device.h>
#include <constant_kernel_arg.h>
#include <kernel_helper.h>

namespace quda
{

  namespace kernel_prefetch
  {
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_1d_v = requires(Functor<Arg> &f) {
      f.prefetch(0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_2d_v = requires(Functor<Arg> &f) {
      f.prefetch(0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_3d_v = requires(Functor<Arg> &f) {
      f.prefetch(0, 0, 0);
    };
  } // namespace kernel_prefetch

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

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if constexpr (Arg::check_bounds) {
      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_1d_v<Functor, Arg>) {
          if (i < arg.threads.x) f.prefetch(i);
        }
      }
      const auto stride = gridDim.x * blockDim.x;
      while (i < arg.threads.x) {
        if constexpr (grid_stride) {
          if constexpr (Arg::grid_stride_unroll > 1u) {
            if (i + (Arg::grid_stride_unroll - 1u) * stride < arg.threads.x) {
#pragma unroll
              for (unsigned e = 0; e < Arg::grid_stride_unroll; e++) {
                if constexpr (kernel_prefetch::kernel_functor_prefetch_1d_v<Functor, Arg>) {
                  const auto i_pf = i + (e + 1u) * stride;
                  if (i_pf < arg.threads.x) f.prefetch(i_pf);
                }
                f(i + e * stride);
              }
              i += Arg::grid_stride_unroll * stride;
              continue;
            }
          }
          if constexpr (kernel_prefetch::kernel_functor_prefetch_1d_v<Functor, Arg>) {
            const auto i_next = i + stride;
            if (i_next < arg.threads.x) f.prefetch(i_next);
          }
        }
        f(i);
        if constexpr (grid_stride) {
          i += stride;
        } else
          break;
      }
    } else {
      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_1d_v<Functor, Arg>) { f.prefetch(i); }
      }
      f(i);
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

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;

    if constexpr (Arg::check_bounds) {
      if (j >= arg.threads.y) return;

      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_2d_v<Functor, Arg>) {
          if (i < arg.threads.x) f.prefetch(i, j);
        }
      }
      const auto stride = gridDim.x * blockDim.x;
      while (i < arg.threads.x) {
        if constexpr (grid_stride) {
          if constexpr (Arg::grid_stride_unroll > 1u) {
            if (i + (Arg::grid_stride_unroll - 1u) * stride < arg.threads.x) {
#pragma unroll
              for (unsigned e = 0; e < Arg::grid_stride_unroll; e++) {
                if constexpr (kernel_prefetch::kernel_functor_prefetch_2d_v<Functor, Arg>) {
                  const auto i_pf = i + (e + 1u) * stride;
                  if (i_pf < arg.threads.x) f.prefetch(i_pf, j);
                }
                f(i + e * stride, j);
              }
              i += Arg::grid_stride_unroll * stride;
              continue;
            }
          }
          if constexpr (kernel_prefetch::kernel_functor_prefetch_2d_v<Functor, Arg>) {
            const auto i_next = i + stride;
            if (i_next < arg.threads.x) f.prefetch(i_next, j);
          }
        }
        f(i, j);
        if constexpr (grid_stride) {
          i += stride;
        } else
          break;
      }
    } else {
      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_2d_v<Functor, Arg>) { f.prefetch(i, j); }
      }
      f(i, j);
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

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z + blockIdx.z * blockDim.z;

    if constexpr (Arg::check_bounds) {
      if (j >= arg.threads.y) return;
      if (k >= arg.threads.z) return;

      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_3d_v<Functor, Arg>) {
          if (i < arg.threads.x) f.prefetch(i, j, k);
        }
      }
      const auto stride = gridDim.x * blockDim.x;
      while (i < arg.threads.x) {
        if constexpr (grid_stride) {
          if constexpr (Arg::grid_stride_unroll > 1u) {
            if (i + (Arg::grid_stride_unroll - 1u) * stride < arg.threads.x) {
#pragma unroll
              for (unsigned e = 0; e < Arg::grid_stride_unroll; e++) {
                if constexpr (kernel_prefetch::kernel_functor_prefetch_3d_v<Functor, Arg>) {
                  const auto i_pf = i + (e + 1u) * stride;
                  if (i_pf < arg.threads.x) f.prefetch(i_pf, j, k);
                }
                f(i + e * stride, j, k);
              }
              i += Arg::grid_stride_unroll * stride;
              continue;
            }
          }
          if constexpr (kernel_prefetch::kernel_functor_prefetch_3d_v<Functor, Arg>) {
            const auto i_next = i + stride;
            if (i_next < arg.threads.x) f.prefetch(i_next, j, k);
          }
        }
        f(i, j, k);
        if constexpr (grid_stride) {
          i += stride;
        } else
          break;
      }
    } else {
      if constexpr (grid_stride) {
        if constexpr (kernel_prefetch::kernel_functor_prefetch_3d_v<Functor, Arg>) { f.prefetch(i, j, k); }
      }
      f(i, j, k);
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
