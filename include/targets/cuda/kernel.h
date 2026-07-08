#pragma once

#include <type_traits>
#include <utility>

#include <target_device.h>
#include <constant_kernel_arg.h>
#include <kernel_helper.h>

namespace quda
{

  /**
     Capability traits for functor \c prefetch methods. The generic kernel drivers do not
     invoke prefetch today; functors may still implement \c prefetch for future wiring.
   */
  namespace kernel_prefetch
  {
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_1d_v = requires(Functor<Arg> &f)
    {
      f.prefetch(0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_2d_v = requires(Functor<Arg> &f)
    {
      f.prefetch(0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_3d_v = requires(Functor<Arg> &f)
    {
      f.prefetch(0, 0, 0);
    };
  } // namespace kernel_prefetch

  namespace kernel_unroll
  {
    /** \c std::integral_constant<int, Arg::work_item_unroll> for batched grid-stride calls. */
    template <typename Arg>
    using work_item_unroll_t = std::integral_constant<int, static_cast<int>(Arg::work_item_unroll)>;

    /**
       True if Functor<Arg> supports work-item unroll: \c operator() templated on
       \c std::integral_constant<int, U> with runtime args \c (i, stride) (Kernel1D).
       Using a type template argument avoids matching \c MultiBlas_::template operator()<bool>.
     */
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_1d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0);
    };

    /**
       True if Functor<Arg> supports work-item unroll with runtime args \c (i, j, stride) (Kernel2D).
     */
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_2d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0);
    };

    /**
       True if Functor<Arg> supports work-item unroll with runtime args \c (i, j, k, stride) (Kernel3D).
     */
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_3d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0, 0);
    };
  } // namespace kernel_unroll

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
      const auto grid_stride_x = gridDim.x * blockDim.x;
      if constexpr (grid_stride) {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_1d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * grid_stride_x < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, grid_stride_x);
              i += Arg::work_item_unroll * grid_stride_x;
            }
          }
        }
      } else {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_1d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * arg.item_stride < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, arg.item_stride);
              i += Arg::work_item_unroll * arg.item_stride;
            }
          }
        }
      }
      constexpr bool scalar_tail
        = grid_stride || (Arg::work_item_unroll <= 1u) || !kernel_unroll::kernel_functor_unroll_1d_v<Functor, Arg>;
      if constexpr (scalar_tail) {
        while (i < arg.threads.x) {
          f(i);
          if constexpr (grid_stride) {
            i += grid_stride_x;
          } else
            break;
        }
      }
    } else {
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

      const auto grid_stride_x = gridDim.x * blockDim.x;
      if constexpr (grid_stride) {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_2d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * grid_stride_x < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, j, grid_stride_x);
              i += Arg::work_item_unroll * grid_stride_x;
            }
          }
        }
      } else {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_2d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * arg.item_stride < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, j, arg.item_stride);
              i += Arg::work_item_unroll * arg.item_stride;
            }
          }
        }
      }
      constexpr bool scalar_tail
        = grid_stride || (Arg::work_item_unroll <= 1u) || !kernel_unroll::kernel_functor_unroll_2d_v<Functor, Arg>;
      if constexpr (scalar_tail) {
        while (i < arg.threads.x) {
          f(i, j);
          if constexpr (grid_stride) {
            i += grid_stride_x;
          } else
            break;
        }
      }
    } else {
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

      const auto grid_stride_x = gridDim.x * blockDim.x;
      if constexpr (grid_stride) {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_3d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * grid_stride_x < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, j, k, grid_stride_x);
              i += Arg::work_item_unroll * grid_stride_x;
            }
          }
        }
      } else {
        if constexpr (Arg::work_item_unroll > 1u) {
          if constexpr (kernel_unroll::kernel_functor_unroll_3d_v<Functor, Arg>) {
            while (i + (Arg::work_item_unroll - 1u) * arg.item_stride < arg.threads.x) {
              f.template operator()<kernel_unroll::work_item_unroll_t<Arg>>(i, j, k, arg.item_stride);
              i += Arg::work_item_unroll * arg.item_stride;
            }
          }
        }
      }
      constexpr bool scalar_tail
        = grid_stride || (Arg::work_item_unroll <= 1u) || !kernel_unroll::kernel_functor_unroll_3d_v<Functor, Arg>;
      if constexpr (scalar_tail) {
        while (i < arg.threads.x) {
          f(i, j, k);
          if constexpr (grid_stride) {
            i += grid_stride_x;
          } else
            break;
        }
      }
    } else {
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
