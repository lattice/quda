#pragma once

#include <type_traits>
#include <utility>
#include <constant_kernel_arg.h>

#include <target_device.h>
#include <kernel_helper.h>
#include <kernel_ops_target.h>

#define OMP_KERNEL(kern)                                                                                               \
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>                                 \
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> kern(Arg arg)                                       \
  {                                                                                                                    \
    QUDA_OMPTARGET_KERNEL_BEGIN(arg)                                                                                   \
    kern##_impl<Functor, Arg, grid_stride>(arg);                                                                       \
    QUDA_OMPTARGET_KERNEL_END                                                                                          \
  }

#define OMP_KERNEL_PTR(kern)                                                                                           \
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>                                 \
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> kern(Arg *argp)                                    \
  {                                                                                                                    \
    QUDA_OMPTARGET_KERNEL_BEGIN_PTR(argp)                                                                              \
    kern##_impl<Functor, Arg, grid_stride>(*argp);                                                                     \
    QUDA_OMPTARGET_KERNEL_END                                                                                          \
  }

namespace quda
{

  /**
     Capability traits for functor \c prefetch methods. The generic kernel drivers do not
     invoke prefetch today; functors may still implement \c prefetch for future wiring.
   */
  namespace kernel_prefetch
  {
    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_1d_v = requires(Functor<Arg> &f) { f.prefetch(0); };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_2d_v = requires(Functor<Arg> &f) { f.prefetch(0, 0); };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_prefetch_3d_v = requires(Functor<Arg> &f) { f.prefetch(0, 0, 0); };
  } // namespace kernel_prefetch

  namespace kernel_unroll
  {
    template <typename Arg>
    using work_item_unroll_t = std::integral_constant<int, static_cast<int>(Arg::work_item_unroll)>;

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_1d_v
      = requires(Functor<Arg> &f) { f.template operator()<work_item_unroll_t<Arg>>(0, 0); };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_2d_v
      = requires(Functor<Arg> &f) { f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0); };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_3d_v
      = requires(Functor<Arg> &f) { f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0, 0); };
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
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if constexpr (Arg::check_bounds && needsFullBlock<Functor<Arg>>) {
      // Inactive threads still enter the functor so every team member reaches its barriers.
      if constexpr (grid_stride) {
        const auto stride = gridDim.x * blockDim.x;
        for (auto base = blockIdx.x * blockDim.x; base < arg.threads.x; base += stride, i += stride) {
          f.template operator()<true>(i, i < arg.threads.x);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); } // finish reading before reuse
        }
      } else if constexpr (Arg::work_item_unroll > 1u) {
        const bool active_item = i < static_cast<unsigned int>(arg.item_stride);
        for (unsigned int e = 0; e < Arg::work_item_unroll; e++, i += arg.item_stride) {
          f.template operator()<true>(i, active_item && i < arg.threads.x);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); }
        }
      } else {
        f.template operator()<true>(i, i < arg.threads.x);
      }
    } else if constexpr (Arg::check_bounds) {
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

    if constexpr (Arg::check_bounds && needsFullBlock<Functor<Arg>>) {
      // Inactive threads still enter the functor so every team member reaches its barriers.
      if constexpr (grid_stride) {
        const auto stride = gridDim.x * blockDim.x;
        for (auto base = blockIdx.x * blockDim.x; base < arg.threads.x; base += stride, i += stride) {
          f.template operator()<true>(i, j, i < arg.threads.x && j < arg.threads.y);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); } // finish reading before reuse
        }
      } else if constexpr (Arg::work_item_unroll > 1u) {
        const bool active_item = i < static_cast<unsigned int>(arg.item_stride);
        for (unsigned int e = 0; e < Arg::work_item_unroll; e++, i += arg.item_stride) {
          f.template operator()<true>(i, j, active_item && i < arg.threads.x && j < arg.threads.y);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); }
        }
      } else {
        f.template operator()<true>(i, j, i < arg.threads.x && j < arg.threads.y);
      }
    } else if constexpr (Arg::check_bounds) {
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

    if constexpr (Arg::check_bounds && needsFullBlock<Functor<Arg>>) {
      // Inactive threads still enter the functor so every team member reaches its barriers.
      if constexpr (grid_stride) {
        const auto stride = gridDim.x * blockDim.x;
        for (auto base = blockIdx.x * blockDim.x; base < arg.threads.x; base += stride, i += stride) {
          f.template operator()<true>(i, j, k, i < arg.threads.x && j < arg.threads.y && k < arg.threads.z);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); } // finish reading before reuse
        }
      } else if constexpr (Arg::work_item_unroll > 1u) {
        const bool active_item = i < static_cast<unsigned int>(arg.item_stride);
        for (unsigned int e = 0; e < Arg::work_item_unroll; e++, i += arg.item_stride) {
          f.template operator()<true>(i, j, k,
                                      active_item && i < arg.threads.x && j < arg.threads.y && k < arg.threads.z);
          if constexpr (needsSharedMem<Functor<Arg>>) { __syncthreads(); }
        }
      } else {
        f.template operator()<true>(i, j, k, i < arg.threads.x && j < arg.threads.y && k < arg.threads.z);
      }
    } else if constexpr (Arg::check_bounds) {
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

} // namespace quda

#undef OMP_KERNEL
#undef OMP_KERNEL_PTR
