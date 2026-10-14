#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#include <utility>

#include <kernel_helper.h>
#include <target_device.h>
#include <constant_kernel_arg.h>

namespace quda
{

  /**
     Capability traits for functor \c prefetch methods. The generic kernel drivers do not
     invoke prefetch today; functors may still implement \c prefetch for future wiring.
   */
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

  namespace kernel_unroll
  {
    template <typename Arg>
    using work_item_unroll_t = std::integral_constant<int, static_cast<int>(Arg::work_item_unroll)>;

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_1d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_2d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool kernel_functor_unroll_3d_v = requires(Functor<Arg> &f)
    {
      f.template operator()<work_item_unroll_t<Arg>>(0, 0, 0, 0);
    };
  } // namespace kernel_unroll

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel1D_impl(const Arg &arg)
  {
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
      constexpr bool scalar_tail = grid_stride || (Arg::work_item_unroll <= 1u)
        || !kernel_unroll::kernel_functor_unroll_1d_v<Functor, Arg>;
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

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel1D_launch_bounds<Arg>()) Kernel1D(Arg arg)
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(arg);
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel1D_launch_bounds<Arg>()) Kernel1D()
  {
    Kernel1D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel2D_impl(const Arg &arg)
  {
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
      constexpr bool scalar_tail = grid_stride || (Arg::work_item_unroll <= 1u)
        || !kernel_unroll::kernel_functor_unroll_2d_v<Functor, Arg>;
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

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel2D_launch_bounds<Arg>()) Kernel2D(Arg arg)
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(arg);
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel2D_launch_bounds<Arg>()) Kernel2D()
  {
    Kernel2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel3D_impl(const Arg &arg)
  {
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
      constexpr bool scalar_tail = grid_stride || (Arg::work_item_unroll <= 1u)
        || !kernel_unroll::kernel_functor_unroll_3d_v<Functor, Arg>;
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

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel3D_launch_bounds<Arg>()) Kernel3D(Arg arg)
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(arg);
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel3D_launch_bounds<Arg>()) Kernel3D()
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(Arg arg)
  {
    Functor<Arg> f(arg);
    f();
  }

} // namespace quda
