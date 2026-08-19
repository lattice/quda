#pragma once

#include <type_traits>
#include <utility>

#include <target_device.h>
#include <constant_kernel_arg.h>
#include <reduce_helper.h>

namespace quda
{

  /**
     Capability traits for transformer \c prefetch methods. The reduction drivers do not
     invoke prefetch today; functors may still implement \c prefetch for future wiring.
   */
  namespace reduction_prefetch
  {
    template <template <typename> class Transformer, typename Arg>
    inline constexpr bool reduction_functor_prefetch_2d_v = requires(Transformer<Arg> &t)
    {
      t.prefetch(0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool reduction_functor_prefetch_3d_v = requires(Functor<Arg> &t)
    {
      t.prefetch(0, 0, 0);
    };
  } // namespace reduction_prefetch

  namespace reduction_unroll
  {
    template <typename Arg>
    using work_item_unroll_t = std::integral_constant<int, static_cast<int>(Arg::work_item_unroll)>;

    template <template <typename> class Transformer, typename Arg>
    inline constexpr bool reduction_functor_unroll_2d_v = requires(Transformer<Arg> &t)
    {
      t.template operator()<work_item_unroll_t<Arg>>(std::declval<typename Transformer<Arg>::reduce_t &>(), 0, 0, 0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool reduction_functor_unroll_3d_v = requires(Functor<Arg> &t)
    {
      t.template operator()<work_item_unroll_t<Arg>>(std::declval<typename Functor<Arg>::reduce_t &>(), 0, 0, 0, 0);
    };
  } // namespace reduction_unroll

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
    using reduce_t = typename Transformer<Arg>::reduce_t;
    using reducer_t = typename Transformer<Arg>::reducer_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = reducer_t::init();

    const auto stride = blockDim.x * gridDim.x;
    if constexpr (grid_stride) {
      if constexpr (Arg::work_item_unroll > 1u) {
        while (idx + (Arg::work_item_unroll - 1u) * stride < arg.threads.x) {
          if constexpr (reduction_unroll::reduction_functor_unroll_2d_v<Transformer, Arg>) {
            value = t.template operator()<reduction_unroll::work_item_unroll_t<Arg>>(value, idx, j, 0, stride);
            idx += Arg::work_item_unroll * stride;
          } else {
#pragma unroll
            for (unsigned e = 0; e < Arg::work_item_unroll; e++) { value = t(value, idx + e * stride, j); }
            idx += Arg::work_item_unroll * stride;
          }
        }
      }
    }
    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      if constexpr (grid_stride) {
        idx += stride;
      } else
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
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_reduction_launch_bounds<Arg>()) Reduction2D(Arg arg)
  {
    Reduction2D_impl<Functor, Arg, grid_stride>(arg);
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
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_reduction_launch_bounds<Arg>()) Reduction2D()
  {
    Reduction2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
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
    using reduce_t = typename Functor<Arg>::reduce_t;
    using reducer_t = typename Functor<Arg>::reducer_t;
    Functor<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto k = threadIdx.y;
    auto j = threadIdx.z + blockIdx.z * blockDim.z;

    if (j >= arg.threads.z) return;

    reduce_t value = reducer_t::init();

    const auto stride = blockDim.x * gridDim.x;
    if constexpr (grid_stride) {
      if constexpr (Arg::work_item_unroll > 1u) {
        while (idx + (Arg::work_item_unroll - 1u) * stride < arg.threads.x) {
          if constexpr (reduction_unroll::reduction_functor_unroll_3d_v<Functor, Arg>) {
            value = t.template operator()<reduction_unroll::work_item_unroll_t<Arg>>(value, idx, k, j, stride);
            idx += Arg::work_item_unroll * stride;
          } else {
#pragma unroll
            for (unsigned e = 0; e < Arg::work_item_unroll; e++) { value = t(value, idx + e * stride, k, j); }
            idx += Arg::work_item_unroll * stride;
          }
        }
      }
    }
    while (idx < arg.threads.x) {
      value = t(value, idx, k, j);
      if constexpr (grid_stride) {
        idx += stride;
      } else
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
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_multireduction_launch_bounds<Arg>()) MultiReduction(Arg arg)
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is copied to the device prior to kernel launch.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_multireduction_launch_bounds<Arg>()) MultiReduction()
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

} // namespace quda
