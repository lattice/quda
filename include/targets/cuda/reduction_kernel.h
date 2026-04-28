#pragma once

#include <target_device.h>
#include <constant_kernel_arg.h>
#include <reduce_helper.h>

namespace quda
{

  namespace reduction_prefetch
  {
    template <template <typename> class Transformer, typename Arg>
    inline constexpr bool reduction_functor_prefetch_2d_v = requires(Transformer<Arg> &t) {
      t.prefetch(0, 0);
    };

    template <template <typename> class Functor, typename Arg>
    inline constexpr bool reduction_functor_prefetch_3d_v = requires(Functor<Arg> &t) {
      t.prefetch(0, 0, 0);
    };
  } // namespace reduction_prefetch

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
#ifdef QUDA_SHARED_MEMORY_SPILL
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
#endif
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = t.init();

    if constexpr (grid_stride) {
      if constexpr (reduction_prefetch::reduction_functor_prefetch_2d_v<Transformer, Arg>) {
        if (idx < arg.threads.x) t.prefetch(idx, j);
      }
    }
    const auto stride = blockDim.x * gridDim.x;
    while (idx < arg.threads.x) {
      if constexpr (grid_stride) {
        if constexpr (Arg::grid_stride_unroll > 1u) {
          if (idx + (Arg::grid_stride_unroll - 1u) * stride < arg.threads.x) {
#pragma unroll
            for (unsigned e = 0; e < Arg::grid_stride_unroll; e++) {
              if constexpr (reduction_prefetch::reduction_functor_prefetch_2d_v<Transformer, Arg>) {
                const auto idx_pf = idx + (e + 1u) * stride;
                if (idx_pf < arg.threads.x) t.prefetch(idx_pf, j);
              }
              value = t(value, idx + e * stride, j);
            }
            idx += Arg::grid_stride_unroll * stride;
            continue;
          }
        }
        if constexpr (reduction_prefetch::reduction_functor_prefetch_2d_v<Transformer, Arg>) {
          const auto idx_next = idx + stride;
          if (idx_next < arg.threads.x) t.prefetch(idx_next, j);
        }
      }
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
     argument struct is passed by value directly to the kernel that
     does specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Reduction2D(Arg arg)
  {
    Reduction2D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is passed by value directly to the kernel that
     does not specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void> Reduction2D(Arg arg)
  {
    Reduction2D_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is copied to the device prior to kernel launch
     that does specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> Reduction2D()
  {
    Reduction2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is copied to the device prior to kernel launch
     that does not specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void> Reduction2D()
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
#ifdef QUDA_SHARED_MEMORY_SPILL
    if constexpr (Arg::spill_shared) asm(".pragma \"enable_smem_spilling\";");
#endif
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto k = threadIdx.y;
    auto j = threadIdx.z + blockIdx.z * blockDim.z;

    if (j >= arg.threads.z) return;

    reduce_t value = t.init();

    if constexpr (grid_stride) {
      if constexpr (reduction_prefetch::reduction_functor_prefetch_3d_v<Functor, Arg>) {
        if (idx < arg.threads.x) t.prefetch(idx, k, j);
      }
    }
    const auto stride = blockDim.x * gridDim.x;
    while (idx < arg.threads.x) {
      if constexpr (grid_stride) {
        if constexpr (Arg::grid_stride_unroll > 1u) {
          if (idx + (Arg::grid_stride_unroll - 1u) * stride < arg.threads.x) {
#pragma unroll
            for (unsigned e = 0; e < Arg::grid_stride_unroll; e++) {
              if constexpr (reduction_prefetch::reduction_functor_prefetch_3d_v<Functor, Arg>) {
                const auto idx_pf = idx + (e + 1u) * stride;
                if (idx_pf < arg.threads.x) t.prefetch(idx_pf, k, j);
              }
              value = t(value, idx + e * stride, k, j);
            }
            idx += Arg::grid_stride_unroll * stride;
            continue;
          }
        }
        if constexpr (reduction_prefetch::reduction_functor_prefetch_3d_v<Functor, Arg>) {
          const auto idx_next = idx + stride;
          if (idx_next < arg.threads.x) t.prefetch(idx_next, k, j);
        }
      }
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
     kernel argument struct is passed by value directly to the kernel
     that specifies a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> MultiReduction(Arg arg)
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel
     that does not specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>() && Arg::max_regs == 0, void> MultiReduction(Arg arg)
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(arg);
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel
     that specifies a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  MAXNREG(Arg::max_regs)
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs > 0), void> MultiReduction()
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel
     that does not specify a max register count.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<(!device::use_kernel_arg<Arg>() && Arg::max_regs == 0), void> MultiReduction()
  {
    MultiReduction_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }

} // namespace quda
