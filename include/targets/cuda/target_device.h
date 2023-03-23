#pragma once

#include <type_traits>
#include <algorithm>
#ifdef _NVHPC_CUDA
#include <nv/target>
#endif

#if defined(__CUDACC__) || defined(_NVHPC_CUDA) || (defined(__clang__) && defined(__CUDA__))
#define QUDA_CUDA_CC
#endif

namespace quda
{

  namespace target
  {

#ifdef _NVHPC_CUDA

    // nvc++: run-time dispatch using if target
    template <template <bool, typename...> class f, typename... Args> __host__ __device__ auto dispatch(Args &&...args)
    {
      if target (nv::target::is_device) {
        return f<true>()(args...);
      } else {
        return f<false>()(args...);
      }
    }

#else

    // nvcc or clang: compile-time dispatch
    template <template <bool, typename...> class f, typename... Args> __host__ __device__ auto dispatch(Args &&...args)
    {
#ifdef __CUDA_ARCH__
      return f<true>()(args...);
#else
      return f<false>()(args...);
#endif
    }

#endif

    template <bool is_device> struct is_device_impl {
      constexpr bool operator()() { return false; }
    };
    template <> struct is_device_impl<true> {
      constexpr bool operator()() { return true; }
    };

    /**
       @brief Helper function that returns if the current execution
       region is on the device
    */
    __device__ __host__ inline bool is_device() { return dispatch<is_device_impl>(); }

    template <bool is_device> struct is_host_impl {
      constexpr bool operator()() { return true; }
    };
    template <> struct is_host_impl<true> {
      constexpr bool operator()() { return false; }
    };

    /**
       @brief Helper function that returns if the current execution
       region is on the host
    */
    __device__ __host__ inline bool is_host() { return dispatch<is_host_impl>(); }

    template <bool is_device> struct block_dim_impl {
      dim3 operator()() { return dim3(1, 1, 1); }
    };
#ifdef QUDA_CUDA_CC
    template <> struct block_dim_impl<true> {
      __device__ dim3 operator()() { return dim3(blockDim.x, blockDim.y, blockDim.z); }
    };
#endif

    /**
       @brief Helper function that returns the thread block
       dimensions.  On CUDA this returns the intrinsic blockDim,
       whereas on the host this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 block_dim() { return dispatch<block_dim_impl>(); }

    template <bool is_device> struct grid_dim_impl {
      dim3 operator()() { return dim3(1, 1, 1); }
    };
#ifdef QUDA_CUDA_CC
    template <> struct grid_dim_impl<true> {
      __device__ dim3 operator()() { return dim3(gridDim.x, gridDim.y, gridDim.z); }
    };
#endif

    /**
       @brief Helper function that returns the grid dimensions.  On
       CUDA this returns the intrinsic blockDim, whereas on the host
       this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 grid_dim() { return dispatch<grid_dim_impl>(); }

    template <bool is_device> struct block_idx_impl {
      dim3 operator()() { return dim3(0, 0, 0); }
    };
#ifdef QUDA_CUDA_CC
    template <> struct block_idx_impl<true> {
      __device__ dim3 operator()() { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
    };
#endif

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 block_idx() { return dispatch<block_idx_impl>(); }

    template <bool is_device> struct thread_idx_impl {
      dim3 operator()() { return dim3(0, 0, 0); }
    };
#ifdef QUDA_CUDA_CC
    template <> struct thread_idx_impl<true> {
      __device__ dim3 operator()() { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
    };
#endif

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       threadIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 thread_idx() { return dispatch<thread_idx_impl>(); }

    /**
       @brief Helper function that returns a linear thread index within a thread block.
    */
    template <int dim> __device__ __host__ inline auto thread_idx_linear()
    {
      switch (dim) {
      case 1: return thread_idx().x;
      case 2: return thread_idx().y * block_dim().x + thread_idx().x;
      case 3:
      default: return (thread_idx().z * block_dim().y + thread_idx().y) * block_dim().x + thread_idx().x;
      }
    }

    /**
       @brief Helper function that returns the total number thread in a thread block
    */
    template <int dim> __device__ __host__ inline auto block_size()
    {
      switch (dim) {
      case 1: return block_dim().x;
      case 2: return block_dim().y * block_dim().x;
      case 3:
      default: return block_dim().z * block_dim().y * block_dim().x;
      }
    }

  } // namespace target

  namespace device
  {

    /**
       @brief Helper function that returns the warp-size of the
       architecture we are running on.
    */
    constexpr int warp_size() { return 32; }

    /**
       @brief Return the thread mask for a converged warp.
    */
    constexpr unsigned int warp_converged_mask() { return 0xffffffff; }

    /**
       @brief Helper function that returns the maximum number of threads
       in a block in the x dimension.
    */
    template <int block_size_y = 1, int block_size_z = 1> constexpr unsigned int max_block_size()
    {
      return std::max(warp_size(), 1024 / (block_size_y * block_size_z));
    }

    /**
       @brief Helper function that returns the maximum size of a
       __constant__ buffer on the target architecture.  For CUDA,
       this is set to the somewhat arbitrary limit of 32 KiB for now.
    */
    constexpr size_t max_constant_size() { return 32764; }

    /**
       @brief Helper function that returns the default max kernel arg
       size when QUDA_LARGE_KERNEL_ARG is not enabled.
     */
    constexpr size_t max_kernel_arg_legacy_size() { return 4096; }

    /**
       @brief Helper function that returns the maximum static size of
       the kernel arguments passed to a kernel on the target
       architecture.
    */
#ifdef QUDA_LARGE_KERNEL_ARG
    constexpr size_t max_kernel_arg_size() { return max_constant_size(); }
#else
    constexpr size_t max_kernel_arg_size() { return max_kernel_arg_legacy_size(); }
#endif

    /**
       @brief Helper function that returns true if we are to pass the
       kernel parameter struct to the kernel as an explicit kernel
       argument.  Otherwise the parameter struct is explicitly copied
       to the device prior to kernel launch.
    */
    template <typename Arg> constexpr bool use_kernel_arg()
    {
      return Arg::always_use_kernel_arg() ||
        (Arg::default_use_kernel_arg() && sizeof(Arg) <= device::max_kernel_arg_size());
    }

    /**
       @brief Helper function that returns kernel argument from
       __constant__ memory.  Note this is the dummy implementation,
       and is present only to keep the compiler happy in the
       translation units where constant memory is not used.
     */
    template <typename Arg> constexpr std::enable_if_t<use_kernel_arg<Arg>(), const Arg &> get_arg()
    {
      return reinterpret_cast<Arg &>(nullptr);
    }

    /**
       @brief Helper function that returns a pointer to the
       __constant__ memory buffer.  Note this is the dummy
       implementation, and is present only to keep the compiler happy
       in the translation units where constant memory is not used.
     */
    template <typename Arg> constexpr std::enable_if_t<use_kernel_arg<Arg>(), void *> get_constant_buffer()
    {
      return nullptr;
    }

  } // namespace device

} // namespace quda

#undef QUDA_CUDA_CC
