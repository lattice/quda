#pragma once

#ifndef QUDA_WARP_SIZE
#define QUDA_WARP_SIZE 16
#endif
#ifndef QUDA_MAX_BLOCK_SIZE
#define QUDA_MAX_BLOCK_SIZE 512
#endif
#ifndef QUDA_MAX_ARGUMENT_SIZE
#define QUDA_MAX_ARGUMENT_SIZE 2048
#endif

namespace quda {

  extern void *arg_buf;
  extern int arg_buf_size;

  namespace target {

    // compile-time dispatch
    template <template <bool, typename ...> class f, typename ...Args>
    auto dispatch(Args &&... args)
    {
#ifdef __SYCL_DEVICE_ONLY__
      return f<true>()(args...);
#else
      return f<false>()(args...);
#endif
    }

    template <bool is_device> struct is_device_impl { constexpr bool operator()() { return false; } };
    template <> struct is_device_impl<true> { constexpr bool operator()() { return true; } };

    /**
       @brief Helper function that returns if the current execution
       region is on the device
    */
    __device__ __host__ inline bool is_device() { return dispatch<is_device_impl>(); }


    template <bool is_device> struct is_host_impl { constexpr bool operator()() { return true; } };
    template <> struct is_host_impl<true> { constexpr bool operator()() { return false; } };

    /**
       @brief Helper function that returns if the current execution
       region is on the host
    */
    __device__ __host__ inline bool is_host() { return dispatch<is_host_impl>(); }


    template <bool is_device> struct block_dim_impl { dim3 operator()() { return dim3(1, 1, 1); } };
    template <> struct block_dim_impl<true> { dim3 operator()() { return getBlockDim(); } };

    /**
       @brief Helper function that returns the block dimensions.  On
       CUDA this returns the intrinsic blockDim, whereas on the host
       this returns (1, 1, 1).
    */
    inline dim3 block_dim() { return getBlockDim(); }

    /**
       @brief Helper function that returns the grid dimensions.  On
       CUDA this returns the intrinsic blockDim, whereas on the host
       this returns (1, 1, 1).
    */
    inline dim3 grid_dim() { return getGridDim(); }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    inline dim3 block_idx() { return getBlockIdx(); }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       threadIdx, whereas on the host this just returns (0, 0, 0).
    */
    inline dim3 thread_idx() { return getThreadIdx(); }

    //inline uint local_linear_id() { return getLocalLinearId(); }

  } // namespace target

  namespace device {

    /**
       @brief Helper function that returns the warp-size of the
       architecture we are running on.
    */
    constexpr int warp_size() { return QUDA_WARP_SIZE; }

    /**
       @brief Return the thread mask for a converged warp.
    */
    constexpr unsigned int warp_converged_mask() { return 0xffffffff; }

    /**
       @brief Helper function that returns the maximum number of threads
       in a block in the x dimension.
    */
    template <int block_size_y = 1, int block_size_z = 1>
      constexpr unsigned int max_block_size()
      {
        //return std::max(warp_size(), 1024 / (block_size_y * block_size_z));
        //return QUDA_MAX_BLOCK_SIZE / (block_size_y * block_size_z);
        return std::max(warp_size(), QUDA_MAX_BLOCK_SIZE / (block_size_y * block_size_z));
      }

    /**
       @brief Helper function that returns the maximum number of threads
       in a block in the x dimension for reduction kernels.
    */
    template <int block_size_y = 1, int block_size_z = 1>
      constexpr unsigned int max_reduce_block_size()
      {
#ifdef QUDA_FAST_COMPILE_REDUCE
        // This is the specialized variant used when we have fast-compilation mode enabled
        return warp_size();
#else
        return max_block_size<block_size_y, block_size_z>();
#endif
      }

    /**
       @brief Helper function that returns the maximum number of threads
       in a block in the x dimension for reduction kernels.
    */
    template <int block_size_y = 1, int block_size_z = 1>
    constexpr unsigned int max_multi_reduce_block_size()
    {
#ifdef QUDA_FAST_COMPILE_REDUCE
      // This is the specialized variant used when we have fast-compilation mode enabled
      return warp_size();
#else
      return max_block_size<block_size_y, block_size_z>();
#endif
    }

    /**
       @brief Helper function that returns the maximum size of a
       __constant__ buffer on the target architecture.  For CUDA,
       this is set to the somewhat arbitrary limit of 32 KiB for now.
    */
    constexpr size_t max_constant_size() { return 32768; }

    /**
       @brief Helper function that returns the maximum size of a
       constant_param_t buffer on the target architecture.  For CUDA,
       this corresponds to the maximum __constant__ buffer size.
    */
    //constexpr size_t max_constant_param_size() { return 8192; }
    constexpr size_t max_constant_param_size() { return 32768; }

    /**
       @brief Helper function that returns the maximum static size of
       the kernel arguments passed to a kernel on the target
       architecture.
    */
    constexpr size_t max_kernel_arg_size() { return QUDA_MAX_ARGUMENT_SIZE; }

    /**
       @brief Helper function that returns the bank width of the
       shared memory bank width on the target architecture.
    */
    constexpr int shared_memory_bank_width() { return 32; }

    /**
       @brief Helper function that returns true if we are to pass the
       kernel parameter struct to the kernel as an explicit kernel
       argument.  Otherwise the parameter struct is explicitly copied
       to the device prior to kernel launch.
    */
    template <typename Arg> constexpr bool use_kernel_arg()
    {
      return (sizeof(Arg) <= device::max_kernel_arg_size() && Arg::use_kernel_arg);
    }

  } // namespace device

} // namespace quda
