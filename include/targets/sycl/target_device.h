#pragma once

namespace quda {

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
#ifdef QUDA_CUDA_CC
    template <> struct block_dim_impl<true> { __device__ dim3 operator()() { return dim3(blockDim.x, blockDim.y, blockDim.z); } };
#endif

    /**
       @brief Helper function that returns the thread block
       dimensions.  On CUDA this returns the intrinsic blockDim,
       whereas on the host this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 block_dim() { return dispatch<block_dim_impl>(); }


    template <bool is_device> struct block_idx_impl { dim3 operator()() { return dim3(0, 0, 0); } };
#ifdef QUDA_CUDA_CC
    template <> struct block_idx_impl<true> { __device__ dim3 operator()() { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); } };
#endif

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 block_idx() { return dispatch<block_idx_impl>(); }


    template <bool is_device> struct thread_idx_impl { dim3 operator()() { return dim3(0, 0, 0); } };
#ifdef QUDA_CUDA_CC
    template <> struct thread_idx_impl<true> { __device__ dim3 operator()() { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); } };
#endif

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       threadIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 thread_idx() { return dispatch<thread_idx_impl>(); }

  }


  namespace device {

    /**
       @brief Helper function that returns the warp-size of the
       architecture we are running on.
    */
    constexpr int warp_size() { return 8; }

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
        return std::max(warp_size(), 1024 / (block_size_y * block_size_z));
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
    constexpr unsigned int max_multi_reduce_block_size()
    {
#ifdef QUDA_FAST_COMPILE_REDUCE
      // This is the specialized variant used when we have fast-compilation mode enabled
      return warp_size();
#else
      return 128;
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
    constexpr size_t max_constant_param_size() { return 8192; }

    /**
       @brief Helper function that returns the maximum static size of
       the kernel arguments passed to a kernel on the target
       architecture.
    */
    constexpr size_t max_kernel_arg_size() { return 4096; }

    /**
       @brief Helper function that returns the bank width of the
       shared memory bank width on the target architecture.
    */
    constexpr int shared_memory_bank_width() { return 32; }

  }

  /**
     @brief This is a convenience wrapper that allows us to perform
     reductions at the warp or sub-warp level
  */
  template <typename T, int width> struct WarpReduce
  {
    static_assert(width <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    //using warp_reduce_t = cub::WarpReduce<T, width>;

    __device__ __host__ inline WarpReduce() {}

    template <bool is_device, typename dummy = void> struct sum { T operator()(const T &value) { return value; } };

    template <typename dummy> struct sum<true, dummy> {
      __device__ inline T operator()(const T &value) {
        //typename warp_reduce_t::TempStorage dummy_storage;
        //warp_reduce_t warp_reduce(dummy_storage);
        //return warp_reduce.Sum(value);
	return value;
      }
    };

    __device__ __host__ inline T Sum(const T &value) { return target::dispatch<sum>(value); }
  };

}
