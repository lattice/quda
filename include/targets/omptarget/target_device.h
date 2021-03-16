#pragma once

namespace quda {

  namespace target {

#pragma omp begin declare variant match(device={kind(host)})
    template <template <bool, typename ...> class f, typename ...Args>
      __host__ __device__ auto dispatch(Args &&... args)
    {
      return f<false>()(args...);
    }
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(nohost)})
    template <template <bool, typename ...> class f, typename ...Args>
      __host__ __device__ auto dispatch(Args &&... args)
    {
      return f<true>()(args...);
    }
#pragma omp end declare variant

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

    /**
       @brief Helper function that returns the thread block
       dimensions.  On CUDA this returns the intrinsic blockDim,
       whereas on the host this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 block_dim()
    {
      return dim3(omp_get_num_threads());
/*
#ifdef __CUDA_ARCH__
      return dim3(blockDim.x, blockDim.y, blockDim.z);
#else
      return dim3(1, 1, 1);
#endif
*/
    }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 block_idx()
    {
      return dim3(omp_get_team_num(),0,0);
/*
#ifdef __CUDA_ARCH__
      return dim3(blockIdx.x, blockIdx.y, blockIdx.z);
#else
      return dim3(0, 0, 0);
#endif
*/
    }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       threadIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 thread_idx()
    {
      return dim3(omp_get_thread_num(),0,0);
/*
#ifdef __CUDA_ARCH__
      return dim3(threadIdx.x, threadIdx.y, threadIdx.z);
#else
      return dim3(0, 0, 0);
#endif
*/
    }

  }

  namespace device {

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

}
