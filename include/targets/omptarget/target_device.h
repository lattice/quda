#pragma once

#ifndef QUDA_WARP_SIZE
#define QUDA_WARP_SIZE 16
#endif

#ifndef QUDA_MAX_BLOCK_SIZE
#define QUDA_MAX_BLOCK_SIZE 1024
#endif

#ifndef QUDA_MAX_SHARED_MEMORY_SIZE
#define QUDA_MAX_SHARED_MEMORY_SIZE 40*1024
#endif

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
      return target::omptarget::launch_param->block;
    }

    /**
       @brief Helper function that returns the grid dimensions.  On
       CUDA this returns the intrinsic blockDim, whereas on the host
       this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 grid_dim()
    {
      return target::omptarget::launch_param->grid;
    }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 block_idx()
    {
      const auto n = (unsigned int)omp_get_team_num();
      return dim3(n%target::omptarget::launch_param->grid.x, (n/target::omptarget::launch_param->grid.x)%target::omptarget::launch_param->grid.y, n/(target::omptarget::launch_param->grid.x*target::omptarget::launch_param->grid.y));
    }

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       threadIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 thread_idx()
    {
      const auto n = (unsigned int)omp_get_thread_num();
      return dim3(n%target::omptarget::launch_param->block.x, (n/target::omptarget::launch_param->block.x)%target::omptarget::launch_param->block.y, n/(target::omptarget::launch_param->block.x*target::omptarget::launch_param->block.y));
    }

    /**
       @brief Helper function that returns a linear thread index within a thread block.
    */
    template <int dim> __device__ __host__ inline auto thread_idx_linear()
    {
      const auto n = (unsigned int)omp_get_thread_num();
      switch (dim) {
      case 1: return n%target::omptarget::launch_param->block.x;
      case 2: return n%(target::omptarget::launch_param->block.x*target::omptarget::launch_param->block.y);
      case 3:
      default: return n;
      }
    }

    /**
       @brief Helper function that returns the total number thread in a thread block
    */
    template <int dim> __device__ __host__ inline auto block_size()
    {
      switch (dim) {
      case 1: return target::omptarget::launch_param->block.x;
      case 2: return target::omptarget::launch_param->block.y * target::omptarget::launch_param->block.x;
      case 3:
      default: return target::omptarget::launch_param->block.z * target::omptarget::launch_param->block.y * target::omptarget::launch_param->block.x;
      }
    }

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
        return std::max(warp_size(), QUDA_MAX_BLOCK_SIZE / (block_size_y * block_size_z));
      }

    /**
       @brief Helper function that returns the maximum size of a
       __constant__ buffer on the target architecture.  For CUDA,
       this is set to the somewhat arbitrary limit of 32 KiB for now.
    */
    constexpr size_t max_constant_size() { return 32768; }

    /**
       @brief Helper function that returns the maximum static size of
       the kernel arguments passed to a kernel on the target
       architecture.
    */
    constexpr size_t max_kernel_arg_size() { return 64; }

    /**
       @brief Helper function that returns the bank width of the
       shared memory bank width on the target architecture.
    */
    constexpr int shared_memory_bank_width() { return 32; }

    /**
       @brief Use a compile time fixed size for the shared local memory,
       until we can find a way to set it dynamically.
    */
    constexpr unsigned int max_shared_memory_size() { return QUDA_MAX_SHARED_MEMORY_SIZE; }

    /**
       @brief Helper function that returns true if we are to pass the
       kernel parameter struct to the kernel as an explicit kernel
       argument.  Otherwise the parameter struct is explicitly copied
       to the device prior to kernel launch.
    */
    template <typename Arg> constexpr bool use_kernel_arg()
    {
      return Arg::use_kernel_arg > 1 || (sizeof(Arg) <= device::max_kernel_arg_size() && Arg::use_kernel_arg);
    }

    /**
       @brief Helper function that returns a pointer to the
       __constant__ memory buffer.  Note this is the dummy
       implementation, and is present only to keep the compiler happy
       in the translation units where constant memory is not used.
     */
    template <typename Arg> constexpr std::enable_if_t<use_kernel_arg<Arg>(), void *> get_constant_buffer() { return nullptr; }

  }

}
