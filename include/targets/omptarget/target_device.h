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

#ifndef QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD
/* 0: racy
   1: if && barrier
   2: single
   3: teams set && parallel
 */
#define QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD 0
#endif

#if QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD==0

  #define QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block) \
    _Pragma("omp parallel num_threads(ld)") \
    { target::omptarget::launch_param_device_set(grid, block);

#elif QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD==1

  #define QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block) \
    _Pragma("omp parallel num_threads(ld)") \
    { if(omp_get_thread_num()==0) \
        target::omptarget::launch_param_device_set(grid, block); \
      _Pragma("omp barrier")

#elif QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD==2

  #define QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block) \
    _Pragma("omp parallel num_threads(ld)") \
    { _Pragma("omp single") \
      target::omptarget::launch_param_device_set(grid, block);

#elif QUDA_OMPTARGET_PARALLEL_LAUNCH_METHOD==3

  #define QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block) \
    target::omptarget::launch_param_device_set(grid, block); \
    _Pragma("omp parallel num_threads(ld)") \
    {

#else

  #error "Allowed values for QUDA_OMPTARGET_LAUNCH_METHOD are 0, 1, 2, or 3."

#endif

#define QUDA_OMPTARGET_KERNEL_BEGIN(arg) \
    const dim3 grid = target::omptarget::launch_param_grid(); \
    const dim3 block = target::omptarget::launch_param_block(); \
    const int gd = grid.x*grid.y*grid.z; \
    const int ld = block.x*block.y*block.z; \
    _Pragma("omp target teams num_teams(gd) thread_limit(ld) firstprivate(arg,grid,block)") \
    { QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block)

#define QUDA_OMPTARGET_KERNEL_BEGIN_PTR(argp) \
    const dim3 grid = target::omptarget::launch_param_grid(); \
    const dim3 block = target::omptarget::launch_param_block(); \
    const int gd = grid.x*grid.y*grid.z; \
    const int ld = block.x*block.y*block.z; \
    _Pragma("omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(argp) firstprivate(grid,block)") \
    { QUDA_OMPTARGET_PARALLEL_LAUNCH(ld,grid,block)

#define QUDA_OMPTARGET_KERNEL_END }} /* closes BEGIN/BEGIN_PTR */

namespace quda {

  namespace target {

    namespace omptarget {
      inline dim3 & launch_param_kernel_block(void)
      {
#if 1
        static char block[sizeof(dim3)];
        #pragma omp groupprivate(block)
        return *reinterpret_cast<dim3*>(block);
#else
        /* omp 6 */
        static char block[sizeof(dim3)];
        #pragma omp threadprivate(block)
        return *reinterpret_cast<dim3*>(block);
#endif
      }
      inline dim3 & launch_param_kernel_grid(void)
      {
#if 1
        static char grid[sizeof(dim3)];
        #pragma omp groupprivate(grid)
        return *reinterpret_cast<dim3*>(grid);
#else
        /* omp 6 */
        static char grid[sizeof(dim3)];
        #pragma omp threadprivate(grid)
        return *reinterpret_cast<dim3*>(grid);
#endif
      }
      inline void launch_param_device_set(dim3 grid, dim3 block)
      {
        dim3 & gref = launch_param_kernel_grid();
        dim3 & bref = launch_param_kernel_block();
        gref = grid;
        bref = block;
      }
    } // namespace omptarget

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

    template <bool is_device> struct block_dim_impl {
      inline dim3 operator()() { return dim3(1, 1, 1); }
    };
    template <> struct block_dim_impl<true> {
      __device__ inline dim3 operator()() { return target::omptarget::launch_param_kernel_block(); }
    };

    /**
       @brief Helper function that returns the thread block
       dimensions.  On CUDA this returns the intrinsic blockDim,
       whereas on the host this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 block_dim() { return dispatch<block_dim_impl>(); }

    template <bool is_device> struct grid_dim_impl {
      inline dim3 operator()() { return dim3(1, 1, 1); }
    };
    template <> struct grid_dim_impl<true> {
      __device__ inline dim3 operator()() { return target::omptarget::launch_param_kernel_grid(); }
    };

    /**
       @brief Helper function that returns the grid dimensions.  On
       CUDA this returns the intrinsic blockDim, whereas on the host
       this returns (1, 1, 1).
    */
    __device__ __host__ inline dim3 grid_dim() { return dispatch<grid_dim_impl>(); }

    template <bool is_device> struct block_idx_impl {
      inline dim3 operator()() { return dim3(0, 0, 0); }
    };
    template <> struct block_idx_impl<true> {
      __device__ inline dim3 operator()() {
        const dim3 & gridDim=target::omptarget::launch_param_kernel_grid();
        const auto n = (unsigned int)omp_get_team_num();
        return dim3(n%gridDim.x, (n/gridDim.x)%gridDim.y, n/(gridDim.x*gridDim.y));
      }
    };

    /**
       @brief Helper function that returns the thread indices within a
       thread block.  On CUDA this returns the intrinsic
       blockIdx, whereas on the host this just returns (0, 0, 0).
    */
    __device__ __host__ inline dim3 block_idx() { return dispatch<block_idx_impl>(); }

    template <bool is_device> struct thread_idx_impl {
      inline dim3 operator()() { return dim3(0, 0, 0); }
    };
    template <> struct thread_idx_impl<true> {
      __device__ inline dim3 operator()() {
        const dim3 & blockDim=target::omptarget::launch_param_kernel_block();
        const auto n = (unsigned int)omp_get_thread_num();
        return dim3(n%blockDim.x, (n/blockDim.x)%blockDim.y, n/(blockDim.x*blockDim.y));
      }
    };

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
      const auto n = (unsigned int)omp_get_thread_num();
      const dim3 & blockDim=target::omptarget::launch_param_kernel_block();
      switch (dim) {
      case 1: return n%blockDim.x;
      case 2: return n%(blockDim.x*blockDim.y);
      case 3:
      default: return n;
      }
    }

    /**
       @brief Helper function that returns the total number thread in a thread block
    */
    template <int dim> __device__ __host__ inline auto block_size()
    {
      const dim3 & blockDim=target::omptarget::launch_param_kernel_block();
      switch (dim) {
      case 1: return blockDim.x;
      case 2: return blockDim.y * blockDim.x;
      case 3:
      default: return blockDim.z * blockDim.y * blockDim.x;
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
      return Arg::always_use_kernel_arg() ||
        (Arg::default_use_kernel_arg() && sizeof(Arg) <= device::max_kernel_arg_size());
    }

    /**
       @brief Helper function that returns a pointer to the
       __constant__ memory buffer.  Note this is the dummy
       implementation, and is present only to keep the compiler happy
       in the translation units where constant memory is not used.
     */
    template <typename Arg> constexpr std::enable_if_t<use_kernel_arg<Arg>(), void *> get_constant_buffer() { return nullptr; }

    /**
       @brief Return the address of the shared local memory for the current thread group.
     */
    inline char *get_shared_cache(void)
    {
      static char s[device::max_shared_memory_size()];
      #pragma omp groupprivate(s)
      return s;
    }
  }

}
