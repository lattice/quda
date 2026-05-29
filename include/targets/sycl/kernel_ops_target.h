#pragma once
#include <kernel_ops.h>
#include <block_reduce_helper.h>

namespace quda
{

  // needsFullBlock
  template <typename T> static constexpr bool needsFullBlockImpl = (T) false;
  template <typename... T> static constexpr bool needsFullBlockImpl<KernelOps<T...>> = (needsFullBlockImpl<T> || ...);
  template <typename T> static constexpr bool needsFullBlock = needsFullBlockImpl<getKernelOps<T>>;
  // needsFullWarp?

  // needsSharedMem
  template <typename T> static constexpr bool needsSharedMemImpl = (T) false;
  template <typename T> static constexpr bool needsSharedMem = needsSharedMem<getKernelOps<T>>;
  template <typename... T> static constexpr bool needsSharedMem<KernelOps<T...>> = (needsSharedMemImpl<T> || ...);

  // KernelOps
  template <typename... T> struct KernelOps : KernelOpsBase<T...> {
    sycl::local_ptr<char> smem = nullptr;

    inline KernelOps() { static_assert(!needsSharedMem<KernelOps<T...>>); }

    inline KernelOps(char *s)
    { // for host
      static_assert(needsSharedMem<KernelOps<T...>>);
      smem = s;
    }

    template <typename... U> inline KernelOps(const KernelOps<U...> &ops)
    {
      checkKernelOps<T...>(ops);
      if constexpr (needsSharedMem<KernelOps<T...>>) { smem = ops.smem; }
    }
  };

  // blockSync
  template <typename... T> inline void blockSync(const KernelOps<T...> &ops)
  {
    checkKernelOps<op_blockSync>(ops);
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(getGroup());
#endif
  }

  // op implementations
  struct op_blockSync {
    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };
  template <> static constexpr bool needsSharedMemImpl<op_blockSync> = false;

  template <typename T>
  struct op_warp_combine {
    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };
  template <typename T> static constexpr bool needsFullBlockImpl<op_warp_combine<T>> = false;
  template <typename T> static constexpr bool needsSharedMemImpl<op_warp_combine<T>> = false;

  // tests
#if 0
  static const int opTestArg = 10;
  static_assert(needsFullBlock<only_SharedMemoryCache<float>> == true);
  static_assert(sharedMemSize<only_SharedMemoryCache<float>>(dim3(2,3,4))==24*sizeof(float));
  static_assert(sharedMemSize<only_SharedMemoryCache<float>>(dim3(2,3,4),opTestArg)==24*sizeof(float));

  template <typename T, typename U> static constexpr bool opTestHasKernelOpType = hasKernelOpType<T,U>;
  template <typename T, int n = 0> static constexpr bool opTestAllHasKernelOpType = false;
  template <typename ...T> static constexpr bool opTestAllHasKernelOpType<KernelOps<T...>,sizeof...(T)> = true;
  template <typename ...T, int n> static constexpr bool opTestAllHasKernelOpType<KernelOps<T...>,n> =
    opTestHasKernelOpType<std::tuple_element_t<n,std::tuple<T...>>,KernelOps<T...>> &&
    opTestAllHasKernelOpType<KernelOps<T...>,n+1>;

  using opTestC1 = op_Concurrent<op_blockSync,op_thread_array<bool,4>>;
  using opTest1 = KernelOps<op_blockSync,op_warp_combine<int>,op_thread_array<float,4>,op_SharedMemoryCache<float>,
    op_SharedMemory<double>,op_SharedMemStatic<char,100>,opTestC1>;
  static_assert(opTestAllHasKernelOpType<opTest1>);
  static_assert(hasKernelOpType<opTestC1,opTest1>);
  static_assert(!hasKernelOpType<op_thread_array<bool,4>,opTest1>);

  static_assert(sharedMemSize<opTest1>(dim3(0,0,0))==std::max((unsigned int)100,0*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(dim3(1,2,5))==std::max({(unsigned int)100,10*sizeof(double),40*sizeof(float)}));
  static_assert(sharedMemSize<opTest1>(dim3(2,5,10))==std::max({(unsigned int)100,100*sizeof(double),400*sizeof(float)}));
#endif

#if 0
  using opTest2 = KernelOps<op_blockSync,op_warp_combine<int>,op_thread_array<float,4>,
			     op_SharedMemoryCache<double>,op_SharedMemory<float>,op_SharedMemStatic<char,100>>;
  static_assert(opTestAllHasKernelOpType<opTest1>);
   template <typename T, typename U> static constexpr bool opTestKernelOpsType =
    //std::is_same_v<KernelOpsType<T,U>,KernelOps<T>;
    hasKernelOpType<T,U>;
  template <typename T, int n = 0> static constexpr bool opTestAllKernelOpsType = false;
  template <typename ...T> static constexpr bool opTestAllKernelOpsType<KernelOps<T...>,sizeof...(T)> = true;
  template <typename ...T, int n> static constexpr bool opTestAllKernelOpsType<KernelOps<T...>,n> =
    opTestKernelOpsType<std::tuple_element_t<n,std::tuple<T...>>,KernelOps<T...>> &&
    opTestAllKernelOpsType<KernelOps<T...>,n+1>;
#endif
} // namespace quda
