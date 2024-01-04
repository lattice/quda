#pragma once
#include <kernel_ops.h>
#include <block_reduce_helper.h>

namespace quda {

  // needsSharedMem
#if 0
  template <typename T> static constexpr bool needsSharedMem = needsSharedMem<getKernelOps<T>>;
  template <typename ...T> static constexpr bool needsSharedMemImpl = (needsSharedMemImpl<T> || ...);
  template <> static constexpr bool needsSharedMemImpl<depNone> = false;
  template <> static constexpr bool needsSharedMemImpl<depFullBlock> = false;
  template <typename T, typename S> static constexpr bool needsSharedMemImpl<depSharedMem<T,S>> = true;
  template <typename ...T> static constexpr bool needsSharedMemImpl<op_Concurrent<T...>> = needsSharedMemImpl<T...>;
  template <typename ...T> static constexpr bool needsSharedMemImpl<op_Sequential<T...>> = needsSharedMemImpl<T...>;
  template <typename T> static constexpr bool needsSharedMemF() {
    if constexpr (std::is_base_of<op_Base,T>::value) {
    //if constexpr (is_instance<T,op_Base>) {
      return needsSharedMemImpl<typename T::dependencies>;
    } else {
      //if constexpr (hasKernelOps<T>) {
      //return needsSharedMem<getKernelOps<T>>;
      //} else {
      //return false;
      return needsSharedMem<typename T::dependentOps>;
      //}
    }
  }
  template <typename T> static constexpr bool needsSharedMemImpl<T> = needsSharedMemF<T>();
  template <> static constexpr bool needsSharedMem<NoKernelOps> = false;
  template <typename ...T> static constexpr bool needsSharedMem<KernelOps<T...>> = needsSharedMemImpl<T...>;
#else
  //template <typename ...T> static constexpr bool needsSharedMemImpl = (needsSharedMemImpl<T> || ...);
  template <typename T> static constexpr bool needsSharedMemImpl = (T::shared_mem_size(dim3{8,8,8}) > 0);
  template <typename... T> static constexpr bool needsSharedMemImpl<KernelOps<T...>> = (needsSharedMemImpl<T> || ...);
  template <typename T> static constexpr bool needsSharedMem = needsSharedMem<getKernelOps<T>>;
  template <typename... T> static constexpr bool needsSharedMem<KernelOps<T...>> = (needsSharedMemImpl<T> || ...);
  template <> static constexpr bool needsSharedMem<NoKernelOps> = false;
#endif

  // KernelOps
  template <typename ...T>
  struct KernelOps : KernelOps_Base<T...> {
  //struct KernelOpsTarget<T...> {
    //using KernelOpsT = op_Sequential<T...>;
    //using KernelOpsT = KernelOps<T...>;
    //using KernelOpsElemType = typename KernelOpsElemTypeS<T...>::type;
    //const sycl::nd_item<3> *ndi = nullptr;
    //char *smem;
    sycl::local_ptr<char> smem = nullptr;

    //KernelOps() = delete;
    inline KernelOps() {
      static_assert(!needsSharedMem<KernelOps<T...>>);
    }
    inline KernelOps(char *s) {  // for host
      static_assert(needsSharedMem<KernelOps<T...>>);
      smem = s;
    }
    //template <typename S>
    //inline KernelOps(S s) {
    //  static_assert(needsSharedMem<KernelOps<T...>>);
    //  smem = s.get();
    //}
    template <typename ...U>
    inline KernelOps(const KernelOps<U...> &ops) {
      checkKernelOps<T...>(ops);
      if constexpr (needsSharedMem<KernelOps<T...>>) {
	smem = ops.smem;
      }
    }

#if 0
    //inline void setNdItem(const sycl::nd_item<3> &i) { ndi = &i; }
    inline void setNdItem(const sycl::nd_item<3> &i) {}
    inline void setSharedMem(char *s) { smem = s; }
    template <typename ...U> inline void setKernelOps(const KernelOps<U...> &ops) {
      static_assert(std::is_same_v<KernelOps<T...>,KernelOps<U...>>);
      //ndi = ops.ndi;
      smem = ops.smem;
    }
#endif
#if 0
    KernelOpsElemType *getSharedMemPtr() {
      static_assert(!std::is_same_v<KernelOpsElemType,void>);
      return reinterpret_cast<KernelOpsElemType*>(smem);
    }
#endif
  };

  // blockSync
  template <typename ...T>
  inline void blockSync(const KernelOps<T...> &) {
    //static_assert(hasBlockSync<T...>);
    checkKernelOp<op_blockSync,T...>();
    //if (ops->ndi == nullptr) {
    //  errorQuda("KernelOps not set");
    //}
#ifdef __SYCL_DEVICE_ONLY__
    //sycl::group_barrier(ops->ndi->get_group());
    sycl::group_barrier(getGroup());
#endif
  }
  //template <typename ...T> inline void blockSync(KernelOps<T...> ops) { blockSync(&ops); }

  //template <typename ...T> static constexpr bool isOpConcurrent = false;
  //template <typename ...T> static constexpr bool isOpConcurrent<op_Concurrent<T...>> = true;

  //template <typename T, typename ...U> static constexpr int getOpIndex = 0;
  //template <typename T, typename ...U> static constexpr int getOpIndex<T,op_Concurrent<U...>> = getOpIndex<T,U...>;
  //template <typename T, typename U, typename ...V> static constexpr int getOpIndex<T, U, V...> =
  //  std::is_same_v<T,U> ? 0 : (1 + getOpIndex<T,V...>);

#if 0
  // getKernelOp
  template <typename U, typename ...T>
  inline U getKernelOp(const KernelOps<T...> &ops) {
    //static_assert(hasKernelOpType<U,T...>);
    checkKernelOp<U,T...>();
    //if (ops->ndi == nullptr || ops->smem == nullptr) {
    //	errorQuda("KernelOps not set");
    //}
    //KernelOpsType<U,n> s;
    KernelOps<U> s;
    //s.ndi = ops.ndi;
    //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
    //s.smem = ops.smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
    s.smem = ops.smem;
    return s;
  }
  template <typename U, typename ...T>
    inline U getKernelOp(const KernelOps<T...> *ops) { return getKernelOp<U>(*ops); }
  template <typename U> struct getKernelOpF {
    template <typename T> inline U operator()(const T &ops) { return getKernelOp<U>(ops); }
  };
#endif

#if 0
  // getDependentOps
  template <typename U, int n = 0, typename ...T>
  inline KernelOpDependencies<KernelOpsType<U,n>> getDependentOps(const KernelOps<T...> &ops) {
    static_assert(hasKernelOpType<U,T...>);
    //if (ops->ndi == nullptr || ops->smem == nullptr) {
    //errorQuda("KernelOps not set");
    //}
    //KernelOpDependencies<KernelOpsType<U,n>> s;
    //s.ndi = ops.ndi;
    //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
    //s.smem = ops.smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
    //return s;
    using R = KernelOpDependencies<KernelOpsType<U,n>>;
    if constexpr (needsSharedMem<R>) {
      auto m = ops.smem + KernelOps<U>::
      R s{};
      return s;
    } else {
      R s{};
      return s;
    }
  }
#endif

  // getSharedMemPtr
#if 0
  template <typename ...T>
  //KernelOpsElemType<T...> *getSharedMemPtr(KernelOps<T...> *ops) {
  sycl::local_ptr<KernelOpsElemType<T...>> getSharedMemPtr(KernelOps<T...> *ops) {
    static_assert(!std::is_same_v<KernelOpsElemType<T...>,void>);
    //return reinterpret_cast<KernelOpsElemType<T...>*>(ops->smem);
    //return reinterpret_cast<KernelOpsElemType<T...>*>(ops->smem.get());
    //sycl::local_ptr<KernelOpsElemType<T...>> smem = ops->smem.get();
    //return smem.get();
    //auto p = ops->smem.get();
    sycl::local_ptr<void> v(ops->smem);
    sycl::local_ptr<KernelOpsElemType<T...>> p(v);
    return p;
    //sycl::local_ptr<KernelOpsElemType<T...>> smem;
    //using LT = decltype(smem.get());
    //LT pt = reinterpret_cast<LT>(p);
    //sycl::local_ptr<KernelOpsElemType<T...>> smem2(pt);
    //return smem2;
    //return reinterpret_cast<KernelOpsElemType<T...>*>(0);
  }
  template <typename ...T>
  inline KernelOpsElemType<T...> *getSharedMemPtr(KernelOps<T...> ops) { return getSharedMemPtr(&ops); }
#endif

#if 0
  template <typename T, typename S, typename O = op_SharedMemory<T,S>>
  inline sycl::local_ptr<T> getSharedMemPtr(const only_SharedMemory<T,S> &ops) {
    //if (ops->ndi == nullptr || ops->smem == nullptr) {
    //errorQuda("KernelOps not set");
    //}
    sycl::local_ptr<void> v(ops.smem);
    sycl::local_ptr<T> p(v);
    return p;
  }
  //template <typename T, typename S>
  //inline sycl::local_ptr<T> getSharedMemPtr(only_SharedMemory<T,S> ops) { return getSharedMemPtr(&ops); }
  template <typename O, typename T, typename U, typename ...V>
  inline auto getSharedMemPtr(const KernelOps<T,U,V...> &ops) {
    KernelOps<O> op = getKernelOp<O>(ops);
    return getSharedMemPtr(op);
  }
#endif

#if 0
  template <typename T, typename O>
  inline auto getSharedMemory(O *ops)
  {
    auto s = getKernelOp<T>(ops);
    return getSharedMemPtr(s);
  }
#endif

  // needsFullBlock
#if 0
  template <typename T> static constexpr bool needsFullBlock = needsFullBlock<getKernelOps<T>>;
  template <typename ...T> static constexpr bool needsFullBlockImpl = (needsFullBlockImpl<T> || ...);
  template <> static constexpr bool needsFullBlockImpl<depNone> = false;
  template <> static constexpr bool needsFullBlockImpl<depFullBlock> = true;
  template <typename T, typename S> static constexpr bool needsFullBlockImpl<depSharedMem<T,S>> = false;
  template <typename ...T> static constexpr bool needsFullBlockImpl<op_Concurrent<T...>> = needsFullBlockImpl<T...>;
  template <typename ...T> static constexpr bool needsFullBlockImpl<op_Sequential<T...>> = needsFullBlockImpl<T...>;
  template <typename T> static constexpr bool needsFullBlockF() {
    if constexpr (std::is_base_of<op_Base,T>::value) {
      return needsFullBlockImpl<typename T::dependencies>;
    } else {
      //if constexpr (hasKernelOps<T>) {
      //return needsFullBlock<getKernelOps<T>>;
      //} else {
      //return false;
      return needsFullBlock<typename T::dependentOps>;
      //}
    }
  }
  template <typename T> static constexpr bool needsFullBlockImpl<T> = needsFullBlockF<T>();
  template <> static constexpr bool needsFullBlock<NoKernelOps> = false;
  template <typename ...T> static constexpr bool needsFullBlock<KernelOps<T...>> = needsFullBlockImpl<T...>;
#else
  template <typename T> static constexpr bool needsFullBlockImpl = (T)false;
  template <typename ...T> static constexpr bool needsFullBlockImpl<KernelOps<T...>> = (needsFullBlockImpl<T> || ...);
  template <> static constexpr bool needsFullBlockImpl<NoKernelOps> = false;
  template <typename T> static constexpr bool needsFullBlock = needsFullBlockImpl<getKernelOps<T>>;
#endif

  // base operation dependencies
  struct depNone {};
  template <> struct sharedMemSizeS<depNone> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3, Arg &...) { return 0; }
  };

  struct depFullBlock {};
  template <> struct sharedMemSizeS<depFullBlock> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3, Arg &...) { return 0; }
  };

  template <typename T, typename S>
  struct depSharedMem {};
  template <typename T, typename S> struct sharedMemSizeS<depSharedMem<T,S>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) { return S().template size<T>(block, arg...); }
  };

  // op implementations
  //struct op_blockSync : op_BaseT<void> {
  struct op_blockSync {
    //using dependencies = depFullBlock;
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };

  template <typename T>
  //struct op_warp_combine : op_BaseT<T> {
  struct op_warp_combine {
    //using dependencies = depNone;
    //using dependencies = depFullBlock;
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };
  template <typename T> static constexpr bool needsFullBlockImpl<op_warp_combine<T>> = false;

#if 0
  template <typename T, int N>
  struct op_thread_array : op_BaseT<T,N> {
    //using dependencies = depNone;
    using dependencies = op_SharedMemory<array<T,N>,opSizeBlock>;
  };

  template <typename T>
  struct op_BlockReduce : op_BaseT<T> {
    using concurrentOps = op_Concurrent<op_blockSync,op_SharedMemory<T,opSizeBlockDivWarp>>;
    using opBlockSync = getKernelOpF<concurrentOps,0>;
    using opSharedMem = getKernelOpF<concurrentOps,1>;
    //using specialOps = KernelOps<concurrentOps>;
    using dependencies = concurrentOps;
  };

  template <typename T, typename D>
  struct op_SharedMemoryCache : op_BaseT<T> {
    template <typename ...Arg> static constexpr dim3 dims(dim3 block, Arg &...arg) { return D::dims(block, arg...); }
    using dependencies = op_Sequential<op_blockSync,op_SharedMemory<T,opSizeDims<D>>>;
  };

  template <typename T, typename S>
  struct op_SharedMemory : op_BaseT<T> {
    using dependencies = depSharedMem<T,S>;
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3 block, Arg &...arg) { return S::template size<T>(block, arg...); }
  };
#endif

  // needsFullWarp?

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
}
