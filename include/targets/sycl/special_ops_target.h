#pragma once
#include <special_ops.h>
#include <block_reduce_helper.h>

namespace quda {

  // SpecialOps
  template <typename ...T>
  struct SpecialOps : SpecialOps_Base<T...> {
    //using SpecialOpsT = op_Sequential<T...>;
    //using SpecialOpsT = SpecialOps<T...>;
    //using SpecialOpsElemType = typename SpecialOpsElemTypeS<T...>::type;
    const sycl::nd_item<3> *ndi = nullptr;
    //char *smem;
    sycl::local_ptr<char> smem = nullptr;
    inline void setNdItem(const sycl::nd_item<3> &i) { ndi = &i; }
    inline void setSharedMem(char *s) { smem = s; }
    template <typename ...U> inline void setSpecialOps(const SpecialOps<U...> &ops) {
      static_assert(std::is_same_v<SpecialOps<T...>,SpecialOps<U...>>);
      ndi = ops.ndi;
      smem = ops.smem;
    }
#if 0
    SpecialOpsElemType *getSharedMemPtr() {
      static_assert(!std::is_same_v<SpecialOpsElemType,void>);
      return reinterpret_cast<SpecialOpsElemType*>(smem);
    }
#endif
  };

  // blockSync
  template <typename ...T>
  inline void blockSync(SpecialOps<T...> *ops) {
    static_assert(hasBlockSync<T...>);
    //if (ops->ndi == nullptr) {
    //  errorQuda("SpecialOps not set");
    //}
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(ops->ndi->get_group());
#endif
  }
  template <typename ...T> inline void blockSync(SpecialOps<T...> ops) { blockSync(&ops); }

  template <typename ...T> static constexpr bool isOpConcurrent = false;
  template <typename ...T> static constexpr bool isOpConcurrent<op_Concurrent<T...>> = true;

  template <typename T, typename ...U> static constexpr int getOpIndex = 0;
  template <typename T, typename ...U> static constexpr int getOpIndex<T,op_Concurrent<U...>> = getOpIndex<T,U...>;
  template <typename T, typename U, typename ...V> static constexpr int getOpIndex<T, U, V...> =
    std::is_same_v<T,U> ? 0 : (1 + getOpIndex<T,V...>);

  // getSpecialOp
  template <typename U, int n = 0, typename ...T>
  inline SpecialOpsType<U,n> getSpecialOp(const SpecialOps<T...> &ops) {
    if constexpr (!isOpConcurrent<U> && sizeof...(T) == 1 && isOpConcurrent<T...>) {
      static constexpr int i = getOpIndex<U, T...>;
      return getSpecialOp<T...,i>(ops);
    } else {
      static_assert(hasSpecialOpType<U,T...>);
      //if (ops->ndi == nullptr || ops->smem == nullptr) {
      //	errorQuda("SpecialOps not set");
      //}
      SpecialOpsType<U,n> s;
      s.ndi = ops.ndi;
      //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
      s.smem = ops.smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
      return s;
    }
  }
  template <typename U, int n = 0, typename ...T>
    inline SpecialOpsType<U,n> getSpecialOp(const SpecialOps<T...> *ops) { return getSpecialOp<U,n>(*ops); }
  template <typename U, int n = 0> struct getSpecialOpF {
    template <typename T> inline SpecialOpsType<U,n> operator()(const T &ops) { return getSpecialOp<U,n>(ops); }
  };

  // getDependentOps
  template <typename U, int n = 0, typename ...T>
  inline SpecialOpDependencies<SpecialOpsType<U,n>> getDependentOps(const SpecialOps<T...> &ops) {
    static_assert(hasSpecialOpType<U,T...>);
    //if (ops->ndi == nullptr || ops->smem == nullptr) {
    //errorQuda("SpecialOps not set");
    //}
    SpecialOpDependencies<SpecialOpsType<U,n>> s;
    s.ndi = ops.ndi;
    //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
    s.smem = ops.smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
    return s;
  }

  // getSharedMemPtr
#if 0
  template <typename ...T>
  //SpecialOpsElemType<T...> *getSharedMemPtr(SpecialOps<T...> *ops) {
  sycl::local_ptr<SpecialOpsElemType<T...>> getSharedMemPtr(SpecialOps<T...> *ops) {
    static_assert(!std::is_same_v<SpecialOpsElemType<T...>,void>);
    //return reinterpret_cast<SpecialOpsElemType<T...>*>(ops->smem);
    //return reinterpret_cast<SpecialOpsElemType<T...>*>(ops->smem.get());
    //sycl::local_ptr<SpecialOpsElemType<T...>> smem = ops->smem.get();
    //return smem.get();
    //auto p = ops->smem.get();
    sycl::local_ptr<void> v(ops->smem);
    sycl::local_ptr<SpecialOpsElemType<T...>> p(v);
    return p;
    //sycl::local_ptr<SpecialOpsElemType<T...>> smem;
    //using LT = decltype(smem.get());
    //LT pt = reinterpret_cast<LT>(p);
    //sycl::local_ptr<SpecialOpsElemType<T...>> smem2(pt);
    //return smem2;
    //return reinterpret_cast<SpecialOpsElemType<T...>*>(0);
  }
  template <typename ...T>
  inline SpecialOpsElemType<T...> *getSharedMemPtr(SpecialOps<T...> ops) { return getSharedMemPtr(&ops); }
#endif

  template <typename T, typename S, typename O = op_SharedMemory<T,S>>
  inline sycl::local_ptr<T> getSharedMemPtr(const only_SharedMemory<T,S> &ops) {
    //if (ops->ndi == nullptr || ops->smem == nullptr) {
    //errorQuda("SpecialOps not set");
    //}
    sycl::local_ptr<void> v(ops.smem);
    sycl::local_ptr<T> p(v);
    return p;
  }
  //template <typename T, typename S>
  //inline sycl::local_ptr<T> getSharedMemPtr(only_SharedMemory<T,S> ops) { return getSharedMemPtr(&ops); }
  template <typename O, typename T, typename U, typename ...V>
  inline auto getSharedMemPtr(const SpecialOps<T,U,V...> &ops) {
    SpecialOps<O> op = getSpecialOp<O>(ops);
    return getSharedMemPtr(op);
  }

  template <typename T, typename O>
  inline auto SharedMemory(O *ops)
  {
    auto s = getSpecialOp<T>(ops);
    return getSharedMemPtr(s);
  }

  // base operation dependencies
  struct depNone {};
  template <> struct sharedMemSizeS<depNone> {
    template <typename ...Arg>
    static constexpr size_t size(dim3 block, Arg &...arg) { return 0; }
  };

  struct depFullBlock {};
  template <> struct sharedMemSizeS<depFullBlock> {
    template <typename ...Arg>
    static constexpr size_t size(dim3 block, Arg &...arg) { return 0; }
  };

  template <typename T, typename S>
  struct depSharedMem {};
  template <typename T, typename S> struct sharedMemSizeS<depSharedMem<T,S>> {
    template <typename ...Arg>
    static constexpr size_t size(dim3 block, Arg &...arg) { return S().template size<T>(block, arg...); }
  };

  // op implementations
  struct op_blockSync : op_BaseT<void> {
    using dependencies = depFullBlock;
  };

  template <typename T>
  struct op_warp_combine : op_BaseT<T> {
    using dependencies = depNone;
    //using dependencies = depFullBlock;
  };

  template <typename T, int N>
  struct op_thread_array : op_BaseT<T,N> {
    //using dependencies = depNone;
    using dependencies = op_SharedMemory<array<T,N>,opSizeBlock>;
  };

  template <typename T>
  struct op_BlockReduce : op_BaseT<T> {
    using concurrentOps = op_Concurrent<op_blockSync,op_SharedMemory<T,opSizeBlockDivWarp>>;
    using opBlockSync = getSpecialOpF<concurrentOps,0>;
    using opSharedMem = getSpecialOpF<concurrentOps,1>;
    //using specialOps = SpecialOps<concurrentOps>;
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
  };

  // needsFullWarp?

  // needsFullBlock
  template <typename T> static constexpr bool needsFullBlock = needsFullBlock<getSpecialOps<T>>;
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
      //if constexpr (hasSpecialOps<T>) {
      //return needsFullBlock<getSpecialOps<T>>;
      //} else {
      //return false;
      return needsFullBlock<typename T::dependentOps>;
      //}
    }
  }
  template <typename T> static constexpr bool needsFullBlockImpl<T> = needsFullBlockF<T>();
  template <> static constexpr bool needsFullBlock<NoSpecialOps> = false;
  template <typename ...T> static constexpr bool needsFullBlock<SpecialOps<T...>> = needsFullBlockImpl<T...>;

  // needsSharedMem
  template <typename T> static constexpr bool needsSharedMem = needsSharedMem<getSpecialOps<T>>;
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
      //if constexpr (hasSpecialOps<T>) {
      //return needsSharedMem<getSpecialOps<T>>;
      //} else {
      //return false;
      return needsSharedMem<typename T::dependentOps>;
      //}
    }
  }
  template <typename T> static constexpr bool needsSharedMemImpl<T> = needsSharedMemF<T>();
  template <> static constexpr bool needsSharedMem<NoSpecialOps> = false;
  template <typename ...T> static constexpr bool needsSharedMem<SpecialOps<T...>> = needsSharedMemImpl<T...>;

  // tests
  static const int opTestArg = 10;
  static_assert(needsFullBlock<only_SharedMemoryCache<float>> == true);
  static_assert(sharedMemSize<only_SharedMemoryCache<float>>(dim3(2,3,4))==24*sizeof(float));
  static_assert(sharedMemSize<only_SharedMemoryCache<float>>(dim3(2,3,4),opTestArg)==24*sizeof(float));

  template <typename T, typename U> static constexpr bool opTestHasSpecialOpType = hasSpecialOpType<T,U>;
  template <typename T, int n = 0> static constexpr bool opTestAllHasSpecialOpType = false;
  template <typename ...T> static constexpr bool opTestAllHasSpecialOpType<SpecialOps<T...>,sizeof...(T)> = true;
  template <typename ...T, int n> static constexpr bool opTestAllHasSpecialOpType<SpecialOps<T...>,n> =
    opTestHasSpecialOpType<std::tuple_element_t<n,std::tuple<T...>>,SpecialOps<T...>> &&
    opTestAllHasSpecialOpType<SpecialOps<T...>,n+1>;

  using opTestC1 = op_Concurrent<op_blockSync,op_thread_array<bool,4>>;
  using opTest1 = SpecialOps<op_blockSync,op_warp_combine<int>,op_thread_array<float,4>,op_SharedMemoryCache<float>,
    op_SharedMemory<double>,op_SharedMemStatic<char,100>,opTestC1>;
  static_assert(opTestAllHasSpecialOpType<opTest1>);
  static_assert(hasSpecialOpType<opTestC1,opTest1>);
  static_assert(!hasSpecialOpType<op_thread_array<bool,4>,opTest1>);

  static_assert(sharedMemSize<opTest1>(dim3(0,0,0))==std::max((size_t)100,0*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(dim3(1,2,5))==std::max({(size_t)100,10*sizeof(double),40*sizeof(float)}));
  static_assert(sharedMemSize<opTest1>(dim3(2,5,10))==std::max({(size_t)100,100*sizeof(double),400*sizeof(float)}));


#if 0
  using opTest2 = SpecialOps<op_blockSync,op_warp_combine<int>,op_thread_array<float,4>,
			     op_SharedMemoryCache<double>,op_SharedMemory<float>,op_SharedMemStatic<char,100>>;
  static_assert(opTestAllHasSpecialOpType<opTest1>);
   template <typename T, typename U> static constexpr bool opTestSpecialOpsType =
    //std::is_same_v<SpecialOpsType<T,U>,SpecialOps<T>;
    hasSpecialOpType<T,U>;
  template <typename T, int n = 0> static constexpr bool opTestAllSpecialOpsType = false;
  template <typename ...T> static constexpr bool opTestAllSpecialOpsType<SpecialOps<T...>,sizeof...(T)> = true;
  template <typename ...T, int n> static constexpr bool opTestAllSpecialOpsType<SpecialOps<T...>,n> =
    opTestSpecialOpsType<std::tuple_element_t<n,std::tuple<T...>>,SpecialOps<T...>> &&
    opTestAllSpecialOpsType<SpecialOps<T...>,n+1>;
#endif
}
