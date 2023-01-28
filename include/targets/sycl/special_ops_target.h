#pragma once
#include <special_ops.h>
#include <block_reduce_helper.h>

namespace quda {

  // SpecialOps
  template <typename ...T>
  struct SpecialOps {
    //using SpecialOpsT = op_Sequential<T...>;
    using SpecialOpsT = SpecialOps<T...>;
    using SpecialOpsElemType = typename SpecialOpsElemTypeS<T...>::type;
    const sycl::nd_item<3> *ndi;
    //char *smem;
    sycl::local_ptr<char> smem;
    void setNdItem(const sycl::nd_item<3> &i) { ndi = &i; }
    void setSharedMem(char *s) { smem = s; }
#if 0
    SpecialOpsElemType *getSharedMemPtr() {
      static_assert(!std::is_same_v<SpecialOpsElemType,void>);
      return reinterpret_cast<SpecialOpsElemType*>(smem);
    }
#endif
  };

  // blockSync
  template <typename ...T>
  void blockSync(SpecialOps<T...> *ops) {
    static_assert(hasBlockSync<T...>);
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(ops->ndi->get_group());
#endif
  }
  template <typename ...T> inline void blockSync(SpecialOps<T...> ops) { blockSync(&ops); }

  // getSpecialOp
  template <typename U, int n = 0, typename ...T>
  SpecialOpsType<U,n> getSpecialOp(const SpecialOps<T...> *ops) {
    static_assert(hasSpecialOpType<U,T...>);
    SpecialOpsType<U,n> s;
    s.ndi = ops->ndi;
    //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
    s.smem = ops->smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
    return s;
  }
  template <typename U, int n = 0, typename ...T>
    SpecialOpsType<U,n> getSpecialOp(SpecialOps<T...> ops) { return getSpecialOp<U,n>(&ops); }
  template <typename U, int n = 0> struct getSpecialOpF {
    template <typename T> SpecialOpsType<U,n> operator()(T ops) { return getSpecialOp<U,n>(ops); }
  };

  // getDependentOps
  template <typename U, int n = 0, typename ...T>
  SpecialOpDependencies<SpecialOpsType<U,n>> getDependentOps(SpecialOps<T...> *ops) {
    static_assert(hasSpecialOpType<U,T...>);
    SpecialOpDependencies<SpecialOpsType<U,n>> s;
    s.ndi = ops->ndi;
    //s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range());  // FIXME: need to pass arg
    s.smem = ops->smem + sharedMemOffset<U,n>()(getBlockDim());  // FIXME: need to pass arg
    return s;
  }

#if 0
  // getSharedMemPtr
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

  template <typename T, typename S>
  sycl::local_ptr<T> getSharedMemPtr(only_SharedMemory<T,S> *ops) {
    sycl::local_ptr<void> v(ops->smem);
    sycl::local_ptr<T> p(v);
    return p;
  }
  template <typename T, typename S>
  sycl::local_ptr<T> getSharedMemPtr(only_SharedMemory<T,S> ops) { return getSharedMemPtr(&ops); }

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
  struct op_blockSync : op_Base {
    using dependencies = depFullBlock;
  };

  template <typename T>
  struct op_warp_combine : op_Base {
    using dependencies = depFullBlock;
  };

  template <typename T>
  struct op_thread_array : op_Base {
    using dependencies = depNone;
  };

  template <typename T>
  struct op_BlockReduce : op_Base {
    using concurrentOps = op_Concurrent<op_blockSync,op_SharedMemory<T,opSizeBlockDivWarp>>;
    using opBlockSync = getSpecialOpF<concurrentOps,0>;
    using opSharedMem = getSpecialOpF<concurrentOps,1>;
    //using specialOps = SpecialOps<concurrentOps>;
    using dependencies = concurrentOps;
  };

  template <typename T, typename D>
  struct op_SharedMemoryCache : op_Base {
    using ElemT = T;
    template <typename ...Arg> static constexpr dim3 dims(dim3 block, Arg &...arg) { return D::dims(block, arg...); }
    using dependencies = op_Sequential<op_blockSync,op_SharedMemory<T,opSizeDims<D>>>;
  };

  template <typename T, typename S>
  struct op_SharedMemory : op_Base {
    using dependencies = depSharedMem<T,S>;
  };

  // needsFullWarp?

  // needsFullBlock
  template <typename ...T> static constexpr bool needsFullBlock = (needsFullBlock<T> || ...);
  template <> static constexpr bool needsFullBlock<depFullBlock> = true;
  template <typename ...T> static constexpr bool needsFullBlock<SpecialOps<T...>> = needsFullBlock<T...>;
  template <typename ...T> static constexpr bool needsFullBlock<op_Concurrent<T...>> = needsFullBlock<T...>;
  template <typename ...T> static constexpr bool needsFullBlock<op_Sequential<T...>> = needsFullBlock<T...>;
  template <typename T> static constexpr bool needsFullBlockF() {
    if constexpr (std::is_base_of<op_Base,T>::value) {
      return needsFullBlock<typename T::dependencies>;
    } else {
      if constexpr (hasSpecialOps<T>) {
	return needsFullBlock<getSpecialOps<T>>;
      } else {
	return false;
      }
    }
  }
  template <typename T> static constexpr bool needsFullBlock<T> = needsFullBlockF<T>();

  // needsSharedMem
  template <typename ...T> static constexpr bool needsSharedMem = (needsSharedMem<T> || ...);
  template <typename T, typename S> static constexpr bool needsSharedMem<depSharedMem<T,S>> = true;
  template <typename ...T> static constexpr bool needsSharedMem<SpecialOps<T...>> = needsSharedMem<T...>;
  template <typename ...T> static constexpr bool needsSharedMem<op_Concurrent<T...>> = needsSharedMem<T...>;
  template <typename ...T> static constexpr bool needsSharedMem<op_Sequential<T...>> = needsSharedMem<T...>;
  template <typename T> static constexpr bool needsSharedMemF() {
    if constexpr (std::is_base_of<op_Base,T>::value) {
      return needsSharedMem<typename T::dependencies>;
    } else {
      if constexpr (hasSpecialOps<T>) {
	return needsSharedMem<getSpecialOps<T>>;
      } else {
	return false;
      }
    }
  }
  template <typename T> static constexpr bool needsSharedMem<T> = needsSharedMemF<T>();

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

  using opTestC1 = op_Concurrent<op_blockSync,op_thread_array<bool>>;
  using opTest1 = SpecialOps<op_blockSync,op_warp_combine<int>,op_thread_array<float>,op_SharedMemoryCache<float>,
    op_SharedMemory<double>,op_SharedMemStatic<char,100>,opTestC1>;
  static_assert(opTestAllHasSpecialOpType<opTest1>);
  static_assert(hasSpecialOpType<opTestC1,opTest1>);
  static_assert(!hasSpecialOpType<op_thread_array<bool>,opTest1>);

  static_assert(sharedMemSize<opTest1>(dim3(0,0,0))==std::max((size_t)100,0*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(dim3(1,2,5))==std::max((size_t)100,10*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(dim3(2,5,10))==std::max((size_t)100,100*sizeof(double)));


#if 0
  using opTest2 = SpecialOps<op_blockSync,op_warp_combine<int>,op_thread_array<float>,
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
