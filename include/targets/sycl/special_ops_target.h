#pragma once

namespace quda {

  // SpecialOps
  template <typename ...T>
  struct SpecialOps {
    using SpecialOpsElemType = typename SpecialOpsElemTypeS<T...>::type;
    const sycl::nd_item<3> *ndi;
    char *smem;
    void setNdItem(const sycl::nd_item<3> &i) { ndi = &i; }
    void setSharedMem(char *s) { smem = s;}
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
  SpecialOpsType<U,n> getSpecialOp(SpecialOps<T...> *ops) {
    static_assert(hasSpecialOpType<U,T...>);
    SpecialOpsType<U,n> s;
    s.ndi = ops->ndi;
    s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range().size());
    return s;
  }
  template <typename U, int n = 0, typename ...T>
    SpecialOpsType<U,n> getSpecialOp(SpecialOps<T...> ops) { return getSpecialOp<U,n>(&ops); }
  template <typename U, int n = 0> struct getSpecialOpF {
    template <typename T> SpecialOpsType<U,n> operator()(T ops) { return getSpecialOp<U,n>(ops); }
  };

  // getDeendentOp
  template <typename U, int n = 0, typename ...T>
  SpecialOpDependencies<SpecialOpsType<U,n>> getDependentOps(SpecialOps<T...> *ops) {
    static_assert(hasSpecialOpType<U,T...>);
    SpecialOpDependencies<SpecialOpsType<U,n>> s;
    s.ndi = ops->ndi;
    s.smem = ops->smem + sharedMemOffset<U,n>()(ops->ndi->get_local_range().size());
    return s;
  }

  // getSharedMemPtr
  template <typename ...T>
  SpecialOpsElemType<T...> *getSharedMemPtr(SpecialOps<T...> *ops) {
    static_assert(!std::is_same_v<SpecialOpsElemType<T...>,void>);
    return reinterpret_cast<SpecialOpsElemType<T...>*>(ops->smem);
  }
  template <typename ...T>
  inline SpecialOpsElemType<T...> *getSharedMemPtr(SpecialOps<T...> ops) { return getSharedMemPtr(&ops); }

  struct FullBlock {};
  template <> struct sharedMemSizeS<FullBlock> {
    static constexpr size_t size(size_t blocksize) { return 0; }
  };
  template <typename T, typename S>
  struct depSharedMem {};
  template <typename T, typename S> struct sharedMemSizeS<depSharedMem<T,S>> {
    static constexpr size_t size(size_t blocksize) { return S().template size<T>(blocksize); }
  };

  struct op_blockSync : op_Base {
    using dependencies = FullBlock;
  };

  template <typename T>
  struct op_warp_combine : op_Base {
    using dependencies = FullBlock;
  };

  template <typename T>
  struct op_thread_array : op_Base {
    using dependencies = FullBlock;
  };

  template <typename T>
  struct op_BlockReduce : op_Base {
    //struct S { size_t operator()(size_t blocksize) { return blocksize * sizeof(T); } };  // FIXME: divide by warp size
    using dependencies = op_Sequential<op_blockSync,op_SharedMemory<T>>;
  };

  template <typename T>
  struct op_SharedMemoryCache : op_Base {
    using dependencies = op_Sequential<op_blockSync,op_SharedMemory<T>>;
  };

  template <typename T, typename S>
  struct op_SharedMemory : op_Base {
    using dependencies = depSharedMem<T,S>;
  };

  // needsFullWarp?

  // needsFullBlock
  template <typename ...T> static constexpr bool needsFullBlock = (needsFullBlock<T> || ...);
  template <> static constexpr bool needsFullBlock<FullBlock> = true;
  template <typename ...T> static constexpr bool needsFullBlock<SpecialOps<T...>> = needsFullBlock<T...>;
  template <typename ...T> static constexpr bool needsFullBlock<op_Concurrent<T...>> = needsFullBlock<T...>;
  template <typename ...T> static constexpr bool needsFullBlock<op_Sequential<T...>> = needsFullBlock<T...>;
  template <typename T> static constexpr bool needsFullBlockF() {
    if constexpr (std::is_base_of<op_Base,T>::value) {
      return needsFullBlock<typename T::dependencies>;
    } else {
      return false;
    }
  }
  template <typename T> static constexpr bool needsFullBlock<T> = needsFullBlockF<T>();

  // needsSharedMem
  template <typename ...T> static constexpr bool needsSharedMem = false;
  template <typename ...T> static constexpr bool needsSharedMem<SpecialOps<T...>> = needsSharedMem<T...>;
  template <typename ...T> static constexpr bool needsSharedMem<op_Concurrent<T...>> = needsSharedMem<T...>;
  template <typename T> static constexpr bool needsSharedMem<T> = false;
  template <typename T, typename ...U> static constexpr bool needsSharedMem<T,U...> = needsSharedMem<T> || needsSharedMem<U...>;
  template <typename T> static constexpr bool needsSharedMem<op_BlockReduce<T>> = true;
  template <typename T> static constexpr bool needsSharedMem<op_SharedMemoryCache<T>> = true;
  template <typename T, typename S> static constexpr bool needsSharedMem<op_SharedMemory<T,S>> = true;

  // tests
  static_assert(needsFullBlock<only_SharedMemoryCache<float>> == true);
  static_assert(sharedMemSize<only_SharedMemoryCache<float>>(100)==100*sizeof(float));

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

  static_assert(sharedMemSize<opTest1>(0)==std::max((size_t)100,0*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(10)==std::max((size_t)100,10*sizeof(double)));
  static_assert(sharedMemSize<opTest1>(100)==std::max((size_t)100,100*sizeof(double)));


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
