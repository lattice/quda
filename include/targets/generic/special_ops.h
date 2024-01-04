#pragma once
#include <target_device.h>
#include <kernel_ops.h>

namespace quda {

  // dimensions functors for SharedMemoryCache
  struct opDimsBlock {
    template <typename ...Arg> static constexpr dim3 dims(dim3 b, const Arg &...) { return b; }
  };
  template <int bx, int by, int bz> struct opDimsStatic {
    template <typename ...Arg> static constexpr dim3 dims(dim3, const Arg &...) { return dim3(bx,by,bz); }
  };

  // size functors for determining shared memory size
  struct opSizeBlock {
    template <typename T, typename ...Arg> static constexpr unsigned int size(dim3 b, const Arg &...) {
      return b.x * b.y * b.z * sizeof(T);
    }
  };
  struct opSizeBlockDivWarp {
    template <typename T, typename ...Arg> static constexpr unsigned int size(dim3 b, const Arg &...) {
      return ((b.x * b.y * b.z + device::warp_size() - 1)/device::warp_size()) * sizeof(T);
    }
  };
  template <unsigned int S> struct opSizeStatic {
    template <typename T, typename ...Arg> static constexpr unsigned int size(dim3, const Arg &...) {
      return S * sizeof(T);
    }
  };
  template <typename D> struct opSizeDims {
    template <typename T, typename ...Arg> static constexpr unsigned int size(dim3 b, const Arg &...arg) {
      return opSizeBlock::size<T>(D::dims(b, arg...));
    }
  };

  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, Arg &...arg);

  // alternative to SpecialOps
  struct NoSpecialOps {
    using SpecialOpsT = NoSpecialOps;
    using KernelOpsT = NoSpecialOps;
  };
  // SpecialOps forward declaration and base type
  template <typename ...T> struct SpecialOps;
  template <typename ...T> using KernelOps = SpecialOps<T...>;
  template <typename ...T> struct SpecialOps_Base {
    using SpecialOpsT = SpecialOps<T...>;
    using KernelOpsT = SpecialOps<T...>;
    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3 block, Arg &...arg) {
      return sharedMemSize<KernelOpsT>(block, arg...);
    }
  };

  // getSpecialOps
  template <typename T, typename U = void> struct getSpecialOpsS { using type = NoSpecialOps; };
  template <typename T> struct getSpecialOpsS<T,std::conditional_t<true,void,typename T::SpecialOpsT>> {
    using type = typename T::SpecialOpsT;
  };
  template <typename T> using getSpecialOps = typename getSpecialOpsS<T>::type;

  // hasSpecialOp: checks if first type matches any of the op
  // <op, SpecialOps<ops...>>
  template <typename T, typename U> static constexpr bool hasSpecialOp = false;
  template <typename T, typename ...U>
  static constexpr bool hasSpecialOp<T,SpecialOps<U...>> = ( std::is_same_v<T,U> || ... );

  //template <typename T, typename Ops> void checkSpecialOps() { static_assert(hasSpecialOp<T,Ops>); }
  //template <typename T, typename Ops> void checkSpecialOps(const Ops &) {
  //static_assert(hasSpecialOp<T,typename Ops::SpecialOpsT>);
  //}
  template <typename ...T, typename Ops> void checkSpecialOps(const Ops &) {
    static_assert((hasSpecialOp<T,typename Ops::SpecialOpsT> || ...));
  }

  // forward declarations of op types
  struct op_blockSync;
  template <typename T> struct op_warp_combine;

  // only types for convenience
  using only_blockSync = SpecialOps<op_blockSync>;
  template <typename T> using only_warp_combine = SpecialOps<op_warp_combine<T>>;

  // explicitSpecialOps
  template <typename T, typename U = void> struct explicitSpecialOpsS : std::false_type {};
  template <typename T>
  struct explicitSpecialOpsS<T,std::conditional_t<true,void,typename T::SpecialOpsT>> : std::true_type {};
  template <typename T> inline constexpr bool explicitSpecialOps = explicitSpecialOpsS<T>::value;

  // hasSpecialOps
#if 1
  template <typename T> inline constexpr bool hasSpecialOps = !std::is_same_v<getSpecialOps<T>,NoSpecialOps>;
#else
  template <typename T> struct hasSpecialOpsImpl { static constexpr bool value = false; };
  template <typename ...U> struct hasSpecialOpsImpl<SpecialOps<U...>> { static constexpr bool value = true; };
  template <typename T> inline constexpr bool hasSpecialOps = hasSpecialOpsImpl<T>::value;
#endif

  // checkSpecialOp
  template <typename T, typename... U> static constexpr void checkSpecialOp() {
    static_assert((std::is_same_v<T,U> || ...) == true);
  }

  // combineOps
  template <typename ...T> struct combineOpsS {};
  template <typename ...T> struct combineOpsS<NoSpecialOps,SpecialOps<T...>> {
    using type = SpecialOps<T...>; };
  template <typename ...T> struct combineOpsS<SpecialOps<T...>,NoSpecialOps> {
    using type = SpecialOps<T...>; };
  template <typename ...T, typename ...U> struct combineOpsS<SpecialOps<T...>,SpecialOps<U...>> {
    using type = SpecialOps<T..., U...>; };
  template <typename T, typename U> using combineOps = typename combineOpsS<T, U>::type;


  // OLD

#if 0
  template <typename ...T> struct op_Concurrent {};  // set of op types used concurrently (needs separate resources)
  template <typename ...T> struct op_Sequential {};  // set of op types used sequentially (can share resources)
  struct op_Base {};  // base type for other op types
  template <typename T, int N = 0> struct op_BaseT : op_Base {
    //using op_ElementT = T;
    using ElemT = T;
    static constexpr int n = N;
  };

  // unwrapSpecialOps
  template <typename T> struct unwrapSpecialOpsS { using type = T; };
  template <typename ...T> struct unwrapSpecialOpsS<SpecialOps<T...>> {
    using type = std::conditional_t<sizeof...(T)==1,std::tuple_element_t<0,std::tuple<T...>>,op_Sequential<T...>>;
  };
  template <typename T> using unwrapSpecialOps = typename unwrapSpecialOpsS<T>::type;

  // hasSpecialOpType: checks if first type matches any of the other sequential operations
  template <typename ...T> static constexpr bool hasSpecialOpType = false;
  template <typename T, typename U> static constexpr bool hasSpecialOpType2 = std::is_same_v<T,U>;
  template <typename T, typename ...U> static constexpr bool hasSpecialOpType2<T,op_Sequential<U...>> = hasSpecialOpType<T,U...>;
  template <typename T, typename U, typename ...V> static constexpr bool hasSpecialOpType<T,U,V...> =
    hasSpecialOpType2<unwrapSpecialOps<T>,unwrapSpecialOps<U>> || hasSpecialOpType<T,V...>;

  // hasBlockSync
  template <typename ...T> static constexpr bool hasBlockSync = hasSpecialOpType<op_blockSync,T...>;
  template <typename ...T> static constexpr bool hasBlockSync<op_Concurrent<T...>> = hasSpecialOpType<op_blockSync,T...>;

  // hasWarpCombine
  template <typename ...T> static constexpr bool hasWarpCombine = (hasWarpCombine<T> || ...);
  template <typename ...T> static constexpr bool hasWarpCombine<SpecialOps<T...>> = hasWarpCombine<T...>;
  template <typename ...T> static constexpr bool hasWarpCombine<op_Sequential<T...>> = hasWarpCombine<T...>;
  template <typename ...T> static constexpr bool hasWarpCombine<op_Concurrent<T...>> = hasWarpCombine<T...>;
  template <typename T> static constexpr bool hasWarpCombine<op_warp_combine<T>> = true;
  template <typename T> static constexpr bool hasWarpCombine<T> = false;

  // isOpSharedMemoryCache
  template <typename T> static constexpr bool isOpSharedMemoryCache = false;
  template <typename T, typename D> static constexpr bool isOpSharedMemoryCache<op_SharedMemoryCache<T,D>> = true;

  // isOpThreadArray
  template <typename T> static constexpr bool isOpThreadArray = false;
  template <typename T, int n> static constexpr bool isOpThreadArray<op_thread_array<T,n>> = true;

  // isOpBlockReduce
  template <typename T> static constexpr bool isOpBlockReduce = false;
  template <typename T> static constexpr bool isOpBlockReduce<op_BlockReduce<T>> = true;

  // SpecialOpsType: returns SpecialOps type from a Concurrent list
  template <typename T, int n> struct SpecialOpsTypeS { using type = std::enable_if_t<n==0,T>; };
  template <typename ...T, int n> struct SpecialOpsTypeS<op_Concurrent<T...>,n> {
    using type = std::tuple_element_t<n,std::tuple<T...>>;
  };
  template <typename T, int n> using SpecialOpsType = SpecialOps<typename SpecialOpsTypeS<unwrapSpecialOps<T>,n>::type>;

  // SpecialOpsElemType: element type from corresponding op types
  //template <typename ...T> struct SpecialOpsElemTypeS { using type = void; };
  //template <typename T> struct SpecialOpsElemTypeS<SpecialOps<T>> { using type = typename SpecialOpsElemTypeS<T>::type; };
  //template <typename T> struct SpecialOpsElemTypeS<op_warp_combine<T>> { using type = T; };
  //template <typename T, int N> struct SpecialOpsElemTypeS<op_thread_array<T,N>> { using type = T; };
  //template <typename T> struct SpecialOpsElemTypeS<op_BlockReduce<T>> { using type = T; };
  //template <typename T, typename D> struct SpecialOpsElemTypeS<op_SharedMemoryCache<T,D>> { using type = T; };
  //template <typename T, typename S> struct SpecialOpsElemTypeS<op_SharedMemory<T,S>> { using type = T; };
  //template <typename ...T> using SpecialOpsElemType = typename SpecialOpsElemTypeS<T...>::type;

  // SpecialOpDependencies: returns dependencies if type has them
  //template <typename T, typename Enabled = void> struct SpecialOpDependS { using deps = NoSpecialOps; };
  template <typename T, typename Enabled = void> struct SpecialOpDependS { using deps = typename T::dependentOps; };
  template <typename T> using SpecialOpDependencies = typename SpecialOpDependS<T>::deps;
  //template <typename T> struct SpecialOpDependS<SpecialOps<T>> { using deps = SpecialOps<SpecialOpDependencies<T>>; };
  template <typename T> struct SpecialOpDependS<SpecialOps<T>> { using deps = SpecialOpDependencies<T>; };
  template <typename T> struct SpecialOpDependS<T,std::enable_if_t<std::is_base_of_v<op_Base,T>,void>> {
  //template <typename T> struct SpecialOpDependS<T,std::enable_if_t<is_instance<T,op_Base>,void>> {
    using deps = SpecialOps<typename T::dependencies>;
  };
#endif

#if 0
  // SpecialOpDependencies: returns SpecialOps<all dependencies>, all Concurrent and Sequential lists are flattened
  template <typename ...T> struct SpecialOpsDependS { using deps = NoSpecialOps; };
  template <typename ...T> using SpecialOpDependencies<...T> = SpecialOpDependS<...T>::deps;
  template <typename ...T> struct SpecialOpDependS<SpecialOps<...T>> { using deps = SpecialOpDependencies<SpecialOpDependencies<...T>>; };
  template <typename ...T> struct SpecialOpDependS<std::tuple<...T>> { using deps = SpecialOps<...T>; };
  template <typename ...T, typename U, typename ...V> struct SpecialOpDependS<std::tuple<...T>,U,...V> {
    using deps = SpecialOpDependencies<...T,SpecialOpDependencies<>; }

  template <typename T, typename ...U> struct SpecialOpDependS< { using deps = NoSpecialOps; }
  template <typename T> using SpecialOpDependencies<T> =
    std::conditional_t<std::is_base_of_v<op_Base,T>,SpecialOps<T::dependencies>,NoSpecialOps>;
  //std::conditional_t<is_instance<T,op_Base>,SpecialOps<T::dependencies>,NoSpecialOps>;
  template <typename ...T> using SpecialOpDependencies = NoSpecialOps;
#endif

  // sharedMemSize
#if 0
  template <typename ...T> struct sharedMemSizeS {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      return std::max({sharedMemSizeS<T>::size(block, arg...)...});
    }
  };
  template <typename ...T, typename ...Arg> static constexpr unsigned int sharedMemSize(dim3 block, Arg &...arg) {
    return sharedMemSizeS<T...>::size(block, arg...);
  }
  template <typename ...T> struct sharedMemSizeS<SpecialOps<T...>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) { return sharedMemSize<T...>(block, arg...); }
  };
  template <typename ...T> struct sharedMemSizeS<op_Sequential<T...>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) { return sharedMemSize<T...>(block, arg...); }
  };
  template <typename ...T> struct sharedMemSizeS<op_Concurrent<T...>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) { return (sharedMemSize<T>(block, arg...) + ...); }
  };
  template <typename T> struct sharedMemSizeS<T> { // T should be of op_Base
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      return sharedMemSize<typename T::dependencies>(block, arg...);
    }
  };
#else
  template <typename T> struct sharedMemSizeS {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      //return 0;
      return T::shared_mem_size(block, arg...);
    }
  };
  template <> struct sharedMemSizeS<NoSpecialOps> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3, Arg &...) {
      return 0;
    }
  };
  template <typename ...T> struct sharedMemSizeS<SpecialOps<T...>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      return std::max({sharedMemSizeS<T>::size(block, arg...)...});
    }
  };
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, Arg &...arg) {
    return sharedMemSizeS<T>::size(block, arg...);
  }
#endif

#if 0
  // sharedMemOffset
  template <typename T, int n> struct sharedMemOffset {
    template <typename ...Arg>
    inline int operator()(dim3, Arg &...) { return 0; }
  };
  template <typename T, int n> struct sharedMemOffset<SpecialOps<T>,n> {
    template <typename ...Arg>
    inline int operator()(dim3 block, Arg &...arg) { return sharedMemOffset<T,n>()(block, arg...); }
  };
  template <typename ...T> struct sharedMemOffset<op_Concurrent<T...>,0> {
    template <typename ...Arg>
    inline int operator()(dim3, Arg &...) { return 0; }
  };
  template <typename ...T, int n> struct sharedMemOffset<op_Concurrent<T...>,n> {
    template <typename ...Arg>
    inline int operator()(dim3 block, Arg &...arg) {
      return sharedMemOffset<op_Concurrent<T...>,n-1>()(block, arg...)
	+ sharedMemSize<std::tuple_element_t<n-1,std::tuple<T...>>>(block, arg...);
    }
  };
#endif

}
