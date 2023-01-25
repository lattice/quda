#pragma once
#include <target_device.h>

namespace quda {

  template <typename, template <typename...> typename>
  struct is_instance_impl : std::false_type {};
  template <template <typename...> typename T, typename ...U>
  struct is_instance_impl<T<U...>, T> : std::true_type {};
  template <typename T, template <typename ...> typename U>
  static constexpr bool is_instance = is_instance_impl<T, U>();

  struct SharedMemDynamic {
    template <typename T> static constexpr size_t size(size_t blocksize) { return blocksize * sizeof(T); }
  };
  struct SharedMemPerWarp {
    template <typename T> static constexpr size_t size(size_t blocksize) {
      return ((blocksize + device::warp_size() - 1)/device::warp_size()) * sizeof(T);
    }
  };
  template <size_t S> struct SharedMemStatic {
    template <typename T> static constexpr size_t size(size_t blocksize) { return S * sizeof(T); }
  };

  struct NoSpecialOps {};
  struct op_Base {};
  template <typename ...T> struct op_Concurrent {};
  template <typename ...T> struct op_Sequential {};

  // forward declarations
  template <typename ...T> struct SpecialOps;
  struct op_blockSync;
  template <typename T> struct op_warp_combine;
  template <typename T> struct op_thread_array;
  template <typename T> struct op_BlockReduce;
  template <typename T> struct op_SharedMemoryCache;
  template <typename T, typename S = SharedMemDynamic> struct op_SharedMemory;
  template <typename T, int S> using op_SharedMemStatic = op_SharedMemory<T,SharedMemStatic<S>>;

  // only types for convenience
  using only_blockSync = SpecialOps<op_blockSync>;
  template <typename T> using only_warp_combine = SpecialOps<op_warp_combine<T>>;
  template <typename T> using only_thread_array = SpecialOps<op_thread_array<T>>;
  template <typename T> using only_BlockReduce = SpecialOps<op_BlockReduce<T>>;
  template <typename T> using only_SharedMemoryCache = SpecialOps<op_SharedMemoryCache<T>>;
  template <typename T, typename S = SharedMemDynamic> using only_SharedMemory = SpecialOps<op_SharedMemory<T,S>>;
  template <typename T, size_t S> using only_SharedMemStatic = only_SharedMemory<T,SharedMemStatic<S>>;

  // getSpecialOps
  template <typename T, typename U = void> struct getSpecialOpsS { using type = NoSpecialOps; };
  template <typename T> struct getSpecialOpsS<T,std::conditional_t<true,void,typename T::SpecialOpsT>> {
    using type = typename T::SpecialOpsT;
  };
  template <typename T> using getSpecialOps = typename getSpecialOpsS<T>::type;

  // hasSpecialOps
  //template <typename T, typename U = void> static constexpr bool hasSpecialOps2 = false;
  //template <typename T> static constexpr bool hasSpecialOps2<T,std::conditional_t<true,void,typename T::SpecialOpsT>> = true;
  //template <typename ...T> static constexpr bool hasSpecialOps = (hasSpecialOps2<T> || ...);
  template <typename T> static constexpr bool hasSpecialOps = !std::is_same_v<getSpecialOps<T>,NoSpecialOps>;

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

  // hasWarpCombine
  template <typename ...T> static constexpr bool hasWarpCombine = (hasWarpCombine<T> || ...);
  template <typename ...T> static constexpr bool hasWarpCombine<SpecialOps<T...>> = hasWarpCombine<T...>;
  template <typename ...T> static constexpr bool hasWarpCombine<op_Sequential<T...>> = hasWarpCombine<T...>;
  template <typename ...T> static constexpr bool hasWarpCombine<op_Concurrent<T...>> = hasWarpCombine<T...>;
  template <typename T> static constexpr bool hasWarpCombine<op_warp_combine<T>> = true;
  template <typename T> static constexpr bool hasWarpCombine<T> = false;

  // SpecialOpsType: returns SpecialOps type from a Concurrent list
  template <typename T, int n> struct SpecialOpsTypeS { using type = std::enable_if_t<n==0,T>; };
  template <typename ...T, int n> struct SpecialOpsTypeS<op_Concurrent<T...>,n> {
    using type = std::tuple_element_t<n,std::tuple<T...>>;
  };
  template <typename T, int n> using SpecialOpsType = SpecialOps<typename SpecialOpsTypeS<unwrapSpecialOps<T>,n>::type>;

  // SpecialOpsElemType: element type from corresponding op types
  template <typename ...T> struct SpecialOpsElemTypeS { using type = void; };
  template <typename T> struct SpecialOpsElemTypeS<SpecialOps<T>> { using type = typename SpecialOpsElemTypeS<T>::type; };
  template <typename T> struct SpecialOpsElemTypeS<op_warp_combine<T>> { using type = T; };
  template <typename T> struct SpecialOpsElemTypeS<op_thread_array<T>> { using type = T; };
  template <typename T> struct SpecialOpsElemTypeS<op_BlockReduce<T>> { using type = T; };
  template <typename T> struct SpecialOpsElemTypeS<op_SharedMemoryCache<T>> { using type = T; };
  template <typename T, typename S> struct SpecialOpsElemTypeS<op_SharedMemory<T,S>> { using type = T; };
  template <typename ...T> using SpecialOpsElemType = typename SpecialOpsElemTypeS<T...>::type;

  // SpecialOpDependencies: returns dependencies if type has them
  template <typename T, typename Enabled = void> struct SpecialOpDependS { using deps = NoSpecialOps; };
  template <typename T> using SpecialOpDependencies = typename SpecialOpDependS<T>::deps;
  template <typename T> struct SpecialOpDependS<SpecialOps<T>> { using deps = SpecialOps<SpecialOpDependencies<T>>; };
  template <typename T> struct SpecialOpDependS<T,std::enable_if_t<std::is_base_of<op_Base,T>::value,void>> {
    using deps = typename T::dependencies;
  };

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
    std::conditional_t<std::is_base_of<op_Base,T>,SpecialOps<T::dependencies>,NoSpecialOps>;
  template <typename ...T> using SpecialOpDependencies = NoSpecialOps;
#endif

  // sharedMemSize
  template <typename ...T> struct sharedMemSizeS {
    static constexpr size_t size(size_t blocksize) { return std::max({sharedMemSizeS<T>::size(blocksize)...}); }
  };
  template <typename ...T> static constexpr size_t sharedMemSize(size_t blocksize) { return sharedMemSizeS<T...>::size(blocksize); }
  template <typename ...T> struct sharedMemSizeS<SpecialOps<T...>> {
    static constexpr size_t size(int blocksize) { return sharedMemSize<T...>(blocksize); }
  };
  template <typename ...T> struct sharedMemSizeS<op_Sequential<T...>> {
    static constexpr size_t size(int blocksize) { return sharedMemSize<T...>(blocksize); }
  };
  template <typename ...T> struct sharedMemSizeS<op_Concurrent<T...>> {
    static constexpr size_t size(int blocksize) { return (sharedMemSize<T>(blocksize) + ...); }
  };
  template <typename T> struct sharedMemSizeS<T> { // T should be of op_Base
    static constexpr size_t size(size_t blocksize) { return sharedMemSize<typename T::dependencies>(blocksize); }
  };

  // sharedMemOffset
  template <typename T, int n> struct sharedMemOffset {
    int operator()(int blocksize) { return 0; }
  };
  template <typename T, int n> struct sharedMemOffset<SpecialOps<T>,n> {
    int operator()(int blocksize) { return sharedMemOffset<T,n>()(blocksize); }
  };
  template <typename ...T> struct sharedMemOffset<op_Concurrent<T...>,0> {
    int operator()(int blocksize) { return 0; }
  };
  template <typename ...T, int n> struct sharedMemOffset<op_Concurrent<T...>,n> {
    int operator()(int blocksize) {
      return sharedMemOffset<op_Concurrent<T...>,n-1>()(blocksize)
	+ sharedMemSize<std::tuple_element_t<n-1,std::tuple<T...>>>(blocksize);
    }
  };

}
