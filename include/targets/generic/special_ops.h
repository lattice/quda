#pragma once
#include <target_device.h>
#include <kernel_ops.h>

namespace quda {

#if 0
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
#endif

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

}
