#pragma once

namespace quda
{

  /**
     @brief Used to declare an object of fixed size.
   */
  template <int N> struct SizeStatic {
    static constexpr unsigned int size(dim3) { return N; }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N.
   */
  template <int N> struct SizePerThread {
    static constexpr unsigned int size(dim3 block) { return N * block.x * block.y * block.z; }
  };

  /**
     @brief Used to declare an object of size equal to the size of the block Z dimension.
   */
  struct SizeZ {
    static constexpr unsigned int size(dim3 block) {
      return block.z;
    }
  };

  /**
     @brief Used to declare an object of size equal to the block size divided by the warp size.
   */
  struct SizeBlockDivWarp {
    static constexpr unsigned int size(dim3 b) {
      return (b.x * b.y * b.z + device::warp_size() - 1)/device::warp_size();
    }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N, with thread dimensions derermined by D.
   */
  template <typename D, int N = 1> struct SizeDims {
    static constexpr unsigned int size(dim3 block)
    {
      dim3 dims = D::dims(block);
      return dims.x * dims.y * dims.z * N;
    }
  };

  /**
     @brief Used to declare an object with dimensions given by the block size.
   */
  struct DimsBlock {
    static constexpr dim3 dims(dim3 block) { return block; }
  };

  /**
     @brief Used to declare an object with fixed dimensions.
   */
  template <int x, int y, int z> struct DimsStatic {
    static constexpr dim3 dims(dim3) { return dim3(x, y, z); }
  };

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or some wrapper of T
   */
  template <class T, class enable = void> struct get_type {
    using type = T;
  };

  // forward declare sharedMemSize
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, Arg &...arg);

  // alternative to KernelOps
  struct NoKernelOps {
    using KernelOpsT = NoKernelOps;
  };
  // KernelOps forward declaration and base type
  template <typename... T> struct KernelOps;
  template <typename... T> struct KernelOps_Base {
    using KernelOpsT = KernelOps<T...>;
    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3 block, Arg &...arg) {
      return sharedMemSize<KernelOpsT>(block, arg...);
    }
  };

  /**
     @brief Used to get KernelOps from a kernel type.  Checks for the existence of KernelOpsT.
   */
  template <typename T, typename U = void> struct getKernelOpsS { using type = NoKernelOps; };
  template <typename T> struct getKernelOpsS<T,std::conditional_t<true,void,typename T::KernelOpsT>> {
    using type = typename T::KernelOpsT;
  };
  template <typename T> using getKernelOps = typename getKernelOpsS<T>::type;

  // hasKernelOp: checks if first type matches any of the op
  // <op, KernelOps<ops...>>
  template <typename T, typename U> static constexpr bool hasKernelOp = false;
  template <typename T, typename... U>
  static constexpr bool hasKernelOp<T,KernelOps<U...>> = ( std::is_same_v<T,U> || ... );

  // checkKernelOps
  template <typename... T, typename Ops> static constexpr void checkKernelOps(const Ops &) {
    static_assert((hasKernelOp<T,typename Ops::KernelOpsT> || ...));
  }

  // hasKernelOps
  template <typename T> inline constexpr bool hasKernelOps = !std::is_same_v<getKernelOps<T>,NoKernelOps>;

  // combineOps
  template <typename... T> struct combineOpsS {};
  template <typename... T> struct combineOpsS<NoKernelOps,KernelOps<T...>> {
    using type = KernelOps<T...>; };
  template <typename ... T> struct combineOpsS<KernelOps<T...>,NoKernelOps> {
    using type = KernelOps<T...>; };
  template <typename ...T, typename ...U> struct combineOpsS<KernelOps<T...>,KernelOps<U...>> {
    using type = KernelOps<T..., U...>; };
  template <typename T, typename U> using combineOps = typename combineOpsS<T, U>::type;

  // sharedMemSize
  template <typename T> struct sharedMemSizeS {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      return T::shared_mem_size(block, arg...);
    }
  };
  template <> struct sharedMemSizeS<NoKernelOps> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3, Arg &...) {
      return 0;
    }
  };
  template <typename ...T> struct sharedMemSizeS<KernelOps<T...>> {
    template <typename ...Arg>
    static constexpr unsigned int size(dim3 block, Arg &...arg) {
      return std::max({sharedMemSizeS<T>::size(block, arg...)...});
    }
  };
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, Arg &...arg) {
    return sharedMemSizeS<T>::size(block, arg...);
  }

  // forward declarations of op types
  struct op_blockSync;
  template <typename T> struct op_warp_combine;

  // only types for convenience
  using only_blockSync = KernelOps<op_blockSync>;
  template <typename T> using only_warp_combine = KernelOps<op_warp_combine<T>>;

  // explicitKernelOps
  //template <typename T, typename U = void> struct explicitKernelOpsS : std::false_type {};
  //template <typename T>
  //struct explicitKernelOpsS<T,std::conditional_t<true,void,typename T::KernelOpsT>> : std::true_type {};
  //template <typename T> inline constexpr bool explicitKernelOps = explicitKernelOpsS<T>::value;

  // checkKernelOp
  //template <typename T, typename... U> static constexpr void checkKernelOp() {
  //  static_assert((std::is_same_v<T,U> || ...) == true);
  //}

} // namespace quda
