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
    static constexpr unsigned int size(dim3 block) { return block.z; }
  };

  /**
     @brief Used to declare an object of size equal to the block size divided by the warp size.
   */
  struct SizeBlockDivWarp {
    static constexpr unsigned int size(dim3 b)
    {
      return (b.x * b.y * b.z + device::warp_size() - 1) / device::warp_size();
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
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, const Arg &...arg);

  /**
     @brief KernelOps forward declaration and KernelOps_Base type,
     which the target specific KernelOps should inherit from.  Kernels
     can inherit from KernelOps to tag kernels with operations that
     may need special resources like shared memory, or have other
     special requirements (e.g. using block sync which may require
     special handling for some targets).  The template arguments,
     T..., specify the types of the operations the kernel uses.
   */
  template <typename... T> struct KernelOps;
  template <typename... T> struct KernelOps_Base {
    using KernelOpsT = KernelOps<T...>;
    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3 block, Arg &...arg)
    {
      return sharedMemSize<KernelOpsT>(block, arg...);
    }
  };

  /**
     @brief Type that specifies a kernel does not have any operations
     that need tagging.  This can be used as an alternative in cases
     where the operations are only conditionally used.
   */
  struct NoKernelOps {
    using KernelOpsT = NoKernelOps;
  };

  /**
     @brief Used to get KernelOps from a kernel type.  Checks for the
     existence of KernelOpsT.
   */
  template <typename T, typename U = void> struct getKernelOpsS {
    using type = NoKernelOps;
  };
  template <typename T> struct getKernelOpsS<T, std::conditional_t<true, void, typename T::KernelOpsT>> {
    using type = typename T::KernelOpsT;
  };
  template <typename T> using getKernelOps = typename getKernelOpsS<T>::type;

  /**
     @brief Checks whether a kernel type is tagged with any KernelOps.
   */
  template <typename T> static constexpr bool hasKernelOps = !std::is_same_v<getKernelOps<T>, NoKernelOps>;

  /**
     @brief Checks if first template type matches any of the ops in
     the second template type, which is a KernelOps template type.
   */
  template <typename T, typename U> static constexpr bool hasKernelOp = false;
  template <typename T, typename... U>
  static constexpr bool hasKernelOp<T, KernelOps<U...>> = (std::is_same_v<T, U> || ...);

  /**
     @brief Function to statically check if the passed kernel functor was tagged with all the
     operations in the template parameters.
   */
  template <typename... T, typename Ops> static constexpr void checkKernelOps(const Ops &)
  {
    static_assert((hasKernelOp<T, typename Ops::KernelOpsT> || ...));
  }

  /**
     @brief Helper to combine two KernelOps or NoKernelOps types.
   */
  template <typename... T> struct combineOpsS {
  };
  template <typename... T> struct combineOpsS<NoKernelOps, KernelOps<T...>> {
    using type = KernelOps<T...>;
  };
  template <typename... T> struct combineOpsS<KernelOps<T...>, NoKernelOps> {
    using type = KernelOps<T...>;
  };
  template <typename... T, typename... U> struct combineOpsS<KernelOps<T...>, KernelOps<U...>> {
    using type = KernelOps<T..., U...>;
  };
  template <typename T, typename U> using combineOps = typename combineOpsS<T, U>::type;

  /**
     @brief Gets the total shared memory size needed for a set of
     kernel operations.  If any ops types have an offset for the
     shared memory, then the offset is included in the size.
   */
  template <typename T> struct sharedMemSizeS {
    template <typename... Arg> static constexpr unsigned int size(dim3 block, const Arg &...arg)
    {
      return T::shared_mem_size(block, arg...);
    }
  };
  template <> struct sharedMemSizeS<NoKernelOps> {
    template <typename... Arg> static constexpr unsigned int size(dim3, const Arg &...) { return 0; }
  };
  template <typename... T> struct sharedMemSizeS<KernelOps<T...>> {
    template <typename... Arg> static constexpr unsigned int size(dim3 block, const Arg &...arg)
    {
      return std::max({sharedMemSizeS<T>::size(block, arg...)...});
    }
  };
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, const Arg &...arg)
  {
    return sharedMemSizeS<T>::size(block, arg...);
  }

  // forward declarations of op types to be defined by target
  struct op_blockSync;
  template <typename T> struct op_warp_combine;

  // only types for convenience
  using only_blockSync = KernelOps<op_blockSync>;
  template <typename T> using only_warp_combine = KernelOps<op_warp_combine<T>>;

} // namespace quda
