#pragma once

/**
   @file kernel_ops.h

   @section This file contains the target-independent structs and
   helpers in support of the KernelOps struct, which is used to tag
   kernels that need special resources (like shared memory) or
   operations (like synchronization).  The target-specific parts
   (including the full definition of the KernelOps struct) are in
   kernel_ops_target.h.
 */

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

  // forward declare sharedMemSize
  template <typename T, typename... Arg> static constexpr unsigned int sharedMemSize(dim3 block, const Arg &...arg);

  /**
     @brief KernelOps forward declaration.  The full definition is
     target specific and defined in kernel_ops_target.h.  Kernels can
     inherit from KernelOps to tag kernels with operations that may
     need special resources like shared memory, or have other special
     requirements (e.g. using block sync which may require special
     handling for some targets).  The template parameters, T...,
     specify the types of the operations the kernel uses.  The
     operations listed are assumed to be able to share resources
     (e.g. they could reuse the same shared memory).  If any
     operations are used concurrently, and thus cannot share the same
     resources, then this needs to be encoded in the operation types
     themselves (i.e. by having one operation type use the other
     operation type as an offset parameter).
   */
  template <typename... T> struct KernelOps;

  /**
     @brief KernelOpsBase type, which the target specific KernelOps
     should inherit from.  This contains all the target-independent
     routines used with KernelOps.
  */
  template <typename... T> struct KernelOpsBase {
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
     @brief getKernelOps<T> is used to get the KernelOps type from a
     kernel type T.  Checks for the existence of KernelOpsT.
   */
  template <typename T, typename U = void> struct getKernelOpsS {
    using type = NoKernelOps;
  };
  template <typename T> struct getKernelOpsS<T, std::conditional_t<true, void, typename T::KernelOpsT>> {
    using type = typename T::KernelOpsT;
  };
  template <typename T> using getKernelOps = typename getKernelOpsS<T>::type;

  /**
     @brief hasKernelOps<T> checks whether a kernel type T is tagged with any KernelOps.
   */
  template <typename T> static constexpr bool hasKernelOps = !std::is_same_v<getKernelOps<T>, NoKernelOps>;

  /**
     @brief hasKernelOp<T, KernelOps<U...>> checks if first template
     type, T, matches any of the ops (U...) in the second template
     type, which is a KernelOps type.
   */
  template <typename T, typename U> static constexpr bool hasKernelOp = false;
  template <typename T, typename... U>
  static constexpr bool hasKernelOp<T, KernelOps<U...>> = (std::is_same_v<T, U> || ...);

  /**
     @brief Function to statically check if the kernel functor passed
     in as an argument was tagged with all the operations in the
     template parameters (T...).
   */
  template <typename... T, typename Ops> static constexpr void checkKernelOps(const Ops &)
  {
    static_assert((hasKernelOp<T, typename Ops::KernelOpsT> || ...));
  }

  /**
     @brief combineOps<T,U> is a helper to combine two KernelOps or
     NoKernelOps types.  The template arguments T and U must either
     either be a KernelOps or NoKernleOps type.  The returned type is
     a single KernelOps type which combines the template parameters of
     the individual KernelOps template parameters.  Specifically
     \verbatim
       combineOps<KernelOps<T...>,KernelOps<U...>> -> KernelOps<T...,U...>
       combineOps<KernelOps<T...>,NoKernelOps> -> KernelOps<T...>
       combineOps<NoKernelOps,KernelOps<U...>> -> KernelOps<U...>
       combineOps<NoKernelOps,NoKernelOps> -> NoKernelOps
     \endverbatim
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
     @brief sharedMemSize gets the total shared memory size needed for
     a set of kernel operations.  Since operations are assumed to be
     able to share resources, we only need to find the largest shared
     memory block needed by any of them.  Any case where operations
     cannot share resources should have been encoded in the
     corresponding operation types, so that an offset for the shared
     memory is already include into operations that need it.  The
     shared memory size for an operation type with an offset will
     include the offset in the returned size.
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

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or some wrapper of T
   */
  template <class T, class enable = void> struct get_type {
    using type = T;
  };

} // namespace quda
