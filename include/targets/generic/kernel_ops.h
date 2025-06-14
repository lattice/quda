#pragma once
#include <kernel_ops_base.h>
#include <target_device.h>

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
    template <typename... Arg> static constexpr unsigned int size(const dim3, const Arg &...) { return N; }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N.
   */
  template <int N> struct SizePerThread {
    template <typename... Arg> static constexpr unsigned int size(const dim3 block, const Arg &...)
    {
      return N * block.x * block.y * block.z;
    }
  };

  /**
     @brief Used to declare an object of size equal to the size of the block Z dimension.
   */
  struct SizeZ {
    template <typename... Arg> static constexpr unsigned int size(const dim3 block, const Arg &...) { return block.z; }
  };

  /**
     @brief Used to declare an object of size equal to the block size divided by the warp size.
   */
  struct SizeBlockDivWarp {
    template <typename... Arg> static constexpr unsigned int size(const dim3 b, const Arg &...)
    {
      return (b.x * b.y * b.z + device::warp_size() - 1) / device::warp_size();
    }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N, with thread dimensions derermined by D.
   */
  template <typename D, int N = 1> struct SizeDims {
    template <typename... Arg> static constexpr unsigned int size(const dim3 block, const Arg &...arg)
    {
      dim3 dims = D::dims(block, arg...);
      return dims.x * dims.y * dims.z * N;
    }
  };

  /**
     @brief Used to declare an object with dimensions given by the block size.
   */
  struct DimsBlock {
    template <typename... Arg> static constexpr dim3 dims(const dim3 block, const Arg &...) { return block; }
  };

  /**
     @brief Used to declare an object with fixed dimensions.
   */
  template <int x, int y, int z> struct DimsStatic {
    template <typename... Arg> static constexpr dim3 dims(dim3, const Arg &...) { return dim3(x, y, z); }
  };

} // namespace quda
