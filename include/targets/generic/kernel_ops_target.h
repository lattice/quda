#pragma once
#include <kernel_ops.h>

/**
   @file kernel_ops_target.h

   @section This file contains the target-specific parts of the
   KernelOps support.  This is the generic implementation which is
   used by default for targets that don't provide a target-specific
   version of this file.  This is a dummy implementation for targets
   that don't need to pass any data (e.g. a shared memory pointer) to
   tagged kernels.
 */

namespace quda
{

  // KernelOps
  template <typename... T> struct KernelOps : KernelOpsBase<T...> {
  };

  // op implementations
  struct op_blockSync {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, const Arg &...) { return 0; }
  };

  template <typename T>
  struct op_warp_combine {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, const Arg &...) { return 0; }
  };

} // namespace quda
