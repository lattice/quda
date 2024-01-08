#pragma once
#include <kernel_ops.h>

namespace quda {

  // KernelOps
  template <typename ...T>
  struct KernelOps : KernelOps_Base<T...> {
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

}
