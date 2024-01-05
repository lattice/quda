#pragma once
#include <kernel_ops.h>

namespace quda {

  // KernelOps
  template <typename ...T>
  struct KernelOps : KernelOps_Base<T...> {
    //template <typename ...U> constexpr void setKernelOps(const KernelOps<U...> &) {
    //  static_assert(std::is_same_v<KernelOps<T...>,KernelOps<U...>>);
    //}
  };

  // op implementations
  struct op_blockSync {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };

  template <typename T>
  struct op_warp_combine {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };

}
