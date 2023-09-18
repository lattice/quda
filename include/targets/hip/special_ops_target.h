#pragma once
#include <special_ops.h>

namespace quda {

  // SpecialOps
  template <typename ...T>
  struct SpecialOps : SpecialOps_Base<T...> {
    template <typename ...U> constexpr void setSpecialOps(const SpecialOps<U...> &) {
      static_assert(std::is_same_v<SpecialOps<T...>,SpecialOps<U...>>);
    }
  };

  // op implementations
  struct op_blockSync : op_BaseT<void> {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };

  template <typename T>
  struct op_warp_combine : op_BaseT<T> {
    template <typename ...Arg>
    static constexpr unsigned int shared_mem_size(dim3, Arg &...) { return 0; }
  };

}
