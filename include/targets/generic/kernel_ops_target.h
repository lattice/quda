#pragma once
#include <kernel_ops.h>

namespace quda {

  // KernelOps
  template <typename ...T>
  struct KernelOps : KernelOps_Base<T...> {
  };

}
