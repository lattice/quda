#pragma once

namespace quda {

  enum KernelType {
    INTERIOR_KERNEL = 5,
    EXTERIOR_KERNEL_ALL = 6,
    EXTERIOR_KERNEL_X = 0,
    EXTERIOR_KERNEL_Y = 1,
    EXTERIOR_KERNEL_Z = 2,
    EXTERIOR_KERNEL_T = 3,
    KERNEL_POLICY = 7
  };

}
