#pragma once
#include <cstddef>

namespace quda
{

  /**
     @brief Compile-time predicate: was this build compiled with MNNVL/fabric
     P2P support?  Always false on the HIP target -- MNNVL is a CUDA-only
     feature.  Mirrors include/targets/cuda/comm_target.h so target-agnostic
     code can branch with `if constexpr (comm_build_is_mnnvl())` without
     scattering #ifdefs.
   */
  constexpr bool comm_build_is_mnnvl() { return false; }

} // namespace quda
