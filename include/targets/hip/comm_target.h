#pragma once
#include <cstddef>

namespace quda
{

  namespace comm_target
  {
    // MNNVL fabric P2P is a CUDA-only feature, so on the HIP target these are
    // inline no-op stubs.  They exist only so target-agnostic code can call the
    // facade under `if constexpr (comm_build_is_mnnvl())` (both branches of an
    // if-constexpr in a non-template context must compile); they are never
    // executed at runtime because comm_build_is_mnnvl() is false here.
    inline size_t fabric_handle_size() { return 0; }
    inline void *open_fabric_probe(void *) { return nullptr; }
    inline bool try_import_fabric_handle(const void *) { return false; }
    inline void close_fabric_probe(void *) { }
  } // namespace comm_target

  /**
     @brief Compile-time predicate: was this build compiled with MNNVL/fabric
     P2P support?  Always false on the HIP target -- MNNVL is a CUDA-only
     feature.  Mirrors include/targets/cuda/comm_target.h so target-agnostic
     code can branch with `if constexpr (comm_build_is_mnnvl())` without
     scattering #ifdefs.
   */
  constexpr bool comm_build_is_mnnvl() { return false; }

} // namespace quda
