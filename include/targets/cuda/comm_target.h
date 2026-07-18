#pragma once
#include <cstddef>

namespace quda
{

  namespace comm_target
  {
    // The MNNVL fabric-reachability primitives are declared unconditionally so
    // target-agnostic code can call them under `if constexpr (comm_build_is_mnnvl())`
    // (both branches of an if-constexpr in a non-template context must compile).
    // The implementations live in lib/targets/cuda/comm_target.cpp: real under
    // QUDA_MNNVL, cheap stubs otherwise (never executed at runtime on non-MNNVL).
    // All use opaque void*/size_t so this header need not pull in <cuda.h>.

    /**
       @brief Size in bytes of the platform's exportable fabric handle
       (CUmemFabricHandle on CUDA).  Use this to size buffers without
       pulling cuda.h into headers that don't want it.
       @return Handle size in bytes, or 0 on non-MNNVL builds
     */
    size_t fabric_handle_size();

    /**
       @brief Allocate a small probe buffer suitable for fabric P2P and
       write its local CUmemFabricHandle into `out_handle`.
       @param[out] out_handle Buffer of at least fabric_handle_size() bytes
       that receives this rank's exportable fabric handle
       @return Opaque probe handle to pass to close_fabric_probe() when done,
       or nullptr on non-MNNVL builds
     */
    void *open_fabric_probe(void *out_handle);

    /**
       @brief Attempt cuMemImportFromShareableHandle on the supplied peer
       fabric handle.  Safe to call on a peer outside this rank's actual
       fabric reach -- the failed import is the truth signal.
       @param[in] peer_handle Peer's fabric handle (fabric_handle_size() bytes)
       @return true if the import succeeds (the imported handle is released
       before returning), false otherwise or on non-MNNVL builds
     */
    bool try_import_fabric_handle(const void *peer_handle);

    /**
       @brief Release the probe buffer returned by open_fabric_probe().
       No-op on non-MNNVL builds.
       @param[in] probe Opaque probe handle returned by open_fabric_probe()
     */
    void close_fabric_probe(void *probe);
  } // namespace comm_target

  /**
     @brief Compile-time predicate: was this build compiled with MNNVL/fabric
     P2P support (QUDA_MNNVL)?  Lets target-agnostic code branch with
     `if constexpr (comm_build_is_mnnvl())` instead of scattering #ifdefs.
     True only on the CUDA target when QUDA_MNNVL is defined.
   */
  constexpr bool comm_build_is_mnnvl()
  {
#ifdef QUDA_MNNVL
    return true;
#else
    return false;
#endif
  }

} // namespace quda
