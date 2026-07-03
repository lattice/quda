#pragma once
#include <cstddef>

namespace quda
{

  namespace comm_target
  {
#ifdef QUDA_MNNVL
    /**
       @brief Return the NVML-reported GPU Fabric clique ID for the current
       device.  Informational only on some systems (e.g. Ptyche reports a
       constant sentinel cluster-wide); use the open/try_import/close
       primitives below for ground-truth fabric reachability.  Returns 0 if
       NVML is unavailable or the device has no fabric info.
     */
    unsigned int get_fabric_clique_id();

    /**
       @brief Size in bytes of the platform's exportable fabric handle
       (CUmemFabricHandle on CUDA).  Use this to size buffers without
       pulling cuda.h into headers that don't want it.
     */
    size_t fabric_handle_size();

    /**
       @brief Allocate a small probe buffer suitable for fabric P2P and
       write its local CUmemFabricHandle into `out_handle` (must point at
       at least fabric_handle_size() bytes).  Returns an opaque handle the
       caller passes to close_fabric_probe() when done.
     */
    void *open_fabric_probe(void *out_handle);

    /**
       @brief Attempt cuMemImportFromShareableHandle on the supplied peer
       fabric handle (fabric_handle_size() bytes).  Returns true on success
       (and releases the imported handle), false otherwise.  Safe to call
       on a peer outside this rank's actual fabric reach -- the failed
       import is the truth signal.
     */
    bool try_import_fabric_handle(const void *peer_handle);

    /**
       @brief Release the probe buffer returned by open_fabric_probe().
     */
    void close_fabric_probe(void *probe);
#endif
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
