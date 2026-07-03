#pragma once

// CUDA-target accessors for the P2P comm-buffer fabric handles.  These return a
// CUmemFabricHandle and are only meaningful under MNNVL, so they are isolated
// here (targets/cuda) rather than in the generic <malloc_quda.h>, keeping
// <cuda.h> and the QUDA_MNNVL #ifdef out of target-agnostic headers.

#include <cstddef>
#include <cstdint>

#ifdef QUDA_MNNVL
#include <cuda.h>

namespace quda
{

  /**
     @brief Return the CUmemFabricHandle for a P2P comm buffer previously
     allocated via comm_buffer_malloc_(DeviceCommBuffer, ...).  Used by
     comm_create_neighbor_memory_p2p to export the local buffer's handle to
     peer ranks across the MNNVL clique via MPI.  Errors if ptr is not a
     P2P comm buffer allocated under QUDA_MNNVL.
   */
  CUmemFabricHandle get_p2p_fabric_handle(void *ptr);

  /** @brief Return the exact padded VMM allocation size for @p ptr. */
  size_t get_p2p_buffer_size(void *ptr);

  /** @brief Return a process-local identifier for this allocation generation. */
  uint64_t get_p2p_buffer_generation(void *ptr);

} // namespace quda
#endif // QUDA_MNNVL
