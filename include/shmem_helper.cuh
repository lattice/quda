#pragma once

/**
   @file shmem_helper.cuh

   @section Description
   Include this file as opposed to nvshmem headers directly to ensure
   correct compilation with NVSHMEM
 */

#if defined(NVSHMEM_COMMS)
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#if defined(__CUDACC__) ||  defined(_NVHPC_CUDA) || (defined(__clang__) && defined(__CUDA__))
// only include if using a CUDA compiler
#include <cuda/atomic>
#endif
#endif
