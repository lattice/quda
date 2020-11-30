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
#include <cuda/atomic>
#endif