#pragma once
#include <quda_define.h>

// Make sure this is not defined
#ifdef QUDA_MMA_AVAILABLE
#undef QUDA_MMA_AVAILABLE
#endif

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

// Define MMA availability
#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#define QUDA_MMA_AVAILABLE	1
#endif

#elif defined(QUDA_TARGET_HIP)
#include <hip/hip_runtime.h>
#endif
