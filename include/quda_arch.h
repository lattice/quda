#pragma once
#include <quda_define.h>

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#define QUDA_MMA_AVAILABLE 1
#endif

#elif defined(QUDA_TARGET_HIP)
#include <hip/hip_runtime.h>
#endif
