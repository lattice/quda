#pragma once
#include <quda_define.h>

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(QUDA_TARGET_HIP)
#include <hip/hip_runtime.h>
#endif
