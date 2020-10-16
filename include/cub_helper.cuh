#pragma once
#include "quda_define.h"
#if defined(QUDA_BUILD_TARGET_CUDA)
#include "targets/cuda/cub_helper_cuda.cuh"
#elif defined(QUDA_BUILD_TARGET_HIP)
#include "targets/hip/cub_helper_hip.cuh"
#endif

