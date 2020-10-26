#pragma once

#include <quda_define.h>

#if defined(QUDA_TARGET_CUDA)
#include "targets/cuda/quda_fp16_cuda.cuh"
#elif defined(QUDA_TARGET_HIP)
#include "targets/hip/quda_fp16_hip.cuh"
#endif
