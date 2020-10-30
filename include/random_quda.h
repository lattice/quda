#pragma once

#include "quda_define.h"

#if defined(QUDA_TARGET_CUDA)
#include "targets/cuda/random_quda_cuda.h"
#elif defined(QUDA_TARGET_HIP)
#include "targets/hip/random_quda_hip.h"
#else
#error "Neither QUDA_TARGET_CUDA or QUDA_TARGET_HIP are defined"
#endif
