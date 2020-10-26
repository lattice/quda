#pragma once

#include "quda_define.h"

#if defined(QUDA_TARGET_CUDA)
#include "targets/cuda/random_quda_cuda.h"
#elif defined(QUDA_TARGET_HIP)
#include "targets/hip/random_quda_hip.h"
#endif
