#pragma once

#include "quda_define.h"

#if defined(QUDA_BUILD_TARGET_CUDA)
#include "targets/cuda/random_quda_cuda.h"
#else
#include "targets/hip/random_quda_hip.h"
#endif
