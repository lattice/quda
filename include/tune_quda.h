#pragma once
#include "quda_define.h"
#if defined( QUDA_BUILD_TARGET_CUDA )
#include "targets/cuda/tune_quda.h"
#elif defined(QUDA_BUILD_TARGET_HIP )
#include "targets/hip/tune_quda.h"
#endif
