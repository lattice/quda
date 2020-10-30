#pragma once
#include "quda_define.h"
#if defined( QUDA_TARGET_CUDA )
#include "targets/cuda/tune_quda.h"
#elif defined(QUDA_TARGET_HIP )
#include "targets/hip/tune_quda.h"
#else
#error "Neither QUDA_TARGET_CUDA nor QUDA_TARGET_HIP are defined"
#endif
