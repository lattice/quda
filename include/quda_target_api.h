#pragma once

#include <quda_target.h>

#ifdef CUDA_TARGET
#include <quda_cuda_api.h>
#endif

#ifdef HIP_TARGET
#include <quda_hip_api.h>
#endif
