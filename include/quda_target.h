#pragma once

#ifdef CUDA_TARGET
#include <cuda.h>
#include <cuda_runtime.h>
#include <quda_cuda_target.h>
#endif

#ifdef HIP_TARGET
#include <hip/hip_runtime.h>
#include <hip/hip_profiler_api.h>
#include <quda_hip_target.h>
#endif
