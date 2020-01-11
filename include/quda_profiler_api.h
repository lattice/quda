#pragma once

#ifdef CUDA_BACKEND
#include <cuda_profiler_api.h>
#endif

#ifdef HIP_BACKEND
#include <hip/hip_profiler_api.h>
#endif
