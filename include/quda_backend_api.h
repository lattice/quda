#pragma once

#include <quda_backend.h>

#ifdef CUDA_BACKEND
#include <quda_cuda_api.h>
#endif

#ifdef HIP_BACKEND
#include <quda_hip_api.h>
#endif
