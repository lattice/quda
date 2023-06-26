#pragma once
#include <quda_define.h>

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#if (__COMPUTE_CAPABILITY__ >= 700) && defined(QUDA_ENABLE_MMA)
#define QUDA_MMA_AVAILABLE
#endif

#elif defined(QUDA_TARGET_HIP)
#include <hip/hip_runtime.h>

#elif defined(QUDA_TARGET_SYCL)
#include <targets/sycl/quda_sycl.h>

#endif
