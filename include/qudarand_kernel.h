#pragma once

#ifdef CUDA_BACKEND
#include <curand_kernel.h>
#endif

#ifdef HIP_BACKEND
#include <hiprand_kernel.h>
#endif
