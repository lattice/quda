#pragma once

#ifdef CUDA_BACKEND
#include <cuComplex.h>
#endif

#ifdef HIP_BACKEND
#include <hipComplex.h>
#endif
