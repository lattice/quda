#pragma once

#ifdef CUDA_BACKEND
#include <cufft.h>
#endif

#ifdef HIP_BACKEND
#include <hipfft.h>
#endif
