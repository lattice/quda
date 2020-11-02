#pragma once

// ensure that we include the quda_define.h file to ensure that __COMPUTE_CAPABILITY__ is set
#include <quda_define.h>

// trove requires the warp shuffle instructions introduced with Kepler and has issues with device debug
#if QUDA_TARGET_CUDA
#if __COMPUTE_CAPABILITY__ >= 300 && !defined(DEVICE_DEBUG)
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif
#else
#define DISABLE_TROVE
#endif