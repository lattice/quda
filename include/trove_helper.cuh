#pragma once

// ensure that we include the quda_define.h file to ensure that __COMPUTE_CAPABILITY__ is set
#include <quda_define.h>

// trove requires CUDA and has issues with device debug
#if defined(TARGET_CUDA) && !defined(DEVICE_DEBUG)
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif
