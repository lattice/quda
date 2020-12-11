#pragma once 
#include <hip/hip_runtime_api.h>

using qudaDeviceProp_t = hipDeviceProp_t;

using qudaEvent_t = hipEvent_t;
using qudaIpcEventHandle_t = hipIpcEventHandle_t;
using qudaIpcMemHandle_t = hipIpcMemHandle_t;

#define QUDA_DYNAMIC_SHARED( type, var )        \
        HIP_DYNAMIC_SHARED(type, var);

