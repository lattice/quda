#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
using qudaDeviceProp_t = cudaDeviceProp;

using qudaEvent_t = cudaEvent_t;
using qudaIpcEventHandle_t = cudaIpcEventHandle_t;
using qudaIpcMemHandle_t = cudaIpcMemHandle_t;

#define QUDA_DYNAMIC_SHARED( type, var )        \
        extern __shared__ type var[] ;
