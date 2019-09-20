#pragma once

#ifdef CUDA_BACKEND
#include <cuda.h>
#include <cuda_runtime.h>
#define qudaStream_t cudaStream_t
#define qudaEvent_t cudaEvent_t
#define qudaError_t cudaError_t
#define qudaMemcpyKind cudaMemcpyKind
#define qudaFuncAttribute cudaFuncAttribute
#define qudaDeviceProp cudaDeviceProp
#endif

#ifdef HIP_BACKEND
#include <hip/hip_runtime.h>
#define qudaStream_t hipStream_t
#define qudaEvent_t hipEvent_t
#define qudaError_t hipError_t
#define qudaMemcpyKind hipMemcpyKind
#define qudaFuncAttribute hipFuncAttribute
#define qudaDeviceProp hipDeviceProp
#endif
