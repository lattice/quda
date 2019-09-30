#pragma once

// TBD: API calls
// cudaCreateTextureObject
// cudaDestroyTextureObject
// cudaEventSynchronize
// cudaGetTextureObjectResourceDesc
// cudaHostGetDevicePointer
// cudaHostRegisterDefault
// cudaHostRegisterMapped
// cudaHostRegisterPortable
// cudaHostUnregister
// cudaHostRegister

#ifdef CUDA_BACKEND
#include <cuda.h>
#include <cuda_runtime.h>

#define qudaChannelFormatDesc cudaChannelFormatDesc
#define qudaChannelFormatKindFloat cudaChannelFormatKindFloat
#define qudaChannelFormatKindSigned cudaChannelFormatKindSigned
#define qudaReadModeElementType cudaReadModeElementType
#define qudaReadModeNormalizedFloat cudaReadModeNormalizedFloat
#define qudaResourceDesc cudaResourceDesc
#define qudaResourceTypeLinear cudaResourceTypeLinear
#define qudaStream_t cudaStream_t
#define qudaSuccess cudaSuccess
#define qudaEvent_t cudaEvent_t
#define qudaError_t cudaError_t
#define qudaTextureObject_t cudaTextureObject_t
#define qudaMemcpyKind cudaMemcpyKind
#define qudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define qudaMemcpyHostToDevice cudaMemcpyHostToDevice
#define qudaMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define qudaFuncAttribute cudaFuncAttribute
#define qudaDeviceProp cudaDeviceProp
#define qudaTextureDesc cudaTextureDesc
#endif

#ifdef HIP_BACKEND
#include <hip/hip_runtime.h>
#define qudaChannelFormatDesc hipChannelFormatDesc
#define qudaChannelFormatKindFloat hipChannelFormatKindFloat
#define qudaChannelFormatKindSigned hipChannelFormatKindSigned
#define qudaReadModeElementType hipReadModeElementType
#define qudaReadModeNormalizedFloat hipReadModeNormalizedFloat
#define qudaResourceDesc hipResourceDesc
#define qudaResourceTypeLinear hipResourceTypeLinear
#define qudaStream_t hipStream_t
#define qudaSuccess hipSuccess
#define qudaEvent_t hipEvent_t
#define qudaError_t hipError_t
#define qudaTextureObject_t hipTextureObject_t
#define qudaMemcpyKind hipMemcpyKind
#define qudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define qudaMemcpyHostToDevice hipMemcpyHostToDevice
#define qudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define qudaFuncAttribute hipFuncAttribute
#define qudaDeviceProp hipDeviceProp
#define qudaTextureDesc hipTextureDesc
#endif
