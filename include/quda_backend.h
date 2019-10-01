#pragma once

// TBD: API calls
// cudaCreateTextureObject

// cudaDestroyTextureObject
// cudaDeviceCanAccessPeer
// cudaDeviceGetStreamPriorityRange
// cudaDeviceReset
// cudaDeviceSetCacheConfig
// cudaDriverGetVersion

// cudaEventSynchronize

// cudaGetTextureObjectResourceDesc
// cudaGetDeviceCount 
// cudaGetDeviceProperties
// cudaGetErrorString

// cudaHostGetDevicePointer

// cudaHostRegister
// cudaHostRegisterDefault
// cudaHostRegisterMapped
// cudaHostRegisterPortable

// cudaHostUnregister

// cudaProfilerStart
// cudaProfilerStop

// cudaRuntimeGetVersion
// cudaSetDevice

// cudaStreamCreateWithPriority
// cudaStreamDestroy


#ifdef CUDA_BACKEND
#include <cuda.h>
#include <cuda_runtime.h>

#define QUDA_SUCCESS CUDA_SUCCESS
#define QUDA_MEMORYTYPE_ARRAY CU_MEMORYTYPE_ARRAY
#define QUDA_MEMORYTYPE_DEVICE CU_MEMORYTYPE_DEVICE
#define QUDA_MEMORYTYPE_UNIFIED CU_MEMORYTYPE_UNIFIED
#define QUDA_MEMORYTYPE_HOST CU_MEMORYTYPE_HOST

#define qudaChannelFormatDesc cudaChannelFormatDesc
#define qudaChannelFormatKindFloat cudaChannelFormatKindFloat
#define qudaChannelFormatKindSigned cudaChannelFormatKindSigned

#define qudaFuncCachePreferL1 cudaFuncCachePreferL1

#define qudaTextureDesc cudaTextureDesc
#define qudaTextureObject_t cudaTextureObject_t

#define qudaReadModeElementType cudaReadModeElementType
#define qudaReadModeNormalizedFloat cudaReadModeNormalizedFloat

#define qudaResourceDesc cudaResourceDesc
#define qudaResourceTypeLinear cudaResourceTypeLinear

#define qudaStreamDefault cudaStreamDefault

#define qudaStream_t cudaStream_t
#define qudaSuccess cudaSuccess
#define qudaEvent_t cudaEvent_t
#define qudaError_t cudaError_t

#define qudaDeviceptr_t CUdeviceptr
#define qudaMemoryType CUmemorytype
#define qudaCUresult CUresult

#define qudaMemcpyKind cudaMemcpyKind
#define qudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define qudaMemcpyHostToDevice cudaMemcpyHostToDevice
#define qudaMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define qudaFuncAttribute cudaFuncAttribute
#define qudaDeviceProp cudaDeviceProp

#endif

#ifdef HIP_BACKEND
#include <hip/hip_runtime.h>
#include <hip/hip_profiler_api.h

#define QUDA_SUCCESS hipSuccess
#define QUDA_MEMORYTYPE_ARRAY hipMemoryTypeArray
#define QUDA_MEMORYTYPE_DEVICE hipMemoryTypeDevice
#define QUDA_MEMORYTYPE_UNIFIED hipMemoryTypeUnified
#define QUDA_MEMORYTYPE_HOST hipMemoryTypeHost

#define qudaChannelFormatDesc hipChannelFormatDesc
#define qudaChannelFormatKindFloat hipChannelFormatKindFloat
#define qudaChannelFormatKindSigned hipChannelFormatKindSigned

#define qudaFuncCachePreferL1 hipFuncCachePreferL1

#define qudaReadModeElementType hipReadModeElementType
#define qudaReadModeNormalizedFloat hipReadModeNormalizedFloat

#define qudaResourceDesc hipResourceDesc
#define qudaResourceTypeLinear hipResourceTypeLinear

#define qudaStreamDefault hipStreamDefault

#define qudaStream_t hipStream_t
#define qudaSuccess hipSuccess
#define qudaEvent_t hipEvent_t
#define qudaError_t hipError_t
#define qudaTextureObject_t hipTextureObject_t
#define qudaDeviceptr_t hipDeviceptr_t
#define qudaMemoryType hipMemoryType
#define qudaCUresult  hipError_t

#define qudaMemcpyKind hipMemcpyKind
#define qudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define qudaMemcpyHostToDevice hipMemcpyHostToDevice
#define qudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define qudaFuncAttribute hipFuncAttribute
#define qudaDeviceProp hipDeviceProp
#define qudaTextureDesc hipTextureDesc



#endif
