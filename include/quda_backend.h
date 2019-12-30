#pragma once

// TBD: API calls

// qurand_normal
// qurand_normal_double
// qurand_uniform
// qurand_uniform_double

// cufftExecC2C
// cufftExecZ2Z
// cufftPlanMany

// cuMemAlloc
// cuMemFree

// cudaFree
// cudaFreeHost

// * cudaCreateTextureObject
// * cudaDestroyTextureObject

// * cudaDeviceCanAccessPeer
// * cudaDeviceGetStreamPriorityRange
// * cudaDeviceReset
// * cudaDeviceSetCacheConfig
// * cudaDeviceSynchronize
// * cudaGetDeviceCount 
// * cudaGetDeviceProperties
// * cudaHostGetDevicePointer
// * cudaSetDevice

// * cudaDriverGetVersion
// * cudaRuntimeGetVersion

// * cudaEventCreate
// * cudaEventDestroy
// * cudaEventElapsedTime
// * cudaEventRecord
// * cudaEventSynchronize
 
// * cudaGetTextureObjectResourceDesc
// * cudaGetErrorString
// * cudaGetLastError

// cudaHostAlloc
// cudaHostRegister
// cudaHostRegisterDefault
// cudaHostRegisterMapped
// cudaHostRegisterPortable
// cudaHostUnregister

// cudaIpcCloseMemHandle
// cudaIpcGetEventHandle
// cudaIpcGetMemHandle
// cudaIpcOpenEventHandle
// cudaIpcOpenMemHandle

// cudaProfilerStart
// cudaProfilerStop

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
#define QUDAFFT_C2C CUFFT_C2C

#define QUdeviceptr CUdeviceptr
#define QUmemorytype CUmemorytype
#define QUresult CUresult

#define qudaChannelFormatDesc cudaChannelFormatDesc
#define qudaChannelFormatKindFloat cudaChannelFormatKindFloat
#define qudaChannelFormatKindSigned cudaChannelFormatKindSigned
#define qudaEventDisableTiming cudaEventDisableTiming
#define qudaEventInterprocess cudaEventInterprocess
#define qudaFuncCache cudaFuncCache
#define qudaFuncCachePreferL1 cudaFuncCachePreferL1
#define qudafftComplex cufftComplex
#define qudafftDoubleComplex cufftDoubleComplex
#define qudafftHandle cufftHandle
#define qudafftResult cufftResult
#define qudaHostRegisterDefault cudaHostRegisterDefault
#define qudaHostRegisterMapped cudaHostRegisterMapped
#define qudaHostRegisterPortable cudaHostRegisterPortable

#define qudaIpcEventHandle_t cudaIpcEventHandle_t
#define qudaIpcMemHandle_t cudaIpcMemHandle_t
#define qudaIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define qudaTextureDesc cudaTextureDesc
#define qudaTextureObject_t cudaTextureObject_t
#define qudaReadModeElementType cudaReadModeElementType
#define qudaReadModeNormalizedFloat cudaReadModeNormalizedFloat
#define qudaResourceDesc cudaResourceDesc
#define qudaResourceTypeLinear cudaResourceTypeLinear
#define qudaResourceViewDesc cudaResourceViewDesc
#define qudaStreamDefault cudaStreamDefault
#define qudaStream_t cudaStream_t
#define qudaSuccess cudaSuccess
#define qudaEvent_t cudaEvent_t
#define qudaError_t cudaError_t
#define qudaMemcpyKind cudaMemcpyKind
#define qudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define qudaMemcpyHostToDevice cudaMemcpyHostToDevice
#define qudaMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define qudaFuncAttribute cudaFuncAttribute
#define qudaDeviceProp cudaDeviceProp
#define qudaWarpSize warpSize

#define qudaFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize 
#define qudaFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributePreferredSharedMemoryCarveout
#define qudaSharedmemCarveoutMaxShared cudaSharedmemCarveoutMaxShared


#endif

#ifdef HIP_BACKEND

#endif
