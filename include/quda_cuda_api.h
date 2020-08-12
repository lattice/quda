#pragma once

#if defined(__HIP__)

#include <hip/hip_runtime.h>
using qudaStream_t = hipStream_t;
using qudaMemcpyKind = hipMemcpyKind;
using qudaError_t = hipError_t;
using qudaEvent_t = hipEvent_t;
using qudaMemoryType = hipMemoryType;
using qudaIpcMemHandle_t = hipIpcMemHandle_t;
using qudaDeviceProp = hipDeviceProp;
#define qudaHostGetDevicePointer hipHostGetDevicePointer
#define qudaStreamCreate hipStreamCreate
#define qudaStreamDestroy hipStreamDestroy
#define qudaEventCreate hipEvenCreate
#define qudaEventRecord hipEvenRecord
#define qudaEventSynchronize hipEventSynchronize
#define qudaEventDisableTiming hipEventDisableTiming
#define qudaEventDestroy hipEventDestroy
#define qudaDeviceSynchronize hipDeviceSynchronize
#define qudaMemset hipMemset
#define qudaSuccess hipSuccess
#define qudaErrorNotReady hipErrorNotReady
#define qudaErrorUnknown hipErrorUnknown
#define qudaSuccessjit hipSuccess
#define qudaEventElapsedTime hipEventElapsedTime
#define qudaEventDestroy hipEventDestroy
#define qudaGetLastError hipGetLastError
#define qudaGetErrorString hipGetErrorString
#define qudaHostRegister hipHostRegister
#define qudaHostAlloc hipHostAlloc
#define qudaHostRegisterMapped hipHostRegisterMapped
#define qudaHostRegisterPortable hipHostRegisterPortable
#define QUDA_ERROR_INVALID_VALUE hipErrorInvalidValue
#define qudaIpcGetMemHandle hipIpcGetMemHandle
#define qudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define qudaIpcCloseMemHandle hipIpcCloseMemHandle
#define qudaEventQuery hipEventQuery
#define qudaMemcpyHostToDevice hipMemcpyHostToDevice
#define qudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define qudaFuncSetCacheConfig hipFuncSetCacheConfig
#define qudaFuncCachePreferL1 hipFuncCachePreferL1
#define qudaGetErrorName hipGetErrorName
#define qudarand hiprand
#define qudaFree hipFree

#else

#ifndef __CUDACC_RTC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using qudaStream_t = cudaStream_t;
using qudaMemcpyKind = cudaMemcpyKind;
using qudaError_t = cudaError_t;
using qudaEvent_t = cudaEvent_t;
using qudaMemoryType = cudaMemoryType;
using qudaIpcMemHandle_t = cudaIpcMemHandle_t;
using qudaDeviceProp = cudaDeviceProp;
#define qudaHostGetDevicePointer cudaHostGetDevicePointer
#define qudaStreamCreate cudaStreamCreate
#define qudaStreamDestroy cudaStreamDestroy
#define qudaEventCreate cudaEvenCreate
#define qudaEventRecord cudaEvenRecord
#define qudaEventSynchronize cudaEventSynchronize
#define qudaEventDisableTiming cudaEventDisableTiming
#define qudaEventDestroy cudaEventDestroy
#define qudaDeviceSynchronize cudaDeviceSynchronize
#define qudaMemset cudaMemset
#define qudaSuccess cudaSuccess
#define qudaErrorNotReady cudaErrorNotReady
#define qudaErrorUnknown cudaErrorUnknown
#define qudaSuccessjit CUDA_SUCCESS
#define qudaEventElapsedTime cudaEventElapsedTime
#define qudaEventDestroy cudaEventDestroy
#define qudaGetLastError cudaGetLastError
#define qudaGetErrorString cudaGetErrorString
#define qudaHostRegister cudaHostRegister
#define qudaHostAlloc cudaHostAlloc
#define qudaHostRegisterMapped cudaHostRegisterMapped
#define qudaHostRegisterPortable cudaHostRegisterPortable
#define QUDA_ERROR_INVALID_VALUE CUDA_ERROR_INVALID_VALUE
#define qudaIpcGetMemHandle cudaIpcGetMemHandle
#define qudaIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define qudaIpcCloseMemHandle cudaIpcCloseMemHandle
#define qudaEventQuery cudaEventQuery
#define qudaMemcpyHostToDevice cudaMemcpyHostToDevice
#define qudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define qudaFuncSetCacheConfig cudaFuncSetCacheConfig
#define qudaFuncCachePreferL1 cudaFuncCachePreferL1
#define qudaGetErrorName cudaGetErrorName
#define qudarand curand
#define qudaFree cudaFree

#endif

/**
   @file quda_cuda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
 */

namespace quda {

  /**
     @brief Wrapper around qudaMemcpy used for auto-profiling.  Do not
     call directly, rather call macro below which will grab the
     location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind,
		   const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamSynchronize or cuStreamSynchronize
     @param[in] stream Stream which we are synchronizing
  */
  qudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line);
}

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

#define qudaStreamSynchronize(stream)                                                                                  \
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpy(dst, src, count, kind) \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpyAsync(dst, src, count, kind, stream) \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream) \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemset(ptr, value, count)                                                                                  \
  ::quda::qudaMemset_(ptr, value, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemsetAsync(ptr, value, count, stream)                                                                     \
  ::quda::qudaMemsetAsync_(ptr, value, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

namespace quda {

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent
     Adds auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpy2DAsync or driver API equivalent
     Adds auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                          qudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line);

  /**
     @brief Wrapper around cudaMemset or driver API equivalent.
     Adds auto-profiling support.
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
   */
  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent.
     Adds auto-profiling support.
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
     @param[in] stream  Stream to issue memset
   */
  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line);

  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] gridDim Grid dimensions
     @param[in] blockDim Block dimensions
     @param[in] args Arguments
     @param[in] sharedMem Shared memory requested per thread block
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
                               qudaStream_t stream);

  /**
     @brief Wrapper around qudaEventQuery or qudaEventQuery
     @param[in] event Event we are querying
     @return Status of event query
   */
  qudaError_t qudaEventQuery(qudaEvent_t &event);

  /**
     @brief Wrapper around hipEventRecord or hipEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream = 0);

  /**
     @brief Wrapper around qudaEventRecord or qudaEventRecord
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  qudaError_t qudaStreamWaitEvent(qudaStream_t stream, qudaEvent_t event, unsigned int flags);

  /**
     @brief Wrapper around qudaEventSynchronize or qudaEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
   */
  qudaError_t qudaEventSynchronize(qudaEvent_t &event);

  /**
     @brief Wrapper around qudaDeviceSynchronize or cuDeviceSynchronize
   */
  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around cudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  cudaError_t qudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);

  /**
     @brief Wrapper around cudaFuncGetAttributes
     @param[in] attr the cudaFuncGetAttributes object to store the output
     @param[in] func Function for which we are setting the attribute
  */
  cudaError_t qudaFuncGetAttributes(cudaFuncAttributes &attr, const void* func);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();
  
} // namespace quda

#define qudaDeviceSynchronize() \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
