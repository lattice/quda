#pragma once

#ifndef __CUDACC_RTC__
#include <quda_backend.h>
#include <quda_cuda_api.h>

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
     @brief Wrapper around qudaMemcpyAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, 
			const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemcpy2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t hieght, qudaMemcpyKind kind, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemset_(void *dst, int val, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemsetAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemsetAsync_(void *dst, int val, size_t count, const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemset2D or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2D_(void* dst, size_t pitch, int val, size_t width, size_t height, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2DAsync_(void* dst, size_t pitch, int val, size_t width, size_t height, const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] gridDim Grid dimensions
     @param[in] blockDim Block dimensions
     @param[in] args Arguments
     @param[in] sharedMem Shared memory requested per thread block
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, qudaStream_t stream);

  /**
     @brief Wrapper around qudaEventQuery or quEventQuery
     @param[in] event Event we are querying
     @return Status of event query
   */
  qudaError_t qudaEventQuery(qudaEvent_t &event);

  /**
     @brief Wrapper around qudaEventRecord or quEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream=0);

  /**
     @brief Wrapper around qudaEventSynchronize or quEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
   */
  qudaError_t qudaEventSynchronize(qudaEvent_t &event);


  /**
     @brief Wrapper around qudaEventRecord or quEventRecord
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  qudaError_t qudaStreamWaitEvent(qudaStream_t stream, qudaEvent_t event, unsigned int flags);

  /**
     @brief Wrapper around qudaStreamSynchronize or quStreamSynchronize
     @param[in] stream Stream which we are synchronizing with respect to
   */
  qudaError_t qudaStreamSynchronize(qudaStream_t &stream);

  /**
     @brief Wrapper around qudaDeviceSynchronize or quDeviceSynchronize
   */
  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  //QUDA texture objects
  /**
     @brief Wrapper around cudaCreateTextureObject
  */
  qudaError_t qudaCreateTextureObject_(qudaTextureObject_t* pTexObject, const qudaResourceDesc* pResDesc, const qudaTextureDesc* pTexDesc, const qudaResourceViewDesc* pResViewDesc, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDestroyTextureObject
  */
  qudaError_t qudaDestroyTextureObject_(qudaTextureObject_t pTexObject, const char *func, const char *file, const char *line);

  //QUDA Device
  /**
     @brief Wrapper around cudaDeviceCanAccessPeer
  */
  qudaError_t qudaDeviceCanAccessPeer_(int* canAccessPeer, int device, int peerDevice, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceGetStreamPriorityRange
  */
  qudaError_t qudaDeviceGetStreamPriorityRange_(int* leastPriority, int* greatestPrioriy, const char *func, const char *file, const char *line);
  
  

#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around qudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  qudaError_t qudaFuncSetAttribute(const void* func, qudaFuncAttribute attr, int value);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

//START Memcpy
//-------------------------------------------------------------------------------------
#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemcpy(dst, src, count, kind) \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemcpyAsync(dst, src, count, kind, stream) \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream) \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//END Memcpy
//-------------------------------------------------------------------------------------

//START Memset
//-------------------------------------------------------------------------------------
#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemset(dst, val, count)					\
  ::quda::qudaMemset_(dst, val, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemsetAsync(dst, val, count, stream)				\
  ::quda::qudaMemsetAsync_(dst, val, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemset2D(dst, val, pitch, width, height)			\
  ::quda::qudaMemset2D_(dst, val, pitch, width, height, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaMemset2DAsync(dst, val, pitch, width, height, stream)		\
  ::quda::qudaMemset2DAsync_(dst, val, pitch, width, height, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//END Memset
//-------------------------------------------------------------------------------------

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaDeviceSynchronize() \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

//START texture
//-------------------------------------------------------------------------------------
#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc) \
  ::quda::qudaCreateTextureObject_(pTexObject, pResDesc, pTexDesc, pResViewDesc, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaDestroyTextureObject(pTexObject) \
  ::quda::qudaDestroyTextureObject_(pTexObject, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//END texture
//-------------------------------------------------------------------------------------

//START Device
//-------------------------------------------------------------------------------------
#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice) \
  ::quda::qudaDeviceCanAccessPeer_(canAccessPeer, device, peerDevice, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority) \
  ::quda::qudaDeviceGetStreamPriorityRange_(leastPriority, greatestPriority, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#endif
