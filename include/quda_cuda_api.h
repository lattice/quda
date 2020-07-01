#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <quda_cuda_api.h>

using qudaStream_t = cudaStream_t;

/**
   @file quda_cuda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
 */

namespace quda {

  /**
     @brief Wrapper around cudaMemcpy used for auto-profiling.  Do not
     call directly, rather call macro below which will grab the
     location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		   const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamSynchronize or cuStreamSynchronize
     @param[in] stream Stream which we are synchronizing
  */
  cudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line);
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
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
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
                          cudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
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
  cudaError_t qudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
                               qudaStream_t stream);

  /**
     @brief Wrapper around cudaEventQuery or cuEventQuery
     @param[in] event Event we are querying
     @return Status of event query
   */
  cudaError_t qudaEventQuery(cudaEvent_t &event);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  cudaError_t qudaEventRecord(cudaEvent_t &event, qudaStream_t stream = 0);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  cudaError_t qudaStreamWaitEvent(qudaStream_t stream, cudaEvent_t event, unsigned int flags);

  /**
     @brief Wrapper around cudaEventSynchronize or cuEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
   */
  cudaError_t qudaEventSynchronize(cudaEvent_t &event);

  /**
     @brief Wrapper around cudaDeviceSynchronize or cuDeviceSynchronize
   */
  cudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around cudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  cudaError_t qudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

#define qudaDeviceSynchronize() \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#endif
