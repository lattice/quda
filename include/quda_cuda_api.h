#pragma once

#ifndef __CUDACC_RTC__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <quda_cuda_api.h>

/**
   @file quda_cuda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
 */

namespace quda {

  /**
     @brief Wrapper around hipMemcpy used for auto-profiling.  Do not
     call directly, rather call macro below which will grab the
     location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, hipMemcpyKind kind,
		   const char *func, const char *file, const char *line);

}

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

namespace quda {

  /**
     @brief Wrapper around hipMemcpyAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, hipMemcpyKind kind, const hipStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around hipMemcpy2DAsync or driver API equivalent
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
                          size_t width, size_t hieght, hipMemcpyKind kind, const hipStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around hipLaunchKernel
     @param[in] func Device function symbol
     @param[in] gridDim Grid dimensions
     @param[in] blockDim Block dimensions
     @param[in] args Arguments
     @param[in] sharedMem Shared memory requested per thread block
     @param[in] stream Stream identifier
  */
  hipError_t qudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, hipStream_t stream);

  /**
     @brief Wrapper around hipEventQuery or hipEventQuery
     @param[in] event Event we are querying
     @return Status of event query
   */
  hipError_t qudaEventQuery(hipEvent_t &event);

  /**
     @brief Wrapper around hipEventRecord or hipEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  hipError_t qudaEventRecord(hipEvent_t &event, hipStream_t stream=0);

  /**
     @brief Wrapper around hipEventRecord or hipEventRecord
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  hipError_t qudaStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);

  /**
     @brief Wrapper around hipStreamSynchronize or hipStreamSynchronize
     @param[in] stream Stream which we are synchronizing with respect to
   */
  hipError_t qudaStreamSynchronize(hipStream_t &stream);

  /**
     @brief Wrapper around hipEventSynchronize or hipEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
   */
  hipError_t qudaEventSynchronize(hipEvent_t &event);

  /**
     @brief Wrapper around hipDeviceSynchronize or cuDeviceSynchronize
   */
  hipError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around cudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  hipError_t qudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)
#define qudaDeviceSynchronize() \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#endif
