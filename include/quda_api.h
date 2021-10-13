#pragma once

#include <quda_define.h>
#include <string>
#include <enum_quda.h>

/**
   @file quda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
 */

enum qudaMemcpyKind {
  qudaMemcpyHostToHost,
  qudaMemcpyHostToDevice,
  qudaMemcpyDeviceToHost,
  qudaMemcpyDeviceToDevice,
  qudaMemcpyDefault
};

namespace quda
{

  class TuneParam;

  struct qudaStream_t {
    int idx;
  };

  struct qudaEvent_t {
    void *event;
  };

  /**
     @brief Wrapper around cudaMemcpy or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line);

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent for peer-to-peer copies
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream, const char *func,
                           const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemset or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
   */
  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemset2D or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] Pitch in bytes
     @param[in] value Value to set for each byte of specified memory
     @param[in] width Width in bytes
     @param[in] height Height in bytes
   */
  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height, const char *func,
                     const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
     @param[in] stream Stream to issue memset
   */
  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] Pitch in bytes
     @param[in] value Value to set for each byte of specified memory
     @param[in] width Width in bytes
     @param[in] height Height in bytes
     @param[in] stream Stream to issue memset
   */
  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemPrefetchAsync or driver API equivalent
     @param[out] ptr Starting address pointer to be prefetched
     @param[in] count Size in bytes to prefetch
     @param[in] mem_space Memory space to prefetch to
     @param[in] stream Stream to issue prefetch
   */
  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line);

  /**
     @brief Return instance of an event.
  */
  qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line);

  /**
     @brief Return instance of an event that can be used for timing.
  */
  qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line);

  /**
     @brief Return elapsed time in seconds between two events
  */
  float qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop, const char *func, const char *file,
                              const char *line);

  /**
     @brief Destroy the event
  */
  void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventQuery or cuEventQuery with built-in error checking
     @param[in] event Event we are querying
     @return true if event has been reached
   */
  bool qudaEventQuery_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord with
     built-in error checking
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  void qudaEventRecord_(qudaEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamWaitEvent or cuStreamWaitEvent
     with built-in error checking
     @param[in,out] stream Stream which we are instructing to wait
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t event, unsigned int flags, const char *func,
                            const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventSynchronize or cuEventSynchronize
     with built-in error checking
     @param[in] event Event which we are synchronizing with respect to
   */
  void qudaEventSynchronize_(const qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamSynchronize or
     cuStreamSynchronize with built-in error checking
     @param[in] stream Stream which we are synchronizing
  */
  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize or
     cuDeviceSynchronize with built-in error checking
   */
  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaGetSymbolAddress with built in error
     checking.  Returns the address of symbol on the device; symbol
     is a variable that resides in global memory space.

     @param[in] symbol Global variable or string symbol to search for
     @return Return device pointer associated with symbol
  */
  void *qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line);

  /**
     @brief Get the last error recorded by the target runtime.  By
     calling this, we reset the last error.
  */
  qudaError_t qudaGetLastError();

  /**
     @brief Get the error string associated with the last error that
     was thrown by the target runtime
  */
  std::string qudaGetLastErrorString();

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

#define qudaMemcpy(dst, src, count, kind)                                                                              \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpyAsync(dst, src, count, kind, stream)                                                                 \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpyP2PAsync(dst, src, count, stream)                                                                    \
  ::quda::qudaMemcpyP2PAsync_(dst, src, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset(ptr, value, count)                                                                                  \
  ::quda::qudaMemset_(ptr, value, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2D(ptr, pitch, value, width, height)                                                                 \
  ::quda::qudaMemset2D_(ptr, pitch, value, width, height, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemsetAsync(ptr, value, count, stream)                                                                     \
  ::quda::qudaMemsetAsync_(ptr, value, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2DAsync(ptr, pitch, value, width, height, stream)                                                    \
  ::quda::qudaMemset2DAsync_(ptr, pitch, value, width, height, stream, __func__, quda::file_name(__FILE__),            \
                             __STRINGIFY__(__LINE__))

#define qudaMemPrefetchAsync(ptr, count, mem_space, stream)                                                            \
  ::quda::qudaMemPrefetchAsync_(ptr, count, mem_space, stream, __func__, quda::file_name(__FILE__),                    \
                                __STRINGIFY__(__LINE__))

#define qudaEventCreate() ::quda::qudaEventCreate_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaChronoEventCreate()                                                                                        \
  ::quda::qudaChronoEventCreate_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventElapsedTime(start, stop)                                                                              \
  ::quda::qudaEventElapsedTime_(start, stop, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventDestroy(event)                                                                                        \
  ::quda::qudaEventDestroy_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventQuery(event)                                                                                          \
  ::quda::qudaEventQuery_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventRecord(event, stream)                                                                                 \
  ::quda::qudaEventRecord_(event, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamWaitEvent(stream, event, flags)                                                                      \
  ::quda::qudaStreamWaitEvent_(stream, event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventSynchronize(event)                                                                                    \
  ::quda::qudaEventSynchronize_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamSynchronize(stream)                                                                                  \
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaDeviceSynchronize()                                                                                        \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaGetSymbolAddress(symbol)                                                                                   \
  ::quda::qudaGetSymbolAddress_(symbol, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
