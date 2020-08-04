#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>

// if this macro is defined then we use the driver API, else use the
// runtime API.  Typically the driver API has 10-20% less overhead
#define USE_DRIVER_API

// if this macro is defined then we profile the CUDA API calls
//#define API_PROFILE

#ifdef API_PROFILE
#define PROFILE(f, idx)                                 \
  apiTimer.TPSTART(idx);				\
  f;                                                    \
  apiTimer.TPSTOP(idx);
#else
#define PROFILE(f, idx) f;
#endif

namespace quda {

#ifdef USE_DRIVER_API
  static TimeProfile apiTimer("CUDA API calls (driver)");
#else
  static TimeProfile apiTimer("CUDA API calls (runtime)");
#endif

  class QudaMem : public Tunable
  {

    void *dst;
    const void *src;
    const size_t count;
    const int value;
    const bool copy;
    const hipMemcpyKind kind;
    const bool async;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, hipMemcpyKind kind, bool async, const char *func,
                   const char *file, const char *line) :
      dst(dst),
      src(src),
      count(count),
      value(0),
      copy(true),
      kind(kind),
      async(async)
    {
      if (!async) {
        switch (kind) {
        case hipMemcpyDeviceToHost:   name = "hipMemcpyDeviceToHost";   break;
        case hipMemcpyHostToDevice:   name = "hipMemcpyHostToDevice";   break;
        case hipMemcpyHostToHost:     name = "hipMemcpyHostToHost";     break;
        case hipMemcpyDeviceToDevice: name = "hipMemcpyDeviceToDevice"; break;
        case hipMemcpyDefault:        name = "hipMemcpyDefault";        break;
        default: errorQuda("Unsupported hipMemcpyKind %d", kind);
        }
      } else {
        switch(kind) {
        case hipMemcpyDeviceToHost:   name = "hipMemcpyAsyncDeviceToHost";   break;
        case hipMemcpyHostToDevice:   name = "hipMemcpyAsyncHostToDevice";   break;
        case hipMemcpyHostToHost:     name = "hipMemcpyAsyncHostToHost";     break;
        case hipMemcpyDeviceToDevice: name = "hipMemcpyAsyncDeviceToDevice"; break;
        case hipMemcpyDefault:        name = "hipMemcpyAsyncDefault";        break;
        default: errorQuda("Unsupported hipMemcpyKind %d", kind);
        }
      }
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);
    }

    inline QudaMem(void *dst, int value, size_t count, bool async, const char *func, const char *file, const char *line) :
      dst(dst),
      src(nullptr),
      count(count),
      value(value),
      copy(false),
      kind(hipMemcpyDefault),
      async(async)
    {
      name = !async ? "cudaMemset" : "cudaMemsetAsync";
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);
    }

    inline void apply(const hipStream_t &stream) {
      tuneLaunch(*this, getTuning(), getVerbosity());
      if (copy) {
        if (async) {
#ifdef USE_DRIVER_API
        switch (kind) {
        case hipMemcpyDeviceToHost:
          PROFILE(hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
          break;
        case hipMemcpyHostToDevice:
          PROFILE(hipMemcpyHtoDAsync((hipDeviceptr_t)dst, const_cast<void *>(src), count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
          break;
        case hipMemcpyDeviceToDevice:
          PROFILE(hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
          break;
        default:
          errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
        }
#else
        PROFILE(hipMemcpyAsync(dst, src, count, kind, stream),
                kind == hipMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
        } else {
#ifdef USE_DRIVER_API
          switch (kind) {
          case hipMemcpyDeviceToHost:   hipMemcpyDtoH(dst, (hipDeviceptr_t)src, count);              break;
          case hipMemcpyHostToDevice:   hipMemcpyHtoD((hipDeviceptr_t)dst, const_cast<void *>(src), count);              break;
          case hipMemcpyHostToHost:     memcpy(dst, src, count);                                 break;
          case hipMemcpyDeviceToDevice: hipMemcpyDtoD((hipDeviceptr_t)dst, (hipDeviceptr_t)src, count); break;
          case hipMemcpyDefault:        hipMemcpy(dst, src, count, kind);     break;
          default:
            errorQuda("Unsupported hipMemcpy %d", kind);
          }
#else
          hipMemcpy(dst, src, count, kind);
#endif
        }
      } else {
#ifdef USE_DRIVER_API
        if (async)
          hipMemsetD32Async((hipDeviceptr_t)dst, value, count / 4, stream);
        else
          hipMemsetD32((hipDeviceptr_t)dst, value, count / 4);
#else
        if (async)
          hipMemsetAsync(dst, value, count, stream);
        else
          hipMemset(dst, value, count);
#endif
      }
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const {
      char vol[128];
      strcpy(vol,"bytes=");
      u64toa(vol+6, (uint64_t)count);
      return TuneKey(vol, name, aux);
    }

    long long flops() const { return 0; }
    long long bytes() const { return kind == hipMemcpyDeviceToDevice ? 2*count : count; }

  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, hipMemcpyKind kind,
                   const char *func, const char *file, const char *line) {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, false, func, file, line);
    copy.apply(0);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, hipMemcpyKind kind, const hipStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == hipMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, true, func, file, line);
      copy.apply(stream);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case hipMemcpyDeviceToHost:
        PROFILE(hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case hipMemcpyHostToDevice:
        PROFILE(hipMemcpyHtoDAsync((hipDeviceptr_t)dst, const_cast<void *>(src), count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case hipMemcpyDeviceToDevice:
        PROFILE(hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(hipMemcpyAsync(dst, src, count, kind, stream),
              kind == hipMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t height, hipMemcpyKind kind, const hipStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    hip_Memcpy2D param;
    param.srcPitch = spitch;
    param.srcY = 0;
    param.srcXInBytes = 0;
    param.dstPitch = dpitch;
    param.dstY = 0;
    param.dstXInBytes = 0;
    param.WidthInBytes = width;
    param.Height = height;

    switch (kind) {
    case hipMemcpyDeviceToHost:
      param.srcDevice = (hipDeviceptr_t)src;
#ifdef __HIP_PLATFORM_HCC__ 
      param.srcMemoryType = hipMemoryTypeDevice;
      param.dstMemoryType = hipMemoryTypeHost;
#else
      param.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      param.dstMemoryType = CU_MEMORYTYPE_HOST;
#endif
      param.dstHost = dst;
      break;
    default:
      errorQuda("Unsupported cuMemcpyType2DAsync %d", kind);
    }
//    printfQuda("driver_api is not supported in hipMemcpy2DAsync\n");
    PROFILE(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);    
#else
    PROFILE(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#endif
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, false, func, file, line);
    set.apply(0);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const hipStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, true, func, file, line);
    copy.apply(0);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  hipError_t qudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, hipStream_t stream)
  {
    // no driver API variant here since we have C++ functions
    printfQuda("this kernel can not be used in HIP\n");hipError_t error = hipErrorUnknown;
//    PROFILE(hipError_t error = hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream), QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != hipSuccess && !activeTuning()) errorQuda("(CUDA) %s", hipGetErrorString(error));
    return error;
  }

  hipError_t qudaEventQuery(hipEvent_t &event)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    case hipErrorNotReady: // this is the only return value care about
      return hipErrorNotReady;
    default:
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("hipEventQuery returned error %s", str);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    return error;
#endif
  }

  hipError_t qudaEventRecord(hipEvent_t &event, hipStream_t stream)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    default: // should always return successful
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("cuEventrecord returned error %s", str);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    return error;
#endif
  }

  hipError_t qudaStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    default: // should always return successful
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("hipStreamWaitEvent returned error %s", str);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    return error;
#endif
  }

  hipError_t qudaStreamSynchronize(hipStream_t &stream)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    default: // should always return successful
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("hipStreamSynchronize returned error %s", str);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    return error;
#endif
  }

  hipError_t qudaEventSynchronize(hipEvent_t &event)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    default: // should always return successful
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("hipEventSynchronize returned error %s", str);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    return error;
#endif
  }

  hipError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(hipError_t error = hipCtxSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    switch (error) {
    case hipSuccess:
      return hipSuccess;
    default: // should always return successful
      const char *str;
      str=hipGetErrorName(error);
      errorQuda("hipCtxSynchronize returned error %s (%s:%s in %s())\n", str, file, line, func);
    }
    return hipErrorUnknown;
#else
    PROFILE(hipError_t error = hipDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != hipSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    return error;
#endif
  }

#if (CUDA_VERSION >= 9000)
  cudaError_t qudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncSetAttribute(func, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    return error;
  }
#endif

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
