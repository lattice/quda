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
    const qudaMemcpyKind kind;
    const bool async;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, qudaMemcpyKind kind, bool async, const char *func,
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
        case qudaMemcpyDeviceToHost:   name = "qudaMemcpyDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "qudaMemcpyHostToDevice";   break;
        case cudaMemcpyHostToHost:     name = "cudaMemcpyHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyDeviceToDevice"; break;
        case cudaMemcpyDefault:        name = "cudaMemcpyDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyKind %d", kind);
        }
      } else {
        switch(kind) {
        case qudaMemcpyDeviceToHost:   name = "cudaMemcpyAsyncDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "cudaMemcpyAsyncHostToDevice";   break;
        case cudaMemcpyHostToHost:     name = "cudaMemcpyAsyncHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "cudaMemcpyAsyncDeviceToDevice"; break;
        case cudaMemcpyDefault:        name = "cudaMemcpyAsyncDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyKind %d", kind);
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
      kind(cudaMemcpyDefault),
      async(async)
    {
      name = !async ? "cudaMemset" : "cudaMemsetAsync";
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);
    }

    inline void apply(const qudaStream_t &stream) {
      tuneLaunch(*this, getTuning(), getVerbosity());
      if (copy) {
        if (async) {
#ifdef USE_DRIVER_API
          switch (kind) {
          case qudaMemcpyDeviceToHost:
            PROFILE(cuMemcpyDtoHAsync(dst, (QUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
            break;
          case qudaMemcpyHostToDevice:
            PROFILE(cuMemcpyHtoDAsync((QUdeviceptr)dst, src, count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
            break;
          case qudaMemcpyDeviceToDevice:
            PROFILE(cuMemcpyDtoDAsync((QUdeviceptr)dst, (QUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
            break;
          default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
          }
#else
          PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream),
                  kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
        } else {
#ifdef USE_DRIVER_API
          switch (kind) {
          case qudaMemcpyDeviceToHost: cuMemcpyDtoH(dst, (QUdeviceptr)src, count); break;
          case qudaMemcpyHostToDevice: cuMemcpyHtoD((QUdeviceptr)dst, src, count); break;
          case cudaMemcpyHostToHost: memcpy(dst, src, count); break;
          case qudaMemcpyDeviceToDevice: cuMemcpyDtoD((QUdeviceptr)dst, (QUdeviceptr)src, count); break;
          case cudaMemcpyDefault: cuMemcpy((QUdeviceptr)dst, (QUdeviceptr)src, count); break;
          default: errorQuda("Unsupported cudaMemcpyType %d", kind);
          }
#else
          cudaMemcpy(dst, src, count, kind);
#endif
        }
      } else {
#ifdef USE_DRIVER_API
        if (async)
          cuMemsetD32Async((QUdeviceptr)dst, value, count / 4, stream);
        else
          cuMemsetD32((QUdeviceptr)dst, value, count / 4);
#else
        if (async)
          cudaMemsetAsync(dst, value, count, stream);
        else
          cudaMemset(dst, value, count);
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
    long long bytes() const { return kind == qudaMemcpyDeviceToDevice ? 2*count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind,
                   const char *func, const char *file, const char *line) {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, false, func, file, line);
    copy.apply(0);
    qudaError_t error = cudaGetLastError();
    if (error != qudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, true, func, file, line);
      copy.apply(stream);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case qudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (QUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case qudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((QUdeviceptr)dst, src, count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case qudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((QUdeviceptr)dst, (QUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream),
              kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t height, qudaMemcpyKind kind, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    CUDA_MEMCPY2D param;
    param.srcPitch = spitch;
    param.srcY = 0;
    param.srcXInBytes = 0;
    param.dstPitch = dpitch;
    param.dstY = 0;
    param.dstXInBytes = 0;
    param.WidthInBytes = width;
    param.Height = height;

    switch (kind) {
    case qudaMemcpyDeviceToHost:
      param.srcDevice = (QUdeviceptr)src;
      param.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      param.dstHost = dst;
      param.dstMemoryType = CU_MEMORYTYPE_HOST;
      break;
    default:
      errorQuda("Unsupported cuMemcpyType2DAsync %d", kind);
    }
    PROFILE(cuMemcpy2DAsync(&param, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#else
    PROFILE(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#endif
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, false, func, file, line);
    set.apply(0);
    qudaError_t error = cudaGetLastError();
    if (error != qudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, true, func, file, line);
    copy.apply(0);
    qudaError_t error = cudaGetLastError();
    if (error != qudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  qudaError_t qudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, qudaStream_t stream)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(qudaError_t error = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream), QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != qudaSuccess && !activeTuning()) errorQuda("(CUDA) %s", cudaGetErrorString(error));
    return error;
  }

  qudaError_t qudaEventQuery(qudaEvent_t &event)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    case CUDA_ERROR_NOT_READY: // this is the only return value care about
      return cudaErrorNotReady;
    default:
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventQuery returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = cudaEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    return error;
#endif
  }

  qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventrecord returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = qudaEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    return error;
#endif
  }

  qudaError_t qudaStreamWaitEvent(qudaStream_t stream, qudaEvent_t event, unsigned int flags)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamWaitEvent returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = cudaStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    return error;
#endif
  }

  qudaError_t qudaStreamSynchronize(qudaStream_t &stream)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamSynchronize returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = cudaStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    return error;
#endif
  }

  qudaError_t qudaEventSynchronize(qudaEvent_t &event)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventSynchronize returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = qudaEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    return error;
#endif
  }

  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(QUresult error = cuCtxSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    switch (error) {
    case QUDA_SUCCESS:
      return qudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuCtxSynchronize returned error %s (%s:%s in %s())\n", str, file, line, func);
    }
    return cudaErrorUnknown;
#else
    PROFILE(qudaError_t error = qudaDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != qudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
#endif
  }

#if (CUDA_VERSION >= 9000)
  qudaError_t qudaFuncSetAttribute(const void* func, qudaFuncAttribute attr, int value)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(qudaError_t error = cudaFuncSetAttribute(func, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    return error;
  }
#endif

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
