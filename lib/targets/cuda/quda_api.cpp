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

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, stream), QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != cudaSuccess && !activeTuning()) errorQuda("(CUDA) %s", cudaGetErrorString(error));
    return error == cudaSuccess ? qudaSuccess : qudaError;
  }

  class QudaMem : public Tunable
  {
    void *dst;
    const void *src;
    const size_t count;
    const int value;
    const bool copy;
    const cudaMemcpyKind kind;
    const bool async;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, cudaMemcpyKind kind, bool async, const char *func,
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
        case cudaMemcpyDeviceToHost:   name = "cudaMemcpyDeviceToHost";   break;
        case cudaMemcpyHostToDevice:   name = "cudaMemcpyHostToDevice";   break;
        case cudaMemcpyHostToHost:     name = "cudaMemcpyHostToHost";     break;
        case cudaMemcpyDeviceToDevice: name = "cudaMemcpyDeviceToDevice"; break;
        case cudaMemcpyDefault:        name = "cudaMemcpyDefault";        break;
        default: errorQuda("Unsupported cudaMemcpyKind %d", kind);
        }
      } else {
        switch(kind) {
        case cudaMemcpyDeviceToHost:   name = "cudaMemcpyAsyncDeviceToHost";   break;
        case cudaMemcpyHostToDevice:   name = "cudaMemcpyAsyncHostToDevice";   break;
        case cudaMemcpyHostToHost:     name = "cudaMemcpyAsyncHostToHost";     break;
        case cudaMemcpyDeviceToDevice: name = "cudaMemcpyAsyncDeviceToDevice"; break;
        case cudaMemcpyDefault:        name = "cudaMemcpyAsyncDefault";        break;
        default: errorQuda("Unsupported cudaMemcpyKind %d", kind);
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

    inline void apply(const qudaStream_t &stream)
    {
      tuneLaunch(*this, getTuning(), getVerbosity());
      if (copy) {
        if (async) {
#ifdef USE_DRIVER_API
          switch (kind) {
          case cudaMemcpyDeviceToHost:
            PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
            break;
          case cudaMemcpyHostToDevice:
            PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
            break;
          case cudaMemcpyDeviceToDevice:
            PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
            break;
          case cudaMemcpyDefault:
            PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
            break;
          default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
          }
#else
          QudaProfileType type;
          switch (kind) {
          case cudaMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
          case cudaMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
          case cudaMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
          case cudaMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
          default: errorQuda("Unsupported cudaMemcpyTypeAsync %d", kind);
          }

          PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream), type);
#endif
        } else {
#ifdef USE_DRIVER_API
          switch (kind) {
          case cudaMemcpyDeviceToHost: cuMemcpyDtoH(dst, (CUdeviceptr)src, count); break;
          case cudaMemcpyHostToDevice: cuMemcpyHtoD((CUdeviceptr)dst, src, count); break;
          case cudaMemcpyHostToHost: memcpy(dst, src, count); break;
          case cudaMemcpyDeviceToDevice: cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count); break;
          case cudaMemcpyDefault: cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count); break;
          default: errorQuda("Unsupported cudaMemcpyType %d", kind);
          }
#else
          cudaMemcpy(dst, src, count, kind);
#endif
        }
      } else {
#ifdef USE_DRIVER_API
        if (async)
          cuMemsetD32Async((CUdeviceptr)dst, value, count / 4, stream);
        else
          cuMemsetD32((CUdeviceptr)dst, value, count / 4);
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
    long long bytes() const { return kind == cudaMemcpyDeviceToDevice ? 2*count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                   const char *func, const char *file, const char *line) {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, false, func, file, line);
    copy.apply(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == cudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, true, func, file, line);
      copy.apply(stream);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case cudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case cudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case cudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      case cudaMemcpyDefault:
        PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream),
              kind == cudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                          cudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line)
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
    case cudaMemcpyDeviceToHost:
      param.srcDevice = (CUdeviceptr)src;
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
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, true, func, file, line);
    copy.apply(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream, const char *func,
                             const char *file, const char *line)
  {
    int dev_id = 0;
    if (mem_space == QUDA_CUDA_FIELD_LOCATION)
      dev_id = comm_gpuid();
    else if (mem_space == QUDA_CPU_FIELD_LOCATION)
      dev_id = cudaCpuDeviceId;
    else
      errorQuda("Invalid QudaFieldLocation.");

    cudaError_t error = cudaMemPrefetchAsync(ptr, count, dev_id, stream);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  bool qudaEventQuery_(cudaEvent_t &event, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case CUDA_SUCCESS:         return true;
    case CUDA_ERROR_NOT_READY: return false;
    default:
      {
        const char *str;
        cuGetErrorName(error, &str);
        errorQuda("cuEventQuery returned error %s", str);
      }
    }
#else
    PROFILE(cudaError_t error = cudaEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case cudaSuccess:       return true;
    case cudaErrorNotReady: return false;
    default:
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    }
#endif
    return false;
  }

  void qudaEventRecord_(cudaEvent_t &event, qudaStream_t stream,
                        const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventRecord returned error %s", str);
    }
#else
    PROFILE(cudaError_t error = cudaEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, cudaEvent_t event, unsigned int flags,
                            const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamWaitEvent returned error %s", str);
    }
#else
    PROFILE(cudaError_t error = cudaStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaEventSynchronize_(cudaEvent_t &event, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventSynchronize returned error %s", str);
    }
#else
    PROFILE(cudaError_t error = cudaEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("(CUDA) cuStreamSynchronize returned error %s\n (%s:%s in %s())\n", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuCtxSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuCtxSynchronize returned error %s (%s:%s in %s())\n", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaFuncSetAttribute_(const void* kernel, cudaFuncAttribute attr, int value,
                             const char *func, const char *file, const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncSetAttribute(kernel, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaFuncGetAttributes_(cudaFuncAttributes &attr, const void* kernel,
                              const char *func, const char *file, const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncGetAttributes(&attr, kernel), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
