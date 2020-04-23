#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>

// if this macro is defined then we use the driver API, else use the
// runtime API.  Typically the driver API has 10-20% less overhead
#define USE_DRIVER_API

// if this macro is defined then we profile the CUDA API calls
//#define API_PROFILE

#ifdef API_PROFILE
#define PROFILE(f, idx)                                                                                                \
  apiTimer.TPSTART(idx);                                                                                               \
  f;                                                                                                                   \
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

    inline void apply(const cudaStream_t &stream) {
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
          default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
          }
#else
          PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream),
                  kind == cudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
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

  void qudaMemPrefetchAsync_(const void *devPtr, size_t count, int dstDevice, const cudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    cudaError_t error = cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const char *func, const char *file,
                   const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, false, func, file, line);
    copy.apply(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpyNoTune_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const char *func,
                         const char *file, const char *line)
  {
    if (count == 0) return;
    cudaError_t error = cudaMemcpy(dst, src, count, kind);
    if (error != cudaSuccess && !activeTuning())
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
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, kind, stream),
              kind == cudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpyAsyncNoTune_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
			      const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

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
    default:
      errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
    }
#else
    cudaMemcpyAsync(dst, src, count, kind, stream);
#endif
}

  
  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                     cudaMemcpyKind kind, const char *func, const char *file, const char *line)
  {
    PROFILE(cudaError_t error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind),
	    kind == cudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_2D : QUDA_PROFILE_MEMCPY_H2D_2D);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpy2DNoTune_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
			   cudaMemcpyKind kind, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
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

  void qudaMemcpy2DAsyncNoTune_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
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
    cuMemcpy2DAsync(&param, stream);
#else
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
#endif
  }


  qudaError_t qudaGetLastError_(const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaGetLastError();
    return error;
  }

  const char *qudaGetErrorString_(qudaError_t &error, const char *func, const char *file, const char *line)
  {
    return cudaGetErrorString(error);
  }

  void qudaMemset_(void *dst, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(dst, value, count, false, func, file, line);
    set.apply(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *dst, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, value, count, true, func, file, line);
    copy.apply(0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemset2D_(void *dst, size_t pitch, int val, size_t width, size_t height, const char *func, const char *file,
                     const char *line)
  {
    if (pitch == 0) return;
    cudaError_t error = cudaMemset2D(dst, pitch, val, width, height);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemset2DAsync_(void *dst, size_t pitch, int val, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
    if (pitch == 0) return;
    cudaError_t error = cudaMemset2DAsync(dst, pitch, val, width, height, stream);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  qudaError_t qudaLaunchKernel_(const void *func_arg, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
                                qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    cudaError_t error;
    // no driver API variant here since we have C++ functions
    PROFILE(error = cudaLaunchKernel(func_arg, gridDim, blockDim, args, sharedMem, stream), QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != cudaSuccess && !activeTuning() && !activeTuning()) errorQuda("(CUDA) %s", cudaGetErrorString(error));
    return error;
  }

  qudaError_t qudaEventCreate_(qudaEvent_t *event, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaEventCreate((CUevent *)event);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaEventCreateWithFlags_(qudaEvent_t *event, unsigned int flags, const char *func, const char *file,
                                        const char *line)
  {
    cudaError_t error = cudaEventCreateWithFlags((CUevent *)event, flags);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaEventDestroy((CUevent)event);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaEventQuery_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaEventQuery(event);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaEventRecord_(qudaEvent_t &event, qudaStream_t stream, const char *func, const char *file,
                               const char *line)
  {
    cudaError_t error = cudaEventRecord(event, stream);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaEventElapsedTime_(float *ms, qudaEvent_t &start, qudaEvent_t &end, const char *func, const char *file,
                                    const char *line)
  {
    cudaError_t error = cudaEventElapsedTime(ms, start, end);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaStreamWaitEvent(qudaStream_t stream, qudaEvent_t event, unsigned int flags)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamWaitEvent returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(cudaError_t error = cudaStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    return error;
#endif
  }

  qudaError_t qudaStreamSynchronize(qudaStream_t &stream)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamSynchronize returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(cudaError_t error = cudaStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    return error;
#endif
  }

  qudaError_t qudaEventSynchronize(qudaEvent_t &event)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventSynchronize returned error %s", str);
    }
    return cudaErrorUnknown;
#else
    PROFILE(cudaError_t error = cudaEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    return error;
#endif
  }

  qudaError_t qudaCreateTextureObject_(qudaTextureObject_t *pTexObject, const qudaResourceDesc *pResDesc,
                                       const qudaTextureDesc *pTexDesc, const qudaResourceViewDesc *pResViewDesc,
                                       const char *func, const char *file, const char *line)
  {
    cudaError_t error;
    PROFILE(error = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc),
            QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDestroyTextureObject_(qudaTextureObject_t pTexObject, const char *func, const char *file,
                                        const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuTexObjectDestroy(pTexObject), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuTexObjectDestroy returned error %s (%s:%s in %s())\n", str, file, line, func);
    }
    return cudaErrorUnknown;
#else
    PROFILE(cudaError_t error = cuTexObjectDestroy(pTexObject), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
#endif
  }

  qudaError_t qudaGetTextureObjectResourceDesc_(qudaResourceDesc *pResDesc, qudaTextureObject_t texObject,
                                                const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaGetTextureObjectResourceDesc(pResDesc, texObject);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDeviceCanAccessPeer_(int *canAccessPeer, int device, int peerDevice, const char *func,
                                       const char *file, const char *line)
  {
    cudaError_t error = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDeviceGetStreamPriorityRange_(int *leastPriority, int *greatestPriority, const char *func,
                                                const char *file, const char *line)
  {
    cudaError_t error = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDeviceReset_(const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaDeviceReset();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDeviceSetCacheConfig_(qudaFuncCache cacheConfig, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaDeviceSetCacheConfig(cacheConfig);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaDeviceSynchronize();
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
      printfQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaSetDevice_(int dev, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaSetDevice(dev);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaGetDeviceCount_(int *count, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaGetDeviceCount(count);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaGetDeviceProperties_(qudaDeviceProp *prop, int device, const char *func, const char *file,
                                       const char *line)
  {
    cudaError_t error = cudaGetDeviceProperties(prop, device);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaHostGetDevicePointer_(void **pDevice, void *pHost, unsigned int flags, const char *func,
                                        const char *file, const char *line)
  {
    cudaError_t error = cudaHostGetDevicePointer(pDevice, pHost, flags);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaDriverGetVersion_(int *driverVersion, const char *func, const char *file, const char *line)
  {
    cudaDriverGetVersion(driverVersion);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaRuntimeGetVersion_(int *runtimeVersion, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaRuntimeGetVersion(runtimeVersion);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaHostRegister_(void *ptr, size_t size, unsigned int flags, const char *func, const char *file,
                                const char *line)
  {
    cudaError_t error = cudaHostRegister(ptr, size, flags);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  qudaError_t qudaHostUnregister_(void *ptr, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaHostUnregister(ptr);
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return error;
  }

  
#if (CUDA_VERSION >= 9000)
  qudaError_t qudaFuncSetAttribute(const void *func, qudaFuncAttribute attr, int value)
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
