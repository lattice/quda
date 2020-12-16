#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <device.h>

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

  // No need to abstract these across the library so keep these definitions local to CUDA target

  /**
     @brief Wrapper around cudaFuncSetAttribute with built-in error checking
     @param[in] kernel Kernel function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  void qudaFuncSetAttribute_(const void *kernel, cudaFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line);

  /**
     @brief Wrapper around cudaFuncGetAttributes with built-in error checking
     @param[in] attr the cudaFuncGetAttributes object to store the output
     @param[in] kernel Kernel function for which we are setting the attribute
  */
  void qudaFuncGetAttributes_(cudaFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line);

#define qudaFuncSetAttribute(kernel, attr, value)                                                                      \
  ::quda::qudaFuncSetAttribute_(kernel, attr, value, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaFuncGetAttributes(attr, kernel)                                                                            \
  ::quda::qudaFuncGetAttributes_(attr, kernel, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#ifdef USE_DRIVER_API
  static TimeProfile apiTimer("CUDA API calls (driver)");
#else
  static TimeProfile apiTimer("CUDA API calls (runtime)");
#endif

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream)
  {
    // if launch requests the maximum shared memory and the device supports it then opt in
    if (tp.set_max_shared_bytes && device::max_dynamic_shared_memory() > device::max_default_shared_memory()) {
      static std::unordered_set<const void *> cache;
      auto search = cache.find(func);
      if (search == cache.end()) {
        cache.insert(func);
        qudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
        cudaFuncAttributes attributes;
        qudaFuncGetAttributes(attributes, func);
        qudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             device::max_dynamic_shared_memory() - attributes.sharedSizeBytes);
      }
    }

    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, device::get_cuda_stream(stream)),
            QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != cudaSuccess && !activeTuning()) errorQuda("(CUDA) %s", cudaGetErrorString(error));
    return error == cudaSuccess ? QUDA_SUCCESS : QUDA_ERROR;
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
    const bool active_tuning;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
                   bool async, const char *func, const char *file, const char *line) :
      dst(dst), src(src), count(count), value(0), copy(true), kind(kind), async(async), active_tuning(activeTuning())
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

      apply(stream);
    }

    inline QudaMem(void *dst, int value, size_t count, const qudaStream_t &stream, bool async, const char *func,
                   const char *file, const char *line) :
      dst(dst),
      src(nullptr),
      count(count),
      value(value),
      copy(false),
      kind(cudaMemcpyDefault),
      async(async),
      active_tuning(activeTuning())
    {
      name = !async ? "cudaMemset" : "cudaMemsetAsync";
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);

      apply(stream);
    }

    inline void apply(const qudaStream_t &stream)
    {
      if (!active_tuning) tuneLaunch(*this, getTuning(), getVerbosity());

      if (copy) {
        if (async) {
#ifdef USE_DRIVER_API
          switch (kind) {
          case cudaMemcpyDeviceToHost:
            PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
            break;
          case cudaMemcpyHostToDevice:
            PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
            break;
          case cudaMemcpyDeviceToDevice:
            PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
            break;
          case cudaMemcpyDefault:
            PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
            break;
          default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
          }
#else
#ifdef API_PROFILE
          QudaProfileType type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC;
          switch (kind) {
          case cudaMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
          case cudaMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
          case cudaMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
          case cudaMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
          default: errorQuda("Unsupported cudaMemcpyTypeAsync %d", kind);
          }
#endif
          PROFILE(cudaMemcpyAsync(dst, src, count, kind, device::get_cuda_stream(stream)), type);
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
          cuMemsetD32Async((CUdeviceptr)dst, value, count / 4, device::get_cuda_stream(stream));
        else
          cuMemsetD32((CUdeviceptr)dst, value, count / 4);
#else
        if (async)
          cudaMemsetAsync(dst, value, count, device::get_cuda_stream(stream));
        else
          cudaMemset(dst, value, count);
#endif
      }
    }

    bool advanceTuneParam(TuneParam &) const { return false; }

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
    QudaMem copy(dst, src, count, kind, device::get_default_stream(), false, func, file, line);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == cudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, stream, true, func, file, line);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case cudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case cudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case cudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      case cudaMemcpyDefault:
        PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, kind, device::get_cuda_stream(stream)),
              kind == cudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream,
                           const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    auto error = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, device::get_cuda_stream(stream));
    if (error != cudaSuccess)
      errorQuda("cudaMemcpyAsync returned %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                     cudaMemcpyKind kind, const char *func, const char *file, const char *line)
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
    default: errorQuda("Unsupported cuMemcpyType2DAsync %d", kind);
    }
    PROFILE(auto error = cuMemcpy2D(&param), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuMemcpy2D returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
#else
    PROFILE(auto error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    if (error != cudaSuccess)
      errorQuda("cudaMemcpy2D returned error %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
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
    PROFILE(auto error = cuMemcpy2DAsync(&param, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuMemcpy2DAsync returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
#else
    PROFILE(auto error = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    if (error != cudaSuccess)
      errorQuda("cudaMemcpy2DAsync returned error %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaMemcpy2DP2PAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                             const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    auto error = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, device::get_cuda_stream(stream));
    if (error != cudaSuccess)
      errorQuda("cudaMemcpy2DAsync returned %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, device::get_default_stream(), false, func, file, line);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, stream, true, func, file, line);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height, const char *func,
                     const char *file, const char *line)
  {
    cudaError_t error = cudaMemset2D(ptr, pitch, value, width, height);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaMemset2DAsync(ptr, pitch, value, width, height, device::get_cuda_stream(stream));
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
    int dev_id = 0;
    if (mem_space == QUDA_CUDA_FIELD_LOCATION)
      dev_id = comm_gpuid();
    else if (mem_space == QUDA_CPU_FIELD_LOCATION)
      dev_id = cudaCpuDeviceId;
    else
      errorQuda("Invalid QudaFieldLocation.");

    cudaError_t error = cudaMemPrefetchAsync(ptr, count, dev_id, device::get_cuda_stream(stream));
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  bool qudaEventQuery_(cudaEvent_t &event, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case CUDA_SUCCESS: return true;
    case CUDA_ERROR_NOT_READY: return false;
    default: {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventQuery returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
    }
#else
    PROFILE(cudaError_t error = cudaEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case cudaSuccess: return true;
    case cudaErrorNotReady: return false;
    default: errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    }
#endif
    return false;
  }

  void qudaEventRecord_(cudaEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventRecord(event, device::get_cuda_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventRecord returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaEventRecord(event, device::get_cuda_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, cudaEvent_t event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamWaitEvent(device::get_cuda_stream(stream), event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuStreamWaitEvent returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaStreamWaitEvent(device::get_cuda_stream(stream), event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaEventSynchronize_(const cudaEvent_t &event, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("cuEventSynchronize returned error %s\n (%s:%s in %s())", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamSynchronize(device::get_cuda_stream(stream)), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    if (error != CUDA_SUCCESS) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("(CUDA) cuStreamSynchronize returned error %s\n (%s:%s in %s())\n", str, file, line, func);
    }
#else
    PROFILE(cudaError_t error = cudaStreamSynchronize(device::get_cuda_stream(stream)), QUDA_PROFILE_STREAM_SYNCHRONIZE);
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

  void* qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    void *ptr;
    cudaError_t error = cudaGetSymbolAddress(&ptr, symbol);
    if (error != cudaSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
    return ptr;
  }

  void qudaFuncSetAttribute_(const void *kernel, cudaFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncSetAttribute(kernel, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaFuncGetAttributes_(cudaFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncGetAttributes(&attr, kernel), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != cudaSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  static std::string error_str("CUDA_SUCCESS");

  void qudaSetErrorString(const std::string &error_str_)
  {
    error_str = error_str_;
  }

  std::string qudaGetLastErrorString()
  {
    return error_str;
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
