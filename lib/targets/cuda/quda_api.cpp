#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>
#include <quda_cuda_api.h>

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

namespace quda
{

  static qudaError_t last_error = QUDA_SUCCESS;
  static std::string last_error_str("CUDA_SUCCESS");

  qudaError_t qudaGetLastError()
  {
    auto rtn = last_error;
    last_error = QUDA_SUCCESS;
    return rtn;
  }

  std::string qudaGetLastErrorString()
  {
    auto rtn = last_error_str;
    last_error_str = "CUDA_SUCCESS";
    return rtn;
  }

  namespace target
  {

    namespace cuda
    {

      void set_runtime_error(cudaError_t error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error)
      {
        if (error == cudaSuccess) return;
        last_error = error == cudaSuccess ? QUDA_SUCCESS : QUDA_ERROR;
        last_error_str = cudaGetErrorString(error);
        if (!allow_error)
          errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, cudaGetErrorString(error), file, line, func);
        else
          cudaGetLastError(); // clear the error state
      }

      void set_driver_error(CUresult error, const char *api_func, const char *func, const char *file, const char *line,
                            bool allow_error)
      {
        if (error == CUDA_SUCCESS) return;
        last_error = error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
        const char *str;
        cuGetErrorName(error, &str);
        last_error_str = str;
        if (!allow_error)
          errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, str, file, line, func);
        else
          cudaGetLastError(); // clear the error state
      }

    } // namespace cuda

  } // namespace target

  using namespace target::cuda;

  // Agnostic way to return a cuda API flag
  namespace
  {
    inline cudaMemcpyKind qudaMemcpyKindToAPI(const qudaMemcpyKind &k)
    {
      switch (k) {
      case qudaMemcpyHostToHost: return cudaMemcpyHostToHost;
      case qudaMemcpyHostToDevice: return cudaMemcpyHostToDevice;
      case qudaMemcpyDeviceToHost: return cudaMemcpyDeviceToHost;
      case qudaMemcpyDeviceToDevice: return cudaMemcpyDeviceToDevice;
      case qudaMemcpyDefault: return cudaMemcpyDefault;
      default:
        errorQuda(" unknown value for qudaMemcpyKind %d", static_cast<int>(k));
        return cudaMemcpyDefault; // keep warnings away
      }
    }
  } // namespace

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

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, const qudaStream_t &stream, const void *arg)
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
    void *args[] = {const_cast<void *>(arg)};
    PROFILE(cudaError_t error = cudaLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, get_stream(stream)),
            QUDA_PROFILE_LAUNCH_KERNEL);
    set_runtime_error(error, __func__, __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
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
    const char *func;
    const char *file;
    const char *line;

  public:
    inline QudaMem(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
                   bool async, const char *func, const char *file, const char *line) :
      dst(dst),
      src(src),
      count(count),
      value(0),
      copy(true),
      kind(kind),
      async(async),
      active_tuning(activeTuning()),
      func(func),
      file(file),
      line(line)
    {
      if (!async) {
        switch (kind) {
        case cudaMemcpyDeviceToHost: name = "cudaMemcpyDeviceToHost"; break;
        case cudaMemcpyHostToDevice: name = "cudaMemcpyHostToDevice"; break;
        case cudaMemcpyHostToHost: name = "cudaMemcpyHostToHost"; break;
        case cudaMemcpyDeviceToDevice: name = "cudaMemcpyDeviceToDevice"; break;
        case cudaMemcpyDefault: name = "cudaMemcpyDefault"; break;
        default: errorQuda("Unsupported cudaMemcpyKind %d", kind);
        }
      } else {
        switch (kind) {
        case cudaMemcpyDeviceToHost: name = "cudaMemcpyAsyncDeviceToHost"; break;
        case cudaMemcpyHostToDevice: name = "cudaMemcpyAsyncHostToDevice"; break;
        case cudaMemcpyHostToHost: name = "cudaMemcpyAsyncHostToHost"; break;
        case cudaMemcpyDeviceToDevice: name = "cudaMemcpyAsyncDeviceToDevice"; break;
        case cudaMemcpyDefault: name = "cudaMemcpyAsyncDefault"; break;
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

    void apply(const qudaStream_t &stream) override
    {
      if (!active_tuning) tuneLaunch(*this, getTuning(), getVerbosity());

      if (copy) {
        if (async) {
#ifdef USE_DRIVER_API
          CUresult error = CUDA_SUCCESS;
          switch (kind) {
          case cudaMemcpyDeviceToHost:
            PROFILE(error = cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, get_stream(stream)),
                    QUDA_PROFILE_MEMCPY_D2H_ASYNC);
            break;
          case cudaMemcpyHostToDevice:
            PROFILE(error = cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, get_stream(stream)),
                    QUDA_PROFILE_MEMCPY_H2D_ASYNC);
            break;
          case cudaMemcpyDeviceToDevice:
            PROFILE(error = cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                    QUDA_PROFILE_MEMCPY_D2D_ASYNC);
            break;
          case cudaMemcpyDefault:
            PROFILE(error = cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                    QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
            break;
          default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
          }
          set_driver_error(error, "cuMemcpyAsync", func, file, line, active_tuning);
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
          cudaError_t error;
          PROFILE(error = cudaMemcpyAsync(dst, src, count, kind, get_stream(stream)), type);
          set_runtime_error(error, "cudaMemcpyAsync", func, file, line, active_tuning);
#endif
        } else {
#ifdef USE_DRIVER_API
          CUresult error = CUDA_SUCCESS;
          switch (kind) {
          case cudaMemcpyDeviceToHost: error = cuMemcpyDtoH(dst, (CUdeviceptr)src, count); break;
          case cudaMemcpyHostToDevice: error = cuMemcpyHtoD((CUdeviceptr)dst, src, count); break;
          case cudaMemcpyHostToHost: memcpy(dst, src, count); break;
          case cudaMemcpyDeviceToDevice: error = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count); break;
          case cudaMemcpyDefault: error = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count); break;
          default: errorQuda("Unsupported cudaMemcpyType %d", kind);
          }
          set_driver_error(error, "cuMemcpy", func, file, line, active_tuning);
#else
          cudaError_t error = cudaMemcpy(dst, src, count, kind);
          set_runtime_error(error, "cudaMemcpy", func, file, line, active_tuning);
#endif
        }
      } else {
#ifdef USE_DRIVER_API
        CUresult error = async ? cuMemsetD32Async((CUdeviceptr)dst, value, count / 4, get_stream(stream)) :
                                 cuMemsetD32((CUdeviceptr)dst, value, count / 4);
        set_driver_error(error, "cuMemset", func, file, line, active_tuning);
#else
        cudaError_t error
          = async ? cudaMemsetAsync(dst, value, count, get_stream(stream)) : cudaMemset(dst, value, count);
        set_runtime_error(error, "cudaMemset", func, file, line, active_tuning);
#endif
      }
    }

    bool advanceTuneParam(TuneParam &) const override { return false; }

    TuneKey tuneKey() const override
    {
      char vol[128];
      strcpy(vol, "bytes=");
      u64toa(vol + 6, (uint64_t)count);
      return TuneKey(vol, name, aux);
    }

    long long bytes() const override { return kind == cudaMemcpyDeviceToDevice ? 2 * count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, src, count, qudaMemcpyKindToAPI(kind), device::get_default_stream(), false, func, file, line);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, qudaMemcpyKindToAPI(kind), stream, true, func, file, line);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case qudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, get_stream(stream)), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case qudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, get_stream(stream)), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case qudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      case qudaMemcpyDefault:
        PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
      default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), get_stream(stream)),
              kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }

  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream, const char *func,
                           const char *file, const char *line)
  {
    if (count == 0) return;
    auto error = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, get_stream(stream));
    set_runtime_error(error, "cudaMemcpyAsync", func, file, line);
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, device::get_default_stream(), false, func, file, line);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, stream, true, func, file, line);
  }

  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height, const char *func,
                     const char *file, const char *line)
  {
    cudaError_t error = cudaMemset2D(ptr, pitch, value, width, height);
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaMemset2DAsync(ptr, pitch, value, width, height, get_stream(stream));
    set_runtime_error(error, __func__, func, file, line);
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

    cudaError_t error = cudaMemPrefetchAsync(ptr, count, dev_id, get_stream(stream));
    set_runtime_error(error, __func__, func, file, line);
  }

  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(quda_event.event);
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case CUDA_SUCCESS: return true;
    case CUDA_ERROR_NOT_READY: return false;
    default: set_driver_error(error, __func__, func, file, line);
    }
#else
    PROFILE(cudaError_t error = cudaEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case cudaSuccess: return true;
    case cudaErrorNotReady: return false;
    default: set_runtime_error(error, __func__, func, file, line);
    }
#endif
    return false;
  }

  void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(quda_event.event);
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventRecord(event, get_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    set_driver_error(error, __func__, func, file, line);
#else
    PROFILE(cudaError_t error = cudaEventRecord(event, get_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    set_runtime_error(error, __func__, func, file, line);
#endif
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t quda_event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
    cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(quda_event.event);
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamWaitEvent(get_stream(stream), event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    set_driver_error(error, __func__, func, file, line);
#else
    PROFILE(cudaError_t error = cudaStreamWaitEvent(get_stream(stream), event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    set_runtime_error(error, __func__, func, file, line);
#endif
  }

  qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line)
  {
    cudaEvent_t cuda_event;
    cudaError_t error = cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
    set_runtime_error(error, __func__, func, file, line);
    qudaEvent_t quda_event;
    quda_event.event = reinterpret_cast<void *>(cuda_event);
    return quda_event;
  }

  qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line)
  {
    cudaEvent_t cuda_event;
    cudaError_t error = cudaEventCreate(&cuda_event);
    set_runtime_error(error, __func__, func, file, line);
    qudaEvent_t quda_event;
    quda_event.event = reinterpret_cast<void *>(cuda_event);
    return quda_event;
  }

  float qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop, const char *func, const char *file,
                              const char *line)
  {
    float elapsed_time;
    cudaError_t error = cudaEventElapsedTime(&elapsed_time, reinterpret_cast<cudaEvent_t>(start.event),
                                             reinterpret_cast<cudaEvent_t>(stop.event));
    set_runtime_error(error, __func__, func, file, line);
    return elapsed_time / 1000;
  }

  void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    cudaError_t error = cudaEventDestroy(reinterpret_cast<cudaEvent_t &>(event.event));
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    const cudaEvent_t &event = reinterpret_cast<const cudaEvent_t &>(quda_event.event);
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    set_driver_error(error, __func__, func, file, line);
#else
    PROFILE(cudaError_t error = cudaEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    set_runtime_error(error, __func__, func, file, line);
#endif
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuStreamSynchronize(get_stream(stream)), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    set_driver_error(error, __func__, file, line, func, activeTuning());
#else
    PROFILE(cudaError_t error = cudaStreamSynchronize(get_stream(stream)), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    set_runtime_error(error, __func__, file, line, func, activeTuning());
#endif
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuCtxSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    set_driver_error(error, __func__, func, file, line, activeTuning());
#else
    PROFILE(cudaError_t error = cudaDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    set_runtime_error(error, __func__, func, file, line, activeTuning());
#endif
  }

  void *qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    void *ptr;
    cudaError_t error = cudaGetSymbolAddress(&ptr, symbol);
    set_runtime_error(error, __func__, func, file, line);
    return ptr;
  }

  void qudaFuncSetAttribute_(const void *kernel, cudaFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncSetAttribute(kernel, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaFuncGetAttributes_(cudaFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(cudaError_t error = cudaFuncGetAttributes(&attr, kernel), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    set_runtime_error(error, __func__, func, file, line);
  }

  void printAPIProfile()
  {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
