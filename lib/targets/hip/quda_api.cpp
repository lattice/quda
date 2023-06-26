#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>
#include <target_device.h>
#include <hip/hip_runtime.h>
#include <quda_hip_api.h>

// if this macro is defined then we profile the HIP API calls
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

  /* This is checked in the tuner */
  static qudaError_t last_error = QUDA_SUCCESS;

  /* This is only ever printed */
  static std::string last_error_str {"HIP_SUCCESS"};

  /* For the tuner to operat correctly we need to clear the last error */
  qudaError_t qudaGetLastError()
  {
    auto rtn = last_error;
    last_error = QUDA_SUCCESS; // Clear the error prior to returning
    return rtn;
  }

  std::string qudaGetLastErrorString()
  {
    auto rtn = last_error_str;
    last_error_str = "HIP_SUCCESS"; // Clear the error prior to returning.
    return rtn;
  }

  namespace target
  {
    namespace hip
    {

      void set_runtime_error(hipError_t error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error)
      {
        if (error == hipSuccess) return;
        last_error = QUDA_ERROR;
        last_error_str = hipGetErrorString(error);
        if (!allow_error) {
          errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, last_error_str.c_str(), file, line, func);
        } else
          hipGetLastError(); // Clear the error
      }

      void set_driver_error(hipError_t error, const char *api_func, const char *func, const char *file,
                            const char *line, bool allow_error)
      {
        if (error == hipSuccess) return;
        last_error = QUDA_ERROR;
        last_error_str = hipGetErrorString(error);
        if (!allow_error) {
          errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, last_error_str.c_str(), file, line, func);
        } else
          hipGetLastError();
      }
    } // namespace hip
  }   // namespace target

  using namespace target::hip;

  // Agnostic way to return a hipAPI flag
  namespace
  {
    inline hipMemcpyKind qudaMemcpyKindToAPI(const qudaMemcpyKind &k)
    {
      switch (k) {
      case qudaMemcpyHostToHost: return hipMemcpyHostToHost; break;
      case qudaMemcpyHostToDevice: return hipMemcpyHostToDevice; break;
      case qudaMemcpyDeviceToHost: return hipMemcpyDeviceToHost; break;
      case qudaMemcpyDeviceToDevice: return hipMemcpyDeviceToDevice; break;
      case qudaMemcpyDefault: return hipMemcpyDefault; break;
      default:
        errorQuda(" unknown value for qudaMemcpyKind %d", static_cast<int>(k));
        return hipMemcpyDefault; // keep warnings away
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
  void qudaFuncSetAttribute_(const void *kernel, hipFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line);

  /**
     @brief Wrapper around cudaFuncGetAttributes with built-in error checking
     @param[in] attr the cudaFuncGetAttributes object to store the output
     @param[in] kernel Kernel function for which we are setting the attribute
  */
  void qudaFuncGetAttributes_(hipFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line);

#define qudaFuncSetAttribute(kernel, attr, value)                                                                      \
  ::quda::qudaFuncSetAttribute_(kernel, attr, value, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaFuncGetAttributes(attr, kernel)                                                                            \
  ::quda::qudaFuncGetAttributes_(attr, kernel, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

  static TimeProfile apiTimer("HIP API calls (runtime)");

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, const qudaStream_t &stream, const void *arg)
  {
    // no driver API variant here since we have C++ functions
    void *args[] = {const_cast<void *>(arg)};
    PROFILE(hipError_t error = hipLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, get_stream(stream)),
            QUDA_PROFILE_LAUNCH_KERNEL);
    set_runtime_error(error, __func__, __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
    return error == hipSuccess ? QUDA_SUCCESS : QUDA_ERROR;
  }

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
    const bool active_tuning;
    const char *func;
    const char *file;
    const char *line;

  public:
    inline QudaMem(void *dst, const void *src, size_t count, hipMemcpyKind kind, const qudaStream_t &stream, bool async,
                   const char *func, const char *file, const char *line) :
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
        case hipMemcpyDeviceToHost: name = "hipMemcpyDeviceToHost"; break;
        case hipMemcpyHostToDevice: name = "hipMemcpyHostToDevice"; break;
        case hipMemcpyHostToHost: name = "hipMemcpyHostToHost"; break;
        case hipMemcpyDeviceToDevice: name = "hipMemcpyDeviceToDevice"; break;
        case hipMemcpyDefault: name = "hipMemcpyDefault"; break;
        default: errorQuda("Unsupported hipMemcpyKind %d", kind);
        }
      } else {
        switch (kind) {
        case hipMemcpyDeviceToHost: name = "hipMemcpyAsyncDeviceToHost"; break;
        case hipMemcpyHostToDevice: name = "hipMemcpyAsyncHostToDevice"; break;
        case hipMemcpyHostToHost: name = "hipMemcpyAsyncHostToHost"; break;
        case hipMemcpyDeviceToDevice: name = "hipMemcpyAsyncDeviceToDevice"; break;
        case hipMemcpyDefault: name = "hipMemcpyAsyncDefault"; break;

        default: errorQuda("Unsupported hipMemcpyKind %d", kind);
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
      kind(hipMemcpyDefault),
      async(async),
      active_tuning(activeTuning())
    {
      name = !async ? "hipMemset" : "hipMemsetAsync";
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
#ifdef API_PROFILE
          QudaProfileType type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC;
          switch (kind) {
          case hipMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
          case hipMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
          case hipMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
          case hipMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
          default: errorQuda("Unsupported cudaMemcpyTypeAsync %d", kind);
          }
#endif
          hipError_t error;
          PROFILE(error = hipMemcpyAsync(dst, src, count, kind, get_stream(stream)), type);
          set_runtime_error(error, "hipMemcpyAsync", func, file, line, active_tuning);
        } else {
          hipError_t error = hipMemcpy(dst, src, count, kind);
          set_runtime_error(error, "hipMemcpy", func, file, line, active_tuning);
        }
      } else {
        hipError_t error = async ? hipMemsetAsync(dst, value, count, get_stream(stream)) : hipMemset(dst, value, count);
        set_runtime_error(error, " hipMemset", func, file, line, active_tuning);
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

    long long bytes() const override { return kind == hipMemcpyDeviceToDevice ? 2 * count : count; }
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
      PROFILE(hipMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), get_stream(stream)),
              kind == hipMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
    }
  }

  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream, const char *func,
                           const char *file, const char *line)
  {
    if (count == 0) return;
    auto error = hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, get_stream(stream));
    set_runtime_error(error, "hipMemcpyAsync", func, file, line);
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
    hipError_t error = hipMemset2D(ptr, pitch, value, width, height);
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemset2DAsync(ptr, pitch, value, width, height, get_stream(stream));
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaMemPrefetchAsync_(void *, size_t, QudaFieldLocation, const qudaStream_t &, const char *, const char *,
                             const char *)
  {
    // No prefetch
  }

#if 0
  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    cudaEvent_t &event = reinterpret_cast<cudaEvent_t&>(quda_event.event);
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
    cudaEvent_t &event = reinterpret_cast<cudaEvent_t&>(quda_event.event);
#ifdef USE_DRIVER_API
    PROFILE(CUresult error = cuEventRecord(event, get_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    set_driver_error(error, __func__, func, file, line);
#else
    PROFILE(cudaError_t error = cudaEventRecord(event, get_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    set_runtime_error(error, __func__, func, file, line);
#endif
  }
#endif

  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    hipEvent_t &event = reinterpret_cast<hipEvent_t &>(quda_event.event);
    PROFILE(hipError_t error = hipEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case hipSuccess: return true;
    case hipErrorNotReady: return false;
    default: set_runtime_error(error, __func__, func, file, line);
    }
    return false;
  }

  void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    hipEvent_t &event = reinterpret_cast<hipEvent_t &>(quda_event.event);
    PROFILE(hipError_t error = hipEventRecord(event, get_stream(stream)), QUDA_PROFILE_EVENT_RECORD);
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t quda_event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
    hipEvent_t &hip_event = reinterpret_cast<hipEvent_t &>(quda_event.event);

    hipError_t hiperr = hipErrorNotReady; // A random not hipSuccess initialization
    do {
      hiperr = hipEventQuery(hip_event);
    } while (hiperr != hipSuccess);

    PROFILE(hipError_t error = hipStreamWaitEvent(get_stream(stream), hip_event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    set_runtime_error(error, __func__, func, file, line);
  }

  qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line)
  {
    hipEvent_t hip_event;
    hipError_t error = hipEventCreateWithFlags(&hip_event, hipEventDisableTiming);
    set_runtime_error(error, __func__, func, file, line);
    qudaEvent_t quda_event;
    quda_event.event = reinterpret_cast<void *>(hip_event);
    return quda_event;
  }

  qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line)
  {
    hipEvent_t hip_event;
    hipError_t error = hipEventCreate(&hip_event);
    set_runtime_error(error, __func__, func, file, line);
    qudaEvent_t quda_event;
    quda_event.event = reinterpret_cast<void *>(hip_event);
    return quda_event;
  }

  float qudaEventElapsedTime_(const qudaEvent_t &quda_start, const qudaEvent_t &quda_end, const char *func,
                              const char *file, const char *line)
  {
    float elapsed_time;
    const hipEvent_t &hip_start = reinterpret_cast<const hipEvent_t &>(quda_start.event);
    const hipEvent_t &hip_end = reinterpret_cast<const hipEvent_t &>(quda_end.event);

    PROFILE(hipError_t error = hipEventElapsedTime(&elapsed_time, hip_start, hip_end), QUDA_PROFILE_EVENT_ELAPSED_TIME);
    set_runtime_error(error, "hipEventElapsedTime", func, file, line);
    return elapsed_time / 1000;
  }

  void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    hipError_t error = hipEventDestroy(reinterpret_cast<hipEvent_t &>(event.event));
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    const hipEvent_t &event = reinterpret_cast<const hipEvent_t &>(quda_event.event);
    PROFILE(hipError_t error = hipEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    set_runtime_error(error, "hipEventSynchronize", func, file, line);
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipStreamSynchronize(get_stream(stream)), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    set_runtime_error(error, "hipStreamSynchronize", func, file, line);
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    set_runtime_error(error, "hipDeviceSynchronize", func, file, line);
  }

  void *qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    void *ptr;
    hipError_t error = hipGetSymbolAddress(&ptr, HIP_SYMBOL((const void *)symbol));
    set_runtime_error(error, "hipGetSymbolAddress", func, file, line);
    return ptr;
  }

  void qudaFuncSetAttribute_(const void *kernel, hipFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipFuncSetAttribute(kernel, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    set_runtime_error(error, "hipFuncSetAttribute", func, file, line);
  }

  void qudaFuncGetAttributes_(hipFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipFuncGetAttributes(&attr, reinterpret_cast<const void *>(kernel)),
            QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    set_runtime_error(error, "hipFuncGetAttributes", func, file, line);
  }

  void printAPIProfile()
  {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
