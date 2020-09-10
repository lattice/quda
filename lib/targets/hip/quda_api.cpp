#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <device.h>

// if this macro is defined then we use the driver API, else use the
// runtime API.  Typically the driver API has 10-20% less overhead

// if this macro is defined then we profile the CUDA API calls
#define API_PROFILE

#ifdef API_PROFILE
#define PROFILE(f, idx)                                 \
  apiTimer.TPSTART(idx);				\
  f;                                                    \
  apiTimer.TPSTOP(idx);
#else
#define PROFILE(f, idx) f;
#endif

namespace quda {

  static TimeProfile apiTimer("HIP API calls (runtime)");

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, stream),
            QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != hipSuccess && !activeTuning()) errorQuda("(HIP) %s", hipGetErrorString(error));
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

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, hipMemcpyKind kind, const hipStream_t &stream,
                   bool async, const char *func, const char *file, const char *line) :
      dst(dst),
      src(src),
      count(count),
      value(0),
      copy(true),
      kind(kind),
      async(async),
      active_tuning(activeTuning())
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

      apply(stream);
    }

    inline QudaMem(void *dst, int value, size_t count, const hipStream_t &stream, bool async,
                   const char *func, const char *file, const char *line) :
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

    inline void apply(const qudaStream_t &stream)
    {
      if (!active_tuning) tuneLaunch(*this, getTuning(), getVerbosity());

      if (copy) {
        if (async) {

          QudaProfileType type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; // This will get overridden

          switch (kind) {
            case hipMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
            case hipMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
            case hipMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
            case hipMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
            default: errorQuda("Unsupported hipMemcpyTypeAsync %d", kind); break;
          }

          PROFILE(hipMemcpyAsync(dst, src, count, kind, stream), type);
        } else {
          hipMemcpy(dst, src, count, kind);
        }
      } else {
        if (async) {
          hipMemsetAsync(dst, value, count, stream);
        }
        else {
          hipMemset(dst, value, count);
        }
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
    QudaMem copy(dst, src, count, kind, 0, false, func, file, line);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess)
      errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, hipMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == hipMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, stream, true, func, file, line);
    } else {
      PROFILE(hipMemcpyAsync(dst, src, count, kind, stream),
              kind == hipMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
    }
  }


  void qudaMemcpyToSymbolAsync_(const void *symbol, const void *src, size_t count, size_t offset, hipMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    // FIXME: Should we use the QudaMem here? 
    // FIXME: Should we use the C++ version where we give a ref to the symbol?
    if (count == 0) return;
    switch(kind) { 
    case hipMemcpyDeviceToHost:
	PROFILE(hipMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
	break;
    case hipMemcpyHostToDevice:
        PROFILE(hipMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
    case hipMemcpyDeviceToDevice:
        PROFILE(hipMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
    case hipMemcpyDefault:
	PROFILE(hipMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
     default: 
	errorQuda("Unsupported hipMemcpyToSymbolAsync: %d", kind);
	break;
     }
   }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                          hipMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line)
  {
    PROFILE(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, 0, false, func, file, line);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess && !activeTuning()) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, stream, true, func, file, line);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height,
                     const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemset2D(ptr, pitch, value, width, height);
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height,
                          const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemset2DAsync(ptr, pitch, value, width, height, stream);
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {

    // For now: HiP Prefetch is a NOP on HIP 
  }

  bool qudaEventQuery_(hipEvent_t &event, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case hipSuccess: return true;
    case hipErrorNotReady: return false;
    default: errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }
    return false;
  }

  void qudaEventRecord_(hipEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, hipEvent_t event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaEventSynchronize_(hipEvent_t &event, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != hipSuccess) errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    if (error != hipSuccess && !activeTuning())
      errorQuda("(HIP) %s\n (%s:%s in %s())", hipGetErrorString(error), file, line, func);
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != hipSuccess)
      errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
