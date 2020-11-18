#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <device.h>
#include <hip/hip_runtime.h>


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


  static TimeProfile apiTimer("CUDA API calls (runtime)");

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream)
  {
#if 0
    if (tp.set_max_shared_bytes) {
      static std::unordered_set<const void *> cache;
      auto search = cache.find(func);
      if (search == cache.end()) {
        cache.insert(func);
        cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
        hipFuncAttributes attributes;
        hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(func));
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             device::max_dynamic_shared_memory() - attributes.sharedSizeBytes);
      }
    }
#endif

    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipLaunchKernel(func, tp.grid, tp.block, args, tp.shared_bytes, stream),
            QUDA_PROFILE_LAUNCH_KERNEL);
    if (error != hipSuccess && !activeTuning()) errorQuda("(CUDA) %s", hipGetErrorString(error));
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
        case hipMemcpyDeviceToHost:   name = "cudaMemcpyAsyncDeviceToHost";   break;
        case hipMemcpyHostToDevice:   name = "cudaMemcpyAsyncHostToDevice";   break;
        case hipMemcpyHostToHost:     name = "cudaMemcpyAsyncHostToHost";     break;
        case hipMemcpyDeviceToDevice: name = "cudaMemcpyAsyncDeviceToDevice"; break;
        case hipMemcpyDefault:        name = "cudaMemcpyAsyncDefault";        break;
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
          QudaProfileType type;
          switch (kind) {
          case hipMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
          case hipMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
          case hipMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
          case hipMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
          default: errorQuda("Unsupported cudaMemcpyTypeAsync %d", kind);
          }

          PROFILE(hipMemcpyAsync(dst, src, count, kind, stream), type);
        } else {
          hipMemcpy(dst, src, count, kind);
        }
      } else {
        if (async)
          hipMemsetAsync(dst, value, count, stream);
        else
          hipMemset(dst, value, count);
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
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
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




   /**
     @brief Wrapper around hipMemcpyToSymbolAsync or driver API equivalent
     @param[out] symbol   Destination symbol
     @param[in] src      Source pointer
     @param[in] count    Size of transfer
     @param[in] offset   Offset from start of symbol
     @param[in] kind     Type of memory copy
     @param[in] stream   Stream to issue copy
  */
  void qudaMemcpyToSymbolAsync_(const void *symbol, const void *src, size_t count, size_t offset,  qudaMemcpyKind kind, const qudaStream_t &stream,
                                const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
    if( error != hipSuccess ) {
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }

  }

  
  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                     hipMemcpyKind kind, const char *func, const char *file, const char *line)
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
      param.srcMemoryType = hipMemoryTypeDevice;
      param.dstHost = dst;
      param.dstMemoryType = hipMemoryTypeHost;
      break;
    default:
      errorQuda("Unsupported cuMemcpyType2DAsync %d", kind);
    }
    PROFILE(hipMemcpyParam2D(&param), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#else
    PROFILE(hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#endif
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                          hipMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line)
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
      param.srcMemoryType = hipMemoryTypeDevice;
      param.dstHost = dst;
      param.dstMemoryType = hipMemoryTypeHost;
      break;
    default:
      errorQuda("Unsupported cuMemcpyType2DAsync %d", kind);
    }
    PROFILE(hipMemcpyParam2DAsync(&param, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#else
    PROFILE(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
#endif
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, 0, false, func, file, line);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess && !activeTuning()) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, stream, true, func, file, line);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height,
                     const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemset2D(ptr, pitch, value, width, height);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height,
                          const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    hipError_t error = hipMemset2DAsync(ptr, pitch, value, width, height, stream);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
	  // No prefetch 
  }

  void qudaEventCreate_(qudaEvent_t *event,  const char *func, const char *file, const char *line)
  {

    PROFILE(hipError_t error = hipEventCreate(event), QUDA_PROFILE_EVENT_CREATE);
    if( error != hipSuccess ) {
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }
    return;
  }

  void qudaEventCreateDisableTiming_(qudaEvent_t *event,  const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventCreateWithFlags(event,hipEventDisableTiming), QUDA_PROFILE_EVENT_CREATE_DISABLED_TIMING);
    if( error != hipSuccess ) {
      errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }
    return;
  }

  void qudaEventCreateIpcDisableTiming_(qudaEvent_t *event,  const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventCreateWithFlags(event,hipEventDisableTiming | hipEventInterprocess), QUDA_PROFILE_EVENT_CREATE_DISABLED_TIMING);
    if( error != hipSuccess ) {
      errorQuda("(HIP) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }

    return;
  }

  void qudaEventDestroy_(qudaEvent_t event,  const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventDestroy(event), QUDA_PROFILE_EVENT_DESTROY);
    if( error != hipSuccess ) {
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }
    return;
  }

  
  bool qudaEventQuery_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventQuery(event), QUDA_PROFILE_EVENT_QUERY);
    switch (error) {
    case hipSuccess: return true;
    case hipErrorNotReady: return false;
    default: errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    }
    return false;
  }

  void qudaEventRecord_(hipEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventRecord(event, stream), QUDA_PROFILE_EVENT_RECORD);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaEventElapsedTime_(float *ms, qudaEvent_t start, qudaEvent_t end, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventElapsedTime(ms,start,end), QUDA_PROFILE_EVENT_ELAPSED_TIME);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  
  void qudaStreamWaitEvent_(qudaStream_t stream, hipEvent_t event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipStreamWaitEvent(stream, event, flags), QUDA_PROFILE_STREAM_WAIT_EVENT);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaEventSynchronize_(hipEvent_t &event, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipEventSynchronize(event), QUDA_PROFILE_EVENT_SYNCHRONIZE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

#if defined(QUDA_ENABLE_P2P)
  void qudaIpcGetEventHandle_(qudaIpcEventHandle_t *handle, qudaEvent_t event, const char *func, const char *file,
                              const char *line)
  {
    // qudaIpcEventHandle_t doesn't convert nicely to ihipIpcEventHandle_t so no driver API
    PROFILE(hipError_t error = hipIpcGetEventHandle(handle,event), QUDA_PROFILE_IPC_GET_EVENT_HANDLE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaIpcGetMemHandle_(qudaIpcMemHandle_t *handle, void *ptr, const char *func, const char *file,
                              const char *line)
  {
    // qudaIpcEventHandle_t doesn't convert nicely to ihipIpcEventHandle_t so no driver API
    PROFILE(hipError_t error = hipIpcGetMemHandle(handle,ptr), QUDA_PROFILE_IPC_GET_MEM_HANDLE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaIpcOpenEventHandle_(qudaEvent_t *event, qudaIpcEventHandle_t handle, const char *func, const char *file,
                              const char *line)
  {
    // qudaIpcEventHandle_t doesn't convert nicely to ihipIpcEventHandle_t so no driver API
    PROFILE(hipError_t error = hipIpcOpenEventHandle(event,handle), QUDA_PROFILE_IPC_OPEN_EVENT_HANDLE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaIpcOpenMemHandle_(void **devPtr, qudaIpcMemHandle_t handle, const char *func, const char *file,
			     const char *line)
  {
    // qudaIpcEventHandle_t doesn't convert nicely to ihipIpcEventHandle_t so no driver API
    PROFILE(hipError_t error = hipIpcOpenMemHandle(devPtr,handle,hipIpcMemLazyEnablePeerAccess),
	    QUDA_PROFILE_IPC_OPEN_MEM_HANDLE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }
  
  void qudaIpcCloseMemHandle_(void *devPtr, const char *func, const char *file, const char *line)
  {
    // qudaIpcEventHandle_t doesn't convert nicely to ihipIpcEventHandle_t so no driver API
    PROFILE(hipError_t error = hipIpcCloseMemHandle(devPtr),QUDA_PROFILE_IPC_CLOSE_MEM_HANDLE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }
#endif

  void qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipStreamSynchronize(stream), QUDA_PROFILE_STREAM_SYNCHRONIZE);
    if (error != hipSuccess && !activeTuning())
      errorQuda("(CUDA) %s\n (%s:%s in %s())", hipGetErrorString(error), file, line, func);
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    PROFILE(hipError_t error = hipDeviceSynchronize(), QUDA_PROFILE_DEVICE_SYNCHRONIZE);
    if (error != hipSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void* qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    void *ptr;
    hipError_t error = hipGetSymbolAddress(&ptr, HIP_SYMBOL((const void *)symbol));
    if (error != hipSuccess)
      errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
    return ptr;
  }

  void qudaFuncSetAttribute_(const void *kernel, hipFuncAttribute attr, int value, const char *func, const char *file,
                             const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipFuncSetAttribute(kernel, attr, value), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  void qudaFuncGetAttributes_(hipFuncAttributes &attr, const void *kernel, const char *func, const char *file,
                              const char *line)
  {
    // no driver API variant here since we have C++ functions
    PROFILE(hipError_t error = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)), QUDA_PROFILE_FUNC_SET_ATTRIBUTE);
    if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }


  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

  /**
   * @brief Create a QUDA Stream
   *
   */
  void qudaStreamCreate_(qudaStream_t* stream, const char *func, const char *file, const char *line)
  {
        hipError_t error = hipStreamCreate( stream );
        if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

  /**
   * @brief Destroy a QUDA Stream
   *
   */
  void qudaStreamDestroy_(qudaStream_t stream, const char *func, const char *file, const char *line)
  {
	hipError_t error = hipStreamDestroy( stream );
	if (error != hipSuccess) errorQuda("(CUDA) %s\n (%s:%s in %s())\n", hipGetErrorString(error), file, line, func);
  }

} // namespace quda
