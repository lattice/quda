#include <tune_quda.h>
#include <uint_to_char.h>

// if this macro is defined then we use the driver API, else use the
// runtime API.  Typically the driver API has 10-20% less overhead
#define USE_DRIVER_API

namespace quda {

  class QudaMemCopy : public Tunable {

    void *dst;
    const void *src;
    const size_t count;
    const cudaMemcpyKind kind;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    inline QudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		       const char *func, const char *file, const char *line)
      : dst(dst), src(src), count(count), kind(kind) {

      switch(kind) {
      case cudaMemcpyDeviceToHost:
	name = "cudaMemcpyDeviceToHost";
	break;
      case cudaMemcpyHostToDevice:
	name = "cudaMemcpyHostToDevice";
	break;
      case cudaMemcpyHostToHost:
	name = "cudaMemcpyHostToHost";
	break;
      case cudaMemcpyDeviceToDevice:
	name = "cudaMemcpyDeviceToDevice";
	break;
      case cudaMemcpyDefault:
        name = "cudaMemcpyDefault";
        break;
      default:
	errorQuda("Unsupported cudaMemcpyType %d", kind);
      }
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);
    }

    virtual ~QudaMemCopy() { }

    inline void apply(const cudaStream_t &stream) {
      tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef USE_DRIVER_API
      switch(kind) {
      case cudaMemcpyDeviceToHost:
        cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
	break;
      case cudaMemcpyHostToDevice:
        cuMemcpyHtoD((CUdeviceptr)dst, src, count);
	break;
      case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
	break;
      case cudaMemcpyDeviceToDevice:
        cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count);
	break;
      case cudaMemcpyDefault:
        cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
      default:
	errorQuda("Unsupported cudaMemcpyType %d", kind);
      }
#else
      cudaMemcpy(dst, src, count, kind);
#endif
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
    if (getVerbosity() == QUDA_DEBUG_VERBOSE)
      printfQuda("%s bytes = %llu\n", __func__, (long long unsigned int)count);

    if (count == 0) return;
#if 1
    QudaMemCopy copy(dst, src, count, kind, func, file, line);
    copy.apply(0);
#else
    cudaMemcpy(dst, src, count, kind);
#endif
    checkCudaError();
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream,
                        const char *func, const char *file, const char *line)
  {
#ifdef USE_DRIVER_API
    switch (kind) {
    case cudaMemcpyDeviceToHost:
      cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream);
      break;
    case cudaMemcpyHostToDevice:
      cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, stream);
      break;
    default:
      errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
    }
#else
    cudaMemcpyAsync(dst, src, count, kind, stream);
#endif
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream,
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
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind);
#endif
  }

  cudaError_t qudaEventQuery(cudaEvent_t event)
  {
#ifdef USE_DRIVER_API
    CUresult error = cuEventQuery(event);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    case CUDA_ERROR_NOT_READY: // this is the only return value care about
      return cudaErrorNotReady;
    default:
      errorQuda("cuEventQuery return error code %d", error);
    }
    return cudaErrorUnknown;
#else
    return cudaEventQuery(event);
#endif
  }

  cudaError_t qudaEventRecord(cudaEvent_t event, cudaStream_t stream)
  {
#ifdef USE_DRIVER_API
    CUresult error = cuEventRecord(event, stream);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      errorQuda("cuEventRecord return error code %d", error);
    }
    return cudaErrorUnknown;
#else
    return cudaEventRecord(event);
#endif
  }

  cudaError_t qudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
  {
#ifdef USE_DRIVER_API
    CUresult error = cuStreamWaitEvent(stream, event, flags);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      errorQuda("cuStreamWaitEvent return error code %d", error);
    }
    return cudaErrorUnknown;
#else
    return cudaStreamWaitEvent(stream, event, flags);
#endif
  }

  cudaError_t qudaStreamSynchronize(cudaStream_t stream)
  {
#ifdef USE_DRIVER_API
    CUresult error = cuStreamSynchronize(stream);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      errorQuda("cuStreamSynchronize return error code %d", error);
    }
    return cudaErrorUnknown;
#else
    return cudaStreamSynchronize(stream);
#endif
  }

  cudaError_t qudaEventSynchronize(cudaEvent_t event)
  {
#ifdef USE_DRIVER_API
    CUresult error = cuEventSynchronize(event);
    switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    default: // should always return successful
      errorQuda("cuEventSynchronize return error code %d", error);
    }
    return cudaErrorUnknown;
#else
    return cudaEventSynchronize(event);
#endif
  }

} // namespace quda
