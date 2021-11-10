#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>

// if this macro is defined then we profile the CUDA API calls
//#define API_PROFILE

#ifdef API_PROFILE
#define PROFILE(f, idx)							\
  apiTimer.TPSTART(idx);						\
  f;									\
  apiTimer.TPSTOP(idx);
#else
#define PROFILE(f, idx) f;
#endif

namespace quda
{

  static qudaError_t last_error = QUDA_SUCCESS;
  static std::string last_error_str("QUDA_SUCCESS");

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

  namespace target {

    namespace sycl {

      void set_error(std::string error_str, const char *api_func, const char *func,
		     const char *file, const char *line, bool allow_error)
      {
        last_error = QUDA_ERROR;
        last_error_str = error_str;
        if (!allow_error) errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, error_str.c_str(), file, line, func);
      }

    }

  }

  using namespace target::sycl;

#if 0
  qudaError_t qudaLaunchKernel_(const char *file, const int line,
				const char *func, const char *kern)
  {
    errorQuda("qudaLaunchKernel_ %s %i %s %s\n", file, line, func, kern);
    return QUDA_ERROR;
  }
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
    const bool active_tuning;
    const char *func;
    const char *file;
    const char *line;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

  public:
    inline QudaMem(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
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
        case qudaMemcpyDeviceToHost:   name = "qudaMemcpyDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "qudaMemcpyHostToDevice";   break;
        case qudaMemcpyHostToHost:     name = "qudaMemcpyHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyDeviceToDevice"; break;
        case qudaMemcpyDefault:        name = "qudaMemcpyDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyKind %d", kind);
        }
      } else {
        switch(kind) {
        case qudaMemcpyDeviceToHost:   name = "qudaMemcpyAsyncDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "qudaMemcpyAsyncHostToDevice";   break;
        case qudaMemcpyHostToHost:     name = "qudaMemcpyAsyncHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyAsyncDeviceToDevice"; break;
        case qudaMemcpyDefault:        name = "qudaMemcpyAsyncDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyKind %d", kind);
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
      kind(qudaMemcpyDefault),
      async(async),
      active_tuning(activeTuning())
    {
      name = !async ? "qudaMemset" : "qudaMemsetAsync";
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
      //warningQuda("QudaMem apply %i %i", copy, async);
      if (copy) {
        if (async) {
#ifdef API_PROFILE
          QudaProfileType type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC;
          switch (kind) {
          case qudaMemcpyDeviceToHost: type = QUDA_PROFILE_MEMCPY_D2H_ASYNC; break;
          case qudaMemcpyHostToDevice: type = QUDA_PROFILE_MEMCPY_H2D_ASYNC; break;
          case qudaMemcpyDeviceToDevice: type = QUDA_PROFILE_MEMCPY_D2D_ASYNC; break;
          case qudaMemcpyDefault: type = QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC; break;
          default: errorQuda("Unsupported qudaMemcpyTypeAsync %d", kind);
          }
#endif
          //cudaError_t error;
          //PROFILE(cudaMemcpyAsync(dst, src, count, kind, device::get_cuda_stream(stream)), type);
          //set_runtime_error(error, "cudaMemcpyAsync", func, file, line, active_tuning);
	  auto q = device::get_target_stream(stream);
	  q.memcpy(dst, src, count);
        } else {
          //qudaMemcpy(dst, src, count, kind);
	  auto q = device::get_target_stream(stream);
	  q.memcpy(dst, src, count);
	  q.wait();
        }
      } else {
        if (async) {
          //qudaMemsetAsync(dst, value, count, device::get_quda_stream(stream));
	  auto q = device::get_target_stream(stream);
	  q.memset(dst, value, count);
	} else {
          //qudaMemset(dst, value, count);
	  auto q = device::get_target_stream(stream);
	  q.memset(dst, value, count);
	  q.wait();
	}
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
    long long bytes() const { return kind == qudaMemcpyDeviceToDevice ? 2*count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind,
                   const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, device::get_default_stream(), false, func, file, line);
    qudaError_t error = qudaGetLastError();
    if (error != QUDA_SUCCESS)
      errorQuda("(QUDA) %s\n (%s:%s in %s())\n", qudaGetLastErrorString().c_str(), file, line, func);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, stream, true, func, file, line);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case qudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, device::get_quda_stream(stream)), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case qudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, device::get_quda_stream(stream)), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case qudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_quda_stream(stream)), QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      case qudaMemcpyDefault:
        PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, device::get_quda_stream(stream)), QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
      default:
        errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      //PROFILE(cudaMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), device::get_cuda_stream(stream)),
      //kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
      auto q = device::get_target_stream(stream);
      q.memcpy(dst, src, count);
#endif
    }
  }

  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream,
                           const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    auto q = device::get_target_stream(stream);
    q.memcpy(dst, src, count);
  }

  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch,
		     size_t width, size_t height, qudaMemcpyKind kind,
		     const char *func, const char *file, const char *line)
  {
    auto q = device::defaultQueue();
    char *d = static_cast<char*>(dst);
    const char *s = static_cast<const char*>(src);
    for(int i=0; i<height; i++) {
      q.memcpy(d, s, width);
      d += dpitch;
      s += spitch;
    }
    q.wait();
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
			  size_t width, size_t height, qudaMemcpyKind kind,
			  const qudaStream_t &stream, const char *func,
			  const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    char *d = static_cast<char*>(dst);
    const char *s = static_cast<const char*>(src);
    for(int i=0; i<height; i++) {
      q.memcpy(d, s, width);
      d += dpitch;
      s += spitch;
    }
  }

  void qudaMemcpy2DP2PAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
			     size_t width, size_t height, const qudaStream_t &stream,
			     const char *func, const char *file, const char *line)
  {
    errorQuda("qudaMemcpy2DP2PAsync_ unimplemented\n");
#if 0
    auto error = qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, qudaMemcpyDeviceToDevice, device::get_quda_stream(stream));
    if (error != qudaSuccess)
      errorQuda("qudaMemcpy2DAsync returned %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func,
		   const char *file, const char *line)
  {
    if (count == 0) return;
    auto stream = device::get_default_stream();
    QudaMem set(ptr, value, count, stream, false, func, file, line);
    qudaError_t error = qudaGetLastError();
    if (error != QUDA_SUCCESS && !activeTuning()) {
      errorQuda("(QUDA) %s\n (%s:%s in %s())\n", qudaGetLastErrorString().c_str(),
		file, line, func);
    }
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream,
			const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, stream, true, func, file, line);
    qudaError_t error = qudaGetLastError();
    if (error != QUDA_SUCCESS) {
      errorQuda("(QUDA) %s\n (%s:%s in %s())\n", qudaGetLastErrorString().c_str(),
		file, line, func);
    }
  }

  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height,
		     const char *func, const char *file, const char *line)
  {
    auto q = device::defaultQueue();
    char *p = static_cast<char*>(ptr);
    for(int i=0; i<height; i++) {
      q.memset(p, value, width);
      p += pitch;
    }
    q.wait();
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width,
			  size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    char *p = static_cast<char*>(ptr);
    for(int i=0; i<height; i++) {
      q.memset(p, value, width);
      p += pitch;
    }
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space,
			     const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
    int dev_id = 0;
    if (mem_space == QUDA_CUDA_FIELD_LOCATION)
      dev_id = comm_gpuid();
    else if (mem_space == QUDA_CPU_FIELD_LOCATION)
      dev_id = 0; //cudaCpuDeviceId;
    else
      errorQuda("Invalid QudaFieldLocation.");

    errorQuda("qudaMemPrefetchAsync_ unimplemented\n");
#if 0
    qudaError_t error = qudaMemPrefetchAsync(ptr, count, dev_id, device::get_quda_stream(stream));
    if (error != qudaSuccess) errorQuda("(QUDA) %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }

  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    auto pe = reinterpret_cast<sycl::event *>(quda_event.event);
    auto status = (*pe).get_info<sycl::info::event::command_execution_status>();
    auto val = false;
    if(status==sycl::info::event_command_status::complete) val=true;
    return val;
  }

  void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    auto pe = reinterpret_cast<sycl::event *>(quda_event.event);
    auto q = device::get_target_stream(stream);
    //*pe = q.submit([&](sycl::handler& cgh) {
    //	     cgh.single_task<class EventRecord>([=](){});
    //		     //cgh.codeplay_host_task([=](){});
    //		   });
    *pe = q.submit_barrier();
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t quda_event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
    auto pe = reinterpret_cast<sycl::event *>(quda_event.event);
    (*pe).wait();
  }

  qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line)
  {
    qudaEvent_t quda_event;
    auto sycl_event = new sycl::event;
    quda_event.event = sycl_event;
    return quda_event;
  }

  qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line)
  {
    qudaEvent_t quda_event;
    auto sycl_event = new sycl::event;
    quda_event.event = sycl_event;
    return quda_event;
  }

  float qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop,
                              const char *func, const char *file, const char *line)
  {
    auto pe0 = reinterpret_cast<sycl::event *>(start.event);
    auto pe1 = reinterpret_cast<sycl::event *>(stop.event);
    (*pe1).wait();
    (*pe0).wait();
    auto t0 = (*pe0).get_profiling_info<sycl::info::event_profiling::command_end>();
    auto t1 = (*pe1).get_profiling_info<sycl::info::event_profiling::command_start>();
    auto elapsed_time = 1e-9*(t1-t0);
    return elapsed_time;
  }

  void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    auto pe = reinterpret_cast<sycl::event *>(event.event);
    delete pe;
  }

  void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    auto pe = reinterpret_cast<sycl::event *>(quda_event.event);
    (*pe).wait();
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    //q.wait();
    q.wait_and_throw();
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    int n = device::get_default_stream_idx();
    for(int i=0; i<=n; i++) {
      auto s = device::get_stream(i);
      qudaStreamSynchronize_(s, func, file, line);
    }
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
