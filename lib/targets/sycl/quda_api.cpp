#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>
#include <quda_sycl_api.h>

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
	  device::wasSynced(stream);
	  q.wait_and_throw();
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
	  device::wasSynced(stream);
	  q.wait_and_throw();
	}
      }
      qudaError_t error = qudaGetLastError();
      if (error != QUDA_SUCCESS)
	errorQuda("(QUDA) %s\n (%s:%s in %s())\n", qudaGetLastErrorString().c_str(), file, line, func);
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
  }

  void qudaMemcpy_(const quda_ptr &dst, const quda_ptr &src, size_t count, qudaMemcpyKind kind, const char *func,
                   const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst.data(), src.data(), count, kind, device::get_default_stream(), false, func, file, line);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, stream, true, func, file, line);
    } else {
      //PROFILE(cudaMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), device::get_cuda_stream(stream)),
      //kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
      auto q = device::get_target_stream(stream);
      q.memcpy(dst, src, count);
    }
  }

  //void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream,
  //                         const char *func, const char *file, const char *line)
  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream,
                           const char *, const char *, const char *)
  {
    if (count == 0) return;
    auto q = device::get_target_stream(stream);
    q.memcpy(dst, src, count);
  }

  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch,
		     size_t width, size_t height, qudaMemcpyKind,
		     const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    auto q = device::defaultQueue();
    char *d = static_cast<char*>(dst);
    const char *s = static_cast<const char*>(src);
    for(size_t i=0; i<height; i++) {
      q.memcpy(d, s, width);
      d += dpitch;
      s += spitch;
    }
    device::wasSynced(device::get_default_stream());
    q.wait_and_throw();
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
			  size_t width, size_t height, qudaMemcpyKind, const qudaStream_t &stream,
			  const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    char *d = static_cast<char*>(dst);
    const char *s = static_cast<const char*>(src);
    for(size_t i=0; i<height; i++) {
      q.memcpy(d, s, width);
      d += dpitch;
      s += spitch;
    }
  }

#if 0
  void qudaMemcpy2DP2PAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
			     size_t width, size_t height, const qudaStream_t &stream,
			     const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    errorQuda("qudaMemcpy2DP2PAsync_ unimplemented\n");
#if 0
    auto error = qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, qudaMemcpyDeviceToDevice, device::get_quda_stream(stream));
    if (error != qudaSuccess)
      errorQuda("qudaMemcpy2DAsync returned %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
#endif
  }
#endif

  void qudaMemset_(void *ptr, int value, size_t count, const char *func,
		   const char *file, const char *line)
  {
    if (count == 0) return;
    auto stream = device::get_default_stream();
    QudaMem set(ptr, value, count, stream, false, func, file, line);
  }

  void qudaMemset_(quda_ptr &ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    if (ptr.is_device()) {
      QudaMem set(ptr.data(), value, count, device::get_default_stream(), false, func, file, line);
    } else {
      memset(ptr.data(), value, count);
    }
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream,
			const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, stream, true, func, file, line);
  }

  void qudaMemsetAsync_(quda_ptr &ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    if (ptr.is_device()) {
      QudaMem set(ptr.data(), value, count, stream, true, func, file, line);
    } else {
      memset(ptr.data(), value, count);
    }
  }

#if 0
  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height,
		     const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    auto q = device::defaultQueue();
    char *p = static_cast<char*>(ptr);
    for(size_t i=0; i<height; i++) {
      q.memset(p, value, width);
      p += pitch;
    }
    device::wasSynced(device::get_default_stream());
    q.wait_and_throw();
  }

  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width,
			  size_t height, const qudaStream_t &stream,
                          const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    char *p = static_cast<char*>(ptr);
    for(size_t i=0; i<height; i++) {
      q.memset(p, value, width);
      p += pitch;
    }
  }
#endif

  void qudaMemset2DAsync_(quda_ptr &ptr, size_t offset, size_t pitch, int value, size_t width, size_t height,
                          const qudaStream_t &stream, const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    if (ptr.is_device()) {
      auto q = device::get_target_stream(stream);
      char *p = static_cast<char*>(ptr.data());
      for(size_t i=0; i<height; i++) {
	q.memset(p, value, width);
	p += pitch;
      }
    } else {
      for (auto i = 0u; i < height; i++) memset(static_cast<char *>(ptr.data()) + offset + i * pitch, value, width);
    }
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation,
			     const qudaStream_t &stream,
                             const char *, const char *, const char *)
  //                         const char *func, const char *file, const char *line)
  {
    auto q = device::get_target_stream(stream);
    q.prefetch(ptr, count);
  }

  typedef struct {
    qudaStream_t stream;
    size_t eventIdx;
    sycl::event event;
  } EventImpl;

  //bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *, const char *, const char *)
  {
    auto pe = reinterpret_cast<EventImpl *>(quda_event.event);
    auto status = pe->event.get_info<sycl::info::event::command_execution_status>();
    auto val = false;
    if(status==sycl::info::event_command_status::complete) val=true;
    return val;
  }

  //void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *func, const char *file, const char *line)
  void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *, const char *, const char *)
  {
    auto pe = reinterpret_cast<EventImpl *>(quda_event.event);
    auto q = device::get_target_stream(stream);
    pe->stream = stream;
    pe->eventIdx = device::getEventIdx(stream);
    //pe->event = q.submit([&](sycl::handler& cgh) {
    //cgh.single_task<class EventRecord>([=](){});
    //cgh.host_task([=](){});
    //});
    //*pe = q.submit_barrier();
    pe->event = q.ext_oneapi_submit_barrier();
  }

  //void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t quda_event, unsigned int flags,
  //const char *func, const char *file, const char *line)
  void qudaStreamWaitEvent_(qudaStream_t, qudaEvent_t quda_event, unsigned int,
			    const char *, const char *, const char *)
  {
    auto pe = reinterpret_cast<EventImpl *>(quda_event.event);
    device::wasSynced(pe->stream, pe->eventIdx);
    pe->event.wait_and_throw();
  }

  //qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line)
  qudaEvent_t qudaEventCreate_(const char *, const char *, const char *)
  {
    qudaEvent_t quda_event;
    auto e = new EventImpl;
    quda_event.event = e;
    return quda_event;
  }

  //qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line)
  qudaEvent_t qudaChronoEventCreate_(const char *, const char *, const char *)
  {
    qudaEvent_t quda_event;
    auto e = new EventImpl;
    quda_event.event = e;
    return quda_event;
  }

  float qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop,
                              const char *, const char *, const char *)
  //const char *func, const char *file, const char *line)
  {
    auto pe0 = reinterpret_cast<EventImpl *>(start.event);
    auto pe1 = reinterpret_cast<EventImpl *>(stop.event);
    device::wasSynced(pe0->stream, pe0->eventIdx);
    pe0->event.wait_and_throw();
    auto t0 = pe0->event.get_profiling_info<sycl::info::event_profiling::command_end>();
    device::wasSynced(pe1->stream, pe1->eventIdx);
    pe1->event.wait_and_throw();
    auto t1 = pe1->event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto elapsed_time = 1e-9*(t1-t0);
    //printfQuda("qudaEventElapsedTime: %lu %lu %g\n", t0, t1, elapsed_time);
    return elapsed_time;
  }

  //void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  void qudaEventDestroy_(qudaEvent_t &event, const char *, const char *, const char *)
  {
    auto pe = reinterpret_cast<EventImpl *>(event.event);
    delete pe;
  }

  //void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *, const char *, const char *)
  {
    auto pe = reinterpret_cast<EventImpl *>(quda_event.event);
    device::wasSynced(pe->stream, pe->eventIdx);
    pe->event.wait_and_throw();
  }

  //void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *, const char *, const char *)
  {
    auto q = device::get_target_stream(stream);
    device::wasSynced(stream);
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
