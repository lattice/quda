#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>
#include <target_device.h>
#include <quda_omptarget_api.h>

namespace quda
{

  static qudaError_t last_error = QUDA_SUCCESS;
  static std::string last_error_str("OMPTARGET_SUCCESS");

  qudaError_t qudaGetLastError()
  {
    auto rtn = last_error;
    last_error = QUDA_SUCCESS;
    return rtn;
  }

  std::string qudaGetLastErrorString()
  {
    auto rtn = last_error_str;
    last_error_str = "OMPTARGET_SUCCESS";
    return rtn;
  }

  namespace device {
    void *constant_arg_buffer;
  }

  namespace target {
    namespace omptarget {
      struct {
        dim3 block, grid;
      } launch_param_host;

      dim3 & launch_param_block(void)
      {
        return launch_param_host.block;
      }

      dim3 & launch_param_grid(void)
      {
        return launch_param_host.grid;
      }

      int qudaSetupLaunchParameter(const TuneParam &tp)
      {
        static int init = 0;
        if(!init){
          int dev = omp_get_default_device();
          device::constant_arg_buffer = omp_target_alloc_host(device::max_constant_size(), dev);
          if(!device::constant_arg_buffer){
            errorQuda("failed to allocate %lu bytes host memory for kernel arguments.", device::max_constant_size());
            return -1;
          }
          init = 1;
        }
        launch_param_host.grid = tp.grid;
        launch_param_host.block = tp.block;
        return 0;
      }

      static inline int
      ompMemset(void *p, unsigned char b, std::size_t s)
      {
        constexpr int max_threads = device::max_block_size();
        constexpr size_t nb = sizeof(float);
        if(s%nb==0){
          float *c = reinterpret_cast<float *>(p);
          float f;
          unsigned char bs[nb];
          for(std::size_t i=0; i<nb; ++i) bs[i] = b;
          memcpy(&f, bs, nb);
          const std::size_t sb = s/nb;
          #pragma omp target teams distribute parallel for simd thread_limit(max_threads) is_device_ptr(c)
          for(std::size_t i=0;i<sb;++i) c[i] = f;
        }else{
          unsigned char *c = reinterpret_cast<unsigned char *>(p);
          #pragma omp target teams distribute parallel for simd thread_limit(max_threads) is_device_ptr(c)
          for(std::size_t i=0;i<s;++i) c[i] = b;
        }
        return 0;
      }

      static inline int
      ompMemsetAsync(void *p, unsigned char b, std::size_t s, qudaStream_t stream)
      {
        return ompMemset(p, b, s);
      }

      static inline int
      ompMemset2D(void *p, size_t pitch, unsigned char b, size_t w, size_t h)
      {
        constexpr int max_threads = device::max_block_size();
        constexpr size_t nb = sizeof(float);
        if(w%nb==0 && pitch%nb==0){
          float *c = reinterpret_cast<float *>(p);
          float f;
          unsigned char bs[nb];
          for(std::size_t i=0; i<nb; ++i) bs[i] = b;
          memcpy(&f, bs, nb);
          const std::size_t wb = w/nb, pitchb = pitch/nb;
          #pragma omp target teams distribute parallel for simd thread_limit(max_threads) is_device_ptr(c) collapse(2)
          for(std::size_t i=0;i<h;++i)
            for(std::size_t j=0;j<wb;++j)
              c[j+i*pitchb] = b;
        }else{
          unsigned char *c = reinterpret_cast<unsigned char *>(p);
          #pragma omp target teams distribute parallel for simd thread_limit(max_threads) is_device_ptr(c) collapse(2)
          for(std::size_t i=0;i<h;++i)
            for(std::size_t j=0;j<w;++j)
              c[j+i*pitch] = b;
        }
        return 0;
      }

      static inline int
      ompMemset2DAsync(void *p, size_t pitch, unsigned char b, size_t w, size_t h, qudaStream_t stream)
      {
        return ompMemset2D(p, pitch, b, w, h);
      }

      static inline int
      ompMemcpy(void *d, void *s, std::size_t c, qudaMemcpyKind k)
      {
        // ompwip("memcpy 0x%p <- 0x%p %d %d\n", d, s, c, k);
        int r = 0;  // return value from omp_target_memcpy, note that no return value is reserved for memcpy.
        switch(k){
        case qudaMemcpyHostToHost:
          memcpy(d,s,c);
          break;
        case qudaMemcpyHostToDevice:
          if(0<omp_get_num_devices()){
            r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_initial_device());
          }else{
            warningQuda("cudaMemcpyHostToDevice without a device, calling memcpy");
            memcpy(d,s,c);
          }
          break;
        case qudaMemcpyDeviceToHost:
          if(0<omp_get_num_devices()){
            r = omp_target_memcpy(d,s,c,0,0,omp_get_initial_device(),omp_get_default_device());
          }else{
            warningQuda("cudaMemcpyDeviceToHost without a device, calling memcpy");
            memcpy(d,s,c);
          }
          break;
        case qudaMemcpyDeviceToDevice:
          r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_default_device());
          break;
        case qudaMemcpyDefault:
          if(0<omp_get_num_devices()){
            if(QUDA_CUDA_FIELD_LOCATION==quda::get_pointer_location(d)){
              if(QUDA_CUDA_FIELD_LOCATION==quda::get_pointer_location(s)){
                r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_default_device());
              }else{
                r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_initial_device());
              }
            }else{
              if(QUDA_CUDA_FIELD_LOCATION==quda::get_pointer_location(s)){
                r = omp_target_memcpy(d,s,c,0,0,omp_get_initial_device(),omp_get_default_device());
              }else{
                memcpy(d,s,c);
              }
            }
          }else{
            warningQuda("cudaMemcpyDefault without a device, calling memcpy");
            memcpy(d,s,c);
          }
          break;
        default: errorQuda("Unsupported qudaMemcpyType %d", k);
        }
        return r;
      }

      static inline int
      ompMemcpyAsync(void *d, void *s, std::size_t c, qudaMemcpyKind k, qudaStream_t stream)
      {
        return ompMemcpy(d, s, c, k);
      }

      void set_runtime_error(int error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error)
      {
        if (error == 0) return;
        last_error = error == 0 ? QUDA_SUCCESS : QUDA_ERROR;
        last_error_str = "OMPTARGET_ERROR";
        if (!allow_error)
          errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, last_error_str.c_str(), file, line, func);
      }
    }
  } // namespace target

  using namespace target::omptarget;

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
        case qudaMemcpyDeviceToHost: name = "qudaMemcpyDeviceToHost"; break;
        case qudaMemcpyHostToDevice: name = "qudaMemcpyHostToDevice"; break;
        case qudaMemcpyHostToHost: name = "qudaMemcpyHostToHost"; break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyDeviceToDevice"; break;
        case qudaMemcpyDefault: name = "qudaMemcpyDefault"; break;
        default: errorQuda("Unsupported qudaMemcpyKind %d", kind);
        }
      } else {
        switch (kind) {
        case qudaMemcpyDeviceToHost: name = "qudaMemcpyAsyncDeviceToHost"; break;
        case qudaMemcpyHostToDevice: name = "qudaMemcpyAsyncHostToDevice"; break;
        case qudaMemcpyHostToHost: name = "qudaMemcpyAsyncHostToHost"; break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyAsyncDeviceToDevice"; break;
        case qudaMemcpyDefault: name = "qudaMemcpyAsyncDefault"; break;
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

    void apply(const qudaStream_t &stream) override
    {
      if (!active_tuning) tuneLaunch(*this, getTuning(), getVerbosity());

      if (copy) {
        if (async) {
          auto error = ompMemcpyAsync(dst, (void*)src, count, kind, stream);
          set_runtime_error(error, "qudaMemcpyAsync", func, file, line, active_tuning);
        } else {
          auto error = ompMemcpy(dst, (void*)src, count, kind);
          set_runtime_error(error, "qudaMemcpy", func, file, line, active_tuning);
        }
      } else {
        auto error
          = async ? ompMemsetAsync(dst, value, count, stream) : ompMemset(dst, value, count);
        set_runtime_error(error, "qudaMemset", func, file, line, active_tuning);
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

    long long bytes() const override { return kind == qudaMemcpyDeviceToDevice ? 2 * count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, src, count, kind, device::get_default_stream(), false, func, file, line);
  }

  void qudaMemcpy_(const quda_ptr &dst, const quda_ptr &src, size_t count, qudaMemcpyKind kind, const char *func,
                   const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst.data(), src.data(), count, kind, device::get_default_stream(), false, func,
                 file, line);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, kind, stream, true, func, file, line);
    } else {
      ompMemcpyAsync(dst, (void*)src, count, kind, stream);
    }
  }

  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream, const char *func,
                           const char *file, const char *line)
  {
    if (count == 0) return;
    auto error = ompMemcpyAsync(dst, (void*)src, count, qudaMemcpyDeviceToDevice, stream);
    set_runtime_error(error, "cudaMemcpyAsync", func, file, line);
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem set(ptr, value, count, device::get_default_stream(), false, func, file, line);
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

  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    QudaMem copy(ptr, value, count, stream, true, func, file, line);
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

  void qudaMemset2DAsync_(quda_ptr &ptr, size_t offset, size_t pitch, int value, size_t width, size_t height,
                          const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    if (ptr.is_device()) {
      auto error = ompMemset2DAsync(static_cast<char *>(ptr.data()) + offset, pitch, value, width, height, stream);
      set_runtime_error(error, __func__, func, file, line);
    } else {
      for (auto i = 0u; i < height; i++) memset(static_cast<char *>(ptr.data()) + offset + i * pitch, value, width);
    }
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
    ompwip("doing nothing");
  }

  constexpr int max_quda_event = 16;
  struct QudaEvent { bool active; double time; };
  static QudaEvent global_quda_event[max_quda_event];

  bool qudaEventQuery_(qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    return true;
  }

  void qudaEventRecord_(qudaEvent_t &quda_event, qudaStream_t stream, const char *func, const char *file, const char *line)
  {
    QudaEvent *e = reinterpret_cast<QudaEvent *>(quda_event.event);
    if(e!=nullptr)
      e->time = omp_get_wtime();
  }

  void qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t quda_event, unsigned int flags, const char *func,
                            const char *file, const char *line)
  {
  }

  qudaEvent_t qudaEventCreate_(const char *func, const char *file, const char *line)
  {
    qudaEvent_t quda_event;
    quda_event.event = nullptr;
    return quda_event;
  }

  qudaEvent_t qudaChronoEventCreate_(const char *func, const char *file, const char *line)
  {
    qudaEvent_t quda_event;
    int i;
    for(i=0;i<max_quda_event;++i)
      if(!global_quda_event[i].active)
        break;
    if(i<max_quda_event){
      global_quda_event[i].active = true;
      global_quda_event[i].time = 0.;
      quda_event.event = reinterpret_cast<void*>(&global_quda_event[i]);
    }else{
      errorQuda("global_quda_event exhausted.");
    }
    return quda_event;
  }

  float qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop, const char *func, const char *file,
                              const char *line)
  {
    return static_cast<float>(reinterpret_cast<QudaEvent *>(stop.event)->time - reinterpret_cast<QudaEvent *>(start.event)->time);
  }

  void qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line)
  {
    QudaEvent *e = reinterpret_cast<QudaEvent *>(event.event);
    if(e!=nullptr){
      e->active = false;
      e->time = 0.;
    }
  }

  void qudaEventSynchronize_(const qudaEvent_t &quda_event, const char *func, const char *file, const char *line)
  {
    // ompwip("unimplemented");
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    // ompwip("unimplemented");
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
    // ompwip("unimplemented");
  }

  void *qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    ompwip("unimplemented");
    return nullptr;
  }

  void printAPIProfile()
  {
    ompwip("unimplemented");
  }

} // namespace quda
