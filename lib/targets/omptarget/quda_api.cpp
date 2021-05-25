#include <unordered_set>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>
#include <timer.h>
#include <device.h>

using cudaError_t = int;
enum {cudaSuccess};
enum cudaMemcpyKind{cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault};

static char FIXME[]="OMP FIXME";

#define cudaGetErrorString(a) "OMP FIXME"
#define cudaGetLastError()
#define cuGetErrorName(a,b) ompwip([&](){(*(b))=FIXME;})

#if 0

#define cudaMemcpy(a,b,c,d) ompwip([&](){ompwipMemcpy(a,(void*)b,c,d);},"memcpy %p <- %p %ld",a,b,c)
#define cudaMemcpyAsync(a,b,c,d,e) ompwip([&](){ompwipMemcpy(a,(void*)b,c,d);},"memcpy %p <- %p %ld",a,b,c)
#define cudaMemcpy2D(d,dp,s,sp,w,h,k) ompwip([&](){ompwipMemcpy2D(d,dp,(void*)s,sp,w,h,k);},"memcpy %p(%ld) <- %p(%ld) %ld %ld",d,dp,s,sp,w,h)
#define cudaMemcpy2DAsync(d,dp,s,sp,w,h,k,st) ompwip([&](){ompwipMemcpy2D(d,dp,(void*)s,sp,w,h,k);},"memcpy %p(%ld) <- %p(%ld) %ld %ld",d,dp,s,sp,w,h)
#define cudaMemset(a,b,c) ompwip([&](){ompwipMemset(a,b,c);},"memset %p %d %ld",a,b,c)
#define cudaMemsetAsync(a,b,c,d) ompwip([&](){ompwipMemset(a,b,c);},"memset %p %d %ld",a,b,c)
#define cudaMemset2D(a,b,c,d,e) ompwip([&](){ompwipMemset2D(a,b,c,d,e);},"memset %p(%ld) %d %ld %ld",a,b,c,d,e)
#define cudaMemset2DAsync(a,b,c,d,e,f) ompwip([&](){ompwipMemset2D(a,b,c,d,e);},"memset %p(%ld) %d %ld %ld",a,b,c,d,e)

#else

#define cudaMemcpy(a,b,c,d) ompwipMemcpy(a,(void*)b,c,d)
#define cudaMemcpyAsync(a,b,c,d,e) ompwipMemcpy(a,(void*)b,c,d)
#define cudaMemcpy2D(d,dp,s,sp,w,h,k) ompwipMemcpy2D(d,dp,(void*)s,sp,w,h,k)
#define cudaMemcpy2DAsync(d,dp,s,sp,w,h,k,st) ompwipMemcpy2D(d,dp,(void*)s,sp,w,h,k)
#define cudaMemset(a,b,c) ompwipMemset(a,b,c)
#define cudaMemsetAsync(a,b,c,d) ompwipMemset(a,b,c)
#define cudaMemset2D(a,b,c,d,e) ompwipMemset2D(a,b,c,d,e)
#define cudaMemset2DAsync(a,b,c,d,e,f) ompwipMemset2D(a,b,c,d,e)

#endif

static inline int
ompwipMemset(void *p, unsigned char b, std::size_t s)
{
  unsigned char *c = reinterpret_cast<unsigned char *>(p);
#pragma omp target teams distribute parallel for simd is_device_ptr(c)
  for(std::size_t i=0;i<s;++i) c[i] = b;
  return 0;
}

static inline int
ompwipMemset2D(void *p, size_t pitch, unsigned char b, size_t w, size_t h)
{
  unsigned char *c = reinterpret_cast<unsigned char *>(p);
#pragma omp target teams distribute parallel for simd is_device_ptr(c) collapse(2)
  for(std::size_t i=0;i<h;++i)
    for(std::size_t j=0;j<w;++j)
      c[j+i*pitch] = b;
  return 0;
}

#if 1
#define printmem(d,m,h)
#else
static inline void
printmem(void *d, std::size_t m, int host)
{
  constexpr size_t loc[] = {0, 1, 2, 3, 5, 7, 11, 13};
  constexpr size_t nloc = sizeof(loc)/sizeof(loc[0]);
  unsigned char buf[nloc*8];
  unsigned char *p = (unsigned char *)d;

  if(host){
    for(size_t i=0; i<nloc; ++i){
      if(i*8>=m) break;
      for(size_t j=0; j<8; ++j)
        buf[j+i*8] = p[j+loc[i]*8];
    }
    printf("host   mem:");
  }else{
    #pragma omp target map(tofrom:buf[:8*nloc]) map(to:loc[:nloc]) is_device_ptr(p)
    {
    for(size_t i=0; i<nloc; ++i){
      if(i*8>=m) break;
      for(size_t j=0; j<8; ++j)
        buf[j+i*8] = p[j+loc[i]*8];
    }
    }
    printf("target mem:");
  }
  for(size_t i=0; i<nloc; ++i){
    printf(" ");
    for(size_t j=0; j<8; ++j)
      printf("%02X", buf[j+i*8]);
  }
  printf("\n");
  fflush(0);
}
#endif

static inline int
ompwipMemcpy(void *d, void *s, std::size_t c, cudaMemcpyKind k)
{
  int r = 0;  // return value from omp_target_memcpy, note that no return value is reserved for memcpy.
  switch(k){
  case cudaMemcpyHostToHost:
    printmem(s,c,1);
    memcpy(d,s,c);
    printmem(d,c,1);
    break;
  case cudaMemcpyHostToDevice:
    printmem(s,c,1);
    if(0<omp_get_num_devices()){
      r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_initial_device());
      printmem(d,c,0);
    }else{
      warningQuda("cudaMemcpyHostToDevice without a device, calling memcpy");
      memcpy(d,s,c);
      printmem(d,c,1);
    }
    break;
  case cudaMemcpyDeviceToHost:
    if(0<omp_get_num_devices()){
      printmem(s,c,0);
      r = omp_target_memcpy(d,s,c,0,0,omp_get_initial_device(),omp_get_default_device());
    }else{
      warningQuda("cudaMemcpyDeviceToHost without a device, calling memcpy");
      printmem(s,c,1);
      memcpy(d,s,c);
    }
    printmem(d,c,1);
    break;
  case cudaMemcpyDeviceToDevice:
    printmem(s,c,0);
    r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_default_device());
    printmem(d,c,0);
    break;
  case cudaMemcpyDefault:
    if(0<omp_get_num_devices()){
      if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(d)){
        if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(s)){
          // ompwip("cudaMemcpyDefault calling device to device");
          printmem(s,c,0);
          r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_default_device());
          printmem(d,c,0);
        }else{
          // ompwip("cudaMemcpyDefault calling host to device");
          printmem(s,c,1);
          r = omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_initial_device());
          printmem(d,c,0);
        }
      }else{
        if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(s)){
          // ompwip("cudaMemcpyDefault calling device to host");
          printmem(s,c,0);
          r = omp_target_memcpy(d,s,c,0,0,omp_get_initial_device(),omp_get_default_device());
          printmem(d,c,1);
        }else{
          // ompwip("cudaMemcpyDefault calling host to host");
          printmem(s,c,1);
          memcpy(d,s,c);
          printmem(d,c,1);
        }
      }
    }else{
      warningQuda("cudaMemcpyDefault without a device, calling memcpy");
      printmem(s,c,1);
      memcpy(d,s,c);
      printmem(d,c,1);
    }
    break;
  default: errorQuda("Unsupported cudaMemcpyType %d", k);
  }
  return r;
}

static inline int
ompwipMemcpy2D(void *d, std::size_t dp, void *s, std::size_t sp, std::size_t w, std::size_t h, cudaMemcpyKind k)
{
  int r = 0;  // return value from omp_target_memcpy, note that no return value is reserved for memcpy.
  std::size_t v[2] = {w,h}, z[2] = {0,0}, dd[2] = {dp,h}, sd[2] = {sp,h};
  unsigned char *cd = reinterpret_cast<unsigned char *>(d), *cs = reinterpret_cast<unsigned char *>(s);
  switch(k){
  case cudaMemcpyHostToHost:
    printmem(s,c,1);
    #pragma omp parallel for
    for(std::size_t i=0;i<h;++i) memcpy(cd+i*dp,cs+i*sp,w);
    printmem(d,c,1);
    break;
  case cudaMemcpyHostToDevice:
    printmem(s,c,1);
    if(0<omp_get_num_devices()){
      r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_default_device(),omp_get_initial_device());
      printmem(d,c,0);
    }else{
      warningQuda("cudaMemcpyHostToDevice without a device, calling memcpy");
      #pragma omp parallel for
      for(std::size_t i=0;i<h;++i) memcpy(cd+i*dp,cs+i*sp,w);
      printmem(d,c,1);
    }
    break;
  case cudaMemcpyDeviceToHost:
    if(0<omp_get_num_devices()){
      printmem(s,c,0);
      r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_initial_device(),omp_get_default_device());
    }else{
      warningQuda("cudaMemcpyDeviceToHost without a device, calling memcpy");
      printmem(s,c,1);
      #pragma omp parallel for
      for(std::size_t i=0;i<h;++i) memcpy(cd+i*dp,cs+i*sp,w);
    }
    printmem(d,c,1);
    break;
  case cudaMemcpyDeviceToDevice:
    printmem(s,c,0);
    r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_default_device(),omp_get_default_device());
    printmem(d,c,0);
    break;
  case cudaMemcpyDefault:
    if(0<omp_get_num_devices()){
      if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(d)){
        if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(s)){
          // ompwip("cudaMemcpyDefault calling device to device");
          printmem(s,c,0);
          r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_default_device(),omp_get_default_device());
          printmem(d,c,0);
        }else{
          // ompwip("cudaMemcpyDefault calling host to device");
          printmem(s,c,1);
          r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_default_device(),omp_get_initial_device());
          printmem(d,c,0);
        }
      }else{
        if(QUDA_CUDA_FIELD_LOCATION==get_pointer_location(s)){
          // ompwip("cudaMemcpyDefault calling device to host");
          printmem(s,c,0);
          r = omp_target_memcpy_rect(d,s,1,2,v,z,z,dd,sd,omp_get_initial_device(),omp_get_default_device());
          printmem(d,c,1);
        }else{
          // ompwip("cudaMemcpyDefault calling host to host");
          printmem(s,c,1);
          #pragma omp parallel for
          for(std::size_t i=0;i<h;++i) memcpy(cd+i*dp,cs+i*sp,w);
          printmem(d,c,1);
        }
      }
    }else{
      warningQuda("cudaMemcpyDefault without a device, calling memcpy");
      printmem(s,c,1);
      #pragma omp parallel for
      for(std::size_t i=0;i<h;++i) memcpy(cd+i*dp,cs+i*sp,w);
      printmem(d,c,1);
    }
    break;
  default: errorQuda("Unsupported cudaMemcpyType %d", k);
  }
  return r;
}

#define cudaGetSymbolAddress(a,b) ompwip("WARNING: unimplemented cudaGetSymbolAddress")

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

LaunchParam launch_param;
#pragma omp declare target to(launch_param)

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
    last_error_str = QUDA_SUCCESS;
    return rtn;
  }

  void qudaSetupLaunchParameter(const TuneParam &tp)
  {
    launch_param.grid = tp.grid;
    launch_param.block = tp.block;
    #pragma omp target update to(launch_param)
  }

  namespace cuda {

    void set_runtime_error(cudaError_t error, const char *api_func, const char *func, const char *file, const char *line,
                           bool allow_error = false)
    {
      if (error == cudaSuccess) return;
      last_error = error == cudaSuccess ? QUDA_SUCCESS : QUDA_ERROR;
      last_error_str = cudaGetErrorString(error);
      if (!allow_error) errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, cudaGetErrorString(error), file, line, func);
      else cudaGetLastError(); // clear the error state
    }

    void set_driver_error(CUresult error, const char *api_func, const char *func, const char *file, const char *line,
                          bool allow_error = false)
    {
      if (error == CUDA_SUCCESS) return;
      last_error = error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
      const char *str;
      cuGetErrorName(error, &str);
      last_error_str = str;
      if (!allow_error) errorQuda("%s returned %s\n (%s:%s in %s())\n", api_func, str, file, line, func);
      else cudaGetLastError(); // clear the error state
    }

  }

  using namespace cuda;

  // Agnostic way to return a cuda API flag
  namespace {
    inline
    cudaMemcpyKind qudaMemcpyKindToAPI( const qudaMemcpyKind& k)
    {
      switch(k) {
      case qudaMemcpyHostToHost : return cudaMemcpyHostToHost;
      case qudaMemcpyHostToDevice : return cudaMemcpyHostToDevice;
      case qudaMemcpyDeviceToHost : return cudaMemcpyDeviceToHost;
      case qudaMemcpyDeviceToDevice : return cudaMemcpyDeviceToDevice;
      case qudaMemcpyDefault : return cudaMemcpyDefault;
      default:
	errorQuda(" unknown value for qudaMemcpyKind %d", static_cast<int>(k));
	return cudaMemcpyDefault; // keep warnings away
      }
    }
  }

#ifdef USE_DRIVER_API
  static TimeProfile apiTimer("CUDA API calls (driver)");
#else
  static TimeProfile apiTimer("CUDA API calls (runtime)");
#endif

  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream)
  {
    ompwip([&](){std::cerr<<"ERROR: unimplemented launch "<<func<<' '<<tp<<' '<<*args<<' '<<device::get_cuda_stream(stream)<<std::endl;});
    // if launch requests the maximum shared memory and the device supports it then opt in
    if (tp.set_max_shared_bytes && device::max_dynamic_shared_memory() > device::max_default_shared_memory()) {
      ompwip("Unimplemented for maximum shared memory");
    }
    return QUDA_SUCCESS;
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

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

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

    inline void apply(const qudaStream_t &stream)
    {
      if (!active_tuning) tuneLaunch(*this, getTuning(), getVerbosity());

      if (copy) {
        if (async) {
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
          PROFILE(error = cudaMemcpyAsync(dst, src, count, kind, device::get_cuda_stream(stream)), type);
          set_runtime_error(error, "cudaMemcpyAsync", func, file, line, active_tuning);
        } else {
          cudaError_t error = cudaMemcpy(dst, src, count, kind);
          set_runtime_error(error, "cudaMemcpy", func, file, line, active_tuning);
        }
      } else {
        cudaError_t error = async ?
          cudaMemsetAsync(dst, value, count, device::get_cuda_stream(stream)) :
          cudaMemset(dst, value, count);
        set_runtime_error(error, "cudaMemset", func, file, line, active_tuning);
      }
    }

    bool advanceTuneParam(TuneParam &) const { return false; }

    TuneKey tuneKey() const
    {
      char vol[128];
      strcpy(vol, "bytes=");
      u64toa(vol + 6, (uint64_t)count);
      return TuneKey(vol, name, aux);
    }

    long long flops() const { return 0; }
    long long bytes() const { return kind == cudaMemcpyDeviceToDevice ? 2 * count : count; }
  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind,
                   const char *func, const char *file, const char *line)
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
      PROFILE(cudaMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), device::get_cuda_stream(stream)),
              kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
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
                     qudaMemcpyKind kind, const char *func, const char *file, const char *line)
  {
    PROFILE(auto error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, qudaMemcpyKindToAPI(kind)), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    if (error != cudaSuccess)
      errorQuda("cudaMemcpy2D returned error %s\n (%s:%s in %s())\n", cudaGetErrorString(error), file, line, func);
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                          qudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line)
  {
    PROFILE(auto error = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, qudaMemcpyKindToAPI(kind), device::get_cuda_stream(stream)), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
    set_runtime_error(error, "cudaMemcpy2DAsync", func, file, line);
  }

  void qudaMemcpy2DP2PAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                             const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
    auto error = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, device::get_cuda_stream(stream));
    set_runtime_error(error, "cudaMemcpy2DAsync", func, file, line);
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
    cudaError_t error = cudaMemset2DAsync(ptr, pitch, value, width, height, device::get_cuda_stream(stream));
    set_runtime_error(error, __func__, func, file, line);
  }

  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line)
  {
    ompwip("doing nothing");
/*
    int dev_id = 0;
    if (mem_space == QUDA_CUDA_FIELD_LOCATION)
      dev_id = comm_gpuid();
    else if (mem_space == QUDA_CPU_FIELD_LOCATION)
      dev_id = cudaCpuDeviceId;
    else
      errorQuda("Invalid QudaFieldLocation.");

    cudaError_t error = cudaMemPrefetchAsync(ptr, count, dev_id, device::get_cuda_stream(stream));
    set_runtime_error(error, __func__, func, file, line);
*/
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

  double qudaEventElapsedTime_(const qudaEvent_t &start, const qudaEvent_t &stop,
                              const char *func, const char *file, const char *line)
  {
    return reinterpret_cast<QudaEvent *>(stop.event)->time - reinterpret_cast<QudaEvent *>(start.event)->time;
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
  }

  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line)
  {
  }

  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line)
  {
  }

  void* qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line)
  {
    void *ptr;
    cudaError_t error = cudaGetSymbolAddress(&ptr, symbol);
    set_runtime_error(error, __func__, func, file, line);
    return ptr;
  }

  void printAPIProfile()
  {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
