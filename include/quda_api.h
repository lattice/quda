#pragma once

#ifdef QUDA_BACKEND_OMPTARGET

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <functional>
#include <string_view>

#include <util_quda.h>

template <typename T>
constexpr auto type_name() noexcept {
  std::string_view name = "Error: unsupported compiler", prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto type_name() [with T = ";
  suffix = "]";
#endif
  name.remove_prefix(prefix.size());
  name.remove_suffix(suffix.size());
  return name;
}

template <typename...Arg>
struct seq_args_call{
  static constexpr size_t n = sizeof...(Arg);
  template <typename F, std::size_t...IX>
  void operator()(F *func, void *args[n], std::index_sequence<IX...>)
  {(*func)(*(Arg*)args[IX]...);}
};

template <typename T>
T * to_device(const T * x, size_t s) {
  if(0<omp_get_num_devices()){
    const int d = omp_get_default_device();
    const int h = omp_get_initial_device();
    void *p = omp_target_alloc(s, d);
    printfQuda("# to_device: host_ptr@%d = %p  device_ptr@%d = %p  size = %zu\n", h, x, d, p, s);
    omp_target_memcpy(p, (void *)x, s, 0, 0, d, h);
    return (T *)p;
  }else{
    return x;
  }
}

template <typename T>
T * to_device(const T& x) {
  constexpr size_t s = sizeof(T);
  return to_device(&x, s);
}

#define __host__
#define __device__
#define __shared__
#define __global__
#define __constant__ static
#define __launch_bounds__(x)
#define __syncthreads() _Pragma("omp barrier")

#define CUDA_SUCCESS QUDA_SUCCESS

using size_t = std::size_t;

#define __forceinline__ inline __attribute__((always_inline))

enum cudaMemcpyKind{cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault};
enum cudaError_t{cudaSuccess,cudaErrorNotReady};
enum {cudaEventDisableTiming,cudaEventInterprocess};
enum {cudaHostRegisterDefault};
enum {cudaIpcMemLazyEnablePeerAccess};
enum {cudaStreamDefault};

static inline cudaError_t
ompwip_(const char * const file, const size_t line, const char * const func, const char * const msg, std::function<void(void)>f = [](){})
{
  if(0==omp_get_team_num()&&0==omp_get_thread_num()) std::cerr<<"OMP WIP:"<<msg<<": "<<file<<':'<<line<<' '<<func<<std::endl;
  f();
  return cudaSuccess;
}
static inline cudaError_t
ompwip_(const char * const file, const size_t line, const char * const func, std::function<void(void)>f = [](){})
{return ompwip_(file,line,func,"",f);}
#define ompwip(...) ompwip_(__FILE__,__LINE__,__PRETTY_FUNCTION__,##__VA_ARGS__)

#define __shfl_down_sync(a,b,c) ompwip("__shfl_down_sync")

#define cuMemcpy(a,b,c) ompwip()
#define cuMemcpyAsync(a,b,c,d) ompwip()
#define cuMemcpyDtoD(a,b,c) ompwip()
#define cuMemcpyDtoDAsync(a,b,c,d) ompwip()
#define cuMemcpyDtoH(a,b,c) ompwip()
#define cuMemcpyDtoHAsync(a,b,c,d) ompwip()
#define cuMemcpyHtoD(a,b,c) ompwip()
#define cuMemcpyHtoDAsync(a,b,c,d) ompwip()
#define cudaMemcpy(a,b,c,d) ompwip([&](){printfQuda("memcpy %p <- %p\n",a,b);ompwipMemcpy(a,(void*)b,c,d);})
#define cudaMemcpy2D(a,b,c,d,e,f,g) ompwip()
#define cudaMemcpy2DAsync(a,b,c,d,e,f,g,h) ompwip()
#define cudaMemcpyAsync(a,b,c,d,e) ompwip()
#define cudaMemcpyToSymbolAsync(a,b,c,d,e,f) ompwip()
#define cudaMemset(a,b,c) ompwip([&](){printfQuda("memset %p\n",a);ompwipMemset(a,b,c);})
#define cudaMemset2D(a,b,c,d,e) ompwip()
#define cudaMemset2DAsync(a,b,c,d,e,f) ompwip()
#define cudaMemsetAsync(a,b,c,d) ompwip()

static inline void
ompwipMemset(void *p, unsigned char b, std::size_t s)
{
#pragma omp target teams distribute parallel for simd is_device_ptr(p)
  for(std::size_t i=0;i<s;++i) *(unsigned char *)p = b;
}

static inline void
ompwipMemcpy(void *d, void *s, std::size_t c, cudaMemcpyKind k)
{
  switch(k){
  case cudaMemcpyHostToHost: memcpy(d,s,c); break;
  case cudaMemcpyHostToDevice:
    if(0<omp_get_num_devices()) omp_target_memcpy(d,s,c,0,0,omp_get_default_device(),omp_get_initial_device());
    else warningQuda("cudaMemcpyHostToDevice without a device");
    break;
  case cudaMemcpyDeviceToHost:
    if(0<omp_get_num_devices()) omp_target_memcpy(d,s,c,0,0,omp_get_initial_device(),omp_get_default_device());
    else warningQuda("cudaMemcpyDeviceToHost without a device");
    break;
  case cudaMemcpyDeviceToDevice:
    warningQuda("unimplemented for cudaMemcpyDeviceToDevice");
    break;
  case cudaMemcpyDefault:
    warningQuda("unimplemented for cudaMemcpyDefault");
    break;
  default: errorQuda("Unsupported cudaMemcpyType %d", k);
  }
}

#define cudaDeviceSynchronize() ompwip()
#define cudaEventCreate(a,...) ompwip([&](){(*(a))=0;})
#define cudaEventCreateWithFlags(a,b) ompwip()
#define cudaEventDestroy(a) ompwip()
#define cudaEventElapsedTime(a,b,c) ompwip([&](){(*(a))=0;})
#define cudaEventQuery(a) ompwip()
#define cudaEventRecord(a,b) ompwip()
#define cudaEventSynchronize(a) ompwip()
#define cudaFuncGetAttributes(a,b) ompwip()
#define cudaFuncSetAttribute(a,b,c) ompwip()
#define cudaGetErrorString(a) "FIXME"
#define cudaGetLastError() ompwip()
#define cudaGetSymbolAddress(a,b) ompwip()
#define cudaHostRegister(a,b,c) ompwip()
#define cudaHostUnregister(a) ompwip()
#define cudaIpcCloseMemHandle(a) ompwip()
#define cudaIpcGetEventHandle(a,b) ompwip([&](){(*(a))=0;})
#define cudaIpcGetMemHandle(a,b) ompwip([&](){(*(a))=0;})
#define cudaIpcOpenEventHandle(a,b) ompwip([&](){(*(a))=0;})
#define cudaIpcOpenMemHandle(a,b,c) ompwip()
#define cudaStreamCreate(a) ompwip([&](){(*(a))=0;})
#define cudaStreamDestroy(a) ompwip()
#define cudaStreamSynchronize(a) ompwip()
#define cudaStreamWaitEvent(a,b,c) ompwip()

static char FIXME[]="FIXME";
#define cuGetErrorString(a,b) ompwip([&](){(*(b))=FIXME;})
#define cuGetErrorName(a,b) ompwip([&](){(*(b))=FIXME;})

using qudaStream_t = int;
using cudaStream_t = int;
using CUresult = int;
using cudaIpcMemHandle_t = int;
using cudaIpcEventHandle_t = int;
using cudaEvent_t = int;

struct dim3 {unsigned int x,y,z;
  constexpr dim3():x(1u),y(1u),z(1u){}
  constexpr dim3(unsigned int x):x(x),y(1u),z(1u){}
  constexpr dim3(unsigned int x,unsigned int y):x(x),y(y),z(1u){}
  constexpr dim3(unsigned int x,unsigned int y,unsigned int z):x(x),y(y),z(z){}
};
struct int2 {int x,y;};
struct int4 {int x,y,z,w;};
static inline int4 make_int4(int x,int y,int z,int w){return int4{x,y,z,w};}
struct uint2 {unsigned int x,y;};
static inline uint2 make_uint2(unsigned int x,unsigned int y){return uint2{x,y};}
struct uint3 {unsigned int x,y,z;};
static inline uint3 make_uint3(unsigned int x,unsigned int y,unsigned int z){return uint3{x,y,z};}
struct char2 {char x,y;};
static inline char2 make_char2(char x,char y){return char2{x,y};}
struct char3 {char x,y,z;};
struct char4 {char x,y,z,w;};
struct short2 {short x,y;};
static inline short2 make_short2(short x,short y){return short2{x,y};}
struct short3 {short x,y,z;};
struct short4 {short x,y,z,w;};
struct float2 {float x,y;};
static inline float2 make_float2(float x,float y){return float2{x,y};}
struct float3 {float x,y,z;};
static inline float3 make_float3(float x,float y,float z){return float3{x,y,z};}
struct float4 {float x,y,z,w;};
static inline float4 make_float4(float x,float y,float z,float w){return float4{x,y,z,w};}
struct double2 {double x,y;};
static inline double2 make_double2(double x,double y){return double2{x,y};}
struct double3 {double x,y,z;};
static inline double3 make_double3(double x,double y,double z){return double3{x,y,z};}
struct double4 {double x,y,z,w;};
static inline double4 make_double4(double x,double y,double z,double w){return double4{x,y,z,w};}

#define QUDA_RT_CONSTS ompwip();const dim3 threadIdx(omp_get_thread_num(),0,0),blockDim(omp_get_num_threads()),blockIdx(omp_get_team_num(),0,0),gridDim(omp_get_num_teams())

#else  // QUDA_BACKEND_OMPTARGET

#define QUDA_RT_CONSTS

#ifndef __CUDACC_RTC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#endif  // QUDA_BACKEND_OMPTARGET

// Target specific definitions in include/targets/XXX/quda_api_target.h
// The correct targets/ directory is set by the build system
// Eventually we want to shrink the contents of that
#include <quda_define.h>

enum qudaMemcpyKind { qudaMemcpyHostToHost, 
	              qudaMemcpyHostToDevice,
		      qudaMemcpyDeviceToHost,
		      qudaMemcpyDeviceToDevice,
		      qudaMemcpyDefault };

#include <string>
#include <enum_quda.h>

/**
   @file quda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
 */

namespace quda
{

  class TuneParam;

  struct qudaStream_t {
    int idx;
    //qudaStream_t(int idx) : idx(idx) {}
  };

  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] args Arguments
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, void **args, qudaStream_t stream);

  /**
     @brief Templated wrapper around qudaLaunchKernel which can accept
     a templated kernel, and expects a kernel with a single Arg argument
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] args Arguments
     @param[in] stream Stream identifier
  */
  template <typename T, typename... Arg>
  qudaError_t qudaLaunchKernel(T *func, const TuneParam &tp, qudaStream_t stream, const Arg &...arg)
  {
    ompwip("directly calling qudaLaunchKernel is unsupported");
    const void *args[] = {&arg...};
    return qudaLaunchKernel(reinterpret_cast<const void *>(func), tp, const_cast<void **>(args), stream);
  }

  /**
     @brief Wrapper around cudaMemcpy or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line);

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent for peer-to-peer copies
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyP2PAsync_(void *dst, const void *src, size_t count, const qudaStream_t &stream,
                           const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpy2DAsync or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch in bytes
     @param[in] src Source pointer
     @param[in] spitch Source pitch in bytes
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                     qudaMemcpyKind kind, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpy2DAsync or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch in bytes
     @param[in] src Source pointer
     @param[in] spitch Source pitch in bytes
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                          qudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line);

  /**
     @brief Wrapper around cudaMemcpy2DAsync or driver API equivalent
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch in bytes
     @param[in] src Source pointer
     @param[in] spitch Source pitch in bytes
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DP2PAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                             const qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemset or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
   */
  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemset2D or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] Pitch in bytes
     @param[in] value Value to set for each byte of specified memory
     @param[in] width Width in bytes
     @param[in] height Height in bytes
   */
  void qudaMemset2D_(void *ptr, size_t pitch, int value, size_t width, size_t height, const char *func,
                     const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
     @param[in] stream Stream to issue memset
   */
  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent
     @param[out] ptr Starting address pointer
     @param[in] Pitch in bytes
     @param[in] value Value to set for each byte of specified memory
     @param[in] width Width in bytes
     @param[in] height Height in bytes
     @param[in] stream Stream to issue memset
   */
  void qudaMemset2DAsync_(void *ptr, size_t pitch, int value, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemPrefetchAsync or driver API equivalent
     @param[out] ptr Starting address pointer to be prefetched
     @param[in] count Size in bytes to prefetch
     @param[in] mem_space Memory space to prefetch to
     @param[in] stream Stream to issue prefetch
   */
  void qudaMemPrefetchAsync_(void *ptr, size_t count, QudaFieldLocation mem_space, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventQuery or cuEventQuery with built-in error checking
     @param[in] event Event we are querying
     @return true if event has been reached
   */
  bool qudaEventQuery_(cudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord with
     built-in error checking
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  void qudaEventRecord_(cudaEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamWaitEvent or cuStreamWaitEvent
     with built-in error checking
     @param[in,out] stream Stream which we are instructing to wait
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  void qudaStreamWaitEvent_(qudaStream_t stream, cudaEvent_t event, unsigned int flags, const char *func,
                            const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventSynchronize or cuEventSynchronize
     with built-in error checking
     @param[in] event Event which we are synchronizing with respect to
   */
  void qudaEventSynchronize_(const cudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaStreamSynchronize or
     cuStreamSynchronize with built-in error checking
     @param[in] stream Stream which we are synchronizing
  */
  void qudaStreamSynchronize_(const qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize or
     cuDeviceSynchronize with built-in error checking
   */
  void qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaGetSymbolAddress with built in error
     checking.  Returns the address of symbol on the device; symbol
     is a variable that resides in global memory space.

     @param[in] symbol Global variable or string symbol to search for
     @return Return device pointer associated with symbol
  */
  void* qudaGetSymbolAddress_(const char *symbol, const char *func, const char *file, const char *line);

  /**
     @brief Get the last error string recorded
  */
  std::string qudaGetLastErrorString();

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

#define qudaMemcpy(dst, src, count, kind)                                                                              \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpyAsync(dst, src, count, kind, stream)                                                                 \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpyP2PAsync(dst, src, count, stream)                     \
  ::quda::qudaMemcpyP2PAsync_(dst, src, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)                                                    \
  ::quda::qudaMemcpy2D_(dst, dpitch, src, spitch, width, height, kind, __func__, quda::file_name(__FILE__),            \
                        __STRINGIFY__(__LINE__))

#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)                                       \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__,                          \
                             quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemcpy2DP2PAsync(dst, dpitch, src, spitch, width, height, stream)                                       \
  ::quda::qudaMemcpy2DP2PAsync_(dst, dpitch, src, spitch, width, height, stream, __func__,                          \
                                quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset(ptr, value, count)                                                                                  \
  ::quda::qudaMemset_(ptr, value, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2D(ptr, pitch, value, width, height)                                                                 \
  ::quda::qudaMemset2D_(ptr, pitch, value, width, height, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemsetAsync(ptr, value, count, stream)                                                                     \
  ::quda::qudaMemsetAsync_(ptr, value, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2DAsync(ptr, pitch, value, width, height, stream)                                                    \
  ::quda::qudaMemset2DAsync_(ptr, pitch, value, width, height, stream, __func__, quda::file_name(__FILE__),            \
                             __STRINGIFY__(__LINE__))

#define qudaMemPrefetchAsync(ptr, count, mem_space, stream)                                                            \
  ::quda::qudaMemPrefetchAsync_(ptr, count, mem_space, stream, __func__, quda::file_name(__FILE__),                    \
                                __STRINGIFY__(__LINE__))

#define qudaEventQuery(event)                                                                                          \
  ::quda::qudaEventQuery_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventRecord(event, stream)                                                                                 \
  ::quda::qudaEventRecord_(event, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamWaitEvent(stream, event, flags)                                                                      \
  ::quda::qudaStreamWaitEvent_(stream, event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventSynchronize(event)                                                                                    \
  ::quda::qudaEventSynchronize_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamSynchronize(stream)                                                                                  \
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaDeviceSynchronize()                                                                                        \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaGetSymbolAddress(symbol)                                    \
  ::quda::qudaGetSymbolAddress_(symbol, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
