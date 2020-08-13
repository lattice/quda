#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <functional>

#ifdef I
#undef I
#endif
#include <CL/sycl.hpp>
//#include <CL/__spirv/spirv_vars.hpp>
//using namespace cl::sycl;

#include "quda_cpu_vec.h"

extern cl::sycl::queue defaultQueue;

#define REDUCE_BLOCK_SIZE 4

/*
struct dim3 {
  unsigned int x,y,z;
  inline dim3() {}
  inline dim3(unsigned int a, unsigned int b, unsigned int c): x(a),y(b),z(c) {}
};
//inline make_dim3(A1 a, A2 b, A3 c) { return vec3<T>{a,b,c}; }
*/

using dim3 = uint3;

//#define gridDim __spirv_BuiltInGlobalSize
//#define gridIdx __spirv_BuiltInGlobalInvocationId
//#define gridDim (__spirv::initNumWorkgroups<3,dim3>())
//#define blockIdx (__spirv::initWorkgroupId<3,dim3>())
//#define blockDim (__spirv::initWorkgroupSize<3,dim3>())
//#define threadIdx (__spirv::initLocalInvocationId<3,dim3>())

//#define gridDim (__spirv::InitSizesSTNumWorkgroups<3,dim3>::initSize())
//#define blockIdx (__spirv::InitSizesSTWorkgroupId<3,dim3>::initSize())
//#define blockDim (__spirv::InitSizesSTWorkgroupSize<3,dim3>::initSize())
//#define threadIdx (__spirv::InitSizesSTLocalInvocationId<3,dim3>::initSize())


//template <typename T>
//int getGlobalId(T ndi) { return ndi.get_global_id(0); }
//template <typename T>
//int getGlobalRange(T ndi) { return ndi.get_global_range(0); }
//#define globalDim_x getGlobalRange(ndi...)
//#define globalIdx_x getGlobalId(ndi...)

template <int N>
using nd_item = cl::sycl::nd_item<N>;
using rid = unsigned int;

template <int N>
inline rid getGrpRange_x(nd_item<N> ndi) { return ndi.get_group_range(0); }
template <int N>
inline rid getGrpRange_y(nd_item<N> ndi) { return ndi.get_group_range(1); }
template <int N>
inline rid getGrpRange_z(nd_item<N> ndi) { return ndi.get_group_range(2); }

template <int N>
inline rid getLocRange_x(nd_item<N> ndi) { return ndi.get_local_range(0); }
template <int N>
inline rid getLocRange_y(nd_item<N> ndi) { return ndi.get_local_range(1); }
template <int N>
inline rid getLocRange_z(nd_item<N> ndi) { return ndi.get_local_range(2); }

template <int N>
inline rid getGrpId_x(nd_item<N> ndi) { return ndi.get_group(0); }
template <int N>
inline rid getGrpId_y(nd_item<N> ndi) { return ndi.get_group(1); }
template <int N>
inline rid getGrpId_z(nd_item<N> ndi) { return ndi.get_group(2); }

template <int N>
inline rid getLocId_x(nd_item<N> ndi) { return ndi.get_local_id(0); }
template <int N>
inline rid getLocId_y(nd_item<N> ndi) { return ndi.get_local_id(1); }
template <int N>
inline rid getLocId_z(nd_item<N> ndi) { return ndi.get_local_id(2); }


#define gridDim_x getGrpRange_x(env_...)
#define gridDim_y getGrpRange_y(env_...)
#define gridDim_z getGrpRange_z(env_...)
#define blockIdx_x getGrpId_x(env_...)
#define blockIdx_y getGrpId_y(env_...)
#define blockIdx_z getGrpId_z(env_...)
#define blockDim_x getLocRange_x(env_...)
#define blockDim_y getLocRange_y(env_...)
#define blockDim_z getLocRange_z(env_...)
#define threadIdx_x getLocId_x(env_...)
#define threadIdx_y getLocId_y(env_...)
#define threadIdx_z getLocId_z(env_...)

#define gridDim ((dim3){gridDim_z,gridDim_y,gridDim_x})
#define blockIdx ((dim3){blockIdx_z,blockIdx_y,blockIdx_x})
#define blockDim ((dim3){blockDim_z,blockDim_y,blockDim_x})
#define threadIdx ((dim3){threadIdx_z,threadIdx_y,threadIdx_x})




using qudaStream_t = int;
using qudaEvent_t = double;

#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__ const
#define __forceinline__ __attribute__((always_inline)) inline
#define __launch_bounds__(x)
#define __fdivdef(x,y) ((x)/(y))

template<typename T>
inline T atomicAdd(T *x, T y) { T z = *x; *x += y; return z; }

template<typename T>
inline T atomicCASImpl(T *addr, T comp, T val) { *addr=val; return comp; }
inline int
atomicCAS(int *addr, int comp, int val) {
  return atomicCASImpl<int>(addr, comp, val);
}
inline unsigned int
atomicCAS(unsigned int *addr, unsigned int comp, unsigned int val) {
  return atomicCASImpl<unsigned int>(addr, comp, val);
}

template<typename T>
inline T atomicMaxImpl(T *addr, T val) {
  T old=*addr;
  *addr=std::max(*addr,val);
  return old;
}
inline unsigned int
atomicMax(unsigned int *addr, unsigned int val) {
  return atomicMaxImpl<unsigned int>(addr, val);
}

#define __float_as_uint(x) ((unsigned int)(x))
#define __uint_as_float(x) ((float)(x))
template <typename T>
inline T rsqrt(T x) { return ((T)1)/sqrt(x); }
template <typename T>
inline void sincos_(const T &a, T *s, T *c) {
  *s = sin(a);
  *c = cos(a);
}


typedef struct {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  int memoryClockRate;
  int memoryBusWidth;                         
  size_t totalConstMem;                       
  int major;                                  
  int minor;                                  
  int multiProcessorCount;                    
  int l2CacheSize;                            
  int maxThreadsPerMultiProcessor;            
  int computeMode;                            
  int clockInstructionRate;                   
  //hipDeviceArch_t arch;                       
  int concurrentKernels;                      
  int pciBusID;                               
  int pciDeviceID;
  size_t maxSharedMemoryPerMultiProcessor;
  int isMultiGpuBoard;                        
  int canMapHostMemory;
  int unifiedAddressing;
} qudaDeviceProp;
typedef qudaDeviceProp cudaDeviceProp;

enum cudaMemcpyKind {
  qudaMemcpyHostToHost = 0,
  qudaMemcpyHostToDevice = 1,
  qudaMemcpyDeviceToHost = 2,
  qudaMemcpyDeviceToDevice = 3,
  qudaMemcpyDefault = 4
};
typedef cudaMemcpyKind qudaMemcpyKind;
#define cudaMemcpyHostToHost qudaMemcpyHostToHost
#define cudaMemcpyHostToDevice qudaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost qudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice qudaMemcpyDeviceToDevice
#define cudaMemcpyDefault qudaMemcpyDefault

enum qudaError_t {
  qudaSuccess,
  qudaErrorInvalidDevice,
  qudaErrorNotReady,
  qudaErrorUnknown
};

typedef qudaError_t cudaError_t;
#define cudaSuccess qudaSuccess
#define cudaErrorNotReady qudaErrorNotReady

typedef int cudaTextureObject_t;
template <typename T>
struct tex1Dfetch {
  cudaTextureObject_t tex;
  int i;
};

qudaError_t qudaEventCreate(qudaEvent_t *event);
qudaError_t qudaEventCreate(qudaEvent_t *event, unsigned int flags);
qudaError_t qudaEventCreateWithFlags(qudaEvent_t *event, unsigned int flags);
qudaError_t qudaEventDestroy(qudaEvent_t event);
qudaError_t qudaEventQuery(qudaEvent_t event);
qudaError_t qudaEventElapsedTime(float *ms, qudaEvent_t start, qudaEvent_t end);
qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream);

qudaError_t qudaGetLastError(void);
qudaError_t qudaPeekAtLastError(void);
const char* qudaGetErrorString(qudaError_t err);

qudaError_t qudaHostRegister(void *hostPtr, size_t sizeBytes,
			     unsigned int flags);
qudaError_t qudaHostUnregister(void *hostPtr);
qudaError_t qudaHostGetDevicePointer(void **devPtr, void *hstPtr,
				     unsigned int flags);

qudaError_t qudaProfilerStart(void);
qudaError_t qudaProfilerStop(void);

qudaError_t qudaDeviceGetStreamPriorityRange(int *leastPriority,
					     int *greatestPriority);
qudaError_t qudaStreamCreateWithPriority(qudaStream_t *pStream,
					 unsigned int flags, int priority);
qudaError_t qudaStreamDestroy(qudaStream_t stream);

qudaError_t qudaGetDeviceCount(int *count);
qudaError_t qudaDeviceReset(void);


qudaError_t qudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
			      size_t spitch, size_t width, size_t height,
			      qudaMemcpyKind kind, qudaStream_t stream);
qudaError_t qudaMemset2D(void *devPtr, size_t pitch, int value, size_t width,
			 size_t height);
qudaError_t qudaMemset2DAsync(void *devPtr, size_t pitch, int value,
			      size_t width, size_t height);
qudaError_t qudaMemset2DAsync(void *devPtr, size_t pitch, int value,
			      size_t width, size_t height, qudaStream_t stream);
qudaError_t qudaMemPrefetchAsync(const void *devPtr, size_t count,
				 int dstDevice, qudaStream_t stream);

typedef int qudaIpcMemHandle_t;
qudaError_t qudaIpcGetMemHandle(qudaIpcMemHandle_t *handle, void *devPtr);
qudaError_t qudaIpcOpenMemHandle(void **devPtr, qudaIpcMemHandle_t handle,
				 unsigned int flags);
qudaError_t qudaIpcCloseMemHandle(void *devPtr);
#define cudaIpcMemLazyEnablePeerAccess 0

typedef int qudaIpcEventHandle_t;
qudaError_t qudaIpcGetEventHandle(qudaIpcEventHandle_t *handle,
				  qudaEvent_t event);
qudaError_t qudaIpcOpenEventHandle(qudaEvent_t *event,
				   qudaIpcEventHandle_t handle);

qudaError_t qudaGetDeviceProperties(qudaDeviceProp *prop, int device);
qudaError_t qudaDeviceCanAccessPeer(int *canAccessPeer, int device,
				    int peerDevice);

struct curandStateXORWOW {
  int state;
};
struct curandStateMRG32k3a {
  int64_t s10,s11,s12;
  int64_t s20,s21,s22;
};

inline void
curand_init(unsigned long long seed, unsigned long long sequence,
	    unsigned long long offset, curandStateMRG32k3a *state)
{
  // FIXME: sequence, offset
  int64_t seed0 = seed;
  state->s10 = seed0;
  state->s11 = seed0;
  state->s12 = seed0;
  state->s20 = seed0;
  state->s21 = seed0;
  state->s22 = seed0;
}

inline int64_t next(curandStateMRG32k3a *state)
{
  const int64_t m1 = 4294967087;
  const int64_t m2 = 4294944443;
  const int32_t a12 = 1403580;
  const int32_t a13 = 810728;
  const int32_t a21 = 527612;
  const int32_t a23 = 1370589;
  const int64_t corr1 = m1 * a13;
  const int64_t corr2 = m2 * a23;
  /* Combination */
  int64_t r = state->s12 - state->s22;
  r -= m1 * ((r - 1) >> 63);
  /* Component 1 */
  int64_t p1 = (a12 * state->s11 - a13 * state->s10 + corr1) % m1;
  state->s10 = state->s11;
  state->s11 = state->s12;
  state->s12 = p1;
  /* Component 2 */
  int64_t p2 = (a21 * state->s22 - a23 * state->s20 + corr2) % m2;
  state->s20 = state->s21;
  state->s21 = state->s22;
  state->s22 = p2;
  return r;
}

inline float curand_uniform(curandStateMRG32k3a *state)
{
  auto r = next(state);
  const float norm = 0x1fp-32;
  return norm*r;
}

inline double curand_uniform_double(curandStateMRG32k3a *state)
{
  auto r = next(state);
  const double norm = 0x1dp-32;
  return norm*r;
}

inline float curand_normal(curandStateMRG32k3a *state)
{
  const float TINY = 9.999999999999999e-38;
  const float TAU = 6.28318531;
  float v = curand_uniform(state);
  float p = curand_uniform(state) * TAU;
  float r = sqrt(-2.0 * log(v + TINY));
  return r * cos(p);
}

inline double curand_normal_double(curandStateMRG32k3a *state)
{
  const double TINY = 9.999999999999999e-308;
  const double TAU = 6.28318530717958648;
  double v = curand_uniform_double(state);
  double p = curand_uniform_double(state) * TAU;
  double r = sqrt(-2.0 * log(v + TINY));
  return r * cos(p);
}


#define cudaEvent_t qudaEvent_t
#define cudaEventCreate qudaEventCreate
#define cudaEventCreateWithFlags qudaEventCreateWithFlags
#define cudaEventDestroy qudaEventDestroy
#define cudaEventQuery qudaEventQuery
#define cudaEventSynchronize qudaEventSynchronize
#define cudaEventRecord qudaEventRecord
#define cudaEventElapsedTime qudaEventElapsedTime
#define cudaEventDisableTiming 0
#define cudaEventInterprocess 1

#define cudaGetLastError qudaGetLastError
#define cudaPeekAtLastError qudaPeekAtLastError
#define cudaGetErrorString qudaGetErrorString

#define cudaHostRegister qudaHostRegister
#define cudaHostUnregister qudaHostUnregister
#define cudaHostRegisterDefault 0
#define cudaHostGetDevicePointer qudaHostGetDevicePointer

#define cudaProfilerStart qudaProfilerStart
#define cudaProfilerStop qudaProfilerStop

#define cudaDeviceGetStreamPriorityRange qudaDeviceGetStreamPriorityRange
#define cudaStreamCreateWithPriority qudaStreamCreateWithPriority
#define cudaStreamDestroy qudaStreamDestroy

#define cudaGetDeviceCount qudaGetDeviceCount
#define cudaDeviceReset qudaDeviceReset
#define cudaCpuDeviceId ((int)-1)

#define cudaMemcpy qudaMemcpy
#define cudaMemcpyAsync qudaMemcpyAsync
#define cudaMemcpy2DAsync qudaMemcpy2DAsync
#define cudaMemset qudaMemset
#define cudaMemset2D qudaMemset2D
#define cudaMemset2DAsync qudaMemset2DAsync
#define cudaMemcpyToSymbolAsync qudaMemcpyToSymbolAsync
#define cudaMemPrefetchAsync qudaMemPrefetchAsync

typedef qudaIpcMemHandle_t cudaIpcMemHandle_t;
#define cudaIpcGetMemHandle qudaIpcGetMemHandle
#define cudaIpcCloseMemHandle qudaIpcCloseMemHandle
#define cudaIpcOpenMemHandle qudaIpcOpenMemHandle

typedef qudaIpcEventHandle_t cudaIpcEventHandle_t;
#define cudaIpcGetEventHandle qudaIpcGetEventHandle
#define cudaIpcOpenEventHandle qudaIpcOpenEventHandle

#define cudaGetDeviceProperties qudaGetDeviceProperties
#define cudaDeviceCanAccessPeer qudaDeviceCanAccessPeer

#define cudaDeviceSynchronize qudaDeviceSynchronize

/**
   @file quda_cpu_api.h

   Wrappers around CPU/OpenMP API function calls.
 */

namespace quda {

  /**
     @brief Wrapper used for auto-profiling.  Do not call directly,
     rather call macro below which will grab the location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		   const char *func, const char *file, const char *line);

  /**
     @brief Wrapper for StreamSynchronize
     @param[in] stream Stream which we are synchronizing
  */
  qudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line);
}

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

#define qudaStreamSynchronize(stream)                                                                                  \
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpy(dst, src, count, kind) \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpyAsync(dst, src, count, kind, stream) \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream) \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemset(ptr, value, count)                                                                                  \
  ::quda::qudaMemset_(ptr, value, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemsetAsync(ptr, value, count, stream)                                                                     \
  ::quda::qudaMemsetAsync_(ptr, value, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

namespace quda {

  /**
     @brief Wrapper around cudaMemcpyAsync or driver API equivalent
     Adds auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemcpy2DAsync or driver API equivalent
     Adds auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                          cudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line);

  qudaError_t qudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, qudaMemcpyKind kind, qudaStream_t stream);

  /**
     @brief Wrapper around cudaMemset or driver API equivalent.
     Adds auto-profiling support.
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
   */
  void qudaMemset_(void *ptr, int value, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemsetAsync or driver API equivalent.
     Adds auto-profiling support.
     @param[out] ptr Starting address pointer
     @param[in] value Value to set for each byte of specified memory
     @param[in] count Size in bytes to set
     @param[in] stream  Stream to issue memset
   */
  void qudaMemsetAsync_(void *ptr, int value, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line);

  /**
     @brief Wrapper around cudaEventQuery or cuEventQuery
     @param[in] event Event we are querying
     @return Status of event query
   */
  qudaError_t qudaEventQuery(qudaEvent_t &event);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
   */
  //qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream = 0);

  /**
     @brief Wrapper around cudaEventRecord or cuEventRecord
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
   */
  cudaError_t qudaStreamWaitEvent(qudaStream_t stream, cudaEvent_t event, unsigned int flags);

  /**
     @brief Wrapper around cudaEventSynchronize or cuEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
   */
  cudaError_t qudaEventSynchronize(cudaEvent_t &event);

  /**
     @brief Wrapper around cudaDeviceSynchronize or cuDeviceSynchronize
   */
  cudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  /**
     @brief Print out the timer profile for CUDA API calls
   */
  void printAPIProfile();

} // namespace quda

#define qudaDeviceSynchronize() \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));


