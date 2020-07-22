#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <functional>

#define REDUCE_BLOCK_SIZE 4

template <typename T>
struct vec1 {
  T x;
} __attribute__((aligned(1*sizeof(T))));
template <typename T>
inline vec1<T> make_vec1(T a) { return vec1<T>{a}; }

template <typename T>
struct vec2 {
  T x,y;
} __attribute__((aligned(2*sizeof(T))));
template <typename T>
inline vec2<T> make_vec2(T a, T b) { return vec2<T>{a,b}; }

template <typename T>
struct vec3 {
  T x,y,z;
  inline vec3() {}
  inline vec3(T a, T b, T c): x(a),y(b),z(c) {}
};
template <typename T, typename A1, typename A2, typename A3>
inline vec3<T> make_vec3(A1 a, A2 b, A3 c) { return vec3<T>{a,b,c}; }
//template <typename T>
//inline vec3<T> vec3<T>::vec3(T a, T b, T c) { return vec3<T>{a,b,c}; }
//inline vec3<unsigned int> vec3<unsigned int>::vec3(unsigned int a, unsigned int b, unsigned int c) { return vec3<unsigned int>{a,b,c}; }

template <typename T>
struct vec4 {
  T x,y,z,w;
} __attribute__((aligned(4*sizeof(T))));
template <typename T>
inline vec4<T> make_vec4(T a, T b, T c, T d) { return vec4<T>{a,b,c,d}; }

typedef vec1<char> char1;
typedef vec2<char> char2;
typedef vec3<char> char3;
typedef vec4<char> char4;
#define make_char1(a) make_vec1<char>(a)
#define make_char2(a,b) make_vec2<char>(a,b)
#define make_char3(a,b,c) make_vec3<char>(a,b,c)
#define make_char4(a,b,c,d) make_vec4<char>(a,b,c,d)

typedef vec1<short> short1;
typedef vec2<short> short2;
typedef vec3<short> short3;
typedef vec4<short> short4;
#define make_short1(a) make_vec1<short>(a)
#define make_short2(a,b) make_vec2<short>(a,b)
#define make_short3(a,b,c) make_vec3<short>(a,b,c)
#define make_short4(a,b,c,d) make_vec4<short>(a,b,c,d)

typedef vec1<int> int1;
typedef vec2<int> int2;
typedef vec3<int> int3;
typedef vec4<int> int4;
#define make_int1(a) make_vec1<int>(a)
#define make_int2(a,b) make_vec2<int>(a,b)
#define make_int3(a,b,c) make_vec3<int>(a,b,c)
#define make_int4(a,b,c,d) make_vec4<int>(a,b,c,d)

typedef vec1<unsigned int> uint1;
typedef vec2<unsigned int> uint2;
typedef vec3<unsigned int> uint3;
typedef vec4<unsigned int> uint4;
#define make_uint1(a) make_vec1<uint>(a)
#define make_uint2(a,b) make_vec2<uint>(a,b)
#define make_uint3(a,b,c) make_vec3<uint>(a,b,c)
#define make_uint4(a,b,c,d) make_vec4<uint>(a,b,c,d)

typedef vec1<float> float1;
typedef vec2<float> float2;
typedef vec3<float> float3;
typedef vec4<float> float4;
#define make_float1(a) make_vec1<float>(a)
#define make_float2(a,b) make_vec2<float>(a,b)
#define make_float3(a,b,c) make_vec3<float>(a,b,c)
#define make_float4(a,b,c,d) make_vec4<float>(a,b,c,d)

typedef vec1<double> double1;
typedef vec2<double> double2;
typedef vec3<double> double3;
typedef vec4<double> double4;
#define make_double1(a) make_vec1<double>(a)
#define make_double2(a,b) make_vec2<double>(a,b)
#define make_double3(a,b,c) make_vec3<double>(a,b,c)
#define make_double4(a,b,c,d) make_vec4<double>(a,b,c,d)

typedef vec3<unsigned int> dim3;

extern dim3 gridDim;
extern dim3 blockDim;
extern dim3 blockIdx;
extern dim3 threadIdx;

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
			      size_t width, size_t height,
			      qudaStream_t stream=0);
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
void curand_init(unsigned long long seed, unsigned long long sequence,
		 unsigned long long offset, curandStateMRG32k3a *state);
float curand_uniform(curandStateMRG32k3a *state);
double curand_uniform_double(curandStateMRG32k3a *state);
float curand_normal(curandStateMRG32k3a *state);
double curand_normal_double(curandStateMRG32k3a *state);

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


