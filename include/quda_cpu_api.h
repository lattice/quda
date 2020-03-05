#pragma once
#include <functional>
/**
   @file quda_cpu_api.h

   Wrappers around CPU API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
*/

//#define _GNU_SOURCE  // for sincos, etc.
//#include <math.h>

#define sincosf(a,s,c) __sincosf(a,s,c)
#define __fdividef(x,y) ((x)/(y))

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <algorithm>
  
#define CUdeviceptr QUdeviceptr
#define cudaSuccess qudaSuccess
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE 0
#define CU_MEMORYTYPE_HOST 0
#define CU_MEMORYTYPE_DEVICE 1
#define CU_MEMORYTYPE_UNIFIED 2
#define CU_MEMORYTYPE_ARRAY 3
#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyDeviceToDevice 1
#define cudaMemcpyHostToDevice 2
#define cudaMemcpyHostToHost 3
#define cudaMemcpyDefault 4
#define cudaMemcpyAsyncDeviceToHost 5
#define cudaMemcpyAsyncHostToDevice 6
#define cudaMemcpyAsyncHostToHost 7
#define cudaMemcpyAsyncDeviceToDevice 8

#define cudaMalloc qudaMalloc
#define cudaHostAlloc qudaHostAlloc
#define cudaFree qudaFree
#define cudaFreeHost qudaFreeHost
#define cudaHostRegister qudaHostRegister
#define cudaHostUnregister qudaHostUnregister
#define cudaMallocManaged qudaMallocManaged
#define cudaProfilerStart qudaProfilerStart
#define cudaProfilerStop qudaProfilerStop
#define cudaStreamCreateWithPriority qudaStreamCreateWithPriority
#define cudaStreamDestroy qudaStreamDestroy
#define cudaMemset2DAsync qudaMemset2DAsync
#define cudaIpcGetMemHandle qudaIpcGetMemHandle
#define cudaIpcOpenMemHandle qudaIpcOpenMemHandle
#define cudaIpcOpenEventHandle qudaIpcOpenEventHandle
#define cudaIpcCloseMemHandle qudaIpcCloseMemHandle
#define cudaIpcGetEventHandle qudaIpcGetEventHandle
#define cudaEventQuery qudaEventQuery
#define cudaGetLastError qudaGetLastError
#define cudaGetErrorString qudaGetErrorString

#define cuMemAlloc quMemAlloc
#define cuMemFree quMemFree
#define cuPointerGetAttributes quPointerGetAttributes
#define CUpointer_attribute QUpointer_attribute
#define CUmemorytype QUmemorytype
#define cuGetErrorString quGetErrorString
#define cuGetErrorName quGetErrorName


#define __device__
#define __host__
#define __forceinline__ inline
#define __global__
#define __shared__
#define __syncthreads()
#define __threadfence()
#define atomicInc(x,y) (*(x) += (y), *(x))

//#define qudaError_t int
#define qudaMemcpyKind int
#define qudaStream_t int
#define qudaEvent_t int
#define qudaTextureObject_t int
#define qudaResourceViewDesc int
#define qudaFuncCache int
//#define qudaTextureDesc int
#define qudaIpcMemHandle_t int
#define qudaIpcEventHandle_t int

#define QUresult int
typedef const void *QUdeviceptr;

typedef struct {
  int f;
  int x,y,z,w;
} qudaChannelFormatDesc;
#define qudaChannelFormatKindFloat 0
#define qudaChannelFormatKindSigned 0

#define qudaEventDisableTiming 0
#define qudaEventInterprocess 1

#define qudaHostRegisterDefault 0
#define qudaHostRegisterMapped 0
#define qudaHostRegisterPortable 0

#define QUDA_SUCCESS 0
enum qudaError_t
{
  qudaSuccess,
  qudaErrorInvalidContext,
  qudaErrorInvalidKernelFile,
  qudaErrorMemoryAllocation,
  qudaErrorInitializationError,
  qudaErrorLaunchFailure,
  qudaErrorLaunchOutOfResources,
  qudaErrorInvalidDevice,
  qudaErrorInvalidValue,
  qudaErrorInvalidDevicePointer,
  qudaErrorInvalidMemcpyDirection,
  qudaErrorUnknown,
  qudaErrorInvalidResourceHandle,
  qudaErrorNotReady,
  qudaErrorNoDevice,
  qudaErrorPeerAccessAlreadyEnabled,
  qudaErrorPeerAccessNotEnabled,
  qudaErrorRuntimeMemory,
  qudaErrorRuntimeOther,
  qudaErrorHostMemoryAlreadyRegistered,
  qudaErrorHostMemoryNotRegistered,
  qudaErrorMapBufferObjectFailed,
  qudaErrorTbd
};


#define qudaFuncCachePreferL1 0
#define qudaStreamDefault 0

#define qudaMemcpyDeviceToHost 0
#define qudaMemcpyDeviceToDevice 1
#define qudaMemcpyHostToDevice 2

typedef int QUmemorytype;
#define QU_POINTER_ATTRIBUTE_MEMORY_TYPE 0
#define QU_MEMORYTYPE_HOST 0
#define QU_MEMORYTYPE_DEVICE 1
#define QU_MEMORYTYPE_UNIFIED 2
#define QU_MEMORYTYPE_ARRAY 3

#define qudaMemcpyDeviceToHost 0
#define qudaMemcpyDeviceToDevice 1
#define qudaMemcpyHostToDevice 2
#define qudaMemcpyHostToHost 3
#define qudaMemcpyDefault 4
#define qudaMemcpyAsyncDeviceToHost 5
#define qudaMemcpyAsyncHostToDevice 6
#define qudaMemcpyAsyncHostToHost 7
#define qudaMemcpyAsyncDeviceToDevice 8


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
  //qudaDeviceArch_t arch;
  int concurrentKernels;
  int pciBusID;
  int pciDeviceID;
  size_t maxSharedMemoryPerMultiProcessor;
  int isMultiGpuBoard;
  int canMapHostMemory;
  int unifiedAddressing;
  int gcnArch;
} qudaDeviceProp;
typedef int QUpointer_attribute;
typedef int QUmemorytype;

typedef struct qudaResourceDesc {
  //enum qudaResourceType resType;
  int resType;
  union {
    //struct {
    //  qudaArray_t array;
    //} array;
    //struct {
    //  qudaMipmappedArray_t mipmap;
    //} mipmap;
    struct {
      void *devPtr;
      qudaChannelFormatDesc desc;
      size_t sizeInBytes;
    } linear;
    struct {
      void *devPtr;
      qudaChannelFormatDesc desc;
      size_t width;
      size_t height;
      size_t pitchInBytes;
    } pitch2D;
  } res;
} qudaResourceDesc;

#define qudaResourceTypeLinear 0

typedef struct qudaTextureDesc {
  //enum qudaTextureAddressMode addressMode[3];
  //enum qudaTextureFilterMode filterMode;
  //enum qudaTextureReadMode readMode;
  int readMode;
  int                         sRGB;
  float                       borderColor[4];
  int                         normalizedCoords;
  unsigned int                maxAnisotropy;
  //enum qudaTextureFilterMode mipmapFilterMode;
  float                       mipmapLevelBias;
  float                       minMipmapLevelClamp;
  float                       maxMipmapLevelClamp;
} qudaTextureDesc;

#define qudaReadModeElementType 0
#define qudaReadModeNormalizedFloat 1

#define QUDA_MAKE_VEC1(C, T) \
  class C { \
    public: \
      T x; \
      C(T a); \
      C(const C &t); \
  };

#define QUDA_MAKE_VEC2(C, T) \
  class C { \
    public: \
      T x,y; \
      C() { x=0; y=0; };			\
      C(T a, T b) { x=a; y=b; };	\
      C(const C &t) { x=t.x; y=t.y; };	\
  }; \
  C inline \
  make_##C(T a, T b) { \
    C t(a,b);	       \
    return t; \
  }

#define QUDA_MAKE_VEC3(C, T) \
  class C { \
    public: \
      T x,y,z; \
      C() { x=0; y=0; z=0; };			\
      C(T a, T b, T c) { x=a; y=b; z=c; };	\
      C(const C &t) { x=t.x; y=t.y; z=t.z; };	\
  }; \
  C inline \
  make_##C(T a, T b, T c) { \
    C t = {a,b,c};	    \
    return t; \
  }

#define QUDA_MAKE_VEC4(C, T) \
  class C { \
    public: \
      T x,y,z,w; \
      C() { x=0; y=0; z=0; w=0; };		\
      C(T a, T b, T c, T d) { x=a; y=b; z=c; w=d; };	\
      C(const C &t) { x=t.x; y=t.y; z=t.z; w=t.w; };	\
  }; \
  C inline \
  make_##C(T a, T b, T c, T d) {			\
    C t(a,b,c,d);					\
    return t; \
  }

#define QUDA_MAKE_VECS(P, T)	    \
  QUDA_MAKE_VEC1(P##1, T)	    \
  QUDA_MAKE_VEC2(P##2, T)	    \
  QUDA_MAKE_VEC3(P##3, T)	    \
  QUDA_MAKE_VEC4(P##4, T)

QUDA_MAKE_VECS(char, char)
QUDA_MAKE_VECS(short, short)
QUDA_MAKE_VECS(int, int)
QUDA_MAKE_VECS(uint, unsigned int)
QUDA_MAKE_VECS(float, float)
QUDA_MAKE_VECS(double, double)

typedef uint3 dim3;

extern dim3 gridDim;
extern dim3 blockDim;
extern dim3 blockIdx;
extern dim3 threadIdx;

#define qudaIpcMemLazyEnablePeerAccess 0

qudaError_t qudaMalloc(void **ptr, size_t size);
qudaError_t qudaFree(void *ptr);

qudaError_t quMemAlloc(QUdeviceptr *ptr, size_t size);
qudaError_t quMemFree(QUdeviceptr ptr);

qudaError_t qudaHostMalloc(void **ptr, size_t size, unsigned int flags);
qudaError_t qudaHostAlloc(void **ptr, size_t size, unsigned int flags);

qudaError_t qudaHostUnregister(void *ptr);

#define qudaMallocManaged qudaMalloc
#define qudaFreeHost qudaFree

QUresult
quPointerGetAttributes(unsigned int numAttributes,
		       QUpointer_attribute* attributes, void **data,
		       QUdeviceptr ptr);

QUresult quGetErrorString(QUresult error, const char** pStr);
void quGetErrorName(int e, const char **str);


#define qudaProfilerStart()
#define qudaProfilerStop()

void qudaStreamCreateWithPriority(int *str, int num, int prio);

void qudaStreamDestroy(int str);

//void qudaMemset2DAsync(void *ptr, int pitch, int n, int pad_bytes, int Npad);

void qudaIpcGetMemHandle(void *h, void *t);
void qudaIpcOpenMemHandle(void *h, int i, int n);
void qudaIpcGetEventHandle(void *h, int i);
void qudaIpcOpenEventHandle(void *h, int i);
void qudaIpcCloseMemHandle(void *i);
int qudaEventQuery(int i);

#define qudaPeekAtLastError() qudaSuccess


namespace quda {

  /**
     @brief Wrapper around qudaGetErrorString
  */
  const char* qudaGetErrorString_(qudaError_t &error, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaGetLastError
  */
  qudaError_t qudaGetLastError_(const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemcpy used for auto-profiling.  Do not
     call directly, rather call macro below which will grab the
     location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  qudaError_t qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind,
			  const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemcpyAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, 
			const qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemcpyToSymbolAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[in] symbol Device symbol address
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] offset Offset from start of symbol in bytes
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyToSymbolAsync_(const void *symbol, const void *src, size_t count, size_t offset, qudaMemcpyKind kind, 
				const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemcpy2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
                          size_t width, size_t hieght, qudaMemcpyKind kind, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemset_(void *dst, int val, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemsetAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemsetAsync_(void *dst, int val, size_t count, const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaMemset2D or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2D_(void* dst, size_t pitch, int val, size_t width, size_t height, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2DAsync_(void* dst, size_t pitch, int val, size_t width, size_t height, const qudaStream_t &stream, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around qudaLaunch
     @param[in] gridDim Grid dimensions
     @param[in] blockDim Block dimensions
     @param[in] sharedMem Shared memory requested per thread block
     @param[in] stream Stream identifier
     @param[in] func Device function symbol
     @param[in] args Arguments
  */
  qudaError_t qudaLaunch_(dim3 gridDim, dim3 blockDim, size_t sharedMem,
			  qudaStream_t stream, const char *func,
			  const char *file, const char *line,
			  std::function<void()> f);

  /**
     @brief Wrapper around qudaEventCreate
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventCreate_(qudaEvent_t *event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventCreateWithFlags
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventCreateWithFlags_(qudaEvent_t *event, unsigned int flags, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventDestroy
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventQuery 
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventQuery_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
  */
  qudaError_t qudaEventRecord_(qudaEvent_t &event, qudaStream_t stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventElapsedTime
     @param[out] ms Time in ms
     @param[in,out] start Start event we are recording
     @param[in,out] end End event we are recording
  */
  qudaError_t qudaEventElapsedTime_(float *ms, qudaEvent_t start, qudaEvent_t end, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventSynchronize 
     @param[in] event Event which we are synchronizing with respect to
  */
  qudaError_t qudaEventSynchronize_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventSynchronize, specialised to use the driver API
     @param[in] event Event which we are synchronizing with respect to
  */
  qudaError_t qudaEventSynchronizeDriver_(qudaEvent_t &event, const char *func, const char *file, const char *line);
    
  /**
     @brief Wrapper around qudaStreamWaitEvent
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
  */
  qudaError_t qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t event, unsigned int flags, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaStreamWaitEvent, Driver enabled variant
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
  */
  qudaError_t qudaStreamWaitEventDriver_(qudaStream_t stream, qudaEvent_t event, unsigned int flags, const char *func, const char *file, const char *line);

  
  /**
     @brief Wrapper around qudaStreamSynchronize or 
     @param[in] stream Stream which we are synchronizing with respect to
  */
  qudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaStreamSynchronize Driver variant 
     @param[in] stream Stream which we are synchronizing with respect to
  */
  qudaError_t qudaStreamSynchronizeDriver_(qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaCreateTextureObject
  */
  qudaError_t qudaCreateTextureObject_(qudaTextureObject_t *pTexObject, const qudaResourceDesc *pResDesc, const qudaTextureDesc *pTexDesc, const qudaResourceViewDesc *pResViewDesc, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around cudaDestroyTextureObject
  */
  qudaError_t qudaDestroyTextureObject_(qudaTextureObject_t pTexObject, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around cudaDestroyTextureObject
  */
  qudaError_t qudaGetTextureObjectResourceDesc_(qudaResourceDesc* pResDesc, qudaTextureObject_t texObject, const char *func, const char *file, const char *line);
  
  //QUDA Device
  /**
     @brief Wrapper around cudaDeviceCanAccessPeer
  */
  qudaError_t qudaDeviceCanAccessPeer_(int* canAccessPeer, int device, int peerDevice, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceGetStreamPriorityRange
  */
  qudaError_t qudaDeviceGetStreamPriorityRange_(int* leastPriority, int* greatestPrioriy, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceGetStreamPriorityRange
  */
  qudaError_t qudaDeviceReset_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSetCacheConfig
  */
  qudaError_t qudaDeviceSetCacheConfig_(qudaFuncCache cacheConfig, const char *func, const char *file, const char *line);
  
  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaGetDeviceCount_(int* count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaSetDevice
  */
  qudaError_t qudaSetDevice_(int dev, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaGetDeviceProperties_(qudaDeviceProp* prop, int  device, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaHostGetDevicePointer
  */
  qudaError_t qudaHostGetDevicePointer_(void** pDevice, void* pHost, unsigned int flags, const char *func, const char *file, const char *line);

  /**
     @brief Call API wrapper
  */
  qudaError_t qudaDriverGetVersion_(int* driverVersion, const char *func, const char *file, const char *line);
  
  qudaError_t qudaRuntimeGetVersion_(int* runtimeVersion, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaHostRegister
  */
  qudaError_t qudaHostRegister_(void* ptr, size_t size, unsigned int flags, const char *func, const char *file, const char *line);
  
#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around qudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  qudaError_t qudaFuncSetAttribute(const void* func, qudaFuncAttribute attr, int value);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
  */
  void printAPIProfile();

} // namespace quda

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

//START Memcpy
//-------------------------------------------------------------------------------------
#define qudaMemcpy(dst, src, count, kind)				\
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpyAsync(dst, src, count, kind, stream)			\
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream) \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream) \
  ::quda::qudaMemcpyToSymbolAsync_(symbol, src, count, offset, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//END Memcpy
//-------------------------------------------------------------------------------------

#define qudaStreamSynchronize(stream)					\
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaStreamSynchronizeDriver(stream)					\
  ::quda::qudaStreamSynchronizeDriver_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));












//START Event
//-------------------------------------------------------------------------------------
#define qudaEventCreate(event)						\
  ::quda::qudaEventCreate_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventCreateWithFlags(event, flags)				\
  ::quda::qudaEventCreateWithFlags_(event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventDestroy(event)						\
  ::quda::qudaEventDestroy_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventQuery(event)						\
  ::quda::qudaEventQuery_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventRecord(event, stream)					\
  ::quda::qudaEventRecord_(event, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventElapsedTime(ms, start, end)				\
  ::quda::qudaEventElapsedTime_(ms, start, end, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventSynchronize(event)					\
  ::quda::qudaEventSynchronize_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaEventSynchronizeDriver(event)				\
  ::quda::qudaEventSynchronizeDriver_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamWaitEvent(stream, event, flags)			\
  ::quda::qudaStreamWaitEvent_(stream, event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaStreamWaitEventDriver(stream, event, flags)			\
  ::quda::qudaStreamWaitEventDriver_(stream, event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
//END Event
//-------------------------------------------------------------------------------------



//START Memset
//-------------------------------------------------------------------------------------
#define qudaMemset(dst, val, count)					\
  ::quda::qudaMemset_(dst, val, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemsetAsync(dst, val, count, stream)			\
  ::quda::qudaMemsetAsync_(dst, val, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2D(dst, val, pitch, width, height)			\
  ::quda::qudaMemset2D_(dst, val, pitch, width, height, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaMemset2DAsync(dst, val, pitch, width, height, stream)	\
  ::quda::qudaMemset2DAsync_(dst, val, pitch, width, height, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
//END Memset
//-------------------------------------------------------------------------------------

//START Texture
//-------------------------------------------------------------------------------------
#define qudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc) \
  ::quda::qudaCreateTextureObject_(pTexObject, pResDesc, pTexDesc, pResViewDesc, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaDestroyTextureObject(pTexObject)				\
  ::quda::qudaDestroyTextureObject_(pTexObject, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaGetTextureObjectResourceDesc(pResDesc, texObject)		\
  ::quda::qudaGetTextureObjectResourceDesc_(pResDesc, texObject, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
//END Texture
//-------------------------------------------------------------------------------------

//START Device
//-------------------------------------------------------------------------------------
#define qudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)	\
  ::quda::qudaDeviceCanAccessPeer_(canAccessPeer, device, peerDevice, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority) \
  ::quda::qudaDeviceGetStreamPriorityRange_(leastPriority, greatestPriority, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaDeviceReset()						\
  ::quda::qudaDeviceReset_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaDeviceSetCacheConfig(cacheConfig)				\
  ::quda::qudaDeviceSetCacheConfig_(cacheConfig, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaDeviceSynchronize()						\
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaDeviceSynchronize()						\
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaGetDeviceCount(count)					\
  ::quda::qudaGetDeviceCount_(count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaSetDevice(dev)						\
  ::quda::qudaSetDevice_(dev, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaGetDeviceProperties(prop, device)				\
  ::quda::qudaGetDeviceProperties_(prop, device, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaHostGetDevicePointer(pDevice, pHost, flags)			\
  ::quda::qudaHostGetDevicePointer_(pDevice, pHost, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaDriverGetVersion(driverVersion)				\
  ::quda::qudaDriverGetVersion_(driverVersion, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaRuntimeGetVersion(runtimeVersion)				\
  ::quda::qudaRuntimeGetVersion_(runtimeVersion, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//END Device
//-------------------------------------------------------------------------------------

//START Host
//-------------------------------------------------------------------------------------
#define qudaHostRegister(ptr, size, flags)				\
  ::quda::qudaHostRegister_(ptr, size, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))
//END Host
//-------------------------------------------------------------------------------------

//START Misc
//-------------------------------------------------------------------------------------
#define qudaGetLastError()						\
  ::quda::qudaGetLastError_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaGetErrorString(error)					\
  ::quda::qudaGetErrorString_(error,__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__))

#define qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
  ::quda::qudaLaunch_(gridDim0, blockDim0, sharedMem0, stream0,  __func__, \
		      quda::file_name(__FILE__), __STRINGIFY__(__LINE__), \
		      [=](){						\
			func0(__VA_ARGS__);				\
		      })

#define qudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream) \
  qudaLaunch(gridDim, blockDim, sharedMem, stream, func, args)

#if 0
#define qudaLaunchX(gridDim0, blockDim0, sharedMem0, stream0, func0, ...) \
  qudaLaunch(gridDim0, blockDim0, sharedMem0, stream0, func0, __VA_ARGS__)
#define qudaLaunchKernelX(func, gridDim, blockDim, args, sharedMem, stream) \
  qudaLaunchX(gridDim, blockDim, sharedMem, stream, func, args)
#define qudaLaunchKernel2(gridDim, blockDim, sharedMem, stream, func, arg1, arg2) qudaLaunchX(gridDim, blockDim, sharedMem, stream, func, arg1, arg2)
#endif

//END Misc
//-------------------------------------------------------------------------------------
//#endif
