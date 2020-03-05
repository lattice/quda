#include <tune_quda.h>
#include <uint_to_char.h>
#include <quda_internal.h>

dim3 gridDim;
dim3 blockDim;
dim3 blockIdx;
dim3 threadIdx;

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

QUresult
quPointerGetAttributes(int d, QUpointer_attribute att[],
		       void *data[], QUdeviceptr ptr)
{
  return qudaSuccess;
}

QUresult
quGetErrorString(QUresult error, const char** pStr)
{
  return qudaSuccess;
}

void
qheckQudaErrorNoSync()
{
}

void
qudaStreamCreateWithPriority(int *str, int num, int prio)
{
  *str = 0;
}

void
qudaStreamDestroy(int str)
{
}

//void
//qudaMemset2DAsync(void *ptr, int pitch, int n, int pad_bytes, int Npad, int stream)
//{
//}

void
qudaIpcGetMemHandle(void *h, void *t)
{
}

void
qudaIpcOpenMemHandle(void *h, int i, int n)
{
}

void
qudaIpcGetEventHandle(void *h, int i)
{
}

void
qudaIpcOpenEventHandle(void *h, int i)
{
}

void
qudaIpcCloseMemHandle(void *i)
{
}

void
quGetErrorName(int e, const char **str)
{
}

void
quMemcpy(QUdeviceptr dst, QUdeviceptr src, int count)
{
  memcpy(const_cast<void*>(dst), src, count);
}

void
quMemcpyDtoD(QUdeviceptr dst, QUdeviceptr src, int count)
{
  memcpy(const_cast<void*>(dst), src, count);
}

void
quMemcpyDtoH(void *dst, QUdeviceptr src, int count)
{
  memcpy(dst, src, count);
}

void
quMemcpyHtoD(QUdeviceptr dst, const void *src, int count)
{
  memcpy(const_cast<void*>(dst), src, count);
}

void
quMemcpyDtoDAsync(QUdeviceptr dst, QUdeviceptr src, int count, int stream)
{
  memcpy(const_cast<void*>(dst), src, count);
}

void
quMemcpyDtoHAsync(void *dst, QUdeviceptr src, int count, int stream)
{
  memcpy(dst, src, count);
}

void
quMemcpyHtoDAsync(QUdeviceptr dst, void *src, int count, int stream)
{
  memcpy(const_cast<void*>(dst), src, count);
}

void
quMemcpyHtoHAsync(void *dst, void *src, int count, int stream)
{
  memcpy(dst, src, count);
}

qudaError_t
qudaMalloc(void **ptr, size_t size)
{
  qudaError_t error = qudaSuccess;

  *ptr = malloc(size);

  if(*ptr == nullptr)
    error = qudaErrorMemoryAllocation;

  return error;
}

qudaError_t
quMemAlloc(QUdeviceptr *ptr, size_t size)
{
  qudaError_t error = qudaSuccess;

  *ptr = malloc(size);

  if(*ptr == nullptr)
    error = qudaErrorMemoryAllocation;

  return error;
}

qudaError_t
qudaFree(void* ptr)
{
  qudaError_t error = qudaSuccess;
  free(ptr);
  return error;
}

qudaError_t
quMemFree(QUdeviceptr ptr)
{
  qudaError_t error = qudaSuccess;
  free(const_cast<void*>(ptr));
  return error;
}

qudaError_t
qudaMallocHost(void** ptr, size_t size)
{
  return qudaMalloc(ptr, size);
}

qudaError_t
qudaHostMalloc(void** ptr, size_t size, unsigned int flags)
{
  return qudaMalloc(ptr, size);
}

qudaError_t qudaHostAlloc(void** ptr, size_t size, unsigned int flags)
{
  return qudaMalloc(ptr, size);
}

qudaError_t
qudaHostUnregister(void *ptr)
{
  qudaError_t error = qudaSuccess;
  return error;
}

QUresult
quPointerGetAttributes(unsigned int numAttributes,
		       QUpointer_attribute* attributes, void **data,
		       QUdeviceptr ptr)
{
  qudaError_t error = qudaSuccess;
  return error;
}


namespace quda {

  static TimeProfile apiTimer("CPU API calls");

  class QudaMemCopy : public Tunable {

    void *dst;
    const void *src;
    const size_t count;
    const qudaMemcpyKind kind;
    const bool async;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {return 0;}

  public:
    inline QudaMemCopy(void *dst, const void *src, size_t count,
		       qudaMemcpyKind kind, bool async, const char *func,
		       const char *file, const char *line)
      : dst(dst), src(src), count(count), kind(kind), async(async) {

      if (!async) {
        switch (kind) {
        case qudaMemcpyDeviceToHost:   name = "qudaMemcpyDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "qudaMemcpyHostToDevice";   break;
        case qudaMemcpyHostToHost:     name = "qudaMemcpyHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyDeviceToDevice"; break;
        case qudaMemcpyDefault:        name = "qudaMemcpyDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyType %d", kind);
        }
      } else {
        switch(kind) {
        case qudaMemcpyDeviceToHost:   name = "qudaMemcpyAsyncDeviceToHost";   break;
        case qudaMemcpyHostToDevice:   name = "qudaMemcpyAsyncHostToDevice";   break;
        case qudaMemcpyHostToHost:     name = "qudaMemcpyAsyncHostToHost";     break;
        case qudaMemcpyDeviceToDevice: name = "qudaMemcpyAsyncDeviceToDevice"; break;
        case qudaMemcpyDefault:        name = "qudaMemcpyAsyncDefault";        break;
        default: errorQuda("Unsupported qudaMemcpyType %d", kind);
        }
      }
      strcpy(aux, func);
      strcat(aux, ",");
      strcat(aux, file);
      strcat(aux, ",");
      strcat(aux, line);
    }

    virtual ~QudaMemCopy() { }

    inline void apply(const qudaStream_t &stream) {
      //tuneLaunch(*this, getTuning(), getVerbosity());
      //printfQuda("QudaMemCopy apply dst: %p  src: %p  count: %i\n", dst, src, count);
      memcpy(dst, src, count);
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const {
      char vol[128];
      strcpy(vol,"bytes=");
      u64toa(vol+6, (uint64_t)count);
      return TuneKey(vol, name, aux);
    }

    long long flops() const { return 0; }
    long long bytes() const { return kind == qudaMemcpyDeviceToDevice ? 2*count : count; }

  };
  
  qudaError_t qudaMemcpy_(void *dst, const void *src, size_t count,
			  qudaMemcpyKind kind, const char *func,
			  const char *file, const char *line) {
    //printfQuda("qudaMemcpy_\n");
    if (count == 0) {
      qudaError_t error = qudaSuccess;
      return error;
    }
    QudaMemCopy copy(dst, src, count, kind, false, func, file, line);
    copy.apply(0);
    qudaError_t error = qudaSuccess;
    //qudaError_t error = qudaGetLastError();
    //if (error != qudaSuccess)
    //  errorQuda("(QUDA) %s\n (%s:%s in %s())\n", qudaGetErrorString(error), file, line, func);
    return error;
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count,
			qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMemCopy copy(dst, src, count, kind, true, func, file, line);
      copy.apply(stream);
    } else {
      //PROFILE(memcpy(dst, src, count, kind, stream),
      PROFILE(memcpy(dst, src, count), 0)
    }
  }

  void
  qudaMemcpyToSymbolAsync_(const void *symbol, const void *src, size_t count,
			   size_t offset, qudaMemcpyKind kind,
			   const qudaStream_t &stream, const char *func,
			   const char *file, const char *line)
  {
    //memcpy(symbol, src, count); //, offset, kind, stream);
  }

  
  void
  qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch,
		     size_t width, size_t height, qudaMemcpyKind kind,
		     const qudaStream_t &stream, const char *func,
		     const char *file, const char *line)
  {
    //PROFILE(memcpy(dst, dpitch, src, spitch, width, height, kind, stream), QUDA_PROFILE_MEMCPY2D_D2H_ASYNC);
  }
  
  qudaError_t
  qudaGetLastError_(const char *func, const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    int e = errno;
    if(e!=0) error = qudaErrorUnknown;
    return error;
  }

  const char* qudaGetErrorString_(qudaError_t &error, const char *func,
				  const char *file, const char *line) {
    return "error";
  }
  
  void
  qudaMemset_(void *dst, int val, size_t count, const char *func,
	      const char *file, const char *line) {
    if (count == 0) return;
    memset(dst, val, count);
  }

  void
  qudaMemsetAsync_(void *dst, int val, size_t count,
		   const qudaStream_t &stream, const char *func,
		   const char *file, const char *line) {
    if (count == 0) return;
    memset(dst, val, count);
  }

  void
  qudaMemset2D_(void *dst, size_t pitch, int val, size_t width, size_t height,
		const char *func, const char *file, const char *line) {
    if (pitch == 0) return;
    printfQuda("qudaMemset2D_\n");
    exit(1);
    //memset(dst, val, pitch, width, height);
  }

  void
  qudaMemset2DAsync_(void *dst, size_t pitch, int val, size_t width,
		     size_t height, const qudaStream_t &stream,
		     const char *func, const char *file, const char *line) {
    if (pitch == 0) return;
    printfQuda("qudaMemset2DAsync_\n");
    exit(1);
    //memset(dst, val, pitch, width, height, stream);
  }
  
  qudaError_t qudaLaunch_(dim3 gridDim0, dim3 blockDim0, size_t sharedMem0,
			  qudaStream_t stream0, const char *func,
			  const char *file, const char *line,
			  std::function<void()> f) {
    //printf("qudaLaunch_ from %s (%s:%s)\n", func, file, line);
    qudaError_t error = qudaSuccess;
    gridDim = gridDim0;
    blockDim = blockDim0;
    int ngx = gridDim0.x;
    int ngy = gridDim0.y;
    int nbx = blockDim0.x;
    int nby = blockDim0.y;
    //printf("ngx: %i  ngy: %i  nbx: %i  nby: %i\n", ngx, ngy, nbx, nby);
    for(int igx=0; igx<ngx; igx++) {
      blockIdx.x = igx;
      for(int igy=0; igy<ngy; igy++) {
	blockIdx.y = igy;
	for(int ibx=0; ibx<nbx; ibx++) {
	  threadIdx.x = ibx;
	  for(int iby=0; iby<nby; iby++) {
	    threadIdx.y = iby;
	    f();
	  }
	}
      }
    }
    return error;
  }

  qudaError_t
  qudaEventCreate_(qudaEvent_t *event, const char *func,
		   const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaEventCreateWithFlags_(qudaEvent_t *event, unsigned int flags,
			    const char *func, const char *file,
			    const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaEventDestroy_(qudaEvent_t &event, const char *func,
		    const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaEventQuery_(qudaEvent_t& event, const char *func,
		  const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaEventRecord_(qudaEvent_t &event, qudaStream_t stream, const char *func,
		   const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaEventElapsedTime_(float *ms, qudaEvent_t start, qudaEvent_t end,
			const char *func, const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaStreamWaitEventDriver_(qudaStream_t stream, qudaEvent_t event,
			     unsigned int flags, const char *func,
			     const char *file, const char *line)
  {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t event,
		       unsigned int flags, const char *func,
		       const char *file, const char *line)
  {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaStreamSynchronize_(qudaStream_t &stream, const char *func,
			 const char *file, const char *line)
  {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaStreamSynchronizeDriver_(qudaStream_t &stream, const char *func,
			       const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  // Driver variant called from lib/quda_gauge_field.cpp
  qudaError_t
  qudaEventSynchronizeDriver_(qudaEvent_t &event, const char *func,
			      const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaEventSynchronize_(qudaEvent_t &event, const char *func,
			const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaCreateTextureObject_(qudaTextureObject_t *pTexObject,
			   const qudaResourceDesc *pResDesc,
			   const qudaTextureDesc *pTexDesc,
			   const qudaResourceViewDesc *pResViewDesc,
			   const char *func, const char *file,
			   const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaDestroyTextureObject_(qudaTextureObject_t pTexObject, const char *func,
			    const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaGetTextureObjectResourceDesc_(qudaResourceDesc *pResDesc,
				    qudaTextureObject_t texObject,
				    const char *func, const char *file,
				    const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaDeviceCanAccessPeer_(int *canAccessPeer, int device, int peerDevice,
			   const char *func, const char *file,
			   const char *line) {
    qudaError_t error = qudaSuccess;
    *canAccessPeer = 0;
    return error;
  }

  qudaError_t
  qudaDeviceGetStreamPriorityRange_(int *leastPriority, int *greatestPriority,
				    const char *func, const char *file,
				    const char *line) {
    qudaError_t error = qudaSuccess;
    *leastPriority = 0;
    *greatestPriority = 0;
    return error;
  }

  qudaError_t
  qudaDeviceReset_(const char *func, const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaDeviceSetCacheConfig_(qudaFuncCache cacheConfig, const char *func,
			    const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaDeviceSynchronize_(const char *func, const char *file,
			 const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  qudaError_t
  qudaSetDevice_(int dev, const char *func,
		 const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }
  
  qudaError_t
  qudaGetDeviceCount_(int *count, const char *func,
		      const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    *count = 1;
    return error;
  }

  qudaError_t
  qudaGetDeviceProperties_(qudaDeviceProp *prop, int device, const char *func,
			   const char *file, const char *line) {
    if(device != 0) return qudaErrorInvalidDevice;

    std::string device_name = "CPU OpenMP host device";
    int max_dim = std::numeric_limits<int>::max();

    //strncpy(prop->name, device_name.c_str(), 256);
    prop->totalGlobalMem = std::numeric_limits<size_t>::max();
    //prop->sharedMemPerBlock = 0;
    prop->regsPerBlock = std::numeric_limits<int>::max();
    prop->warpSize = 1;
    prop->maxThreadsPerBlock = 8;  // FIXME
    prop->maxThreadsDim[0] = max_dim;
    prop->maxThreadsDim[1] = max_dim;
    prop->maxThreadsDim[2] = max_dim;
    prop->maxGridSize[0] = max_dim;
    prop->maxGridSize[1] = max_dim;
    prop->maxGridSize[2] = max_dim;
    prop->clockRate = 1;
    prop->memoryClockRate = 1;
    prop->memoryBusWidth = 1;
    prop->totalConstMem = std::numeric_limits<std::size_t>::max();
    prop->major = 1;
    prop->minor = 0;
    prop->multiProcessorCount = 1;  // FIXME: number of threads
    prop->l2CacheSize = std::numeric_limits<int>::max();
    prop->maxThreadsPerMultiProcessor = prop->maxThreadsPerBlock;
    prop->computeMode = 0;
    prop->clockInstructionRate = prop->clockRate;

#if 0
    qudaDeviceArch_t arch;
    arch.hasGlobalInt32Atomics = 1;
    arch.hasGlobalFloatAtomicExch = 1;
    arch.hasSharedInt32Atomics = 1;
    arch.hasSharedFloatAtomicExch = 1;
    arch.hasFloatAtomicAdd = 1;
    arch.hasGlobalInt64Atomics = 1;
    arch.hasSharedInt64Atomics = 1;
    arch.hasDoubles = 1;
    arch.hasWarpVote = 0;
    arch.hasWarpBallot = 0;
    arch.hasWarpShuffle = 0;
    arch.hasFunnelShift = 0;
    arch.hasThreadFenceSystem = 1;
    arch.hasSyncThreadsExt = 1;
    arch.hasSurfaceFuncs = 0;
    arch.has3dGrid = 1;
    arch.hasDynamicParallelism = 0;
    prop->arch = arch;
#endif

    prop->concurrentKernels = 1;
    prop->pciBusID = 0;
    prop->pciDeviceID = 0;
    prop->maxSharedMemoryPerMultiProcessor = prop->sharedMemPerBlock;
    prop->isMultiGpuBoard = 0;
    prop->canMapHostMemory = 1;
    prop->gcnArch = 0;

    return qudaSuccess;
  }

  qudaError_t
  qudaHostGetDevicePointer_(void **pDevice, void *pHost, unsigned int flags,
			    const char *func, const char *file,
			    const char *line) {
    qudaError_t error = qudaSuccess;
    *pDevice = pHost;
    return error;
  }
  
  qudaError_t
  qudaDriverGetVersion_(int* driverVersion, const char *func,
			const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    driverVersion = 0;
    return error;
  }

  qudaError_t
  qudaRuntimeGetVersion_(int* runtimeVersion, const char *func,
			 const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    runtimeVersion = 0;
    return error;
  }

  qudaError_t
  qudaHostRegister_(void* ptr, size_t size, unsigned int flags,
		    const char *func, const char *file, const char *line) {
    qudaError_t error = qudaSuccess;
    return error;
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
