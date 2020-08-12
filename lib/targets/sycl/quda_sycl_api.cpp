#include <quda_internal.h>
#include <sys/time.h>

//dim3 gridDim;
//dim3 blockDim;
//dim3 blockIdx;
//dim3 threadIdx;

// if this macro is defined then we profile the API calls
//#define API_PROFILE

#ifdef API_PROFILE
#define PROFILE(f, idx)                                 \
  apiTimer.TPSTART(idx);				\
  f;                                                    \
  apiTimer.TPSTOP(idx);
#else
#define PROFILE(f, idx) f;
#endif

qudaError_t
qudaGetLastError(void) {
  qudaError_t error = qudaSuccess;
  //int e = errno;
  //if(e!=0) {
  //  printf("errno: %i\n", errno);
  //  error = qudaErrorUnknown;
  //}
  return error;
}

qudaError_t
qudaPeekAtLastError(void) {
  qudaError_t error = qudaSuccess;
  return error;
}

const char*
qudaGetErrorString(qudaError_t error)
{
  return "error";
}

qudaError_t
qudaHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
{
  return qudaSuccess;
}

qudaError_t
qudaHostUnregister(void *hostPtr)
{
  return qudaSuccess;
}

qudaError_t
qudaHostGetDevicePointer(void **devPtr, void *hstPtr, unsigned int flags)
{
  *devPtr = hstPtr;
  return qudaSuccess;
}

qudaError_t
qudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
  *leastPriority = 0;
  *greatestPriority = 1;
  return qudaSuccess;
}

qudaError_t
qudaGetDeviceCount(int *count)
{
  *count = 1;
  return qudaSuccess;
}

qudaError_t
qudaDeviceReset(void)
{
  return qudaSuccess;
}

qudaError_t
qudaStreamCreateWithPriority(qudaStream_t *pStream,
			     unsigned int flags, int priority)
{
  *pStream = 0;
  return qudaSuccess;
}

qudaError_t
qudaStreamDestroy(qudaStream_t stream)
{
  return qudaSuccess;
}

qudaError_t
qudaEventCreate(qudaEvent_t *event)
{
  *event = 0.0;
  return qudaSuccess;
}

qudaError_t
qudaEventCreate(qudaEvent_t *event, unsigned int flags)
{
  *event = 0.0;
  return qudaSuccess;
}

qudaError_t
qudaEventCreateWithFlags(qudaEvent_t *event, unsigned int flags)
{
  *event = 0.0;
  return qudaSuccess;
}

qudaError_t
qudaEventDestroy(qudaEvent_t event)
{
  return qudaSuccess;
}

qudaError_t
qudaEventQuery(qudaEvent_t event)
{
  return qudaSuccess;
}

qudaError_t
qudaEventRecord(qudaEvent_t &event, qudaStream_t stream)
{
  //event = omp_get_wtime();
  struct timeval tv;
  gettimeofday(&tv, NULL);
  event = tv.tv_sec + 1e-6*tv.tv_usec;
  return qudaSuccess;
}

qudaError_t
qudaEventElapsedTime(float *ms, qudaEvent_t start, qudaEvent_t end)
{
  *ms = 1000.0*(end - start);
  return qudaSuccess;
}

qudaError_t
qudaIpcGetMemHandle(qudaIpcMemHandle_t *handle, void *devPtr)
{
  return qudaSuccess;
}

qudaError_t
qudaIpcOpenMemHandle(void **devPtr, qudaIpcMemHandle_t handle,
		     unsigned int flags)
{
  return qudaSuccess;
}

qudaError_t
qudaIpcCloseMemHandle(void *devPtr)
{
  return qudaSuccess;
}

qudaError_t
qudaIpcGetEventHandle(qudaIpcEventHandle_t *handle, qudaEvent_t event)
{
  return qudaSuccess;
}

qudaError_t
qudaIpcOpenEventHandle(qudaEvent_t *event, qudaIpcEventHandle_t handle)
{
  return qudaSuccess;
}

qudaError_t
qudaMemset2D(void *devPtr, size_t pitch, int value, size_t width,
	     size_t height)
{
  /// TODO
  printf("ERROR: qudaMemset2D unimplemented\n");
  return qudaSuccess;
}

qudaError_t
qudaMemset2DAsync(void *devPtr, size_t pitch, int value,
		  size_t width, size_t height)
{
  /// TODO
  printf("ERROR: qudaMemset2DAsync unimplemented\n");
  return qudaSuccess;
}

qudaError_t
qudaMemset2DAsync(void *devPtr, size_t pitch, int value,
		  size_t width, size_t height, qudaStream_t stream)
{
  /// TODO
  printf("ERROR: qudaMemset2DAsync unimplemented\n");
  return qudaSuccess;
}

qudaError_t
qudaMemPrefetchAsync(const void *devPtr, size_t count,
		     int dstDevice, qudaStream_t stream)
{
  return qudaSuccess;
}

qudaError_t
qudaGetDeviceProperties(qudaDeviceProp *prop, int device)
{
  if(device != 0) return qudaErrorInvalidDevice;

  std::string device_name = "CPU OpenMP host device";
  int max_dim = 1<<30; //std::numeric_limits<int>::max();

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
  prop->major = 2;
  prop->minor = 0;
  prop->multiProcessorCount = 4;  // FIXME: number of threads
  prop->l2CacheSize = std::numeric_limits<int>::max();
  prop->maxThreadsPerMultiProcessor = prop->maxThreadsPerBlock;
  prop->computeMode = 0;
  prop->clockInstructionRate = prop->clockRate;
  prop->concurrentKernels = 1;
  prop->pciBusID = 0;
  prop->pciDeviceID = 0;
  prop->maxSharedMemoryPerMultiProcessor = prop->sharedMemPerBlock;
  prop->isMultiGpuBoard = 0;
  prop->canMapHostMemory = 1;

  return qudaSuccess;
}

qudaError_t
qudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
  *canAccessPeer = 0;
  return qudaSuccess;
}

namespace quda {

  static TimeProfile apiTimer("API calls");

  void qudaMemcpy_(void *dst, const void *src, size_t count,
		   qudaMemcpyKind kind, const char *func, const char *file,
		   const char *line) {
    if (count == 0) return;
    memcpy(dst, src, count);
  }

  void qudaMemcpyAsync_(void *dst, const void *src, size_t count,
			qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line)
  {
    if (count == 0) return;
    memcpy(dst, src, count);
  }

  qudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func,
				     const char *file, const char *line)
  {
    return cudaSuccess;
  }

  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src,
			  size_t spitch, size_t width, size_t height,
                          qudaMemcpyKind kind, const qudaStream_t &stream,
			  const char *func, const char *file, const char *line)
  {
    // TODO
    errorQuda("qudaMemcpy2DAsync_ %s %s %s\n", func, file, line);
  }

  void qudaMemset_(void *ptr, int value, size_t count, const char *func,
		   const char *file, const char *line)
  {
    memset(ptr, value, count);
  }

  void qudaMemsetAsync_(void *ptr, int value, size_t count,
			const qudaStream_t &stream, const char *func,
                        const char *file, const char *line)
  {
    if (count == 0) return;
    memset(ptr, value, count);
  }

  qudaError_t
  qudaLaunchKernel(const void *func, dim3 gridDim0, dim3 blockDim0,
		   void **args, size_t sharedMem, qudaStream_t stream)
  {
    qudaError_t error = qudaSuccess;
    errorQuda("qudaLaunchKernel\n");
#if 0
    if(gridDim0.z>0 || blockDim0.z>0) {
      errorQuda("qudaLaunchKernel gz: %i  bz: %i\n", gridDim0.z, blockDim0.z);
    }
    auto cf = const_cast<void *>(func);
    typedef void (*fp)(void *);
    fp f = reinterpret_cast<fp>(cf); // must assume single pointer arg
#if 0
    sycl::range<3> globalSize{gridDim0.x*blockDim0.x, gridDim0.y*blockDim0.y, gridDim0.z*blockDim0.z};
    sycl::range<3> localSize{blockDim0.x, blockDim0.y, blockDim0.z};
    defaultQueue.submit([&] (sycl::handler &h)
			{ h.parallel_for<class test>(
						     (globalSize, localSize), [=](sycl::nd_item<3> ndi)
						    {
						      f(args[0]);
						    });
			    });
#endif
    //(globalSize, localSize), [=](nd_item<1> ndi) {
    //func(, ndi);
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
	    f(args[0]);
	  }
	}
      }
    }
#endif
    return error;
  }

#if 0
  qudaError_t
  qudaLaunch_(dim3 gridDim0, dim3 blockDim0, size_t sharedMem0,
	      qudaStream_t stream0, const char *func, const char *file,
	      const char *line, std::function<void()> f)
  {
    qudaError_t error = qudaSuccess;
    gridDim = gridDim0;
    blockDim = blockDim0;
    int ngx = gridDim0.x;
    int ngy = gridDim0.y;
    int ngz = gridDim0.z;
    int nbx = blockDim0.x;
    int nby = blockDim0.y;
    int nbz = blockDim0.z;
    //printf("ngx: %i  ngy: %i  nbx: %i  nby: %i\n", ngx, ngy, nbx, nby);
    for(int igx=0; igx<ngx; igx++) {
      blockIdx.x = igx;
      for(int igy=0; igy<ngy; igy++) {
	blockIdx.y = igy;
	for(int igz=0; igz<ngz; igz++) {
	  blockIdx.z = igz;
	  for(int ibx=0; ibx<nbx; ibx++) {
	    threadIdx.x = ibx;
	    for(int iby=0; iby<nby; iby++) {
	      threadIdx.y = iby;
	      for(int ibz=0; ibz<nbz; ibz++) {
		threadIdx.z = ibz;
		f();
	      }
	    }
	  }
	}
      }
    }
    return error;
  }
#endif

  qudaError_t qudaEventQuery(qudaEvent_t &event)
  {
    return cudaSuccess;
  }

  qudaError_t qudaEventRecord(qudaEvent_t &event, qudaStream_t stream)
  {
    return cudaSuccess;
  }

  qudaError_t qudaStreamWaitEvent(qudaStream_t stream, qudaEvent_t event,
				  unsigned int flags)
  {
    return cudaSuccess;
  }

  qudaError_t qudaEventSynchronize(qudaEvent_t &event)
  {
    return cudaSuccess;
  }

  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file,
				     const char *line)
  {
    return cudaSuccess;
  }

  void printAPIProfile() {
#ifdef API_PROFILE
    apiTimer.Print();
#endif
  }

} // namespace quda
