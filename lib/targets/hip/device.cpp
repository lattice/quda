#include <hip/hip_runtime.h>
#include <util_quda.h>
#include <quda_internal.h>

#include <hip/hip_runtime_api.h>

static hipDeviceProp_t deviceProp;
static hipStream_t *streams;
static const int Nstream = 9;

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    void init(int dev)
    {
      
      if (initialized) return;
      initialized = true;
      printfQuda("*** HIP BACKEND ***\n");

      int driver_version=0;
      hipDriverGetVersion(&driver_version);
      printfQuda("HIP Driver version = %d\n", driver_version);

      int runtime_version=0;
      hipRuntimeGetVersion(&runtime_version);
      printfQuda("HIP Runtime version = %d\n", runtime_version);

      int deviceCount;
      hipGetDeviceCount(&deviceCount);
      if (deviceCount == 0) {
        errorQuda("No HIP devices found");
      }

      for (int i = 0; i < deviceCount; i++) {
        hipGetDeviceProperties(&deviceProp, i);
        checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
        if (getVerbosity() >= QUDA_SUMMARIZE) { printfQuda("Found device %d: %s\n", i, deviceProp.name); }
      }

      
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Using device %d: %s\n", dev, deviceProp.name);
      }
      hipSetDevice(dev);
      checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode


      hipDeviceSetCacheConfig(hipFuncCachePreferL1);
      //hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);
      hipGetDeviceProperties(&deviceProp, dev);
      checkCudaErrorNoSync();
    }

    void create_context()
    {
      streams = new hipStream_t[Nstream];

      int greatestPriority;
      int leastPriority;
      hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      for (int i=0; i<Nstream-1; i++) {
        hipStreamCreateWithPriority(&streams[i], hipStreamDefault, greatestPriority);
      }
      hipStreamCreateWithPriority(&streams[Nstream-1], hipStreamDefault, leastPriority);
      checkCudaError();
    }

    void destroy()
    {
      if (streams) {
        for (int i=0; i<Nstream; i++) hipStreamDestroy(streams[i]);
        delete []streams;
        streams = nullptr;
      }

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env, "1") == 0) {
        // end this CUDA context
        hipDeviceReset();
      }
    }


    hipStream_t get_cuda_stream(const qudaStream_t &stream)
    {
      return streams[stream.idx];
    }

    qudaStream_t get_stream(unsigned int i)
    {
      if (i > Nstream) errorQuda("Invalid stream index %u", i);
      qudaStream_t stream;
      stream.idx = i;
      return stream;
      //return qudaStream_t(i);
      // return streams[i];
    }

    qudaStream_t get_default_stream()
    {
      qudaStream_t stream;
      stream.idx = Nstream - 1;
      return stream;
      //return qudaStream_t(Nstream - 1);
      //return streams[Nstream - 1];
    }

    unsigned int get_default_stream_idx()
    {
      return Nstream - 1;
    }

    bool managed_memory_supported()
    {
      // managed memory is not supported
      return false;
    }

    bool shared_memory_atomic_supported()
    {
      // shared memory atomics are supported  in the HIP API
      return true;
    }

    size_t max_default_shared_memory() { return deviceProp.sharedMemPerBlock; } 

    size_t max_dynamic_shared_memory()
    {
      static int max_shared_bytes = 0;
      if (!max_shared_bytes) hipDeviceGetAttribute(&max_shared_bytes,  hipDeviceAttributeMaxSharedMemoryPerBlock, comm_gpuid());
      return max_shared_bytes;
    }

    unsigned int max_threads_per_block() { return deviceProp.maxThreadsPerBlock; }

    unsigned int max_threads_per_processor() { return deviceProp.maxThreadsPerMultiProcessor; }

    unsigned int max_threads_per_block_dim(int i) { return deviceProp.maxThreadsDim[i]; }

    unsigned int max_grid_size(int i) { return deviceProp.maxGridSize[i]; }

    unsigned int processor_count() { return deviceProp.multiProcessorCount; }

    unsigned int max_blocks_per_processor()
    {
	    return 40;
    }

    namespace profile {

      void start()
      {
	// FIXME: HIP PROFILER DEPRECATED
      }

      void stop()
      {
	// FIXME: HIP PROFILER DEPRECARED
      }

    } // namespace profile

  } // namespace device
} // namespace quda
