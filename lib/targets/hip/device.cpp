#include <hip/hip_runtime.h>
#include <util_quda.h>
#include <quda_internal.h>

#include <hip/hip_runtime_api.h>

hipDeviceProp_t deviceProp;
qudaStream_t *streams;

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
      int driver_version;
      
      hipDriverGetVersion(&driver_version);
      printfQuda("HIP Driver version = %d\n", driver_version);

      int runtime_version;
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
        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printfQuda("Found device %d: %s\n", i, deviceProp.name);
        }
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Using device %d: %s\n", dev, deviceProp.name);
      }
#ifndef USE_QDPJIT
      hipSetDevice(dev);
      checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
#endif


      hipDeviceSetCacheConfig(hipFuncCachePreferL1);
      //hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);
      // hipGetDeviceProperties(&deviceProp, dev);
    }

    void create_context()
    {
      streams = new qudaStream_t[Nstream];

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
      if (device_reset_env && strcmp(device_reset_env,"1") == 0) {
        // end this CUDA context
        hipDeviceReset();
      }
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

    }

  }
}
