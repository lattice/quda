#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <util_quda.h>
#include <quda_internal.h>

#ifdef QUDA_NVML
#include <nvml.h>
#endif

#ifdef NUMA_NVML
#include <numa_affinity.h>
#endif

cudaDeviceProp deviceProp;
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
      cudaDriverGetVersion(&driver_version);
      printfQuda("CUDA Driver version = %d\n", driver_version);

      int runtime_version;
      cudaRuntimeGetVersion(&runtime_version);
      printfQuda("CUDA Runtime version = %d\n", runtime_version);

#ifdef QUDA_NVML
      nvmlReturn_t result = nvmlInit();
      if (NVML_SUCCESS != result) errorQuda("NVML Init failed with error %d", result);
      const int length = 80;
      char graphics_version[length];
      result = nvmlSystemGetDriverVersion(graphics_version, length);
      if (NVML_SUCCESS != result) errorQuda("nvmlSystemGetDriverVersion failed with error %d", result);
      printfQuda("Graphic driver version = %s\n", graphics_version);
      result = nvmlShutdown();
      if (NVML_SUCCESS != result) errorQuda("NVML Shutdown failed with error %d", result);
#endif

      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      if (deviceCount == 0) { errorQuda("No CUDA devices found"); }

      for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&deviceProp, i);
        checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
        if (getVerbosity() >= QUDA_SUMMARIZE) { printfQuda("Found device %d: %s\n", i, deviceProp.name); }
      }

      cudaGetDeviceProperties(&deviceProp, dev);
      checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
      if (deviceProp.major < 1) { errorQuda("Device %d does not support CUDA", dev); }

      // Check GPU and QUDA build compatibiliy
      // 4 cases:
      // a) QUDA and GPU match: great
      // b) QUDA built for higher compute capability: error
      // c) QUDA built for lower major compute capability: warn if QUDA_ALLOW_JIT, else error
      // d) QUDA built for same major compute capability but lower minor: warn

      const int my_major = __COMPUTE_CAPABILITY__ / 100;
      const int my_minor = (__COMPUTE_CAPABILITY__ - my_major * 100) / 10;
      // b) UDA was compiled for a higher compute capability
      if (deviceProp.major * 100 + deviceProp.minor * 10 < __COMPUTE_CAPABILITY__)
        errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. ** \n --- "
                  "Please set the correct QUDA_GPU_ARCH when running cmake.\n",
                  deviceProp.major, deviceProp.minor, my_major, my_minor);

      // c) QUDA was compiled for a lower compute capability
      if (deviceProp.major < my_major) {
        char *allow_jit_env = getenv("QUDA_ALLOW_JIT");
        if (allow_jit_env && strcmp(allow_jit_env, "1") == 0) {
          if (getVerbosity() > QUDA_SILENT)
            warningQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- "
                        "Jitting the PTX since QUDA_ALLOW_JIT=1 was set. Note that this will take some time.\n",
                        deviceProp.major, deviceProp.minor, my_major, my_minor);
        } else {
          errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n --- "
                    "Please set the correct QUDA_GPU_ARCH when running cmake.\n If you want the PTX to be jitted for "
                    "your current GPU arch please set the enviroment variable QUDA_ALLOW_JIT=1.",
                    deviceProp.major, deviceProp.minor, my_major, my_minor);
        }
      }
      // d) QUDA built for same major compute capability but lower minor
      if (deviceProp.major == my_major and deviceProp.minor > my_minor) {
        warningQuda(
          "** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- This might "
          "result in a lower performance. Please consider adjusting QUDA_GPU_ARCH when running cmake.\n",
          deviceProp.major, deviceProp.minor, my_major, my_minor);
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) { printfQuda("Using device %d: %s\n", dev, deviceProp.name); }
#ifndef USE_QDPJIT
      cudaSetDevice(dev);
      checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
#endif

#ifdef NUMA_NVML
      char *enable_numa_env = getenv("QUDA_ENABLE_NUMA");
      if (enable_numa_env && strcmp(enable_numa_env, "0") == 0) {
        if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling numa_affinity\n");
      } else {
        setNumaAffinityNVML(dev);
      }
#endif

      cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
      // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
      // cudaGetDeviceProperties(&deviceProp, dev);
    }

    void create_context()
    {
      streams = new qudaStream_t[Nstream];

      int greatestPriority;
      int leastPriority;
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      for (int i = 0; i < Nstream - 1; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, greatestPriority);
      }
      cudaStreamCreateWithPriority(&streams[Nstream - 1], cudaStreamDefault, leastPriority);

      checkCudaError();
    }

    void destroy()
    {
      if (streams) {
        for (int i = 0; i < Nstream; i++) cudaStreamDestroy(streams[i]);
        delete[] streams;
        streams = nullptr;
      }

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env, "1") == 0) {
        // end this CUDA context
        cudaDeviceReset();
      }
    }

    namespace profile
    {

      void start() { cudaProfilerStart(); }

      void stop() { cudaProfilerStop(); }

    } // namespace profile

  } // namespace device
} // namespace quda
