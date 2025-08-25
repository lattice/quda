#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <util_quda.h>
#include <quda_internal.h>
#include <quda_cuda_api.h>
#include <nvml.h>
#include "monitor.h"
#include "mnnvl_helper.h"

static cudaDeviceProp deviceProp;
static cudaStream_t *streams;
static const int Nstream = 9;

#define CHECK_CUDA_ERROR(func)                                                                                         \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

#undef CU_CHECK
#define CU_CHECK(stmt)                                                                  \
    do {                                                                                \
        CUresult result = (stmt);                                                       \
        const char *str;                                                                \
        if (CUDA_SUCCESS != result) {                                                   \
            CUresult ret = cuGetErrorString(result, &str);                              \
            if (ret == CUDA_ERROR_INVALID_VALUE) str = "Unknown error";                 \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, str); \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)

#define NVML_CHECK(func)                                                                                               \
  {                                                                                                                    \
    nvmlReturn_t ret = func;                                                                                           \
    if (ret == NVML_ERROR_NOT_SUPPORTED) {                                                                             \
      warningQuda("%s not supported on this GPU\n", nvmlErrorString(ret));                                             \
    } else if (ret != NVML_SUCCESS) {                                                                                  \
      errorQuda(" NVML returns %s", nvmlErrorString(ret));                                                             \
    }                                                                                                                  \
  }


namespace quda
{

  namespace device
  {

    static bool initialized = false;

    static int device_id = -1;

    static nvmlDevice_t monitor_device_id;

    // forward declare
    bool is_mnnvl_supported(int dev_id);
    static bool mnnvl_capable = false;

    // This is exposed via mnnvl_helper.h
    bool get_mnnvl_capable()
    {
       return mnnvl_capable;
    } 

    int get_driver_version()
    {
      int driver_version;
      CHECK_CUDA_ERROR(cudaDriverGetVersion(&driver_version));
      return driver_version;
    }

    int get_runtime_version()
    {
      int runtime_version;
      CHECK_CUDA_ERROR(cudaRuntimeGetVersion(&runtime_version));
      return runtime_version;
    }

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;

      printfQuda("CUDA Driver version = %d\n", get_driver_version());
      printfQuda("CUDA Runtime version = %d\n", get_runtime_version());

#ifdef QUDA_LARGE_KERNEL_ARG
      if (get_driver_version() < 12010) errorQuda("Large kernel arguments not supported on pre CUDA 12.1 driver");
#endif

      NVML_CHECK(nvmlInit());
      const int length = 80;
      char graphics_version[length];
      NVML_CHECK(nvmlSystemGetDriverVersion(graphics_version, length));
      printfQuda("Graphic driver version = %s\n", graphics_version);

      for (int i = 0; i < get_device_count(); i++) {
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, i));
        logQuda(QUDA_SUMMARIZE, "Found device %d: %s\n", i, deviceProp.name);
      }

      CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
      if (deviceProp.major < 1) errorQuda("Device %d does not support CUDA", dev);

      // Check GPU and QUDA build compatibiliy
      // 4 cases:
      // a) QUDA and GPU match: great
      // b) QUDA built for higher compute capability: error
      // c) QUDA built for lower major compute capability: warn if QUDA_ALLOW_JIT, else error
      // d) QUDA built for same major compute capability but lower minor: warn

      const int my_major = __COMPUTE_CAPABILITY__ / 100;
      const int my_minor = (__COMPUTE_CAPABILITY__ - my_major * 100) / 10;
      // b) QUDA was compiled for a higher compute capability
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

      if (!deviceProp.unifiedAddressing) errorQuda("Device %d does not support unified addressing", dev);

      logQuda(QUDA_SUMMARIZE, "Using device %d: %s\n", dev, deviceProp.name);
#ifndef USE_QDPJIT
      CHECK_CUDA_ERROR(cudaSetDevice(dev));
#endif

      CHECK_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
      //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
      // cudaGetDeviceProperties(&deviceProp, dev);

      device_id = dev;

      char pciBusId[13];
      CHECK_CUDA_ERROR(cudaDeviceGetPCIBusId(pciBusId, 13, device_id));
      NVML_CHECK(nvmlDeviceGetHandleByPciBusId(pciBusId, &monitor_device_id));
      char name[NVML_DEVICE_NAME_BUFFER_SIZE];
      NVML_CHECK(nvmlDeviceGetName(monitor_device_id, name, NVML_DEVICE_NAME_BUFFER_SIZE));

      printf("Initializing monitoring on device %d with pciBusId %s: %s\n", device_id, pciBusId, name);
      monitor::init();
      mnnvl_capable=is_mnnvl_supported(device_id);
    }

    void init_thread()
    {
      if (device_id == -1) errorQuda("No CUDA device has been initialized for this process");
      CHECK_CUDA_ERROR(cudaSetDevice(device_id));
    }

    auto get_power()
    {
      unsigned int power = 0;
      NVML_CHECK(nvmlDeviceGetPowerUsage(monitor_device_id, &power));
      return 1e-3 * power;
    }

    auto get_clock()
    {
      // other clocks available NVML_CLOCK_MEM and NVML_CLOCK_GRAPHICS
      unsigned int clock = 0;
      NVML_CHECK(nvmlDeviceGetClockInfo(monitor_device_id, NVML_CLOCK_SM, &clock));
      return clock;
    }

    auto get_temperature()
    {
      unsigned int temp = 0;
      NVML_CHECK(nvmlDeviceGetTemperature(monitor_device_id, NVML_TEMPERATURE_GPU, &temp));
      return temp;
    }

    state_t get_state()
    {
      state_t state;
      state.time = std::chrono::high_resolution_clock::now();
      state.power = get_power();
      state.clock = get_clock();
      state.temp = get_temperature();
      return state;
    }

    int get_device_count()
    {
      static int device_count = 0;
      if (device_count == 0) {
        CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
        if (device_count == 0) errorQuda("No CUDA devices found");
      }
      return device_count;
    }

    void get_visible_devices_string(char device_list_string[128])
    {
      char *device_order_env = getenv("CUDA_VISIBLE_DEVICES");

      if (device_order_env) {
        std::stringstream device_list_raw(device_order_env); // raw input
        std::stringstream device_list;                       // formatted (no commas)

        int device;
        while (device_list_raw >> device) {
          // check this is a valid policy choice
          if (device < 0) { errorQuda("Invalid CUDA_VISIBLE_DEVICES ordinal %d", device); }

          device_list << device;
          if (device_list_raw.peek() == ',') device_list_raw.ignore();
        }
        snprintf(device_list_string, 128, "%s", device_list.str().c_str());
      }
    }

    void print_device_properties()
    {
      for (int device = 0; device < get_device_count(); device++) {

        // cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
        printfQuda("%d - name:                    %s\n", device, deviceProp.name);
        printfQuda("%d - totalGlobalMem:          %lu bytes ( %.2f Gbytes)\n", device, deviceProp.totalGlobalMem,
                   deviceProp.totalGlobalMem / (float)(1024 * 1024 * 1024));
        printfQuda("%d - sharedMemPerBlock:       %lu bytes ( %.2f Kbytes)\n", device, deviceProp.sharedMemPerBlock,
                   deviceProp.sharedMemPerBlock / (float)1024);
        printfQuda("%d - regsPerBlock:            %d\n", device, deviceProp.regsPerBlock);
        printfQuda("%d - warpSize:                %d\n", device, deviceProp.warpSize);
        printfQuda("%d - memPitch:                %lu\n", device, deviceProp.memPitch);
        printfQuda("%d - maxThreadsPerBlock:      %d\n", device, deviceProp.maxThreadsPerBlock);
        printfQuda("%d - maxThreadsDim[0]:        %d\n", device, deviceProp.maxThreadsDim[0]);
        printfQuda("%d - maxThreadsDim[1]:        %d\n", device, deviceProp.maxThreadsDim[1]);
        printfQuda("%d - maxThreadsDim[2]:        %d\n", device, deviceProp.maxThreadsDim[2]);
        printfQuda("%d - maxGridSize[0]:          %d\n", device, deviceProp.maxGridSize[0]);
        printfQuda("%d - maxGridSize[1]:          %d\n", device, deviceProp.maxGridSize[1]);
        printfQuda("%d - maxGridSize[2]:          %d\n", device, deviceProp.maxGridSize[2]);
        printfQuda("%d - totalConstMem:           %lu bytes ( %.2f Kbytes)\n", device, deviceProp.totalConstMem,
                   deviceProp.totalConstMem / (float)1024);
        printfQuda("%d - compute capability:      %d.%d\n", device, deviceProp.major, deviceProp.minor);
        printfQuda("%d - multiProcessorCount      %d\n", device, deviceProp.multiProcessorCount);
#if CUDA_VERSION <= 12090
        printfQuda("%d - kernelExecTimeoutEnabled %s\n", device,
                   (deviceProp.kernelExecTimeoutEnabled ? "true" : "false"));
#endif
        printfQuda("%d - integrated               %s\n", device, (deviceProp.integrated ? "true" : "false"));
        printfQuda("%d - canMapHostMemory         %s\n", device, (deviceProp.canMapHostMemory ? "true" : "false"));
        int deviceComputeMode;
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&deviceComputeMode, cudaDevAttrComputeMode, device));
        switch (deviceComputeMode) {
        case 0: printfQuda("%d - computeMode              0: cudaComputeModeDefault\n", device); break;
        case 1: printfQuda("%d - computeMode              1: cudaComputeModeExclusive\n", device); break;
        case 2: printfQuda("%d - computeMode              2: cudaComputeModeProhibited\n", device); break;
        case 3: printfQuda("%d - computeMode              3: cudaComputeModeExclusiveProcess\n", device); break;
        default: errorQuda("Unknown deviceProp.computeMode.");
        }

        printfQuda("%d - surfaceAlignment         %lu\n", device, deviceProp.surfaceAlignment);
        printfQuda("%d - concurrentKernels        %s\n", device, (deviceProp.concurrentKernels ? "true" : "false"));
        printfQuda("%d - ECCEnabled               %s\n", device, (deviceProp.ECCEnabled ? "true" : "false"));
        printfQuda("%d - pciBusID                 %d\n", device, deviceProp.pciBusID);
        printfQuda("%d - pciDeviceID              %d\n", device, deviceProp.pciDeviceID);
        printfQuda("%d - pciDomainID              %d\n", device, deviceProp.pciDomainID);
        printfQuda("%d - tccDriver                %s\n", device, (deviceProp.tccDriver ? "true" : "false"));
        switch (deviceProp.asyncEngineCount) {
        case 0: printfQuda("%d - asyncEngineCount         1: host -> device only\n", device); break;
        case 1: printfQuda("%d - asyncEngineCount         2: host <-> device\n", device); break;
        case 2: printfQuda("%d - asyncEngineCount         0: not supported\n", device); break;
        default: errorQuda("Unknown deviceProp.asyncEngineCount.");
        }
        printfQuda("%d - unifiedAddressing        %s\n", device, (deviceProp.unifiedAddressing ? "true" : "false"));
#if CUDA_VERSION <= 12090
        printfQuda("%d - memoryClockRate          %d kilohertz\n", device, deviceProp.memoryClockRate);
#endif
        printfQuda("%d - memoryBusWidth           %d bits\n", device, deviceProp.memoryBusWidth);
        printfQuda("%d - l2CacheSize              %d bytes\n", device, deviceProp.l2CacheSize);
        printfQuda("%d - maxThreadsPerMultiProcessor          %d\n\n", device, deviceProp.maxThreadsPerMultiProcessor);
      }
    }

    void create_context()
    {
      streams = new cudaStream_t[Nstream];

      int greatestPriority;
      int leastPriority;
      CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      for (int i=0; i<Nstream-1; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, greatestPriority));
      }
      CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&streams[Nstream - 1], cudaStreamDefault, leastPriority));
    }

    void destroy()
    {
      if (streams) {
        for (int i = 0; i < Nstream; i++) CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        delete []streams;
        streams = nullptr;
      }

      monitor::destroy();

      NVML_CHECK(nvmlShutdown());

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env, "1") == 0) {
        // end this CUDA context
        CHECK_CUDA_ERROR(cudaDeviceReset());
      }
    }

    qudaStream_t get_stream(unsigned int i)
    {
      if (i >= Nstream) errorQuda("Invalid stream index %u", i);
      qudaStream_t stream;
      stream.idx = i;
      return stream;
      // return qudaStream_t(i);
      // return streams[i];
    }

    qudaStream_t get_default_stream()
    {
      qudaStream_t stream;
      stream.idx = Nstream - 1;
      return stream;
      // return qudaStream_t(Nstream - 1);
      // return streams[Nstream - 1];
    }

    unsigned int get_default_stream_idx() { return Nstream - 1; }

    bool managed_memory_supported()
    {
      // managed memory is supported on Pascal and up
      return deviceProp.major >= 6;
    }

    bool shared_memory_atomic_supported()
    {
      // shared memory atomics are supported on Maxwell and up
      return deviceProp.major >= 5;
    }

    size_t max_default_shared_memory() { return deviceProp.sharedMemPerBlock; }

    size_t max_dynamic_shared_memory()
    {
      static int max_shared_bytes = 0;
      if (!max_shared_bytes)
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, comm_gpuid()));
      return max_shared_bytes;
    }

    unsigned int max_threads_per_block() { return deviceProp.maxThreadsPerBlock; }

    unsigned int max_threads_per_processor() { return deviceProp.maxThreadsPerMultiProcessor; }

    unsigned int max_threads_per_block_dim(int i) { return deviceProp.maxThreadsDim[i]; }

    unsigned int max_grid_size(int i) { return deviceProp.maxGridSize[i]; }

    unsigned int processor_count() { return deviceProp.multiProcessorCount; }

    unsigned int max_blocks_per_processor()
    {
      static int max_blocks_per_sm = 0;
      if (!max_blocks_per_sm)
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, comm_gpuid()));
      return max_blocks_per_sm;
    }

    namespace profile
    {

      void start() { cudaProfilerStart(); }

      void stop() { cudaProfilerStop(); }

    } // namespace profile

    bool is_mnnvl_supported(int dev_id) {
      nvmlGpuFabricInfo_t fabricInfo = {};
      const unsigned char zero[NVML_GPU_FABRIC_UUID_LEN] = {0};
      cudaDeviceProp prop;
      char pcie_bdf[50] = {0};
      int nbytes = 0;
      int attr;
      nvmlReturn_t nvml_status;
      nvmlDevice_t local_device;
      CUdevice my_dev;
      int cuda_drv_version;
#if 0
      fabricInfo.version = nvmlGpuFabricInfo_v2;
#endif
      CHECK_CUDA_ERROR(cudaDriverGetVersion(&cuda_drv_version));
      CU_CHECK(cuDeviceGet(&my_dev, dev_id));

      CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev_id));
      nbytes =
        snprintf(pcie_bdf, 50, "%x:%x:%x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
      if (nbytes < 0 || nbytes > 50) {
        logQuda(QUDA_DEBUG_VERBOSE, "Unable to set device pcie bdf for our local device, disabling MNNVL\n");
        return false;
      }

      if (cuda_drv_version >= 12040 && prop.major >= 9) {
        nvml_status = nvmlDeviceGetHandleByPciBusId(pcie_bdf, &local_device);
        if (nvml_status != NVML_SUCCESS) {
            logQuda(QUDA_DEBUG_VERBOSE, "nvmlDeviceGetHandleByPciBusId failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

#if 0
        /* Some platforms with older driver may not support this API, so bypass MNNVL discovery */
        if (nvmlDeviceGetGpuFabricInfoV == NULL) {
            logQuda(QUDA_DEBUG_VERBOSE, "nvmlDeviceGetGpuFabricInfoV not found, MNNVL not supported\n");
            return false;
        }

        nvml_status = nvmlDeviceGetGpuFabricInfoV(local_device, &fabricInfo);
#endif

        nvml_status = nvmlDeviceGetGpuFabricInfo(local_device, &fabricInfo);
        if (nvml_status != NVML_SUCCESS) {
            logQuda(QUDA_DEBUG_VERBOSE, "nvmlDeviceGetGpuFabricInfoV failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

        CU_CHECK(
            cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, my_dev));
        if (attr <= 0) {
            logQuda(QUDA_DEBUG_VERBOSE, "CUDA EGM fabric not supported\n");
            return false;
        }

        if (fabricInfo.state < NVML_GPU_FABRIC_STATE_COMPLETED ||
            memcmp(fabricInfo.clusterUuid, zero, NVML_GPU_FABRIC_UUID_LEN) == 0) {
            logQuda(QUDA_DEBUG_VERBOSE, "MNNVL not supported\n");
            return false;
        }
    } else {
        logQuda(QUDA_VERBOSE, "MNNVL disabled\n");
        return false;
    }

    return true;
}



  } // namespace device

  namespace target
  {

    namespace cuda
    {

      cudaStream_t get_stream(const qudaStream_t &stream) { return streams[stream.idx]; }

    } // namespace cuda

  } // namespace target

} // namespace quda
