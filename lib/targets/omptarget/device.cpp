#include <omp.h>
#include <util_quda.h>
#include <quda_internal.h>
#include <quda_cuda_api.h>

#ifdef QUDA_NVML
#include <nvml.h>
#endif

#ifdef NUMA_NVML
#include <numa_affinity.h>
#endif

static char ompdevname[] = "OpenMP Target Device";
static char omphostname[] = "OpenMP Host Device";
struct cudaDeviceProp{
  char*name;
  int major,minor;
  size_t sharedMemPerBlock;
  unsigned int maxThreadsPerBlock;
  unsigned int maxThreadsPerMultiProcessor;
  unsigned int maxThreadsDim[3];
  unsigned int maxGridSize[3];
  unsigned int multiProcessorCount;
  unsigned int warpSize;
  unsigned long totalGlobalMem;
  unsigned long totalConstMem;
  unsigned long memPitch;
  unsigned int regsPerBlock;
  int unifiedAddressing;
  int deviceOverlap;
};

static int cudaDriverGetVersion(int*v){*v=0;return 0;}
static int cudaRuntimeGetVersion(int*v){*v=0;return 0;}
static int cudaGetDeviceCount(int*c){*c=omp_get_num_devices();return 0;}
static int cudaGetDeviceProperties(cudaDeviceProp*p,int dev)
{
  /* FIXME totally fake numbers */
  if(0<omp_get_num_devices()){
    p->name = ompdevname;
    p->major = 7;
    p->minor = 0;
    p->sharedMemPerBlock = 0;
    p->maxThreadsPerBlock = 1024;
    p->maxThreadsPerMultiProcessor = 1024;
    p->maxThreadsDim[0] = 1024;
    p->maxThreadsDim[1] = 1024;
    p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 65536;
    p->maxGridSize[1] = 65536;
    p->maxGridSize[2] = 65536;
    p->multiProcessorCount = 16;
  }else{
    p->name = omphostname;
    p->major = 7;
    p->minor = 0;
    p->sharedMemPerBlock = 0;
    p->maxThreadsPerBlock = 1024;
    p->maxThreadsPerMultiProcessor = 1024;
    p->maxThreadsDim[0] = 1024;
    p->maxThreadsDim[1] = 1024;
    p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 65536;
    p->maxGridSize[1] = 65536;
    p->maxGridSize[2] = 65536;
    p->multiProcessorCount = 16;
  }
  p->warpSize = 32;
  p->totalGlobalMem = 1u<34;
  p->totalConstMem = 65536;
  p->memPitch = 1u<31;
  p->regsPerBlock = 32768;
  p->unifiedAddressing = 1;
  p->deviceOverlap = 0;

  /* Now actually querying the device in some way. */
  int m = 0;
  #pragma omp target teams map(tofrom:m)
  if(omp_get_team_num()==0)
    m = omp_get_max_threads();
  p->maxThreadsPerMultiProcessor = m;
  p->maxThreadsPerBlock = m;
  p->maxThreadsDim[0] = m;
  p->maxThreadsDim[1] = m;
  p->maxThreadsDim[2] = m;
  #pragma omp target map(tofrom:m)
  m = omp_get_num_procs();
  p->multiProcessorCount = m/32;
  return 0;
}

enum {cudaFuncCachePreferL1};
enum {cudaStreamDefault};
static int cudaSetDevice(int d){omp_set_default_device(d); return 0;}
static int cudaDeviceSetCacheConfig(int a){ompwip();return 0;}
static int cudaDeviceGetStreamPriorityRange(int*lo,int*hi){ompwip();lo=0;hi=0;return 0;}
static int cudaStreamCreateWithPriority(cudaStream_t*s,int lo,int hi){ompwip();*s=0;return 0;}
static int cudaStreamDestroy(cudaStream_t s){ompwip();return 0;}
static int cudaDeviceReset(){ompwip();return 0;}

static cudaDeviceProp deviceProp;
static cudaStream_t *streams;
static const int Nstream = 9;

#define CHECK_CUDA_ERROR(func)                                          \
  cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;

      int driver_version;
      CHECK_CUDA_ERROR(cudaDriverGetVersion(&driver_version));
      printfQuda("CUDA Driver version = %d\n", driver_version);

      int runtime_version;
      CHECK_CUDA_ERROR(cudaRuntimeGetVersion(&runtime_version));
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

      if(get_device_count() == 0)  /* TODO: get CPU device properties */
        cudaGetDeviceProperties(&deviceProp, -1);
      for (int i = 0; i < get_device_count(); i++) {
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, i));
        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printfQuda("Found device %d: %s\n", i, deviceProp.name);
        }
      }

/*
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
*/

      if (!deviceProp.unifiedAddressing) warningQuda("Device %d does not support unified addressing", dev);

      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Using device %d: %s\n", dev, deviceProp.name);
#ifndef USE_QDPJIT
      CHECK_CUDA_ERROR(cudaSetDevice(dev));
#endif

#ifdef NUMA_NVML
      char *enable_numa_env = getenv("QUDA_ENABLE_NUMA");
      if (enable_numa_env && strcmp(enable_numa_env, "0") == 0) {
        if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling numa_affinity\n");
      } else {
        setNumaAffinityNVML(dev);
      }
#endif

      CHECK_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
      //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
      // cudaGetDeviceProperties(&deviceProp, dev);
    }

    int get_device_count()
    {
      static int device_count = 0;
      if (device_count == 0) {
        CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
        if (device_count == 0) warningQuda("No CUDA devices found");
      }
      return device_count;
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
        printfQuda("%d - deviceOverlap            %s\n", device, (deviceProp.deviceOverlap ? "true" : "false"));
        printfQuda("%d - multiProcessorCount      %d\n", device, deviceProp.multiProcessorCount);
/*
        printfQuda("%d - kernelExecTimeoutEnabled %s\n", device,
                   (deviceProp.kernelExecTimeoutEnabled ? "true" : "false"));
        printfQuda("%d - integrated               %s\n", device, (deviceProp.integrated ? "true" : "false"));
        printfQuda("%d - canMapHostMemory         %s\n", device, (deviceProp.canMapHostMemory ? "true" : "false"));
        switch (deviceProp.computeMode) {
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
        printfQuda("%d - memoryClockRate          %d kilohertz\n", device, deviceProp.memoryClockRate);
        printfQuda("%d - memoryBusWidth           %d bits\n", device, deviceProp.memoryBusWidth);
        printfQuda("%d - l2CacheSize              %d bytes\n", device, deviceProp.l2CacheSize);
*/
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
      CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&streams[Nstream-1], cudaStreamDefault, leastPriority));
    }

    void destroy()
    {
      if (streams) {
        for (int i = 0; i < Nstream; i++) CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        delete []streams;
        streams = nullptr;
      }

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env, "1") == 0) {
        // end this CUDA context
        CHECK_CUDA_ERROR(cudaDeviceReset());
      }
    }

    cudaStream_t get_cuda_stream(const qudaStream_t &stream)
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

    bool managed_memory_supported() { return false; }

    bool shared_memory_atomic_supported() { return false; }

    size_t max_default_shared_memory() { return deviceProp.sharedMemPerBlock; }

    size_t max_dynamic_shared_memory() { return 98304; }

    unsigned int max_threads_per_block() { return deviceProp.maxThreadsPerBlock; }

    unsigned int max_threads_per_processor() { return deviceProp.maxThreadsPerMultiProcessor; }

    unsigned int max_threads_per_block_dim(int i) { return deviceProp.maxThreadsDim[i]; }

    unsigned int max_grid_size(int i) { return deviceProp.maxGridSize[i]; }

    unsigned int processor_count() { return deviceProp.multiProcessorCount; }

    unsigned int max_blocks_per_processor() { return 1; }

    namespace profile
    {

      void start() { /* cudaProfilerStart(); */ }

      void stop() { /* cudaProfilerStop(); */ }

    } // namespace profile

  } // namespace device
} // namespace quda
