#include <hip/hip_runtime.h>
#include <util_quda.h>
#include <quda_internal.h>
#include <quda_hip_api.h>
#include <target_device.h>

static hipDeviceProp_t deviceProp;
static hipStream_t *streams;
static const int Nstream = 9;

#define CHECK_HIP_ERROR(func) target::hip::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    static int device_id = -1;

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;
      printfQuda("*** HIP BACKEND ***\n");

      int driver_version = 0;
      CHECK_HIP_ERROR(hipDriverGetVersion(&driver_version));
      printfQuda("HIP Driver version = %d\n", driver_version);

      int runtime_version = 0;
      CHECK_HIP_ERROR(hipRuntimeGetVersion(&runtime_version));
      printfQuda("HIP Runtime version = %d\n", runtime_version);

      for (int i = 0; i < get_device_count(); i++) {
        CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, i));
        if (getVerbosity() >= QUDA_SUMMARIZE) { printfQuda("Found device %d: %s\n", i, deviceProp.name); }
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) { printfQuda("Using device %d: %s\n", dev, deviceProp.name); }

      CHECK_HIP_ERROR(hipSetDevice(dev));

      // FIXME: Commenting this out now until it is fixed in a newer ROCm as it seems
      // Broken in recent ROCms. I am not sure it does anything anyway on RedTeam
      // CHECK_HIP_ERROR(hipDeviceSetCacheConfig(hipFuncCachePreferL1));
      CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, dev));

      device_id = dev;
    }

    void init_thread()
    {
      if (device_id == -1) errorQuda("No HIP device has been initialized for this process");
      CHECK_HIP_ERROR(hipSetDevice(device_id));
    }

    int get_device_count()
    {
      static int device_count = 0;
      if (device_count == 0) {
        CHECK_HIP_ERROR(hipGetDeviceCount(&device_count));
        if (device_count == 0) errorQuda("No HIP devices found");
      }
      return device_count;
    }

    void get_visible_devices_string(char device_list_string[128])
    {
      char *device_order_env = getenv("ROCR_VISIBLE_DEVICES");

      if (!device_order_env) {
        device_order_env = getenv("HIP_VISIBLE_DEVICES");
      }

      if (device_order_env) {
        std::stringstream device_list_raw(device_order_env); // raw input
        std::stringstream device_list;                       // formatted (no commas)

        int device;
        while (device_list_raw >> device) {
          // check this is a valid policy choice
          if (device < 0) { errorQuda("Invalid ROCR/HIP_VISIBLE_DEVICES ordinal %d", device); }

          device_list << device;
          if (device_list_raw.peek() == ',') device_list_raw.ignore();
        }
        snprintf(device_list_string, 128, "%s", device_list.str().c_str());
      }
    }

    void create_context()
    {
      streams = new hipStream_t[Nstream];

      int greatestPriority;
      int leastPriority;
      CHECK_HIP_ERROR(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      for (int i = 0; i < Nstream - 1; i++) {
        CHECK_HIP_ERROR(hipStreamCreateWithPriority(&streams[i], hipStreamDefault, greatestPriority));
      }
      CHECK_HIP_ERROR(hipStreamCreateWithPriority(&streams[Nstream - 1], hipStreamDefault, leastPriority));
    }

    void destroy()
    {
      if (streams) {
        for (int i = 0; i < Nstream; i++) CHECK_HIP_ERROR(hipStreamDestroy(streams[i]));
        delete[] streams;
        streams = nullptr;
      }

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env, "1") == 0) {
        // end this HIP context
        CHECK_HIP_ERROR(hipDeviceReset());
      }
    }

    qudaStream_t get_stream(unsigned int i)
    {
      if (i > Nstream) errorQuda("Invalid stream index %u", i);
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
      // managed memory is not supported
      return false;
    }

    bool shared_memory_atomic_supported()
    {
      // shared memory atomics are supported  in the HIP API
      return true;
    }

#if 0
    size_t max_default_shared_memory() { return deviceProp.sharedMemPerBlock-1; } 

    size_t max_dynamic_shared_memory()
    {
      static int max_shared_bytes = 0;
      if (!max_shared_bytes) hipDeviceGetAttribute(&max_shared_bytes,  hipDeviceAttributeMaxSharedMemoryPerBlock, comm_gpuid());
      return max_shared_bytes-1;
    }
#else
    size_t max_default_shared_memory() { return 32000; }

    size_t max_dynamic_shared_memory() { return 32000; }
#endif

    unsigned int max_threads_per_block() { return deviceProp.maxThreadsPerBlock; }

    unsigned int max_threads_per_processor() { return deviceProp.maxThreadsPerMultiProcessor; }

    unsigned int max_threads_per_block_dim(int i) { return deviceProp.maxThreadsDim[i]; }

    unsigned int max_grid_size(int i) { return deviceProp.maxGridSize[i]; }

    unsigned int processor_count() { return deviceProp.multiProcessorCount; }

    unsigned int max_blocks_per_processor() { return 32; }

    namespace profile
    {

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

  namespace target
  {

    namespace hip
    {

      hipStream_t get_stream(const qudaStream_t &stream) { return streams[stream.idx]; }

    } // namespace hip

  } // namespace target

} // namespace quda
