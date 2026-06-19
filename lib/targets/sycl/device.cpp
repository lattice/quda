#include <util_quda.h>
#include <quda_internal.h>
#include <target_device.h>
#include <algorithm>

static sycl::device myDevice;
static sycl::queue *streams;
static const int Nstream = 9;
static size_t eventCount[Nstream]; // counts event recording and stream syncs
static size_t syncStamp[Nstream];  // eventCount of last sync
static void *argBufD[Nstream];
static size_t argBufSizeD[Nstream];
static bool print = false;
static bool syncKernels = false;

int mySelectorT(const sycl::device &device)
{
  int score = 1;
  if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu) score += 10;
  if (!device.has(sycl::aspect::fp64)) score = -1; // require fp64
  if (print) { printfQuda("Selector score: %2i %s\n", score, device.get_info<sycl::info::device::name>().c_str()); }
  return score;
}
static auto mySelector = mySelectorT;

void exception_handler(sycl::exception_list exceptions)
{
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      errorQuda("Caught asynchronous SYCL exception:\n %s\n", e.what());
    }
  }
}

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;
      print = true;

      auto ps = sycl::platform::get_platforms();
      printfQuda("SYCL platforms available:\n");
      for (auto p : ps) {
        printfQuda("  %s %s %s\n", p.get_info<sycl::info::platform::name>().c_str(),
                   p.get_info<sycl::info::platform::vendor>().c_str(),
                   p.get_info<sycl::info::platform::version>().c_str());
      }

      auto p = sycl::platform(mySelector);
      printfQuda("Selected platform: %s\n", p.get_info<sycl::info::platform::name>().c_str());
      printfQuda("  Vendor: %s\n", p.get_info<sycl::info::platform::vendor>().c_str());
      printfQuda("  Version: %s\n", p.get_info<sycl::info::platform::version>().c_str());

      auto ds = p.get_devices();
      int ndev = ds.size();
      printfQuda("  Number of devices: %d\n", ndev);
      if (dev >= ndev) { errorQuda("Requested device(%d) out of range(%d)", dev, ndev); }

      printfQuda("Selected device number: %i\n", dev);
      myDevice = ds[dev];
      printfQuda("  Name: %s\n", myDevice.get_info<sycl::info::device::name>().c_str());
      printfQuda("  Version: %s\n", myDevice.get_info<sycl::info::device::version>().c_str());
      printfQuda("  Driver version: %s\n", myDevice.get_info<sycl::info::device::driver_version>().c_str());
      printfQuda("  Max compute units: %u\n", myDevice.get_info<sycl::info::device::max_compute_units>());
      printfQuda("  Max work item dimensions: %u\n", myDevice.get_info<sycl::info::device::max_work_item_dimensions>());
      printfQuda("  Max work item sizes: %s\n",
                 str(myDevice.get_info<sycl::info::device::max_work_item_sizes<3>>()).c_str());
      printfQuda("  Max work group size: %lu\n", myDevice.get_info<sycl::info::device::max_work_group_size>());
      printfQuda("  Max num sub groups: %u\n", myDevice.get_info<sycl::info::device::max_num_sub_groups>());
      printfQuda("  Sub group independent forward progress: %s\n",
                 myDevice.get_info<sycl::info::device::sub_group_independent_forward_progress>() ? "true" : "false");
      printfQuda("  Sub group sizes: %s\n", str(myDevice.get_info<sycl::info::device::sub_group_sizes>()).c_str());
      printfQuda("  Preferred vector width float: %u\n",
                 myDevice.get_info<sycl::info::device::preferred_vector_width_float>());
      printfQuda("  Preferred vector width double: %u\n",
                 myDevice.get_info<sycl::info::device::preferred_vector_width_double>());
      printfQuda("  Native vector width float: %u\n", myDevice.get_info<sycl::info::device::native_vector_width_float>());
      printfQuda("  Native vector width double: %u\n",
                 myDevice.get_info<sycl::info::device::native_vector_width_double>());
      printfQuda("  Max clock frequency: %u MHz\n", myDevice.get_info<sycl::info::device::max_clock_frequency>());
      printfQuda("  Address bits: %u\n", myDevice.get_info<sycl::info::device::address_bits>());
      printfQuda("  Max mem alloc size: %lu\n", myDevice.get_info<sycl::info::device::max_mem_alloc_size>());
      printfQuda("  Max parameter size: %lu\n", myDevice.get_info<sycl::info::device::max_parameter_size>());
      printfQuda("  Mem base addr align: %u\n", myDevice.get_info<sycl::info::device::mem_base_addr_align>());
      printfQuda("  Global mem cache line size: %u\n",
                 myDevice.get_info<sycl::info::device::global_mem_cache_line_size>());
      printfQuda("  Global mem cache size: %lu\n", myDevice.get_info<sycl::info::device::global_mem_cache_size>());
      printfQuda("  Global mem size: %lu\n", myDevice.get_info<sycl::info::device::global_mem_size>());
      printfQuda("  Local mem size: %lu\n", myDevice.get_info<sycl::info::device::local_mem_size>());
      printfQuda("  Error correction support: %s\n",
                 myDevice.get_info<sycl::info::device::error_correction_support>() ? "true" : "false");
      auto moc = myDevice.get_info<sycl::info::device::atomic_memory_order_capabilities>();
      printfQuda("  Atomic memory orders:");
      for (auto mo : moc) {
        switch (mo) {
        case sycl::memory_order::relaxed: printfQuda(" relaxed"); break;
        case sycl::memory_order::acquire: printfQuda(" acquire"); break;
        case sycl::memory_order::release: printfQuda(" release"); break;
        case sycl::memory_order::acq_rel: printfQuda(" acq_rel"); break;
        case sycl::memory_order::seq_cst: printfQuda(" seq_cst"); break;
        default: printfQuda(" unknown"); break;
        }
      }
      printfQuda("\n");
      auto msc = myDevice.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
      printfQuda("  Atomic memory scopes:");
      for (auto ms : msc) {
        switch (ms) {
        case sycl::memory_scope::work_item: printfQuda(" work_item"); break;
        case sycl::memory_scope::sub_group: printfQuda(" sub_group"); break;
        case sycl::memory_scope::work_group: printfQuda(" work_group"); break;
        case sycl::memory_scope::device: printfQuda(" device"); break;
        case sycl::memory_scope::system: printfQuda(" system"); break;
        default: printfQuda(" unknown"); break;
        }
      }
      printfQuda("\n");

      bool err = false;
      auto warps = myDevice.get_info<sycl::info::device::sub_group_sizes>();
      if (std::find(warps.begin(), warps.end(), QUDA_WARP_SIZE) == warps.end()) {
        err = true;
        warningQuda("Warp size %d not in sub group sizes %s", QUDA_WARP_SIZE, str(warps).c_str());
      }
      if (err) { errorQuda("Device checks failed"); }

      char *sync_kernels_env = getenv("QUDA_SYNC_KERNELS");
      if (sync_kernels_env) {
        int sync_kernels_int = atoi(sync_kernels_env);
        switch (sync_kernels_int) {
        case 0:
          syncKernels = false;
          if (getVerbosity() > QUDA_SILENT) warningQuda("Using asynchronous kernels");
          break;
        case 1:
          syncKernels = true;
          if (getVerbosity() > QUDA_SILENT) warningQuda("Using synchronous kernels");
          break;
        default: errorQuda("Unexpected value QUDA_SYNC_KERNELS=%s\n", sync_kernels_env);
        }
      }
    }

    void init_thread() { }

    state_t get_state() { return {}; }

    int get_device_count()
    {
      auto p = sycl::platform(mySelector);
      auto ds = p.get_devices();
      auto device_count = ds.size();
      return device_count;
    }

    void get_visible_devices_string(char device_list_string[128])
    {
      char default_list[] = "";
      char *device_order_env = getenv("SYCL_DEVICE_FILTER");
      if (device_order_env == nullptr) { device_order_env = getenv("ONEAPI_DEVICE_SELECTOR"); }
      if (device_order_env == nullptr) { device_order_env = default_list; }
      snprintf(device_list_string, 128, "%s", device_order_env);
    }

    void print_device_properties()
    {
      auto p = sycl::platform(mySelector);
      auto ds = p.get_devices();
      int dev_count = ds.size();
      for (int device = 0; device < dev_count; device++) {
        namespace id = sycl::info::device;
        auto d = ds[device];
        printfQuda("%d - name:                    %s\n", device, d.get_info<id::name>().c_str());
      }
    }

    void create_context()
    {
      printfQuda("Creating context...");
      auto ctx = sycl::context(myDevice);
      streams = new sycl::queue[Nstream];
      sycl::property_list props {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
      for (int i = 0; i < Nstream - 1; i++) {
        // streams[i] = sycl::queue(ctx, myDevice, props);
        streams[i] = sycl::queue(ctx, myDevice, exception_handler, props);
      }
      streams[Nstream - 1] = sycl::queue(ctx, myDevice, exception_handler, props);
      printfQuda(" done\n");
      for (int i = 0; i < Nstream; i++) {
        eventCount[i] = 0;
        syncStamp[i] = 0;
        argBufD[i] = nullptr;
        argBufSizeD[i] = 0;
      }
    }

    void destroy()
    {
      if (streams) {
        delete[] streams;
        streams = nullptr;
      }
    }

    sycl::queue get_target_stream(const qudaStream_t &stream)
    {
      return streams[stream.idx];
    }

    qudaStream_t get_stream(unsigned int i)
    {
      if (i > Nstream) errorQuda("Invalid stream index %u", i);
      qudaStream_t stream;
      stream.idx = i;
      return stream;
    }

    qudaStream_t get_default_stream()
    {
      qudaStream_t stream;
      stream.idx = Nstream - 1;
      return stream;
    }

    unsigned int get_default_stream_idx() { return Nstream - 1; }

    sycl::queue defaultQueue(void)
    {
      return streams[Nstream - 1];
    }

    size_t getEventIdx(const qudaStream_t &stream)
    {
      eventCount[stream.idx]++;
      return eventCount[stream.idx];
    }

    void wasSynced(const qudaStream_t &stream)
    {
      eventCount[stream.idx]++;
      syncStamp[stream.idx] = eventCount[stream.idx];
    }

    void wasSynced(const qudaStream_t &stream, size_t eventIdx)
    {
      syncStamp[stream.idx] = std::max(syncStamp[stream.idx], eventIdx);
    }

    bool managed_memory_supported()
    {
      // auto val = myDevice.has(sycl::aspect::usm_restricted_shared_allocations);
      auto val = true;
      return val;
    }

    bool shared_memory_atomic_supported()
    {
      // auto val = myDevice.has(sycl::aspect::int64_base_atomics);
      // return val;
      // auto caps = myDevice.get_info<sycl::info::device::atomic_memory_scope_capabilities>;
      //  work_item, sub_group, work_group, device and system
      // return false;  // used in coarse_op, but not portable yet
      return false;
      // return true;
    }

    size_t max_default_shared_memory()
    {
      static size_t max_shared_bytes = 0;
      if (max_shared_bytes == 0) { max_shared_bytes = myDevice.get_info<sycl::info::device::local_mem_size>(); }
      return max_shared_bytes;
    }

    size_t max_dynamic_shared_memory()
    {
      static size_t max_shared_bytes = 0;
      if (max_shared_bytes == 0) { max_shared_bytes = myDevice.get_info<sycl::info::device::local_mem_size>(); }
      return max_shared_bytes;
    }

    unsigned int max_threads_per_block()
    {
      static unsigned int max_threads = 0;
      if (max_threads == 0) {
        max_threads = myDevice.get_info<sycl::info::device::max_work_group_size>();
        max_threads = std::min(max_threads, device::max_block_size());
      }
      return max_threads;
    }

    unsigned int max_threads_per_processor()
    { // not in portable SYCL
      static unsigned int max_threads = 0;
      if (max_threads == 0) {
        max_threads = max_threads_per_block();
        max_threads *= 2;
      }
      return max_threads;
    }

    unsigned int max_threads_per_block_dim(int i)
    {
      auto val = myDevice.get_info<sycl::info::device::max_work_item_sizes<3>>();
      return val[2 - i]; // reverse order, should be consistent with RANGE_{X,Y,Z}
    }

    // unsigned int max_grid_size(int i) { // not in portable SYCL?
    unsigned int max_grid_size(int)
    { // not in portable SYCL?
      // auto val = myDevice.get_info<sycl::info::device::max_work_item_sizes>();
      // return val[i];
      //  FIXME: address_bits / mwgs(i) ?
      return 65536;
    }

    unsigned int processor_count()
    {
      auto val = myDevice.get_info<sycl::info::device::max_compute_units>();
      return val;
    }

    unsigned int max_blocks_per_processor()
    { // FIXME
      static unsigned int max_blocks_per_sm = 2;
      return max_blocks_per_sm;
    }

    bool shared_carve_out_supported() { return false; }

    unsigned int max_parameter_size()
    {
      static unsigned int max_parameter_size = 0;
      if (max_parameter_size == 0) { max_parameter_size = myDevice.get_info<sycl::info::device::max_parameter_size>(); }
      return max_parameter_size;
    }

    namespace profile
    {

      void start()
      {
      }

      void stop()
      {
      }

    } // namespace profile

    // buffer for kernel argument
    typedef struct {
      void *buf;
      size_t size;
      size_t sync;
      qudaStream_t stream;
    } ArgBufT;
    std::vector<ArgBufT> argBuf {};

    void *try_get_arg_buf(qudaStream_t stream, size_t size)
    {
      for (auto &b : argBuf) {
        // printfQuda("  Arg buf stream %i size %i\n", b.stream.idx, b.size);
        if (syncStamp[b.stream.idx] > b.sync) {
          b.stream = stream;
          b.sync = eventCount[stream.idx];
          if (size > b.size) {
            // if(b.buf!=nullptr) device_free(b.buf);
            // b.buf = device_malloc(size);
            if (b.buf != nullptr) host_free(b.buf);
            b.buf = host_pinned_malloc(size);
            // if(b.buf!=nullptr) managed_free(b.buf);
            // b.buf = managed_malloc(size);
            b.size = size;
          }
          return b.buf;
        }
      }
      return nullptr;
    }

    void *get_arg_buf(qudaStream_t stream, size_t size)
    {
      // printfQuda("Adding buf stream %i size %i\n", stream.idx, size);
      auto buf = try_get_arg_buf(stream, size);
      if (buf == nullptr && argBuf.size() >= 10) { // arbitrary max
        qudaStreamSynchronize(stream);
        buf = try_get_arg_buf(stream, size);
      }
      if (buf == nullptr) {
        ArgBufT a;
        a.stream = stream;
        a.sync = eventCount[stream.idx];
        a.size = size;
        // buf = device_malloc(size);
        buf = host_pinned_malloc(size);
        // buf = managed_malloc(size);
        a.buf = buf;
        argBuf.push_back(a);
        // printfQuda("Added buf stream %i size %i\n", a.stream.idx, a.size);
      }
      return buf;
    }

    void *get_arg_buf_d(qudaStream_t stream, size_t size)
    {
      auto buf = argBufD[stream.idx];
      if (size > argBufSizeD[stream.idx]) {
        if (buf != nullptr) device_free(buf);
        buf = device_malloc(size);
        argBufD[stream.idx] = buf;
        argBufSizeD[stream.idx] = size;
      }
      return buf;
    }

    void free_arg_buf()
    {
      printfQuda("Arg buf count %lu\n", argBuf.size());
      for (const auto &b : argBuf) {
        printfQuda("  stream %i size %lu\n", b.stream.idx, b.size);
        // if(b.buf!=nullptr) device_free(b.buf);
        if (b.buf != nullptr) host_free(b.buf);
        // if(b.buf!=nullptr) managed_free(b.buf);
      }
      for (int i = 0; i < Nstream; i++) {
        if (argBufD[i] != nullptr) device_free(argBufD[i]);
      }
    }

    bool sync_kernels() { return syncKernels; }

  } // namespace device
} // namespace quda
