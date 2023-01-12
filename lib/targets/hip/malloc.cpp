#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <unistd.h>   // for getpagesize()
#include <execinfo.h> // for backtrace
#include <quda_internal.h>
#include <device.h>

#include <hip/hip_runtime.h>
#ifdef USE_QDPJIT
#include "qdp_cache.h"
#endif

namespace quda
{

  enum AllocType { DEVICE, DEVICE_PINNED, HOST, PINNED, MAPPED, MANAGED, N_ALLOC_TYPE };

  class MemAlloc
  {

  public:
    std::string func;
    std::string file;
    int line;
    size_t size;
    size_t base_size;

    MemAlloc() : line(-1), size(0), base_size(0) { }

    MemAlloc(std::string func, std::string file, int line) : func(func), file(file), line(line), size(0), base_size(0)
    {
    }

    MemAlloc(const MemAlloc &) = default;
    MemAlloc(MemAlloc &&) = default;
    virtual ~MemAlloc() = default;
    MemAlloc &operator=(const MemAlloc &) = default;
    MemAlloc &operator=(MemAlloc &&) = default;
  };

  static std::map<void *, MemAlloc> alloc[N_ALLOC_TYPE];
  static size_t total_bytes[N_ALLOC_TYPE] = {0};
  static size_t max_total_bytes[N_ALLOC_TYPE] = {0};
  static size_t total_host_bytes, max_total_host_bytes;
  static size_t total_pinned_bytes, max_total_pinned_bytes;

  size_t device_allocated() { return total_bytes[DEVICE]; }

  size_t pinned_allocated() { return total_bytes[PINNED]; }

  size_t mapped_allocated() { return total_bytes[MAPPED]; }

  size_t managed_allocated() { return total_bytes[MANAGED]; }

  size_t host_allocated() { return total_bytes[HOST]; }

  size_t device_allocated_peak() { return max_total_bytes[DEVICE]; }

  size_t pinned_allocated_peak() { return max_total_bytes[PINNED]; }

  size_t mapped_allocated_peak() { return max_total_bytes[MAPPED]; }

  size_t managed_allocated_peak() { return max_total_bytes[MANAGED]; }

  size_t host_allocated_peak() { return max_total_bytes[HOST]; }

  static void print_trace(void)
  {
    void *array[10];
    size_t size;
    char **strings;
    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    printfQuda("Obtained %zd stack frames.\n", size);
    for (size_t i = 0; i < size; i++) printfQuda("%s\n", strings[i]);
    free(strings);
  }

  static void print_alloc_header()
  {
    printfQuda("Type    Pointer          Size             Location\n");
    printfQuda("----------------------------------------------------------\n");
  }

  static void print_alloc(AllocType type)
  {
    const char *type_str[] = {"Device", "Device Pinned", "Host  ", "Pinned", "Mapped", "Managed"};
    std::map<void *, MemAlloc>::iterator entry;

    for (auto entry : alloc[type]) {
      void *ptr = entry.first;
      MemAlloc a = entry.second;
      printfQuda("%s  %15p  %15lu  %s(), %s:%d\n", type_str[type], ptr, (unsigned long)a.base_size, a.func.c_str(),
                 a.file.c_str(), a.line);
    }
  }

  static void track_malloc(const AllocType &type, const MemAlloc &a, void *ptr)
  {
    total_bytes[type] += a.base_size;
    if (total_bytes[type] > max_total_bytes[type]) { max_total_bytes[type] = total_bytes[type]; }
    if (type != DEVICE && type != DEVICE_PINNED) {
      total_host_bytes += a.base_size;
      if (total_host_bytes > max_total_host_bytes) { max_total_host_bytes = total_host_bytes; }
    }
    if (type == PINNED || type == MAPPED) {
      total_pinned_bytes += a.base_size;
      if (total_pinned_bytes > max_total_pinned_bytes) { max_total_pinned_bytes = total_pinned_bytes; }
    }
    alloc[type][ptr] = a;
  }

  static void track_free(const AllocType &type, void *ptr)
  {
    size_t size = alloc[type][ptr].base_size;
    total_bytes[type] -= size;
    if (type != DEVICE && type != DEVICE_PINNED) { total_host_bytes -= size; }
    if (type == PINNED || type == MAPPED) { total_pinned_bytes -= size; }
    alloc[type].erase(ptr);
  }

  /**
   * Under CUDA 4.0, hipHostRegister seems to require that both the
   * beginning and end of the buffer be aligned on page boundaries.
   * This local function takes care of the alignment and gets called
   * by pinned_malloc_() and mapped_malloc_()
   */
  static void *aligned_malloc(MemAlloc &a, size_t size)
  {
    void *ptr = nullptr;

    a.size = size;

    static int page_size = 2 * getpagesize();
    a.base_size = ((size + page_size - 1) / page_size) * page_size; // round up to the nearest multiple of page_size
    int align = posix_memalign(&ptr, page_size, a.base_size);
    if (!ptr || align != 0) {
      errorQuda("Failed to allocate aligned host memory of size %zu (%s:%d in %s())\n", size, a.file.c_str(), a.line,
                a.func.c_str());
    }
    return ptr;
  }

  bool use_managed_memory()
  {
    static bool managed = false;
    static bool init = false;

    if (!init) {
      char *enable_managed_memory = getenv("QUDA_ENABLE_MANAGED_MEMORY");
      if (enable_managed_memory && strcmp(enable_managed_memory, "1") == 0) {
        warningQuda("Using managed memory for HIP allocations");
        managed = true;

        if (!device::managed_memory_supported()) warningQuda("Target device does not report supporting managed memory");
      }

      init = true;
    }

    return managed;
  }

  bool use_qdp_managed()
  {
#if defined(QDP_USE_CUDA_MANAGED_MEMORY) || defined(QDP_ENABLE_MANAGED_MEMORY)
    return true;
#else
    return false;
#endif
  }

  bool is_prefetch_enabled()
  {
    static bool prefetch = false;
    static bool init = false;

    if (!init) {
      if (use_managed_memory()) {
        char *enable_managed_prefetch = getenv("QUDA_ENABLE_MANAGED_PREFETCH");
        if (enable_managed_prefetch && strcmp(enable_managed_prefetch, "1") == 0) {
          // BJoo: Managed Memory Prefetch is not supported currently under HIP
          warningQuda("HIP Does not currently allow prefetch support for managed memory. Setting prefetch to false");
          prefetch = false;
        }
      }

      init = true;
    }

    return prefetch;
  }

  /**
   * Perform a standard hipMalloc() with error-checking.  This
   * function should only be called via the device_malloc() macro,
   * defined in malloc_quda.h
   */
  void *device_malloc_(const char *func, const char *file, int line, size_t size)
  {
    if (use_managed_memory()) return managed_malloc_(func, file, line, size);

    MemAlloc a(func, file, line);
    void *ptr;

    a.size = a.base_size = size;

#ifndef USE_QDPJIT
    // Regular version
    hipError_t err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
      errorQuda("Failed to allocate device memory of size %zu (%s:%d in %s())\n", size, file, line, func);
    }
#else
    // QDPJIT version
    QDP::QDP_get_global_cache().addDeviceStatic(&ptr, size, true);
#endif
    track_malloc(DEVICE, a, ptr);
#ifdef HOST_DEBUG
    hipMemset(ptr, 0xff, size);
#endif

    return ptr;
  }

  /**
   * Perform a hipMalloc with error-checking.  This function is to
   * guarantee a unique memory allocation on the device, since
   * hipMalloc can be redirected (as is the case with QDPJIT).  This
   * should only be called via the device_pinned_malloc() macro,
   * defined in malloc_quda.h.
   */
  void *device_pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {

    if (!comm_peer2peer_present()) return device_malloc_(func, file, line, size);

    MemAlloc a(func, file, line);
    void *ptr;

    a.size = a.base_size = size;
    hipError_t err = hipMalloc(&ptr, size);

    if (err != hipSuccess) {
      errorQuda("Failed to allocate device memory of size %zu (%s:%d in %s())\n", size, file, line, func);
    }
    track_malloc(DEVICE_PINNED, a, ptr);
#ifdef HOST_DEBUG
    hipMemset(ptr, 0xff, size);
#endif
    return ptr;
  }

  /**
   * Perform a standard malloc() with error-checking.  This function
   * should only be called via the safe_malloc() macro, defined in
   * malloc_quda.h
   */
  void *safe_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    a.size = a.base_size = size;

    void *ptr = malloc(size);
    if (!ptr) { errorQuda("Failed to allocate host memory of size %zu (%s:%d in %s())\n", size, file, line, func); }
    track_malloc(HOST, a, ptr);
#ifdef HOST_DEBUG
    // memset(ptr, 0xff, size);
#endif
    return ptr;
  }

  /**
   * Allocate page-locked ("pinned") host memory.  This function
   * should only be called via the pinned_malloc() macro, defined in
   * malloc_quda.h
   *
   * Note that we do not rely on hipHostMalloc(), since buffers
   * allocated in this way have been observed to cause problems when
   * shared with MPI via GPU Direct on some systems.
   */
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr = aligned_malloc(a, size);

    hipError_t err = hipHostRegister(ptr, a.base_size, hipHostRegisterDefault);
    if (err != hipSuccess) {
      errorQuda("Failed to register pinned memory of size %zu (%s:%d in %s())\n", size, file, line, func);
    }
    track_malloc(PINNED, a, ptr);
#ifdef HOST_DEBUG
    memset(ptr, 0xff, a.base_size);
#endif
    return ptr;
  }

  /**
   * Allocate page-locked ("pinned") host memory, and map it into the
   * GPU address space.  This function should only be called via the
   * mapped_malloc() macro, defined in malloc_quda.h
   */
  void *mapped_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);

    void *ptr = aligned_malloc(a, size);
    hipError_t err = hipHostRegister(ptr, a.base_size, hipHostRegisterMapped | hipHostRegisterPortable);
    if (err != hipSuccess) {
      errorQuda("Failed to register host-mapped memory of size %zu (%s:%d in %s())\n", size, file, line, func);
    }

    track_malloc(MAPPED, a, ptr);
#ifdef HOST_DEBUG
    memset(ptr, 0xff, a.base_size);
#endif
    return ptr;
  }

  /**
   * Perform a standard hipMallocManaged() with error-checking.  This
   * function should only be called via the managed_malloc() macro,
   * defined in malloc_quda.h
   */
  void *managed_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr;

    a.size = a.base_size = size;

    hipError_t err = hipMallocManaged(&ptr, size);
    if (err != hipSuccess) {
      errorQuda("Failed to allocate managed memory of size %zu (%s:%d in %s())\n", size, file, line, func);
    }
    track_malloc(MANAGED, a, ptr);
#ifdef HOST_DEBUG
    hipMemset(ptr, 0xff, size);
#endif
    return ptr;
  }

  /**
   * Round to the nearest 2MiB
   *
   */
  size_t align2MiB(const size_t size) noexcept
  {
    constexpr size_t TwoMiB = (1 << 21);
    constexpr size_t LowBits = TwoMiB - 1;
    constexpr size_t HighBits = ~LowBits;

    // If there are low bits, round to nearest 2MiB
    size_t align_remainder = (size & LowBits) ? TwoMiB : 0;

    // Add high bits
    return (size & HighBits) + align_remainder;
  }

  /**
   * Allocate pinned or symmetric (shmem) device memory for comms. Should only be called via the
   * device_comms_pinned_malloc macro, defined in malloc_quda.h
   */
  void *device_comms_pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {

    //#ifdef NVSHMEM_COMMS
    //   return shmem_malloc_(func, file, line, size);
    //#else
    return device_pinned_malloc_(func, file, line, align2MiB(size));
    //#endif
  }
  /**
   * Free device memory allocated with device_malloc().  This function
   * should only be called via the device_free() macro, defined in
   * malloc_quda.h
   */
  void device_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (use_managed_memory()) {
      managed_free_(func, file, line, ptr);
      return;
    }

    if (!ptr) { errorQuda("Attempt to free NULL device pointer (%s:%d in %s())\n", file, line, func); }
    if (!alloc[DEVICE].count(ptr)) {
      errorQuda("Attempt to free invalid device pointer (%s:%d in %s())\n", file, line, func);
    }

#ifndef USE_QDPJIT
    // Regular
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) { errorQuda("Failed to free device memory (%s:%d in %s())\n", file, line, func); }
#else
    // QDPJIT version
    // It will barf if it goes wrong
    QDP::QDP_get_global_cache().signoffViaPtr(ptr);
#endif

    track_free(DEVICE, ptr);
  }

  /**
   * Free device memory allocated with device_pinned malloc().  This
   * function should only be called via the device_pinned_free()
   * macro, defined in malloc_quda.h
   */
  void device_pinned_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!comm_peer2peer_present()) {
      device_free_(func, file, line, ptr);
      return;
    }

    if (!ptr) { errorQuda("Attempt to free NULL device pointer (%s:%d in %s())\n", file, line, func); }
    if (!alloc[DEVICE_PINNED].count(ptr)) {
      errorQuda("Attempt to free invalid device pointer (%s:%d in %s())\n", file, line, func);
    }

    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) { printfQuda("Failed to free device memory (%s:%d in %s())\n", file, line, func); }
    track_free(DEVICE_PINNED, ptr);
  }

  /**
   * Free device memory allocated with device_malloc().  This function
   * should only be called via the device_free() macro, defined in
   * malloc_quda.h
   */
  void managed_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!ptr) { errorQuda("Attempt to free NULL managed pointer (%s:%d in %s())\n", file, line, func); }
    if (!alloc[MANAGED].count(ptr)) {
      errorQuda("Attempt to free invalid managed pointer (%s:%d in %s())\n", file, line, func);
    }
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) { errorQuda("Failed to free device memory (%s:%d in %s())\n", file, line, func); }
    track_free(MANAGED, ptr);
  }

  /**
   * Free host memory allocated with safe_malloc(), pinned_malloc(),
   * or mapped_malloc().  This function should only be called via the
   * host_free() macro, defined in malloc_quda.h
   */
  void host_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!ptr) { errorQuda("Attempt to free NULL host pointer (%s:%d in %s())\n", file, line, func); }
    if (alloc[HOST].count(ptr)) {
      track_free(HOST, ptr);
      free(ptr);
    } else if (alloc[PINNED].count(ptr)) {
      hipError_t err = hipHostUnregister(ptr);
      if (err != hipSuccess) { errorQuda("Failed to unregister pinned memory (%s:%d in %s())\n", file, line, func); }
      track_free(PINNED, ptr);
      free(ptr);
    } else if (alloc[MAPPED].count(ptr)) {
#ifdef HOST_ALLOC
      hipError_t err = hipFreeHost(ptr);
      if (err != hipSuccess) { errorQuda("Failed to free host memory (%s:%d in %s())\n", file, line, func); }
      track_free(MAPPED, ptr);
#else
      hipError_t err = hipHostUnregister(ptr);
      if (err != hipSuccess) {
        errorQuda("Failed to unregister host-mapped memory (%s:%d in %s())\n", file, line, func);
      }
      track_free(MAPPED, ptr);
      free(ptr);
#endif
    } else {
      printfQuda("ERROR: Attempt to free invalid host pointer (%s:%d in %s())\n", file, line, func);
      print_trace();
      errorQuda("Aborting");
    }
  }

  /**
   * Free device comms memory allocated with device_comms_pinned_malloc(). This function should only be
   * called via the device_comms_pinned_free() macro, defined in malloc_quda.h
   */
  void device_comms_pinned_free_(const char *func, const char *file, int line, void *ptr)
  {
    // #ifdef NVSHMEM_COMMS
    //    shmem_free_(func, file, line, ptr);
    // #else
    device_pinned_free_(func, file, line, ptr);
    // #endif
  }

  void printPeakMemUsage()
  {
    printfQuda("Device memory used = %.1f MiB\n", max_total_bytes[DEVICE] / (double)(1 << 20));
    printfQuda("Pinned device memory used = %.1f MiB\n", max_total_bytes[DEVICE_PINNED] / (double)(1 << 20));
    printfQuda("Managed memory used = %.1f MiB\n", max_total_bytes[MANAGED] / (double)(1 << 20));
    //    printfQuda("Shmem memory used = %.1f MiB\n", max_total_bytes[SHMEM] / (double)(1 << 20));
    printfQuda("Page-locked host memory used = %.1f MiB\n", max_total_pinned_bytes / (double)(1 << 20));
    printfQuda("Total host memory used >= %.1f MiB\n", max_total_host_bytes / (double)(1 << 20));
  }

  void assertAllMemFree()
  {
    if (!alloc[DEVICE].empty() || !alloc[DEVICE_PINNED].empty() || !alloc[HOST].empty() || !alloc[PINNED].empty()
        || !alloc[MAPPED].empty()) {
      warningQuda("The following internal memory allocations were not freed.");
      printfQuda("\n");
      print_alloc_header();
      print_alloc(DEVICE);
      print_alloc(DEVICE_PINNED);
      print_alloc(HOST);
      print_alloc(PINNED);
      print_alloc(MAPPED);
      printfQuda("\n");
    }
  }

  QudaFieldLocation get_pointer_location(const void *ptr)
  {
    hipPointerAttribute_t attr;
    hipError_t error = hipPointerGetAttributes(&attr, ptr);

    // hipReturnInvalidValue is not an error here it means that
    // hipPointerGetAttrributes was passed a pointer not knwon to hip
    // This is therefore assumed to be a host pointer, and attr is
    // appropriately filled out
    if (error != hipSuccess && error != hipErrorInvalidValue) {
      errorQuda("hipPointerGetAttributes returned error: %s\n", hipGetErrorString(error));
    }

    switch (attr.memoryType) {
    case hipMemoryTypeHost: return QUDA_CPU_FIELD_LOCATION;
    case hipMemoryTypeDevice: return QUDA_CUDA_FIELD_LOCATION;
    case hipMemoryTypeArray: return QUDA_CUDA_FIELD_LOCATION;
    case hipMemoryTypeUnified: return QUDA_CUDA_FIELD_LOCATION; ///< Not used currently
    default: errorQuda("Unknown memory type %d\n", attr.memoryType); return QUDA_INVALID_FIELD_LOCATION;
    }
  }

  void *get_mapped_device_pointer_(const char *func, const char *file, int line, const void *host)
  {
    void *device;
    auto error = hipHostGetDevicePointer(&device, const_cast<void *>(host), 0);
    if (error != hipSuccess) {
      errorQuda("hipHostGetDevicePointer failed with error %s (%s:%d in %s()", hipGetErrorString(error), file, line,
                func);
    }
    return device;
  }

  void register_pinned_(const char *func, const char *file, int line, void *ptr, size_t bytes)
  {
    auto error = hipHostRegister(ptr, bytes, hipHostRegisterDefault);
    if (error != hipSuccess) {
      errorQuda("hipHostRegister failed with error %s (%s:%d in %s()", hipGetErrorString(error), file, line, func);
    }
  }

  void unregister_pinned_(const char *func, const char *file, int line, void *ptr)
  {
    auto error = hipHostUnregister(ptr);
    if (error != hipSuccess) {
      errorQuda("hipHostUnregister failed with error %s (%s:%d in %s()", hipGetErrorString(error), file, line, func);
    }
  }

  namespace pool
  {

    /** Cache of inactive pinned-memory allocations.  We cache pinned
        memory allocations so that fields can reuse these with minimal
        overhead.*/
    static std::multimap<size_t, void *> pinnedCache;

    /** Sizes of active pinned-memory allocations.  For convenience,
        we keep track of the sizes of active allocations (i.e., those not
        in the cache). */
    static std::map<void *, size_t> pinnedSize;

    /** Cache of inactive device-memory allocations.  We cache pinned
        memory allocations so that fields can reuse these with minimal
        overhead.*/
    static std::multimap<size_t, void *> deviceCache;

    /** Sizes of active device-memory allocations.  For convenience,
        we keep track of the sizes of active allocations (i.e., those not
        in the cache). */
    static std::map<void *, size_t> deviceSize;

    static bool pool_init = false;

    /** whether to use a memory pool allocator for device memory */
    static bool device_memory_pool = true;

    /** whether to use a memory pool allocator for pinned memory */
    static bool pinned_memory_pool = true;

    void init()
    {
      if (!pool_init) {
        // device memory pool
        char *enable_device_pool = getenv("QUDA_ENABLE_DEVICE_MEMORY_POOL");
        if (!enable_device_pool || strcmp(enable_device_pool, "0") != 0) {
          warningQuda("Using device memory pool allocator");
          device_memory_pool = true;
        } else {
          warningQuda("Not using device memory pool allocator");
          device_memory_pool = false;
        }

        // pinned memory pool
        char *enable_pinned_pool = getenv("QUDA_ENABLE_PINNED_MEMORY_POOL");
        if (!enable_pinned_pool || strcmp(enable_pinned_pool, "0") != 0) {
          warningQuda("Using pinned memory pool allocator");
          pinned_memory_pool = true;
        } else {
          warningQuda("Not using pinned memory pool allocator");
          pinned_memory_pool = false;
        }
        pool_init = true;
      }
    }

    void *pinned_malloc_(const char *func, const char *file, int line, size_t nbytes)
    {
      void *ptr = nullptr;
      if (pinned_memory_pool) {
        std::multimap<size_t, void *>::iterator it;

        if (pinnedCache.empty()) {
          ptr = quda::pinned_malloc_(func, file, line, nbytes);
        } else {
          it = pinnedCache.lower_bound(nbytes);
          if (it != pinnedCache.end()) { // sufficiently large allocation found
            nbytes = it->first;
            ptr = it->second;
            pinnedCache.erase(it);
          } else { // sacrifice the smallest cached allocation
            it = pinnedCache.begin();
            ptr = it->second;
            pinnedCache.erase(it);
            host_free(ptr);
            ptr = quda::pinned_malloc_(func, file, line, nbytes);
          }
        }
        pinnedSize[ptr] = nbytes;
      } else {
        ptr = quda::pinned_malloc_(func, file, line, nbytes);
      }
      return ptr;
    }

    void pinned_free_(const char *func, const char *file, int line, void *ptr)
    {
      if (pinned_memory_pool) {
        if (!pinnedSize.count(ptr)) { errorQuda("Attempt to free invalid pointer"); }
        pinnedCache.insert(std::make_pair(pinnedSize[ptr], ptr));
        pinnedSize.erase(ptr);
      } else {
        quda::host_free_(func, file, line, ptr);
      }
    }

    void *device_malloc_(const char *func, const char *file, int line, size_t nbytes)
    {
      void *ptr = nullptr;
      if (device_memory_pool) {
        std::multimap<size_t, void *>::iterator it;

        if (deviceCache.empty()) {
          ptr = quda::device_malloc_(func, file, line, nbytes);
        } else {
          it = deviceCache.lower_bound(nbytes);
          if (it != deviceCache.end()) { // sufficiently large allocation found
            nbytes = it->first;
            ptr = it->second;
            deviceCache.erase(it);
          } else { // sacrifice the smallest cached allocation
            it = deviceCache.begin();
            ptr = it->second;
            deviceCache.erase(it);
            quda::device_free_(func, file, line, ptr);
            ptr = quda::device_malloc_(func, file, line, nbytes);
          }
        }
        deviceSize[ptr] = nbytes;
      } else {
        ptr = quda::device_malloc_(func, file, line, nbytes);
      }
      return ptr;
    }

    void device_free_(const char *func, const char *file, int line, void *ptr)
    {
      if (device_memory_pool) {
        if (!deviceSize.count(ptr)) { errorQuda("Attempt to free invalid pointer"); }
        deviceCache.insert(std::make_pair(deviceSize[ptr], ptr));
        deviceSize.erase(ptr);
      } else {
        quda::device_free_(func, file, line, ptr);
      }
    }

    void flush_pinned()
    {
      if (pinned_memory_pool) {
        std::multimap<size_t, void *>::iterator it;
        for (it = pinnedCache.begin(); it != pinnedCache.end(); it++) {
          void *ptr = it->second;
          host_free(ptr);
        }
        pinnedCache.clear();
      }
    }

    void flush_device()
    {
      if (device_memory_pool) {
        std::multimap<size_t, void *>::iterator it;
        for (it = deviceCache.begin(); it != deviceCache.end(); it++) {
          void *ptr = it->second;
          device_free(ptr);
        }
        deviceCache.clear();
      }
    }

  } // namespace pool

} // namespace quda
