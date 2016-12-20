#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <unistd.h> // for getpagesize()
#include <execinfo.h> // for backtrace
#include <quda_internal.h>

#ifdef USE_QDPJIT
#include "qdp_quda.h"
#endif

namespace quda {

  enum AllocType {
    DEVICE,
    HOST,
    PINNED,
    MAPPED,
    N_ALLOC_TYPE
  };

  class MemAlloc {

  public:
    std::string func;
    std::string file;
    int line;
    size_t size;
    size_t base_size;

    MemAlloc()
      : line(-1), size(0), base_size(0) { }

    MemAlloc(std::string func, std::string file, int line)
      : func(func), file(file), line(line), size(0), base_size(0) { }

    MemAlloc& operator=(const MemAlloc &a) {
      if (&a != this) {
	func = a.func;
	file = a.file;
	line = a.line;
	size = a.size;
	base_size = a.base_size;
      }
      return *this;
    }
  };


  static std::map<void *, MemAlloc> alloc[N_ALLOC_TYPE];
  static long total_bytes[N_ALLOC_TYPE] = {0};
  static long max_total_bytes[N_ALLOC_TYPE] = {0};
  static long total_host_bytes, max_total_host_bytes;
  static long total_pinned_bytes, max_total_pinned_bytes;

  static void print_trace (void) {
    void *array[10];
    size_t size;
    char **strings;
    size = backtrace (array, 10);
    strings = backtrace_symbols (array, size);
    printfQuda("Obtained %zd stack frames.\n", size);
    for (size_t i=0; i<size; i++) printfQuda("%s\n", strings[i]);
    free(strings);
  }

  static void print_alloc_header()
  {
    printfQuda("Type    Pointer          Size             Location\n");
    printfQuda("----------------------------------------------------------\n");
  }


  static void print_alloc(AllocType type)
  {
    const char *type_str[] = {"Device", "Host  ", "Pinned", "Mapped"};
    std::map<void *, MemAlloc>::iterator entry;

    for (entry = alloc[type].begin(); entry != alloc[type].end(); entry++) {
      void *ptr = entry->first;
      MemAlloc a = entry->second;
      printfQuda("%s  %15p  %15lu  %s(), %s:%d\n", type_str[type], ptr, (unsigned long) a.base_size,
		 a.func.c_str(), a.file.c_str(), a.line);
    }
  }


  static void track_malloc(const AllocType &type, const MemAlloc &a, void *ptr)
  {
    total_bytes[type] += a.base_size;
    if (total_bytes[type] > max_total_bytes[type]) {
      max_total_bytes[type] = total_bytes[type];
    }
    if (type != DEVICE) {
      total_host_bytes += a.base_size;
      if (total_host_bytes > max_total_host_bytes) {
	max_total_host_bytes = total_host_bytes;
      }
    }
    if (type == PINNED || type == MAPPED) {
      total_pinned_bytes += a.base_size;
      if (total_pinned_bytes > max_total_pinned_bytes) {
	max_total_pinned_bytes = total_pinned_bytes;
      }
    }
    alloc[type][ptr] = a;
  }


  static void track_free(const AllocType &type, void *ptr)
  {
    size_t size = alloc[type][ptr].base_size;
    total_bytes[type] -= size;
    if (type != DEVICE) {
      total_host_bytes -= size;
    }
    if (type == PINNED || type == MAPPED) {
      total_pinned_bytes -= size;
    }
    alloc[type].erase(ptr);
  }


  /**
   * Under CUDA 4.0, cudaHostRegister seems to require that both the
   * beginning and end of the buffer be aligned on page boundaries.
   * This local function takes care of the alignment and gets called
   * by pinned_malloc_() and mapped_malloc_()
   */
  static void *aligned_malloc(MemAlloc &a, size_t size)
  {
    void *ptr;

    a.size = size;

#if (CUDA_VERSION > 4000)
    a.base_size = size;
    ptr = malloc(size);
#else
    static int page_size = getpagesize();
    a.base_size = ((size + page_size - 1) / page_size) * page_size; // round up to the nearest multiple of page_size
    posix_memalign(&ptr, page_size, a.base_size);
#endif
    if (!ptr) {
      printfQuda("ERROR: Failed to allocate aligned host memory of size %zu (%s:%d in %s())\n", size, a.file.c_str(), a.line, a.func.c_str());
      errorQuda("Aborting");
    }
    return ptr;
  }


  /**
   * Perform a standard cudaMalloc() with error-checking.  This
   * function should only be called via the device_malloc() macro,
   * defined in malloc_quda.h
   */
  void *device_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr;

    a.size = a.base_size = size;

    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      printfQuda("ERROR: Failed to allocate device memory of size %zu (%s:%d in %s())\n", size, file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(DEVICE, a, ptr);
#ifdef HOST_DEBUG
    cudaMemset(ptr, 0xff, size);
#endif
    return ptr;
  }


  /**
   * Perform a cuMemAlloc with error-checking.  This function is to
   * guarantee a unique memory allocation on the device, since
   * cudaMalloc can be redirected (as is the case with QDPJIT).  This
   * should only be called via the device_pinned_malloc() macro,
   * defined in malloc_quda.h.
   */
  void *device_pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr;

    a.size = a.base_size = size;

    CUresult err = cuMemAlloc((CUdeviceptr*)&ptr, size);
    if (err != CUDA_SUCCESS) {
      printfQuda("ERROR: Failed to allocate device memory of size %zu (%s:%d in %s())\n", size, file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(DEVICE, a, ptr);
#ifdef HOST_DEBUG
    cudaMemset(ptr, 0xff, size);
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
    if (!ptr) {
      printfQuda("ERROR: Failed to allocate host memory of size %zu (%s:%d in %s())\n", size, file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(HOST, a, ptr);
#ifdef HOST_DEBUG
    memset(ptr, 0xff, size);
#endif
    return ptr;
  }


  /**
   * Allocate page-locked ("pinned") host memory.  This function
   * should only be called via the pinned_malloc() macro, defined in
   * malloc_quda.h
   *
   * Note that we do not rely on cudaHostAlloc(), since buffers
   * allocated in this way have been observed to cause problems when
   * shared with MPI via GPU Direct on some systems.
   */
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr = aligned_malloc(a, size);
    
    cudaError_t err = cudaHostRegister(ptr, a.base_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      printfQuda("ERROR: Failed to register pinned memory of size %zu (%s:%d in %s())\n", size, file, line, func);
      errorQuda("Aborting");
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
    
    cudaError_t err = cudaHostRegister(ptr, a.base_size, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
      printfQuda("ERROR: Failed to register host-mapped memory of size %zu (%s:%d in %s())\n", size, file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(MAPPED, a, ptr);
#ifdef HOST_DEBUG
    memset(ptr, 0xff, a.base_size);
#endif
    return ptr;
  }  


  /**
   * Free device memory allocated with device_malloc().  This function
   * should only be called via the device_free() macro, defined in
   * malloc_quda.h
   */
  void device_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!ptr) {
      printfQuda("ERROR: Attempt to free NULL device pointer (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    if (!alloc[DEVICE].count(ptr)) {
      printfQuda("ERROR: Attempt to free invalid device pointer (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      printfQuda("ERROR: Failed to free device memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_free(DEVICE, ptr);
  }


  /**
   * Free device memory allocated with device_pinned malloc().  This
   * function should only be called via the device_pinned_free()
   * macro, defined in malloc_quda.h
   */
  void device_pinned_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!ptr) {
      printfQuda("ERROR: Attempt to free NULL device pointer (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    if (!alloc[DEVICE].count(ptr)) {
      printfQuda("ERROR: Attempt to free invalid device pointer (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    CUresult err = cuMemFree((CUdeviceptr)ptr);
    if (err != CUDA_SUCCESS) {
      printfQuda("ERROR: Failed to free device memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_free(DEVICE, ptr);
  }


  /**
   * Free host memory allocated with safe_malloc(), pinned_malloc(),
   * or mapped_malloc().  This function should only be called via the
   * host_free() macro, defined in malloc_quda.h
   */
  void host_free_(const char *func, const char *file, int line, void *ptr)
  {
    if (!ptr) {
      printfQuda("ERROR: Attempt to free NULL host pointer (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    if (alloc[HOST].count(ptr)) {
      track_free(HOST, ptr);
    } else if (alloc[PINNED].count(ptr)) {
      cudaError_t err = cudaHostUnregister(ptr);
      if (err != cudaSuccess) {
	printfQuda("ERROR: Failed to unregister pinned memory (%s:%d in %s())\n", file, line, func);
	errorQuda("Aborting");
      }
      track_free(PINNED, ptr);
    } else if (alloc[MAPPED].count(ptr)) {
      cudaError_t err = cudaHostUnregister(ptr);
      if (err != cudaSuccess) {
	printfQuda("ERROR: Failed to unregister host-mapped memory (%s:%d in %s())\n", file, line, func);
	errorQuda("Aborting");
      }
      track_free(MAPPED, ptr);
    } else {
      printfQuda("ERROR: Attempt to free invalid host pointer (%s:%d in %s())\n", file, line, func);
      print_trace();
      errorQuda("Aborting");
    }
    free(ptr);
  }


  void printPeakMemUsage()
  {
    printfQuda("Device memory used = %.1f MB\n", max_total_bytes[DEVICE] / (double)(1<<20));
    printfQuda("Page-locked host memory used = %.1f MB\n", max_total_pinned_bytes / (double)(1<<20));
    printfQuda("Total host memory used >= %.1f MB\n", max_total_host_bytes / (double)(1<<20));
  }


  void assertAllMemFree()
  {
    if (!alloc[DEVICE].empty() || !alloc[HOST].empty() || !alloc[PINNED].empty() || !alloc[MAPPED].empty()) {
      warningQuda("The following internal memory allocations were not freed.");
      printfQuda("\n");
      print_alloc_header();
      print_alloc(DEVICE);
      print_alloc(HOST);
      print_alloc(PINNED);
      print_alloc(MAPPED);
      printfQuda("\n");
    }
  }


  namespace pool {

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

    void init() {
      if (!pool_init) {
	// device memory pool
	char *enable_device_pool = getenv("QUDA_ENABLE_DEVICE_MEMORY_POOL");
	if (!enable_device_pool || strcmp(enable_device_pool,"0")!=0) {
	  warningQuda("Using device memory pool allocator");
	  device_memory_pool = true;
	} else {
	  warningQuda("Not using device memory pool allocator");
	  device_memory_pool = false;
	}

	// pinned memory pool
	char *enable_pinned_pool = getenv("QUDA_ENABLE_PINNED_MEMORY_POOL");
	if (!enable_pinned_pool || strcmp(enable_pinned_pool,"0")!=0) {
	  warningQuda("Using pinned memory pool allocator");
	  pinned_memory_pool = true;
	} else {
	  warningQuda("Not using pinned memory pool allocator");
	  pinned_memory_pool = false;
	}
	pool_init = true;
      }
    }

    void* pinned_malloc_(const char *func, const char *file, int line, size_t nbytes)
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
	if (!pinnedSize.count(ptr)) {
	  errorQuda("Attempt to free invalid pointer");
	}
	pinnedCache.insert(std::make_pair(pinnedSize[ptr], ptr));
	pinnedSize.erase(ptr);
      } else {
	quda::host_free_(func, file, line, ptr);
      }
    }

    void* device_malloc_(const char *func, const char *file, int line, size_t nbytes)
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
	if (!deviceSize.count(ptr)) {
	  errorQuda("Attempt to free invalid pointer");
	}
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
