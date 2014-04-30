#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <unistd.h> // for getpagesize()
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
      printfQuda("ERROR: Failed to allocate aligned host memory (%s:%d in %s())\n", a.file.c_str(), a.line, a.func.c_str());
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
      printfQuda("ERROR: Failed to allocate device memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(DEVICE, a, ptr);
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
      printfQuda("ERROR: Failed to allocate host memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(HOST, a, ptr);
    return ptr;
  }


  /**
   * Allocate page-locked ("pinned") host memory.  This function
   * should only be called via the pinned_malloc() macro, defined in
   * malloc_quda.h
   *
   * Note that we do rely on cudaHostAlloc(), since buffers allocated
   * in this way have been observed to cause problems when shared with
   * MPI via GPU Direct on some systems.
   */
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size)
  {
    MemAlloc a(func, file, line);
    void *ptr = aligned_malloc(a, size);
    
    cudaError_t err = cudaHostRegister(ptr, a.base_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      printfQuda("ERROR: Failed to register pinned memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(PINNED, a, ptr);
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
      printfQuda("ERROR: Failed to register host-mapped memory (%s:%d in %s())\n", file, line, func);
      errorQuda("Aborting");
    }
    track_malloc(MAPPED, a, ptr);
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

} // namespace quda
