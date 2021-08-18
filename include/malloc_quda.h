#pragma once

#include <cstdlib>
#include <cstdint>
#include <enum_quda.h>

namespace quda {

  void printPeakMemUsage();
  void assertAllMemFree();

  /**
     @return device memory allocated
   */
  size_t device_allocated();

  /**
     @return pinned memory allocated
   */
  size_t pinned_allocated();

  /**
     @return mapped memory allocated
   */
  size_t mapped_allocated();

  /**
     @return host memory allocated
   */
  size_t host_allocated();

  /**
     @return peak device memory allocated
   */
  size_t device_allocated_peak();

  /**
     @return peak pinned memory allocated
   */
  size_t pinned_allocated_peak();

  /**
     @return peak mapped memory allocated
   */
  size_t mapped_allocated_peak();

  /**
     @return peak host memory allocated
   */
  size_t host_allocated_peak();

  /**
     @return are we using managed memory for device allocations
  */
  bool use_managed_memory();

  /**
     @return is prefetching support enabled (assumes managed memory is enabled)
  */
  bool is_prefetch_enabled();

  /*
   * The following functions should not be called directly.  Use the
   * macros below instead.
   */
  void *device_malloc_(const char *func, const char *file, int line, size_t size);
  void *device_pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *device_comms_pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *safe_malloc_(const char *func, const char *file, int line, size_t size);
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *mapped_malloc_(const char *func, const char *file, int line, size_t size);
  void *managed_malloc_(const char *func, const char *file, int line, size_t size);
  void device_free_(const char *func, const char *file, int line, void *ptr);
  void device_pinned_free_(const char *func, const char *file, int line, void *ptr);
  void device_comms_pinned_free_(const char *func, const char *file, int line, void *ptr);
  void managed_free_(const char *func, const char *file, int line, void *ptr);
  void host_free_(const char *func, const char *file, int line, void *ptr);
  void register_pinned_(const char *func, const char *file, int line, void *ptr, size_t bytes);
  void unregister_pinned_(const char *func, const char *file, int line, void *ptr);

  QudaFieldLocation get_pointer_location(const void *ptr);

  /*
    @brief Get device view of a host-mapped pointer
   */
  void *get_mapped_device_pointer_(const char *func, const char *file, int line, const void *ptr);

  /**
   * @return whether the pointer is aligned
   */
  inline bool is_aligned(const void *ptr, size_t alignment)
  {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
  }

} // namespace quda

#define device_malloc(size) quda::device_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_pinned_malloc(size) quda::device_pinned_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_comms_pinned_malloc(size)                                                                               \
  quda::device_comms_pinned_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define safe_malloc(size) quda::safe_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define pinned_malloc(size) quda::pinned_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define mapped_malloc(size) quda::mapped_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define managed_malloc(size) quda::managed_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_free(ptr) quda::device_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define device_pinned_free(ptr) quda::device_pinned_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define device_comms_pinned_free(ptr)                                                                                  \
  quda::device_comms_pinned_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define managed_free(ptr) quda::managed_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define host_free(ptr) quda::host_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define get_mapped_device_pointer(ptr)                                                                                 \
  quda::get_mapped_device_pointer_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define register_pinned(ptr, bytes) quda::register_pinned_(__func__, quda::file_name(__FILE__), __LINE__, ptr, bytes)
#define unregister_pinned(size) quda::unregister_pinned_(__func__, quda::file_name(__FILE__), __LINE__, ptr)

namespace quda {

  namespace pool {

    /**
       @brief Initialize the memory pool allocator
    */
    void init();

    /**
       @brief Allocate device-memory.  If free pre-existing allocation exists
       reuse this.
       @param size Size of allocation
       @return Pointer to allocated memory
    */
    void *device_malloc_(const char *func, const char *file, int line, size_t size);

    /**
       @brief Virtual free of pinned-memory allocation.
       @param ptr Pointer to be (virtually) freed
    */
    void device_free_(const char *func, const char *file, int line, void *ptr);

    /**
       @brief Allocate pinned-memory.  If a free pre-existing allocation exists
       reuse this.
       @param size Size of allocation
       @return Pointer to allocated memory
    */
    void *pinned_malloc_(const char *func, const char *file, int line, size_t size);

    /**
       @brief Virtual free of pinned-memory allocation.
       @param ptr Pointer to be (virtually) freed
    */
    void pinned_free_(const char *func, const char *file, int line, void *ptr);

    /**
       @brief Free all outstanding device-memory allocations.
    */
    void flush_device();

    /**
       @brief Free all outstanding pinned-memory allocations.
    */
    void flush_pinned();

  } // namespace pool

}

#define pool_device_malloc(size) quda::pool::device_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_device_free(ptr) quda::pool::device_free_(__func__, __FILE__, __LINE__, ptr)
#define pool_pinned_malloc(size) quda::pool::pinned_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_pinned_free(ptr) quda::pool::pinned_free_(__func__, __FILE__, __LINE__, ptr)
