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
     @return host pinned (page-locked, GPU-mapped) memory allocated
   */
  size_t host_pinned_allocated();

  /**
     @return host memory allocated
   */
  size_t host_allocated();

  /**
     @return peak device memory allocated
   */
  size_t device_allocated_peak();

  /**
     @return peak host pinned memory allocated
   */
  size_t host_pinned_allocated_peak();

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
  // Tag-dispatched communication-buffer allocator. The DeviceCommBuffer overload uses the
  // driver-API cuMemAlloc (avoids the cudaMalloc runtime-API hijack risk and
  // gives physically-contiguous memory), so the buffer is P2P-capable and
  // RDMA-ready -- suitable for cudaIPC handle export and, under MNNVL, fabric
  // export. NVSHMEM overload uses shmem_malloc_. Each allocation is tagged in
  // the alloc[] tracker so the per-kind *_comm_buffer_free can verify the kind.
  // (MPI comm buffers share the DeviceCommBuffer kind -- they resolve to the
  // same primitive -- so there is no separate MPI tag.)
  namespace comm
  {
    struct DeviceCommBuffer {
    };
    struct QudaCommTypeNVSHMEM {
    };
  } // namespace comm

  void *comm_buffer_malloc_(const char *func, const char *file, int line, comm::DeviceCommBuffer, size_t size);
#ifdef NVSHMEM_COMMS
  void *comm_buffer_malloc_(const char *func, const char *file, int line, comm::QudaCommTypeNVSHMEM, size_t size);
#endif
  // Per-kind free: each asserts the ptr is in alloc[KIND] and dispatches to
  // the matching free primitive.  Callers pair their *_comm_buffer_malloc
  // with the matching *_comm_buffer_free.
  void device_comm_buffer_free_(const char *func, const char *file, int line, void *ptr);
#ifdef NVSHMEM_COMMS
  void nvshmem_comm_buffer_free_(const char *func, const char *file, int line, void *ptr);
#endif

  // The P2P fabric-handle accessors (get_p2p_fabric_handle / get_p2p_buffer_size /
  // get_p2p_buffer_generation) are CUDA/MNNVL-specific and return a CUmemFabricHandle,
  // so they live in <malloc_target.h> (targets/cuda) to keep <cuda.h> and the
  // QUDA_MNNVL #ifdef out of this generic header.

  void *safe_malloc_(const char *func, const char *file, int line, size_t size);
  void *host_pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *managed_malloc_(const char *func, const char *file, int line, size_t size);
  void device_free_(const char *func, const char *file, int line, void *ptr);
  void device_pinned_free_(const char *func, const char *file, int line, void *ptr);
  void managed_free_(const char *func, const char *file, int line, void *ptr);
  void host_free_(const char *func, const char *file, int line, void *ptr);
  void register_pinned_(const char *func, const char *file, int line, void *ptr, size_t bytes);
  void unregister_pinned_(const char *func, const char *file, int line, void *ptr);

  QudaFieldLocation get_pointer_location(const void *ptr);

  /*
    @brief Get device view of a host-pinned mapped pointer
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
#define safe_malloc(size) quda::safe_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define host_pinned_malloc(size) quda::host_pinned_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define managed_malloc(size) quda::managed_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_free(ptr) quda::device_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define device_pinned_free(ptr) quda::device_pinned_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define device_comm_buffer_malloc(size)                                                                                \
  quda::comm_buffer_malloc_(__func__, quda::file_name(__FILE__), __LINE__, quda::comm::DeviceCommBuffer {}, size)
#define nvshmem_comm_buffer_malloc(size)                                                                               \
  quda::comm_buffer_malloc_(__func__, quda::file_name(__FILE__), __LINE__, quda::comm::QudaCommTypeNVSHMEM {}, size)
#define device_comm_buffer_free(ptr) quda::device_comm_buffer_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define nvshmem_comm_buffer_free(ptr)                                                                                  \
  quda::nvshmem_comm_buffer_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define managed_free(ptr) quda::managed_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define host_free(ptr) quda::host_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define get_mapped_device_pointer(ptr)                                                                                 \
  quda::get_mapped_device_pointer_(__func__, quda::file_name(__FILE__), __LINE__, ptr)
#define register_pinned(ptr, bytes) quda::register_pinned_(__func__, quda::file_name(__FILE__), __LINE__, ptr, bytes)
#define unregister_pinned(size) quda::unregister_pinned_(__func__, quda::file_name(__FILE__), __LINE__, ptr)

#define quda_malloc(size) quda::quda_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define quda_free(ptr) quda::quda_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)

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
       @brief Allocate host-pinned memory.  If a free pre-existing allocation exists
       reuse this.
       @param size Size of allocation
       @return Pointer to allocated memory
    */
    void *host_pinned_malloc_(const char *func, const char *file, int line, size_t size);

    /**
       @brief Virtual free of host-pinned allocation.
       @param ptr Pointer to be (virtually) freed
    */
    void host_pinned_free_(const char *func, const char *file, int line, void *ptr);

    /**
       @brief Free all outstanding device-memory allocations.
    */
    void flush_device();

    /**
       @brief Free all outstanding host-pinned allocations.
    */
    void flush_host_pinned();

  } // namespace pool

}

#define pool_device_malloc(size) quda::pool::device_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_device_free(ptr) quda::pool::device_free_(__func__, __FILE__, __LINE__, ptr)
#define pool_host_pinned_malloc(size) quda::pool::host_pinned_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_host_pinned_free(ptr) quda::pool::host_pinned_free_(__func__, __FILE__, __LINE__, ptr)
