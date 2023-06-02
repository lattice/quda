#pragma once

#include "malloc_quda.h"

namespace quda {

  /**
     Object that stores a memory allocation with different views for
     host or device.  Depending on the nature of the underlying memory
     type, both views may not be defined

     type                       defined views
     QUDA_MEMORY_DEVICE         device only
     QUDA_MEMORY_DEVICE_PINNED  device only
     QUDA_MEMORY_HOST           host only
     QUDA_MEMORY_HOST_PINNED    both
     QUDA_MEMORY_MAPPED         both (pinned to host)
     QUDA_MEMORY_MANAGED        both
   */
  class quda_ptr {
    QudaMemoryType type = QUDA_MEMORY_INVALID;
    size_t size = 0;
    bool pool = false;
    void *device = nullptr;
    void *host = nullptr;

  public:
    quda_ptr() = default;
    quda_ptr(quda_ptr &&) = default;
    quda_ptr &operator=(quda_ptr &&);

    /**
       @brief Constructor for quda_ptr
       @param[in] type The memory type of the allocation
       @param[in] size The size of the allocation
       @param[in] pool Whether the allocation should be in the memory pool (default is true)
    */
    quda_ptr(QudaMemoryType type, size_t size, bool pool = true);

    /**
       @brief Constructor for quda_ptr where we are wrapping a non-owned pointer
       @param[in] ptr Raw base pointer
       @param[in] type The memory type of the allocation
    */
    quda_ptr(void *ptr, QudaMemoryType type);

    /**
       @brief Destructor for the quda_ptr
    */
    virtual ~quda_ptr();

    /**
       @return Returns true if allocation is visible to the device
    */
    bool is_device() const;

    /**
       @return Returns true if allocation is visible to the host
    */
    bool is_host() const;

    /**
       Return view of the pointer.  For mapped memory we return the device view.
     */
    void *data() const;

    /**
       Return the device view of the pointer
     */
    void *data_device() const;

    /**
       Return the host view of the pointer
     */
    void *data_host() const;
  };

}
