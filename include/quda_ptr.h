#pragma once

#include <ostream>
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
    friend std::ostream& operator<<(std::ostream& output, const quda_ptr& ptr);
    QudaMemoryType type = QUDA_MEMORY_INVALID; /** Memory type of the allocation */
    size_t size = 0;                           /** Size of the allocation */
    bool pool = false;                         /** Is the allocation is pooled */
    void *device = nullptr;                    /** Device-view of the allocation */
    void *host = nullptr;                      /** Host-view of the allocation */
    bool reference = false;                    /** Is this a reference to another allocation */

    /**
       @brief Internal deallocation routine
     */
    void destroy();

  public:
    quda_ptr() = default;
    quda_ptr(quda_ptr &&) = default;
    quda_ptr &operator=(quda_ptr &&);
    quda_ptr(const quda_ptr &) = delete;
    quda_ptr &operator=(const quda_ptr &) = delete;

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
       @brief Specialized exchange function to use in place of
       std::exchange when exchanging quda_ptr objects: moves obj to
       *this, and moves new_value to obj
       @param[in,out] obj
       @param[in] new_value New value for obj to take
     */
    void exchange(quda_ptr &obj, quda_ptr &&new_value);

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

    /**
       Return if the instance is a reference rather than an allocation
     */
    bool is_reference() const;
  };

  std::ostream& operator<<(std::ostream& output, const quda_ptr& ptr);

}
