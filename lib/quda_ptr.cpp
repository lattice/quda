#include <utility>
#include "quda_ptr.h"
#include "util_quda.h"
#include "timer.h"

namespace quda
{

  quda_ptr::quda_ptr(QudaMemoryType type, size_t size, bool pool) : type(type), size(size), pool(pool)
  {
    getProfile().TPSTART(QUDA_PROFILE_INIT);
    if (pool && (type != QUDA_MEMORY_DEVICE && type != QUDA_MEMORY_HOST_PINNED && type != QUDA_MEMORY_HOST))
      errorQuda("Memory pool not available for memory type %d", type);

    if (size > 0) {
      switch (type) {
      case QUDA_MEMORY_DEVICE: device = pool ? pool_device_malloc(size) : device_malloc(size); break;
      case QUDA_MEMORY_DEVICE_PINNED: device = device_pinned_malloc(size); break;
      case QUDA_MEMORY_HOST: host = safe_malloc(size); break;
      case QUDA_MEMORY_HOST_PINNED: host = pool ? pool_pinned_malloc(size) : pinned_malloc(size); break;
      case QUDA_MEMORY_MAPPED:
        host = mapped_malloc(size);
        device = get_mapped_device_pointer(host);
        break;
      case QUDA_MEMORY_MANAGED:
        host = managed_malloc(size);
        device = host;
        break;
      default: errorQuda("Unknown memory type %d", type);
      }
    }
    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  quda_ptr::quda_ptr(void *ptr, QudaMemoryType type) : type(type), reference(true)
  {
    getProfile().TPSTART(QUDA_PROFILE_INIT);
    switch (type) {
    case QUDA_MEMORY_DEVICE:
    case QUDA_MEMORY_DEVICE_PINNED:
      device = ptr;
      host = nullptr;
      break;
    case QUDA_MEMORY_HOST:
    case QUDA_MEMORY_HOST_PINNED:
      device = nullptr;
      host = ptr;
      break;
    case QUDA_MEMORY_MANAGED:
      device = ptr;
      host = ptr;
      break;
    default: errorQuda("Unsupported memory type %d", type);
    }
    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  quda_ptr &quda_ptr::operator=(quda_ptr &&other)
  {
    if (&other != this) {
      if (size > 0) errorQuda("Cannot move to already initialized quda_ptr");
      type = std::exchange(other.type, QUDA_MEMORY_INVALID);
      size = std::exchange(other.size, 0);
      pool = std::exchange(other.pool, false);
      device = std::exchange(other.device, nullptr);
      host = std::exchange(other.host, nullptr);
    }
    return *this;
  }

  void quda_ptr::destroy()
  {
    if (size > 0) {
      switch (type) {
      case QUDA_MEMORY_DEVICE: pool ? pool_device_free(device) : device_free(device); break;
      case QUDA_MEMORY_DEVICE_PINNED: device_pinned_free(device); break;
      case QUDA_MEMORY_HOST: host_free(host); break;
      case QUDA_MEMORY_HOST_PINNED: pool ? pool_pinned_free(host) : host_free(host); break;
      case QUDA_MEMORY_MAPPED: host_free(host); break;
      default: errorQuda("Unknown memory type %d", type);
      }
    }

    size = 0;
    device = nullptr;
    host = nullptr;
  }

  quda_ptr::~quda_ptr()
  {
    getProfile().TPSTART(QUDA_PROFILE_FREE);
    destroy();
    getProfile().TPSTOP(QUDA_PROFILE_FREE);
  }

  void quda_ptr::exchange(quda_ptr &obj, quda_ptr &&new_value)
  {
    destroy();
    *this = std::move(obj);
    obj = std::move(new_value);
  }

  bool quda_ptr::is_device() const
  {
    switch (type) {
    case QUDA_MEMORY_DEVICE:
    case QUDA_MEMORY_DEVICE_PINNED:
    case QUDA_MEMORY_MAPPED:
    case QUDA_MEMORY_MANAGED: return true;
    default: return false;
    }
  }

  bool quda_ptr::is_host() const
  {
    switch (type) {
    case QUDA_MEMORY_HOST:
    case QUDA_MEMORY_HOST_PINNED:
    case QUDA_MEMORY_MANAGED: return true;
    default: return false;
    }
  }

  void *quda_ptr::data() const
  {
    void *ptr = nullptr;

    switch (type) {
    case QUDA_MEMORY_DEVICE:
    case QUDA_MEMORY_DEVICE_PINNED:
    case QUDA_MEMORY_MAPPED:
    case QUDA_MEMORY_MANAGED: ptr = device; break;
    case QUDA_MEMORY_HOST:
    case QUDA_MEMORY_HOST_PINNED: ptr = host; break;
    default: errorQuda("Unknown memory type %d", type);
    }

    return ptr;
  }

  void *quda_ptr::data_device() const
  {
    if (!device) errorQuda("Device view not defined");
    return device;
  }

  void *quda_ptr::data_host() const
  {
    if (!host) errorQuda("Host view not defined");
    return host;
  }

  bool quda_ptr::is_reference() const { return reference; }

  std::ostream &operator<<(std::ostream &output, const quda_ptr &ptr)
  {
    output << "{type = " << ptr.type << ", size = " << ptr.size << ", pool = " << ptr.pool
           << ", device = " << ptr.device << ", host = " << ptr.host << ", reference = " << ptr.reference << "}";
    return output;
  }

} // namespace quda
