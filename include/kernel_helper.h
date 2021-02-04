#pragma once

#include <target_device.h>

namespace quda {

  struct constant_param_t {
    static constexpr size_t max_size = device::max_constant_param_size();
    size_t bytes;
    alignas(16) char host[max_size];
    void *device_ptr;
    char device_name[128];
  };

  static std::vector<constant_param_t> dummy_param;

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) :
      func(func),
      name(name) {}
  };

#ifdef JITIFY
#define KERNEL(kernel) kernel_t(nullptr, #kernel)
#else
#define KERNEL(kernel) kernel_t(reinterpret_cast<const void *>(kernel<Functor, Arg, grid_stride>), #kernel)
#endif

}
