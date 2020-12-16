#pragma once

#include <target_device.h>

namespace quda {

  struct constant_param_t {
    static constexpr size_t max_size = device::max_constant_param_size();
    size_t bytes;
    char host[max_size] alignas(16);
    void *device_ptr;
    char device_name[128];
  };

  static std::vector<constant_param_t> dummy_param;

}
