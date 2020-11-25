#pragma once

/**
   @file jitify_helper.cuh

   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#include <quda_define.h>
#if defined(JITIFY) && !defined(QUDA_TARGET_CUDA)
#error "Jitify compilation cannot be enabled unless targeting CUDA"
#endif

#if defined(JITIFY)

#ifdef HOST_DEBUG

// display debugging info
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE        1
#define JITIFY_PRINT_LOG           1
#define JITIFY_PRINT_PTX           1
#define JITIFY_PRINT_LINKER_LOG    1
#define JITIFY_PRINT_LAUNCH        1
#define JITIFY_PRINT_HEADER_PATHS  1

#else // !HOST_DEBUG

// hide debugging info
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE        0
#ifdef DEVEL
#define JITIFY_PRINT_LOG           1
#else
#define JITIFY_PRINT_LOG           0
#endif
#define JITIFY_PRINT_PTX           0
#define JITIFY_PRINT_LINKER_LOG    0
#define JITIFY_PRINT_LAUNCH        0
#define JITIFY_PRINT_HEADER_PATHS  0

#endif // !HOST_DEBUG

#include "jitify_options.hpp"
#include <jitify.hpp>
#include <device.h>

#endif

namespace quda {

#ifdef JITIFY

  static jitify::JitCache *kernel_cache = nullptr;
  static jitify::Program *program = nullptr;
  static bool jitify_init = false;

  static void create_jitify_program(const std::string &file, const std::vector<std::string> extra_options = {})
  {
    if (!jitify_init) {
      kernel_cache = new jitify::JitCache;

      std::vector<std::string> options = {"-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-remove-unused-globals"};

#ifdef DEVICE_DEBUG
      options.push_back(std::string("-G"));
#endif

#if __cplusplus >= 201703L
      options.push_back(std::string("-std=c++17"));
#else
      options.push_back(std::string("-std=c++14"));
#endif

      // add an extra compilation options specific to this instance
      for (auto option : extra_options) options.push_back(option);

      program = new jitify::Program(kernel_cache->program(file, 0, options));
      jitify_init = true;
    }
  }

  template <typename instance_t>
  void set_max_shared_bytes(instance_t &instance)
  {
    instance.set_func_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
    auto shared_size = instance.get_func_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    instance.set_func_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                device::max_dynamic_shared_memory() - shared_size);
  }

#endif

} // namespace quda
