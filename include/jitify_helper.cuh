#pragma once

/**
   @file jitify_helper.cuh

   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#ifdef JITIFY

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

      std::vector<std::string> options = {"-std=c++14", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};

#ifdef DEVICE_DEBUG
      options.push_back(std::string("-G"));
#endif

      // add an extra compilation options specific to this instance
      for (auto option : extra_options) options.push_back(option);

      program = new jitify::Program(kernel_cache->program(file, 0, options));
      jitify_init = true;
    }
  }

#endif

} // namespace quda
