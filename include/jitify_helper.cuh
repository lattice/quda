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
#define JITIFY_PRINT_LAUNCH        1
#else
// display debugging info
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE        0
#define JITIFY_PRINT_LOG           0
#define JITIFY_PRINT_PTX           0
#define JITIFY_PRINT_LAUNCH        0
#endif

#include <jitify.hpp>

namespace quda {

  static jitify::JitCache *kernel_cache;
  static jitify::Program *program;
  static bool jitify_init = false;

  static void create_jitify_program(const char *file) {
    if (!jitify_init) {
      kernel_cache = new jitify::JitCache;
#ifdef DEVICE_DEBUG
      std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-G"};
#else
      std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};
#endif
      program = new jitify::Program(kernel_cache->program(file, 0, options));
      jitify_init = true;
    }
  }

} // namespace quda

const constexpr char *compile_type_str = "jitify";

#else // else not JITIFY

const constexpr char *compile_type_str = "offline";

#endif

