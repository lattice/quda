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

#else // !HOST_DEBUG

// hide debugging info
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE        0
#define JITIFY_PRINT_LOG           0
#define JITIFY_PRINT_PTX           0
#define JITIFY_PRINT_LAUNCH        0

#endif // !HOST_DEBUG

#include "jitify_options.hpp"
#include <jitify.hpp>

#endif

namespace quda {

#ifdef JITIFY

  static jitify::JitCache *kernel_cache = nullptr;
  static jitify::Program *program = nullptr;
  static bool jitify_init = false;

  static void create_jitify_program(const char *file, const std::vector<std::string> extra_options = {}) {

    if (!jitify_init) {
      kernel_cache = new jitify::JitCache;

      std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};
      options.push_back( std::string("-D__COMPUTE_CAPABILITY__=") + std::to_string(__COMPUTE_CAPABILITY__) );
      options.push_back( std::string("-DMAX_MULTI_BLAS_N=") + std::to_string(MAX_MULTI_BLAS_N) );

#ifdef USE_TEXTURE_OBJECTS
      options.push_back( std::string("-DUSE_TEXTURE_OBJECTS") );
#endif
#ifdef DYNAMIC_CLOVER
      options.push_back( std::string("-DDYNAMIC_CLOVER") );
#endif

#ifdef DEVICE_DEBUG
      options.push_back( std::string("-G") );
#endif

      // add an extra compilation options specific to this instance
      for (auto option : extra_options) options.push_back(option);

      program = new jitify::Program(kernel_cache->program(file, 0, options));
      jitify_init = true;
    }
  }

#endif

} // namespace quda
