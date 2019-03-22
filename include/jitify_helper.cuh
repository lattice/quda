#pragma once

/**
   @file jitify_helper.cuh

   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#include <lattice_field.h>

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


#include "jitify_options.hpp"
#include <jitify.hpp>

#endif

namespace quda {

#ifdef JITIFY

  static jitify::JitCache *kernel_cache;
  static jitify::Program *program;
  static bool jitify_init = false;

  static void create_jitify_program(const char *file, const std::vector<std::string> extra_options = {}) {
    if (!jitify_init) {
      kernel_cache = new jitify::JitCache;

#ifdef DEVICE_DEBUG
      std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-G"};
#else
      std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};
#endif
      options.push_back( std::string("-D__COMPUTE_CAPABILITY__=") + std::to_string(__COMPUTE_CAPABILITY__) );

      // add an extra compilation options specific to this instance
      for (auto option : extra_options) options.push_back(option);

      program = new jitify::Program(kernel_cache->program(file, 0, options));
      jitify_init = true;
    }
  }

#endif

  /**
     @brief Helper function for setting auxilary string
     @param[in] meta LatticeField used for querying field location
     @return String containing location and compilation type
   */

  inline const char* compile_type_str(const LatticeField &meta, QudaFieldLocation location_ = QUDA_INVALID_FIELD_LOCATION) {
    QudaFieldLocation location = (location_ == QUDA_INVALID_FIELD_LOCATION ? meta.Location() : location_);
#ifdef JITIFY
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-jitify," : "CPU,";
#else
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-offline," : "CPU,";
#endif
  }

} // namespace quda
