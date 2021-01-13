#pragma once
#include <hip/hip_runtime.h>

namespace quda {

  /**
     @file quda_hip_api.h
     @brief Header file that declares some functions that will be called from within the CUDA target
  */

  namespace hip {

    void set_runtime_error(hipError_t error, const char *api_func, const char *func, const char *file, const char *line,
                           bool allow_error = false);
    
    // defined in quda_api.cpp
    void set_driver_error(hipError_t error, const char *api_func, const char *func, const char *file, const char *line,
                          bool allow_error = false);
    
    
  }
}
