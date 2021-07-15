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
  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] arg Host address of argument struct
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, const qudaStream_t &stream, const void *arg);
}
