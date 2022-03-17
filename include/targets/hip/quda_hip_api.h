#pragma once
#include <hip/hip_runtime.h>

namespace quda
{

  /**
     @file quda_hip_api.h
     @brief Header file that declares some functions that will be called from within the CUDA target
  */
  namespace target
  {
    namespace hip
    {

      /**
           @brief Return HIP stream from QUDA stream.  This is only for
           use inside target/cuda.
           @param stream QUDA stream we which to convert to CUDA stream
           @return CUDA stream
        */
      hipStream_t get_stream(const qudaStream_t &stream);

      void set_runtime_error(hipError_t error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error = false);

      // defined in quda_api.cpp
      void set_driver_error(hipError_t error, const char *api_func, const char *func, const char *file,
                            const char *line, bool allow_error = false);
    } // namespace hip
  }   // namespace target
} // namespace quda
