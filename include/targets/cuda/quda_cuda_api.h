#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
   @file quda_cuda_api.h
   @brief Header file that declares some functions that will be called from within the CUDA target
*/

namespace quda
{

  namespace target
  {

    namespace cuda
    {

      /**
         @brief Return CUDA stream from QUDA stream.  This is only for
         use inside target/cuda.
         @param stream QUDA stream we which to convert to CUDA stream
         @return CUDA stream
      */
      cudaStream_t get_stream(const qudaStream_t &stream);

      void set_runtime_error(cudaError_t error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error = false);

      // defined in quda_api.cpp
      void set_driver_error(CUresult error, const char *api_func, const char *func, const char *file, const char *line,
                            bool allow_error = false);

    } // namespace cuda
  }   // namespace target

} // namespace quda
