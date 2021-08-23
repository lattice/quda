#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
   @file quda_cuda_api.h
   @brief Header file that declares some functions that will be called from within the CUDA target
*/

namespace quda {

  namespace device {

    /**
       @brief Return CUDA stream from QUDA stream.  This is a
       temporary addition until all kernels have been made generic.
       @param stream QUDA stream we which to convert to CUDA stream
       @return CUDA stream
    */
    cudaStream_t get_cuda_stream(const qudaStream_t &stream);

  }

  namespace cuda {

    void set_runtime_error(cudaError_t error, const char *api_func, const char *func, const char *file, const char *line,
                           bool allow_error = false);

    // defined in quda_api.cpp
    void set_driver_error(CUresult error, const char *api_func, const char *func, const char *file, const char *line,
                          bool allow_error = false);


  }

}
