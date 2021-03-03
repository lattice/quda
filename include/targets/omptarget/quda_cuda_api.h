using cudaError_t = int;

namespace quda {

  /**
     @file quda_cuda_api.h
     @brief Header file that declares some functions that will be called from within the CUDA target
  */

  namespace cuda {

    void set_runtime_error(cudaError_t error, const char *api_func, const char *func, const char *file, const char *line,
                           bool allow_error = false);

    // defined in quda_api.cpp
    void set_driver_error(CUresult error, const char *api_func, const char *func, const char *file, const char *line,
                          bool allow_error = false);


  }
}
