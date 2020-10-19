#include <quda.h>
#include "util_quda.h"
#include <cuda_runtime.h>

#include "quda_fortran.h"

void register_pinned_quda_(void *ptr, size_t *bytes) {
  cudaHostRegister(ptr, *bytes, cudaHostRegisterDefault);
  checkCudaError();
}

void unregister_pinned_quda_(void *ptr) {
  cudaHostUnregister(ptr);
  checkCudaError();
}
