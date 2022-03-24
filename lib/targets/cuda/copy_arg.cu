#include <util_quda.h>
#include <quda_api.h>
#include <quda_cuda_api.h>

#define CHECK_CUDA_ERROR(func)                                          \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

namespace quda {

  __global__ void copy_arg_kernel(void *dst, void *src, size_t size)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t n = size / sizeof(int4);
    while (idx < n) {
      reinterpret_cast<int4*>(dst)[idx] = reinterpret_cast<int4*>(src)[idx];
      idx += gridDim.x * blockDim.x;
    }
  }

  void copy_arg(void *dst, void *src, size_t size, cudaStream_t stream)
  {
    if (size % 16) errorQuda("Copy size is not a multiple of 16-bytes");
    copy_arg_kernel<<<4, 256, 0, stream>>>(dst, src, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
  }

}
