#include <util_quda.h>
#include <quda_api.h>
#include <quda_cuda_api.h>

#define CHECK_CUDA_ERROR(func)                                          \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

namespace quda {

  template <int n_unroll>
  __global__ void copy_arg_kernel(void *dst, void *src, size_t n)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < n) {
#pragma unroll
      for (int i = 0; i < n_unroll; i++) {
        reinterpret_cast<int4*>(dst)[i * n + idx] = reinterpret_cast<int4*>(src)[i * n + idx];
      }
      idx += gridDim.x * blockDim.x;
    }
  }

  void copy_arg(void *dst, void *src, size_t size, cudaStream_t stream)
  {
    constexpr int n_unroll = 1;
    if (size % (n_unroll * sizeof(int4)) != 0) errorQuda("Copy size %lu is not a multiple of %lu bytes", size, n_unroll * sizeof(int4));
    size_t n = size / (n_unroll * sizeof(int4));
    copy_arg_kernel<n_unroll> <<<8, 256, 0, stream>>>(dst, src, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
  }

}
