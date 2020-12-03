#pragma once

#include "quda_define.h"

#if defined(QUDA_TARGET_CUDA)
#include <cuda_fp16.h>
#elif defined(QUDA_TARGET_CUDA)
#include <hip/hip_fp16.h>
#endif
namespace quda
{

// Hip supports this now supposedly -- come back to
  __device__ inline half2 habs2(half2 input) {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10020)
    return __habs2(input);
#else
    static constexpr uint32_t maximum_mask = 0x7fff7fffu; // 0111 1111 1111 1111 0111 1111 1111 1111

    uint32_t input_masked = *reinterpret_cast<const uint32_t *>(&input) & maximum_mask;
    return *reinterpret_cast<half2 *>(&input_masked);
#endif
  }

} // namespace quda
