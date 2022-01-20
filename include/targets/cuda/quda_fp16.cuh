#pragma once

#include <cuda_fp16.h>

namespace quda
{

  __device__ inline half2 habs2(half2 input) {
#if !(defined(__clang__) && defined(__CUDA__))
    return __habs2(input);
#else
    static constexpr uint32_t maximum_mask = 0x7fff7fffu; // 0111 1111 1111 1111 0111 1111 1111 1111

    uint32_t input_masked = *reinterpret_cast<const uint32_t *>(&input) & maximum_mask;
    return *reinterpret_cast<half2 *>(&input_masked);
#endif
  }

} // namespace quda
