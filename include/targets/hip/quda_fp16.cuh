#pragma once

#include <hip/hip_fp16.h>

namespace quda
{

  __device__ inline half2 habs2(half2 input) {
    static constexpr uint32_t maximum_mask = 0x7fff7fffu; // 0111 1111 1111 1111 0111 1111 1111 1111

    uint32_t input_masked = *reinterpret_cast<const uint32_t *>(&input) & maximum_mask;
    return *reinterpret_cast<half2 *>(&input_masked);
  }

} // namespace quda
