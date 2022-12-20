#pragma once

// This macro determines whether or not we are using the fp16 accumulation of the MMA instruction.
// #define USE_FP16_HMMA_ACCUMULATE

constexpr QudaPrecision accumulate_precision()
{
#ifdef USE_FP16_HMMA_ACCUMULATE
  return QUDA_HALF_PRECISION;
#else
  return QUDA_SINGLE_PRECISION;
#endif
}

namespace quda
{
  namespace hmma
  {
    template <int m, int n, int k, class compute_type, class load_type> struct hmma_t { };
  }
}

#if (__COMPUTE_CAPABILITY__ == 700)
#include <mma_tensor_op/hmma_m16n16k4_sm70.cuh>
#else
#include <mma_tensor_op/hmma_m16n8k8_sm80.cuh>
#include <mma_tensor_op/smma_m16n8_sm80.cuh>
#endif

namespace quda
{
  namespace mma
  {
#if (__COMPUTE_CAPABILITY__ == 700)
    using hmma_t = hmma::hmma_t<16, 16, 4, half, half2>;
#else
    using hmma_t = hmma::hmma_t<16, 8, 8, half, half2>;
#endif

    template <class T>
    struct smma_dispatch { };

    template <>
    struct smma_dispatch <float> {
      using type = smma::smma_t<smma::tfloat32, 4, 1, 1>;
    };

    template <>
    struct smma_dispatch <short> {
      using type = smma::smma_t<smma::bfloat16, 8, 1, 1>;
    };
  }
}
