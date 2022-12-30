#pragma once

// Uncomment this macro to use fp16 accumulation in the HMMA instruction.
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
    template <int m, int n, int k, class compute_t, class load_t> struct hmma_t {
    };
  } // namespace hmma
} // namespace quda

#include <mma_tensor_op/hmma_m16n16k4_sm70.cuh>
#include <mma_tensor_op/hmma_m16n8k8_sm80.cuh>

#include <mma_tensor_op/hmma_tfloat32_sm80.cuh>

#include <mma_tensor_op/smma_m16n8_sm80.cuh>
#include <mma_tensor_op/smma_m16n16k4_sm70.cuh>

#include <mma_tensor_op/simt.cuh>

namespace quda
{
  namespace mma
  {
#if (__COMPUTE_CAPABILITY__ == 700)
    using hmma_t = hmma::hmma_t<16, 16, 4, half, half2>;
#else
    using hmma_t = hmma::hmma_t<16, 8, 8, half, half2>;
#endif

    template <class T> struct smma_dispatch {
    };

    template <> struct smma_dispatch<float> {
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
    };

    template <> struct smma_dispatch<short> {
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
    };
  } // namespace mma
} // namespace quda
