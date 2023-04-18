#pragma once

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
#include <mma_tensor_op/simt_half.cuh>

namespace quda
{
  namespace mma
  {
#if (__COMPUTE_CAPABILITY__ == 700)
    using hmma_t = hmma::hmma_t<16, 16, 4, half, half2>;
    using smma_half_t = smma::smma_t<half, 4, 1, 1>;
#else
    using hmma_t = hmma::hmma_t<16, 8, 8, half, half2>;
    using smma_half_t = smma::smma_t<half, 8, 1, 1>;
#endif

#if (__COMPUTE_CAPABILITY__ >= 800)
    template <class T> struct smma_dispatch {
    };

    template <> struct smma_dispatch<float> {
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
    };

    template <> struct smma_dispatch<short> {
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
    };
#else
    template <class T> struct smma_dispatch {
      using type = smma_half_t;
    };
#endif

    template <class T> struct mg_mma_dispatch_t {
#ifdef QUDA_MULTIGRID_SETUP_USE_SMMA
      using type = typename smma_dispatch<T>::type; // 3xBF16/3xTF32
#else
      using type = hmma_t;
#endif
      // using type = smma_half_t;                          // 3xFP16
      // using type = smma::smma_t<mma::tfloat32, 4, 1, 1>; // 3xTF32
      // using type = smma::smma_t<mma::bfloat32, 8, 1, 1>; // 3xBF16
      // using type = simt::simt_t<float, 8, 4, 2, 2>;      // SIMT
    };

  } // namespace mma
} // namespace quda
