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

#include <quda_define.h>

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

    // QUDA_MULTIGRID_MMA_PROLONGATOR modulo 8
    // - 0 DEFAULT
    // - 1 SIMT
    // - 2 SMMA
    // - 3 1xFP16
    // - 4 3xFP16
    // - 5 1xTF32
    // - 6 3xTF32
    // - 7 3xBF16

    template <class T> struct mg_mma_dslash_t {
    };

    // @brief half precision specialization
    template <> struct mg_mma_dslash_t<short> {
#if (QUDA_MULTIGRID_MMA_DSLASH_HALF == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 2)
      using type = typename smma_dispatch<short>::type;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 3)
      using type = hmma_t;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 4)
      using type = smma_half_t;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_HALF == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    // @brief single precision specialization
    template <> struct mg_mma_dslash_t<float> {
#if (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 2)
      using type = typename smma_dispatch<float>::type;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 3)
      using type = hmma_t;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 4)
      using type = smma_half_t;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_DSLASH_SINGLE == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    template <class T> struct mg_mma_prolongator_t {
    };

    // @brief half precision specialization
    template <> struct mg_mma_prolongator_t<short> {
#if (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 2)
      using type = typename smma_dispatch<short>::type;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 3)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
#else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
#endif
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 4)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
#else
      using type = smma_half_t;
#endif
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_HALF == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    // @brief single precision specialization
    template <> struct mg_mma_prolongator_t<float> {
#if (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 2)
      using type = typename smma_dispatch<float>::type;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 3)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
#else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
#endif
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 4)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
#else
      using type = smma_half_t;
#endif
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    template <class T> struct mg_mma_restrictor_t {
    };

    // @brief half precision specialization
    template <> struct mg_mma_restrictor_t<short> {
#if (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 2)
      using type = typename smma_dispatch<short>::type;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 3)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
#else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
#endif
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 4)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
#else
      using type = smma_half_t;
#endif
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_HALF == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    // @brief single precision specialization
    template <> struct mg_mma_restrictor_t<float> {
#if (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 2)
      using type = typename smma_dispatch<float>::type;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 3)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
#else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
#endif
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 4)
#if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
#else
      using type = smma_half_t;
#endif
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE == 0)
#if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
#else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#endif
#endif
    };

    template <class T> struct mg_mma_setup_t {
    };

    // @brief half precision specialization
    template <> struct mg_mma_setup_t<short> {
#if (QUDA_MULTIGRID_MMA_SETUP_HALF == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 2)
      using type = typename smma_dispatch<short>::type;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 3)
      using type = hmma_t;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 4)
      using type = smma_half_t;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_HALF == 0)
      using type = smma_half_t;
#endif
    };

    // @brief single precision specialization
    template <> struct mg_mma_setup_t<float> {
#if (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 2)
      using type = typename smma_dispatch<float>::type;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 3)
      using type = hmma_t;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 4)
      using type = smma_half_t;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif (QUDA_MULTIGRID_MMA_SETUP_SINGLE == 0)
      using type = smma_half_t;
#endif
    };

  } // namespace mma
} // namespace quda
