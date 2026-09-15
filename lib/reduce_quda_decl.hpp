#pragma once

#include <blas_quda.h>

/**
   Host-visible declarations of the per-store reduction entry points.
   Definitions live in reduce_quda_inst.hpp and are explicitly
   instantiated in reduce_quda_{double,single,half,quarter}.cu.
   This header must not include reduce_quda.hpp (CUDA templates).
*/

namespace quda
{

  namespace blas
  {

    template <typename store_t> cvector<real_t> max_t(cvector_ref<const ColorSpinorField> &x);

    template <typename store_t>
    cvector<array<real_t, 2>> max_deviation_t(cvector_ref<const ColorSpinorField> &x,
                                              cvector_ref<const ColorSpinorField> &y);

    template <typename store_t> cvector<real_t> norm1_t(cvector_ref<const ColorSpinorField> &x);

    template <typename store_t> cvector<real_t> norm2_t(cvector_ref<const ColorSpinorField> &x);

    template <typename store_t>
    cvector<real_t> reDotProduct_t(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);

    template <typename store_t>
    cvector<real_t> axpbyzNorm_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    template <typename store_t>
    cvector<real_t> axpyReDot_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    cvector<real_t> caxpbyNorm_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                 cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    cvector<real_t> cabxpyzAxNorm_t(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    template <typename store_t>
    cvector<complex_t> cDotProduct_t(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);

    template <typename store_t>
    cvector<complex_t> caxpyDotzy_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                    cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    cvector<array<real_t, 4>> cDotProductNormAB_t(cvector_ref<const ColorSpinorField> &x,
                                                  cvector_ref<const ColorSpinorField> &y);

    template <typename store_t>
    cvector<array<real_t, 3>>
    caxpbypzYmbwcDotProductUYNormY_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                     cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                     cvector_ref<const ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);

    template <typename store_t>
    cvector<array<real_t, 2>> axpyCGNorm_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_t(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &r);

    template <typename store_t>
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_t(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &y,
                                                          cvector_ref<const ColorSpinorField> &r);

    template <typename store_t>
    cvector<array<real_t, 3>> tripleCGReduction_t(cvector_ref<const ColorSpinorField> &x,
                                                  cvector_ref<const ColorSpinorField> &y,
                                                  cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    cvector<array<real_t, 4>> quadrupleCGReduction_t(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y,
                                                     cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    cvector<real_t> quadrupleCG3InitNorm_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);

    template <typename store_t>
    cvector<real_t> quadrupleCG3UpdateNorm_t(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                             cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                             cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);

  } // namespace blas

} // namespace quda
