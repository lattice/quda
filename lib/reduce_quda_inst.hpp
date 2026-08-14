#pragma once

#include <type_traits>

/**
   Instantiation helpers for a single x-field store type.
   Each reduce_quda_*.cu defines REDUCE_SUFFIX, REDUCE_PREC, REDUCE_STORE
   and includes this file.
*/

#include "reduce_quda.hpp"

namespace quda
{

  namespace blas
  {

#define REDUCE_CONCAT_I(a, b) a##_##b
#define REDUCE_CONCAT(a, b) REDUCE_CONCAT_I(a, b)
#define REDUCE_NAME(name) REDUCE_CONCAT(name, REDUCE_SUFFIX)

    // Double stays instantiable for host fields even when GPU double is off
    // (same convention as instantiate<> in blas_helper.cuh).
    template <typename store_t> constexpr bool reduce_prec_enabled(QudaPrecision prec)
    {
      if constexpr (std::is_same_v<store_t, double>) return true;
      return is_enabled(prec);
    }

    // `call` must be parenthesized so commas in the expression stay one macro argument.
#define REDUCE_WRAP(name, args, call)                                                                                  \
  auto REDUCE_NAME(name) args->decltype call                                                                           \
  {                                                                                                                    \
    if constexpr (reduce_prec_enabled<REDUCE_STORE>(REDUCE_PREC)) {                                                    \
      /* Init bins in this .cu; ReduceArg's header ctor can be emitted in another TU. */                               \
      reducer::init_rfa_device_bins<device_reduce_t>();                                                                \
      return call;                                                                                                     \
    } else {                                                                                                           \
      errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);                          \
      return {};                                                                                                       \
    }                                                                                                                  \
  }

    REDUCE_WRAP(max, (cvector_ref<const ColorSpinorField> &x), (max_impl<REDUCE_STORE>(x)))

    REDUCE_WRAP(max_deviation, (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y),
                (max_deviation_impl<REDUCE_STORE>(x, y)))

    REDUCE_WRAP(norm1, (cvector_ref<const ColorSpinorField> &x), (norm1_impl<REDUCE_STORE>(x)))

    REDUCE_WRAP(norm2, (cvector_ref<const ColorSpinorField> &x), (norm2_impl<REDUCE_STORE>(x)))

    REDUCE_WRAP(reDotProduct, (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y),
                (reDotProduct_impl<REDUCE_STORE>(x, y)))

    REDUCE_WRAP(axpbyzNorm,
                (cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z),
                (axpbyzNorm_impl<REDUCE_STORE>(a, x, b, y, z)))

    REDUCE_WRAP(axpyReDot, (cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y),
                (axpyReDot_impl<REDUCE_STORE>(a, x, y)))

    REDUCE_WRAP(caxpbyNorm,
                (cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                 cvector_ref<ColorSpinorField> &y),
                (caxpbyNorm_impl<REDUCE_STORE>(a, x, b, y)))

    REDUCE_WRAP(cabxpyzAxNorm,
                (cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z),
                (cabxpyzAxNorm_impl<REDUCE_STORE>(a, b, x, y, z)))

    REDUCE_WRAP(cDotProduct, (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y),
                (cDotProduct_impl<REDUCE_STORE>(x, y)))

    REDUCE_WRAP(caxpyDotzy,
                (cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                 cvector_ref<const ColorSpinorField> &z),
                (caxpyDotzy_impl<REDUCE_STORE>(a, x, y, z)))

    REDUCE_WRAP(cDotProductNormAB, (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y),
                (cDotProductNormAB_impl<REDUCE_STORE>(x, y)))

    REDUCE_WRAP(caxpbypzYmbwcDotProductUYNormY,
                (cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                 cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                 cvector_ref<const ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v),
                (caxpbypzYmbwcDotProductUYNormY_impl<REDUCE_STORE>(a, x, b, y, z, w, v)))

    REDUCE_WRAP(axpyCGNorm, (cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y),
                (axpyCGNorm_impl<REDUCE_STORE>(a, x, y)))

    REDUCE_WRAP(HeavyQuarkResidualNorm,
                (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &r),
                (HeavyQuarkResidualNorm_impl<REDUCE_STORE>(x, r)))

    REDUCE_WRAP(xpyHeavyQuarkResidualNorm,
                (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                 cvector_ref<const ColorSpinorField> &r),
                (xpyHeavyQuarkResidualNorm_impl<REDUCE_STORE>(x, y, r)))

    REDUCE_WRAP(tripleCGReduction,
                (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                 cvector_ref<const ColorSpinorField> &z),
                (tripleCGReduction_impl<REDUCE_STORE>(x, y, z)))

    REDUCE_WRAP(quadrupleCGReduction,
                (cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                 cvector_ref<const ColorSpinorField> &z),
                (quadrupleCGReduction_impl<REDUCE_STORE>(x, y, z)))

    REDUCE_WRAP(quadrupleCG3InitNorm,
                (cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                 cvector_ref<ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w,
                 cvector_ref<const ColorSpinorField> &v),
                (quadrupleCG3InitNorm_impl<REDUCE_STORE>(a, x, y, z, w, v)))

    REDUCE_WRAP(quadrupleCG3UpdateNorm,
                (cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                 cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w,
                 cvector_ref<const ColorSpinorField> &v),
                (quadrupleCG3UpdateNorm_impl<REDUCE_STORE>(a, b, x, y, z, w, v)))

#undef REDUCE_WRAP
#undef REDUCE_NAME
#undef REDUCE_CONCAT
#undef REDUCE_CONCAT_I

  } // namespace blas

} // namespace quda
