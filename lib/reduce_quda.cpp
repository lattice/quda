#include "reduce_store.hpp"
#include "reduce_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    cvector<real_t> max(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return max_t<store_t>(x); });
    }

    cvector<array<real_t, 2>> max_deviation(cvector_ref<const ColorSpinorField> &x,
                                            cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return max_deviation_t<store_t>(x, y); });
    }

    cvector<real_t> norm1(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return norm1_t<store_t>(x); });
    }

    cvector<real_t> norm2(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return norm2_t<store_t>(x); });
    }

    cvector<real_t> reDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return reDotProduct_t<store_t>(x, y); });
    }

    cvector<real_t> axpbyzNorm(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                               cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return axpbyzNorm_t<store_t>(a, x, b, y, z); });
    }

    cvector<real_t> axpyReDot(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                              cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return axpyReDot_t<store_t>(a, x, y); });
    }

    cvector<real_t> caxpbyNorm(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                               cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return caxpbyNorm_t<store_t>(a, x, b, y); });
    }

    cvector<real_t> cabxpyzAxNorm(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return cabxpyzAxNorm_t<store_t>(a, b, x, y, z); });
    }

    cvector<complex_t> cDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return cDotProduct_t<store_t>(x, y); });
    }

    cvector<complex_t> caxpyDotzy(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                  cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return caxpyDotzy_t<store_t>(a, x, y, z); });
    }

    cvector<array<real_t, 4>> cDotProductNormAB(cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return cDotProductNormAB_t<store_t>(x, y); });
    }

    cvector<array<real_t, 3>>
    caxpbypzYmbwcDotProductUYNormY(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                   cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                   cvector_ref<const ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() {
        return caxpbypzYmbwcDotProductUYNormY_t<store_t>(a, x, b, y, z, w, v);
      });
    }

    cvector<array<real_t, 2>> axpyCGNorm(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { return axpyCGNorm_t<store_t>(a, x, y); });
    }

    cvector<array<real_t, 3>> HeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &r)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return HeavyQuarkResidualNorm_t<store_t>(x, r); });
    }

    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y,
                                                        cvector_ref<const ColorSpinorField> &r)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return xpyHeavyQuarkResidualNorm_t<store_t>(x, y, r); });
    }

    cvector<array<real_t, 3>> tripleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<const ColorSpinorField> &y,
                                                cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return tripleCGReduction_t<store_t>(x, y, z); });
    }

    cvector<array<real_t, 4>> quadrupleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                                   cvector_ref<const ColorSpinorField> &y,
                                                   cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return quadrupleCGReduction_t<store_t>(x, y, z); });
    }

    cvector<real_t> quadrupleCG3InitNorm(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                         cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec(x.Precision(),
                                  [&]<typename store_t>() { return quadrupleCG3InitNorm_t<store_t>(a, x, y, z, w, v); });
    }

    cvector<real_t> quadrupleCG3UpdateNorm(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec(
        x.Precision(), [&]<typename store_t>() { return quadrupleCG3UpdateNorm_t<store_t>(a, b, x, y, z, w, v); });
    }

  } // namespace blas

} // namespace quda
