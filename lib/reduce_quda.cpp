#include <blas_quda.h>
#include <util_quda.h>

namespace quda
{

  namespace blas
  {

    cvector<real_t> max_double(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> max_single(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> max_half(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> max_quarter(cvector_ref<const ColorSpinorField> &x);

    cvector<array<real_t, 2>> max_deviation_double(cvector_ref<const ColorSpinorField> &x,
                                                   cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 2>> max_deviation_single(cvector_ref<const ColorSpinorField> &x,
                                                   cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 2>> max_deviation_half(cvector_ref<const ColorSpinorField> &x,
                                                 cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 2>> max_deviation_quarter(cvector_ref<const ColorSpinorField> &x,
                                                    cvector_ref<const ColorSpinorField> &y);

    cvector<real_t> norm1_double(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm1_single(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm1_half(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm1_quarter(cvector_ref<const ColorSpinorField> &x);

    cvector<real_t> norm2_double(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm2_single(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm2_half(cvector_ref<const ColorSpinorField> &x);
    cvector<real_t> norm2_quarter(cvector_ref<const ColorSpinorField> &x);

    cvector<real_t> reDotProduct_double(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<real_t> reDotProduct_single(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<real_t> reDotProduct_half(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<real_t> reDotProduct_quarter(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);

    cvector<real_t> axpbyzNorm_double(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                      cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> axpbyzNorm_single(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                      cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> axpbyzNorm_half(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> axpbyzNorm_quarter(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                       cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    cvector<real_t> axpyReDot_double(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                     cvector_ref<ColorSpinorField> &y);
    cvector<real_t> axpyReDot_single(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                     cvector_ref<ColorSpinorField> &y);
    cvector<real_t> axpyReDot_half(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                   cvector_ref<ColorSpinorField> &y);
    cvector<real_t> axpyReDot_quarter(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                      cvector_ref<ColorSpinorField> &y);

    cvector<real_t> caxpbyNorm_double(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                      cvector<complex_t> &b, cvector_ref<ColorSpinorField> &y);
    cvector<real_t> caxpbyNorm_single(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                      cvector<complex_t> &b, cvector_ref<ColorSpinorField> &y);
    cvector<real_t> caxpbyNorm_half(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                    cvector_ref<ColorSpinorField> &y);
    cvector<real_t> caxpbyNorm_quarter(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                       cvector<complex_t> &b, cvector_ref<ColorSpinorField> &y);

    cvector<real_t> cabxpyzAxNorm_double(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> cabxpyzAxNorm_single(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> cabxpyzAxNorm_half(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                       cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);
    cvector<real_t> cabxpyzAxNorm_quarter(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                          cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    cvector<complex_t> cDotProduct_double(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<complex_t> cDotProduct_single(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<complex_t> cDotProduct_half(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y);
    cvector<complex_t> cDotProduct_quarter(cvector_ref<const ColorSpinorField> &x,
                                           cvector_ref<const ColorSpinorField> &y);

    cvector<complex_t> caxpyDotzy_double(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z);
    cvector<complex_t> caxpyDotzy_single(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z);
    cvector<complex_t> caxpyDotzy_half(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                       cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z);
    cvector<complex_t> caxpyDotzy_quarter(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                          cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z);

    cvector<array<real_t, 4>> cDotProductNormAB_double(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 4>> cDotProductNormAB_single(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 4>> cDotProductNormAB_half(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y);
    cvector<array<real_t, 4>> cDotProductNormAB_quarter(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y);

    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_double(
      cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z, cvector_ref<const ColorSpinorField> &w,
      cvector_ref<const ColorSpinorField> &v);
    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_single(
      cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z, cvector_ref<const ColorSpinorField> &w,
      cvector_ref<const ColorSpinorField> &v);
    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_half(cvector<complex_t> &a,
                                                                 cvector_ref<const ColorSpinorField> &x,
                                                                 cvector<complex_t> &b, cvector_ref<ColorSpinorField> &y,
                                                                 cvector_ref<ColorSpinorField> &z,
                                                                 cvector_ref<const ColorSpinorField> &w,
                                                                 cvector_ref<const ColorSpinorField> &v);
    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_quarter(
      cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z, cvector_ref<const ColorSpinorField> &w,
      cvector_ref<const ColorSpinorField> &v);

    cvector<array<real_t, 2>> axpyCGNorm_double(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y);
    cvector<array<real_t, 2>> axpyCGNorm_single(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y);
    cvector<array<real_t, 2>> axpyCGNorm_half(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                              cvector_ref<ColorSpinorField> &y);
    cvector<array<real_t, 2>> axpyCGNorm_quarter(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                                 cvector_ref<ColorSpinorField> &y);

    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_double(cvector_ref<const ColorSpinorField> &x,
                                                            cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_single(cvector_ref<const ColorSpinorField> &x,
                                                            cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_half(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_quarter(cvector_ref<const ColorSpinorField> &x,
                                                             cvector_ref<const ColorSpinorField> &r);

    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_double(cvector_ref<const ColorSpinorField> &x,
                                                               cvector_ref<const ColorSpinorField> &y,
                                                               cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_single(cvector_ref<const ColorSpinorField> &x,
                                                               cvector_ref<const ColorSpinorField> &y,
                                                               cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_half(cvector_ref<const ColorSpinorField> &x,
                                                             cvector_ref<const ColorSpinorField> &y,
                                                             cvector_ref<const ColorSpinorField> &r);
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_quarter(cvector_ref<const ColorSpinorField> &x,
                                                                cvector_ref<const ColorSpinorField> &y,
                                                                cvector_ref<const ColorSpinorField> &r);

    cvector<array<real_t, 3>> tripleCGReduction_double(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &y,
                                                       cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 3>> tripleCGReduction_single(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &y,
                                                       cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 3>> tripleCGReduction_half(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y,
                                                     cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 3>> tripleCGReduction_quarter(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y,
                                                        cvector_ref<const ColorSpinorField> &z);

    cvector<array<real_t, 4>> quadrupleCGReduction_double(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &y,
                                                          cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 4>> quadrupleCGReduction_single(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &y,
                                                          cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 4>> quadrupleCGReduction_half(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y,
                                                        cvector_ref<const ColorSpinorField> &z);
    cvector<array<real_t, 4>> quadrupleCGReduction_quarter(cvector_ref<const ColorSpinorField> &x,
                                                           cvector_ref<const ColorSpinorField> &y,
                                                           cvector_ref<const ColorSpinorField> &z);

    cvector<real_t> quadrupleCG3InitNorm_double(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3InitNorm_single(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3InitNorm_half(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                              cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                              cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3InitNorm_quarter(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                                 cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                 cvector_ref<ColorSpinorField> &w,
                                                 cvector_ref<const ColorSpinorField> &v);

    cvector<real_t> quadrupleCG3UpdateNorm_double(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                                  cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                  cvector_ref<ColorSpinorField> &w,
                                                  cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3UpdateNorm_single(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                                  cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                  cvector_ref<ColorSpinorField> &w,
                                                  cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3UpdateNorm_half(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v);
    cvector<real_t> quadrupleCG3UpdateNorm_quarter(cvector<real_t> &a, cvector<real_t> &b,
                                                   cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                                                   cvector_ref<ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w,
                                                   cvector_ref<const ColorSpinorField> &v);

    template <typename T, typename DoubleFn, typename SingleFn, typename HalfFn, typename QuarterFn>
    T dispatch_reduce_prec(QudaPrecision prec, DoubleFn &&d, SingleFn &&s, HalfFn &&h, QuarterFn &&q)
    {
      switch (prec) {
      case QUDA_DOUBLE_PRECISION: return d();
      case QUDA_SINGLE_PRECISION: return s();
      case QUDA_HALF_PRECISION: return h();
      case QUDA_QUARTER_PRECISION: return q();
      default: errorQuda("Unsupported precision %d", prec);
      }
      return {};
    }

    cvector<real_t> max(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return max_double(x); }, [&] { return max_single(x); }, [&] { return max_half(x); },
        [&] { return max_quarter(x); });
    }

    cvector<array<real_t, 2>> max_deviation(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 2>>>(
        x.Precision(), [&] { return max_deviation_double(x, y); }, [&] { return max_deviation_single(x, y); },
        [&] { return max_deviation_half(x, y); }, [&] { return max_deviation_quarter(x, y); });
    }

    cvector<real_t> norm1(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return norm1_double(x); }, [&] { return norm1_single(x); }, [&] { return norm1_half(x); },
        [&] { return norm1_quarter(x); });
    }

    cvector<real_t> norm2(cvector_ref<const ColorSpinorField> &x)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return norm2_double(x); }, [&] { return norm2_single(x); }, [&] { return norm2_half(x); },
        [&] { return norm2_quarter(x); });
    }

    cvector<real_t> reDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return reDotProduct_double(x, y); }, [&] { return reDotProduct_single(x, y); },
        [&] { return reDotProduct_half(x, y); }, [&] { return reDotProduct_quarter(x, y); });
    }

    cvector<real_t> axpbyzNorm(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                               cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return axpbyzNorm_double(a, x, b, y, z); },
        [&] { return axpbyzNorm_single(a, x, b, y, z); }, [&] { return axpbyzNorm_half(a, x, b, y, z); },
        [&] { return axpbyzNorm_quarter(a, x, b, y, z); });
    }

    cvector<real_t> axpyReDot(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return axpyReDot_double(a, x, y); }, [&] { return axpyReDot_single(a, x, y); },
        [&] { return axpyReDot_half(a, x, y); }, [&] { return axpyReDot_quarter(a, x, y); });
    }

    cvector<real_t> caxpbyNorm(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                               cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return caxpbyNorm_double(a, x, b, y); }, [&] { return caxpbyNorm_single(a, x, b, y); },
        [&] { return caxpbyNorm_half(a, x, b, y); }, [&] { return caxpbyNorm_quarter(a, x, b, y); });
    }

    cvector<real_t> cabxpyzAxNorm(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return cabxpyzAxNorm_double(a, b, x, y, z); },
        [&] { return cabxpyzAxNorm_single(a, b, x, y, z); }, [&] { return cabxpyzAxNorm_half(a, b, x, y, z); },
        [&] { return cabxpyzAxNorm_quarter(a, b, x, y, z); });
    }

    cvector<complex_t> cDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<complex_t>>(
        x.Precision(), [&] { return cDotProduct_double(x, y); }, [&] { return cDotProduct_single(x, y); },
        [&] { return cDotProduct_half(x, y); }, [&] { return cDotProduct_quarter(x, y); });
    }

    cvector<complex_t> caxpyDotzy(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                  cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec<cvector<complex_t>>(
        x.Precision(), [&] { return caxpyDotzy_double(a, x, y, z); }, [&] { return caxpyDotzy_single(a, x, y, z); },
        [&] { return caxpyDotzy_half(a, x, y, z); }, [&] { return caxpyDotzy_quarter(a, x, y, z); });
    }

    cvector<array<real_t, 4>> cDotProductNormAB(cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<const ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 4>>>(
        x.Precision(), [&] { return cDotProductNormAB_double(x, y); }, [&] { return cDotProductNormAB_single(x, y); },
        [&] { return cDotProductNormAB_half(x, y); }, [&] { return cDotProductNormAB_quarter(x, y); });
    }

    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                                            cvector<complex_t> &b, cvector_ref<ColorSpinorField> &y,
                                                            cvector_ref<ColorSpinorField> &z,
                                                            cvector_ref<const ColorSpinorField> &w,
                                                            cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 3>>>(
        x.Precision(), [&] { return caxpbypzYmbwcDotProductUYNormY_double(a, x, b, y, z, w, v); },
        [&] { return caxpbypzYmbwcDotProductUYNormY_single(a, x, b, y, z, w, v); },
        [&] { return caxpbypzYmbwcDotProductUYNormY_half(a, x, b, y, z, w, v); },
        [&] { return caxpbypzYmbwcDotProductUYNormY_quarter(a, x, b, y, z, w, v); });
    }

    cvector<array<real_t, 2>> axpyCGNorm(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 2>>>(
        x.Precision(), [&] { return axpyCGNorm_double(a, x, y); }, [&] { return axpyCGNorm_single(a, x, y); },
        [&] { return axpyCGNorm_half(a, x, y); }, [&] { return axpyCGNorm_quarter(a, x, y); });
    }

    cvector<array<real_t, 3>> HeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &r)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 3>>>(
        x.Precision(), [&] { return HeavyQuarkResidualNorm_double(x, r); },
        [&] { return HeavyQuarkResidualNorm_single(x, r); }, [&] { return HeavyQuarkResidualNorm_half(x, r); },
        [&] { return HeavyQuarkResidualNorm_quarter(x, r); });
    }

    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y,
                                                        cvector_ref<const ColorSpinorField> &r)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 3>>>(
        x.Precision(), [&] { return xpyHeavyQuarkResidualNorm_double(x, y, r); },
        [&] { return xpyHeavyQuarkResidualNorm_single(x, y, r); },
        [&] { return xpyHeavyQuarkResidualNorm_half(x, y, r); },
        [&] { return xpyHeavyQuarkResidualNorm_quarter(x, y, r); });
    }

    cvector<array<real_t, 3>> tripleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                                cvector_ref<const ColorSpinorField> &y,
                                                cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 3>>>(
        x.Precision(), [&] { return tripleCGReduction_double(x, y, z); },
        [&] { return tripleCGReduction_single(x, y, z); }, [&] { return tripleCGReduction_half(x, y, z); },
        [&] { return tripleCGReduction_quarter(x, y, z); });
    }

    cvector<array<real_t, 4>> quadrupleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                                   cvector_ref<const ColorSpinorField> &y,
                                                   cvector_ref<const ColorSpinorField> &z)
    {
      return dispatch_reduce_prec<cvector<array<real_t, 4>>>(
        x.Precision(), [&] { return quadrupleCGReduction_double(x, y, z); },
        [&] { return quadrupleCGReduction_single(x, y, z); }, [&] { return quadrupleCGReduction_half(x, y, z); },
        [&] { return quadrupleCGReduction_quarter(x, y, z); });
    }

    cvector<real_t> quadrupleCG3InitNorm(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                         cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return quadrupleCG3InitNorm_double(a, x, y, z, w, v); },
        [&] { return quadrupleCG3InitNorm_single(a, x, y, z, w, v); },
        [&] { return quadrupleCG3InitNorm_half(a, x, y, z, w, v); },
        [&] { return quadrupleCG3InitNorm_quarter(a, x, y, z, w, v); });
    }

    cvector<real_t> quadrupleCG3UpdateNorm(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return dispatch_reduce_prec<cvector<real_t>>(
        x.Precision(), [&] { return quadrupleCG3UpdateNorm_double(a, b, x, y, z, w, v); },
        [&] { return quadrupleCG3UpdateNorm_single(a, b, x, y, z, w, v); },
        [&] { return quadrupleCG3UpdateNorm_half(a, b, x, y, z, w, v); },
        [&] { return quadrupleCG3UpdateNorm_quarter(a, b, x, y, z, w, v); });
    }

  } // namespace blas

} // namespace quda
