#pragma once

#include <blas_quda.h>

/**
   Host-visible declarations of the per-store BLAS entry points.
   Definitions live in blas_quda_inst.hpp and are explicitly
   instantiated in blas_quda_{double,single,half,quarter}.cu.
   This header must not include blas_quda.hpp (CUDA templates).
*/

namespace quda
{

  namespace blas
  {

    template <typename store_t>
    void axpbyz_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    template <typename store_t>
    void axy_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    void caxpyz_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                  cvector_ref<ColorSpinorField> &z);

    template <typename store_t>
    void caxpby_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                  cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    void axpbypczw_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                     cvector_ref<const ColorSpinorField> &y, cvector<real_t> &c, cvector_ref<const ColorSpinorField> &z,
                     cvector_ref<ColorSpinorField> &w);

    template <typename store_t>
    void caxpbypzw_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z,
                     cvector_ref<ColorSpinorField> &w);

    template <typename store_t>
    void axpyBzpcx_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<real_t> &b, cvector_ref<const ColorSpinorField> &z, cvector<real_t> &c);

    template <typename store_t>
    void axpyZpbx_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                    cvector_ref<const ColorSpinorField> &z, cvector<real_t> &b);

    template <typename store_t>
    void caxpyBzpx_t(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<complex_t> &b, cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    void caxpyBxpz_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<complex_t> &b, cvector_ref<ColorSpinorField> &z);

    template <typename store_t>
    void caxpbypzYmbw_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                        cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                        cvector_ref<const ColorSpinorField> &w);

    template <typename store_t>
    void cabxpyAx_t(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y);

    template <typename store_t>
    void caxpyXmaz_t(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    void caxpyXmazMR_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                       cvector_ref<const ColorSpinorField> &z);

    template <typename store_t>
    void tripleCGUpdate_t(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                          cvector_ref<ColorSpinorField> &w);

  } // namespace blas

} // namespace quda
