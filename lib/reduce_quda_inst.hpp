#pragma once

/**
   Per-store reduction entry points. Included only by the precision
   .cu files. Host dispatch in reduce_quda.cpp sees declarations only
   (reduce_quda_decl.hpp) and must not instantiate these CUDA templates.
*/

#include "reduce_store.hpp"
#include "reduce_quda.hpp"
#include "reduce_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    template <typename store_t> cvector<real_t> max_t(cvector_ref<const ColorSpinorField> &x)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return max_impl<store_t>(x);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 2>> max_deviation_t(cvector_ref<const ColorSpinorField> &x,
                                              cvector_ref<const ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return max_deviation_impl<store_t>(x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t> cvector<real_t> norm1_t(cvector_ref<const ColorSpinorField> &x)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return norm1_impl<store_t>(x);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t> cvector<real_t> norm2_t(cvector_ref<const ColorSpinorField> &x)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return norm2_impl<store_t>(x);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> reDotProduct_t(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return reDotProduct_impl<store_t>(x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> axpbyzNorm_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return axpbyzNorm_impl<store_t>(a, x, b, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> axpyReDot_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return axpyReDot_impl<store_t>(a, x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> caxpbyNorm_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                 cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return caxpbyNorm_impl<store_t>(a, x, b, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> cabxpyzAxNorm_t(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return cabxpyzAxNorm_impl<store_t>(a, b, x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<complex_t> cDotProduct_t(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return cDotProduct_impl<store_t>(x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<complex_t> caxpyDotzy_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                    cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return caxpyDotzy_impl<store_t>(a, x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 4>> cDotProductNormAB_t(cvector_ref<const ColorSpinorField> &x,
                                                  cvector_ref<const ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return cDotProductNormAB_impl<store_t>(x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 3>>
    caxpbypzYmbwcDotProductUYNormY_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                     cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                     cvector_ref<const ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return caxpbypzYmbwcDotProductUYNormY_impl<store_t>(a, x, b, y, z, w, v);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 2>> axpyCGNorm_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return axpyCGNorm_impl<store_t>(a, x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_t(cvector_ref<const ColorSpinorField> &x,
                                                       cvector_ref<const ColorSpinorField> &r)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return HeavyQuarkResidualNorm_impl<store_t>(x, r);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_t(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &y,
                                                          cvector_ref<const ColorSpinorField> &r)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return xpyHeavyQuarkResidualNorm_impl<store_t>(x, y, r);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 3>> tripleCGReduction_t(cvector_ref<const ColorSpinorField> &x,
                                                  cvector_ref<const ColorSpinorField> &y,
                                                  cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return tripleCGReduction_impl<store_t>(x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<array<real_t, 4>> quadrupleCGReduction_t(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y,
                                                     cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return quadrupleCGReduction_impl<store_t>(x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> quadrupleCG3InitNorm_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return quadrupleCG3InitNorm_impl<store_t>(a, x, y, z, w, v);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    template <typename store_t>
    cvector<real_t> quadrupleCG3UpdateNorm_t(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                             cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                             cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      if constexpr (reduce_prec_enabled<store_t>()) {
        return quadrupleCG3UpdateNorm_impl<store_t>(a, b, x, y, z, w, v);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        return {};
      }
    }

    // Repeat instantiations only; names and bodies are the templates above.
#define INSTANTIATE_REDUCE_STORE(store_t)                                                                               \
  template cvector<real_t> max_t<store_t>(cvector_ref<const ColorSpinorField> &);                                       \
  template cvector<array<real_t, 2>> max_deviation_t<store_t>(cvector_ref<const ColorSpinorField> &,                    \
                                                              cvector_ref<const ColorSpinorField> &);                   \
  template cvector<real_t> norm1_t<store_t>(cvector_ref<const ColorSpinorField> &);                                     \
  template cvector<real_t> norm2_t<store_t>(cvector_ref<const ColorSpinorField> &);                                     \
  template cvector<real_t> reDotProduct_t<store_t>(cvector_ref<const ColorSpinorField> &,                               \
                                                   cvector_ref<const ColorSpinorField> &);                              \
  template cvector<real_t> axpbyzNorm_t<store_t>(cvector<real_t> &, cvector_ref<const ColorSpinorField> &,              \
                                                 cvector<real_t> &, cvector_ref<const ColorSpinorField> &,              \
                                                 cvector_ref<ColorSpinorField> &);                                      \
  template cvector<real_t> axpyReDot_t<store_t>(cvector<real_t> &, cvector_ref<const ColorSpinorField> &,               \
                                                cvector_ref<ColorSpinorField> &);                                       \
  template cvector<real_t> caxpbyNorm_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,           \
                                                 cvector<complex_t> &, cvector_ref<ColorSpinorField> &);                \
  template cvector<real_t> cabxpyzAxNorm_t<store_t>(                                                                    \
    cvector<real_t> &, cvector<complex_t> &, cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,    \
    cvector_ref<ColorSpinorField> &);                                                                                   \
  template cvector<complex_t> cDotProduct_t<store_t>(cvector_ref<const ColorSpinorField> &,                             \
                                                     cvector_ref<const ColorSpinorField> &);                            \
  template cvector<complex_t> caxpyDotzy_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,        \
                                                    cvector_ref<ColorSpinorField> &,                                    \
                                                    cvector_ref<const ColorSpinorField> &);                             \
  template cvector<array<real_t, 4>> cDotProductNormAB_t<store_t>(cvector_ref<const ColorSpinorField> &,                \
                                                                  cvector_ref<const ColorSpinorField> &);               \
  template cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_t<store_t>(                                         \
    cvector<complex_t> &, cvector_ref<const ColorSpinorField> &, cvector<complex_t> &, cvector_ref<ColorSpinorField> &, \
    cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);     \
  template cvector<array<real_t, 2>> axpyCGNorm_t<store_t>(cvector<real_t> &, cvector_ref<const ColorSpinorField> &,    \
                                                           cvector_ref<ColorSpinorField> &);                            \
  template cvector<array<real_t, 3>> HeavyQuarkResidualNorm_t<store_t>(cvector_ref<const ColorSpinorField> &,           \
                                                                       cvector_ref<const ColorSpinorField> &);          \
  template cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_t<store_t>(cvector_ref<const ColorSpinorField> &,        \
                                                                          cvector_ref<const ColorSpinorField> &,        \
                                                                          cvector_ref<const ColorSpinorField> &);       \
  template cvector<array<real_t, 3>> tripleCGReduction_t<store_t>(cvector_ref<const ColorSpinorField> &,                \
                                                                  cvector_ref<const ColorSpinorField> &,                \
                                                                  cvector_ref<const ColorSpinorField> &);               \
  template cvector<array<real_t, 4>> quadrupleCGReduction_t<store_t>(cvector_ref<const ColorSpinorField> &,             \
                                                                     cvector_ref<const ColorSpinorField> &,             \
                                                                     cvector_ref<const ColorSpinorField> &);            \
  template cvector<real_t> quadrupleCG3InitNorm_t<store_t>(                                                             \
    cvector<real_t> &, cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &,                                \
    cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);           \
  template cvector<real_t> quadrupleCG3UpdateNorm_t<store_t>(                                                           \
    cvector<real_t> &, cvector<real_t> &, cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &,             \
    cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);

  } // namespace blas

} // namespace quda
