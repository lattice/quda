#pragma once

/**
   Per-store BLAS entry points. Included only by the precision .cu
   files. Host dispatch in blas_quda.cpp sees declarations only
   (blas_quda_decl.hpp) and must not instantiate these CUDA templates.
*/

#include "reduce_store.hpp"
#include "blas_quda.hpp"
#include "blas_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    template <typename store_t>
    void axpbyz_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        axpbyz_impl<store_t>(a, x, b, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void axy_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        axy_impl<store_t>(a, x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpyz_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                  cvector_ref<ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpyz_impl<store_t>(a, x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpby_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                  cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpby_impl<store_t>(a, x, b, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void axpbypczw_t(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                     cvector_ref<const ColorSpinorField> &y, cvector<real_t> &c, cvector_ref<const ColorSpinorField> &z,
                     cvector_ref<ColorSpinorField> &w)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        axpbypczw_impl<store_t>(a, x, b, y, c, z, w);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpbypzw_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z,
                     cvector_ref<ColorSpinorField> &w)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpbypzw_impl<store_t>(a, x, b, y, z, w);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void axpyBzpcx_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<real_t> &b, cvector_ref<const ColorSpinorField> &z, cvector<real_t> &c)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        axpyBzpcx_impl<store_t>(a, x, y, b, z, c);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void axpyZpbx_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                    cvector_ref<const ColorSpinorField> &z, cvector<real_t> &b)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        axpyZpbx_impl<store_t>(a, x, y, z, b);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpyBzpx_t(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<complex_t> &b, cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpyBzpx_impl<store_t>(a, x, y, b, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpyBxpz_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector<complex_t> &b, cvector_ref<ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpyBxpz_impl<store_t>(a, x, y, b, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpbypzYmbw_t(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                        cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                        cvector_ref<const ColorSpinorField> &w)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpbypzYmbw_impl<store_t>(a, x, b, y, z, w);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void cabxpyAx_t(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        cabxpyAx_impl<store_t>(a, b, x, y);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpyXmaz_t(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpyXmaz_impl<store_t>(a, x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void caxpyXmazMR_t(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                       cvector_ref<const ColorSpinorField> &z)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        caxpyXmazMR_impl<store_t>(a, x, y, z);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    template <typename store_t>
    void tripleCGUpdate_t(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                          cvector_ref<ColorSpinorField> &w)
    {
      if constexpr (blas_prec_enabled<store_t>()) {
        tripleCGUpdate_impl<store_t>(a, b, x, y, z, w);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
      }
    }

    // Repeat instantiations only; names and bodies are the templates above.
#define INSTANTIATE_BLAS_STORE(store_t)                                                                                \
  template void axpbyz_t<store_t>(cvector<real_t> &, cvector_ref<const ColorSpinorField> &, cvector<real_t> &,          \
                                  cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);             \
  template void axy_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                             \
                               cvector_ref<ColorSpinorField> &);                                                       \
  template void caxpyz_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                          \
                                  cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);             \
  template void caxpby_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &, cvector<complex_t> &,    \
                                  cvector_ref<ColorSpinorField> &);                                                    \
  template void axpbypczw_t<store_t>(cvector<real_t> &, cvector_ref<const ColorSpinorField> &, cvector<real_t> &,       \
                                     cvector_ref<const ColorSpinorField> &, cvector<real_t> &,                         \
                                     cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);          \
  template void caxpbypzw_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                       \
                                     cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                      \
                                     cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);          \
  template void axpyBzpcx_t<store_t>(cvector<real_t> &, cvector_ref<ColorSpinorField> &,                                \
                                     cvector_ref<ColorSpinorField> &, cvector<real_t> &,                               \
                                     cvector_ref<const ColorSpinorField> &, cvector<real_t> &);                        \
  template void axpyZpbx_t<store_t>(cvector<real_t> &, cvector_ref<ColorSpinorField> &,                                 \
                                    cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,            \
                                    cvector<real_t> &);                                                                \
  template void caxpyBzpx_t<store_t>(cvector<complex_t> &, cvector_ref<ColorSpinorField> &,                             \
                                     cvector_ref<ColorSpinorField> &, cvector<complex_t> &,                            \
                                     cvector_ref<const ColorSpinorField> &);                                           \
  template void caxpyBxpz_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                       \
                                     cvector_ref<ColorSpinorField> &, cvector<complex_t> &,                            \
                                     cvector_ref<ColorSpinorField> &);                                                 \
  template void caxpbypzYmbw_t<store_t>(cvector<complex_t> &, cvector_ref<const ColorSpinorField> &,                    \
                                        cvector<complex_t> &, cvector_ref<ColorSpinorField> &,                         \
                                        cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);       \
  template void cabxpyAx_t<store_t>(cvector<real_t> &, cvector<complex_t> &, cvector_ref<ColorSpinorField> &,           \
                                    cvector_ref<ColorSpinorField> &);                                                  \
  template void caxpyXmaz_t<store_t>(cvector<complex_t> &, cvector_ref<ColorSpinorField> &,                             \
                                     cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);          \
  template void caxpyXmazMR_t<store_t>(cvector<real_t> &, cvector_ref<ColorSpinorField> &,                              \
                                       cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &);        \
  template void tripleCGUpdate_t<store_t>(cvector<real_t> &, cvector<real_t> &, cvector_ref<const ColorSpinorField> &,  \
                                          cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &,            \
                                          cvector_ref<ColorSpinorField> &);

  } // namespace blas

} // namespace quda
