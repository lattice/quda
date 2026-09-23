#pragma once

/**
   Per-store multi-BLAS entry points. Included only by the precision
   .cu files. Host dispatch in multi_blas_quda.cpp sees declarations
   only (multi_blas_quda_decl.hpp) and must not instantiate these
   CUDA templates.
*/

#include "reduce_store.hpp"
#include "multi_blas_quda.hpp"
#include "multi_blas_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      template <typename store_t>
      void axpy_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpy_U_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_U_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpy_L_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_L_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpy_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpy_U_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_U_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpy_L_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpy_L_impl<store_t>(a, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpyz_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpyz_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpyz_U_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpyz_U_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpyz_L_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpyz_L_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void caxpyz_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          caxpyz_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void caxpyz_U_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          caxpyz_U_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpyz_L_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpyz_L_impl<store_t>(a, x, y, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void axpyBzpcx_t(const std::vector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                       const std::vector<real_t> &b, ColorSpinorField &z, const std::vector<real_t> &c)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          axpyBzpcx_impl<store_t>(a, x, y, b, z, c);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void caxpyBxpz_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, ColorSpinorField &y,
                       const std::vector<complex_t> &b, ColorSpinorField &z)
      {
        if constexpr (multi_blas_prec_enabled<store_t>()) {
          caxpyBxpz_impl<store_t>(a, x, y, b, z);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      // Repeat instantiations only; names and bodies are the templates above.
#define INSTANTIATE_MULTI_BLAS_STORE(store_t)                                                                          \
  template void axpy_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                    \
                                cvector_ref<ColorSpinorField> &);                                                      \
  template void axpy_U_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                  \
                                  cvector_ref<ColorSpinorField> &);                                                    \
  template void axpy_L_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                  \
                                  cvector_ref<ColorSpinorField> &);                                                    \
  template void axpy_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,                 \
                                cvector_ref<ColorSpinorField> &);                                                      \
  template void axpy_U_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,               \
                                  cvector_ref<ColorSpinorField> &);                                                    \
  template void axpy_L_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,               \
                                  cvector_ref<ColorSpinorField> &);                                                    \
  template void axpyz_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                   \
                                 cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);              \
  template void axpyz_U_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                 \
                                   cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);            \
  template void axpyz_L_t<store_t>(const std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                 \
                                   cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);            \
  template void caxpyz_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,               \
                                  cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);             \
  template void caxpyz_U_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,             \
                                    cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);           \
  template void axpyz_L_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,              \
                                   cvector_ref<const ColorSpinorField> &, cvector_ref<ColorSpinorField> &);            \
  template void axpyBzpcx_t<store_t>(const std::vector<real_t> &, cvector_ref<ColorSpinorField> &,                     \
                                     cvector_ref<ColorSpinorField> &, const std::vector<real_t> &, ColorSpinorField &, \
                                     const std::vector<real_t> &);                                                    \
  template void caxpyBxpz_t<store_t>(const std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,            \
                                     ColorSpinorField &, const std::vector<complex_t> &, ColorSpinorField &);

    } // namespace block

  } // namespace blas

} // namespace quda
