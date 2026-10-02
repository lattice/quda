#pragma once

#include <blas_quda.h>

/**
   Host-visible declarations of the per-store multi-BLAS entry points.
   Definitions live in multi_blas_quda_inst.hpp and are explicitly
   instantiated in multi_blas_quda_{double,single,half,quarter}.cu.
   This header must not include multi_blas_quda.hpp (CUDA templates).
*/

namespace quda
{

  namespace blas
  {

    namespace block
    {

      template <typename store_t>
      void axpy_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpy_U_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpy_L_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpy_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpy_U_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpy_L_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<ColorSpinorField> &y);

      template <typename store_t>
      void axpyz_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void axpyz_U_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void axpyz_L_t(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void caxpyz_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void caxpyz_U_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void axpyz_L_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

      template <typename store_t>
      void axpyBzpcx_t(const std::vector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                       const std::vector<real_t> &b, ColorSpinorField &z, const std::vector<real_t> &c);

      template <typename store_t>
      void caxpyBxpz_t(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, ColorSpinorField &y,
                       const std::vector<complex_t> &b, ColorSpinorField &z);

    } // namespace block

  } // namespace blas

} // namespace quda
