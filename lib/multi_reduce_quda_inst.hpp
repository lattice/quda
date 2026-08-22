#pragma once

/**
   Per-store multi-reduce entry points. Included only by the precision
   .cu files. Host dispatch in multi_reduce_quda.cpp sees declarations
   only (multi_reduce_quda_decl.hpp) and must not instantiate these
   CUDA templates.
*/

#include "reduce_store.hpp"
#include "multi_reduce_quda.hpp"
#include "multi_reduce_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      template <typename store_t>
      void reDotProduct_t(std::vector<real_t> &result, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<const ColorSpinorField> &y)
      {
        if constexpr (multi_reduce_prec_enabled<store_t>()) {
          reDotProduct_impl<store_t>(result, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void cDotProduct_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                         cvector_ref<const ColorSpinorField> &y)
      {
        if constexpr (multi_reduce_prec_enabled<store_t>()) {
          cDotProduct_impl<store_t>(result, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void hDotProduct_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                         cvector_ref<const ColorSpinorField> &y)
      {
        if constexpr (multi_reduce_prec_enabled<store_t>()) {
          hDotProduct_impl<store_t>(result, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      template <typename store_t>
      void hDotProduct_Anorm_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                               cvector_ref<const ColorSpinorField> &y)
      {
        if constexpr (multi_reduce_prec_enabled<store_t>()) {
          hDotProduct_Anorm_impl<store_t>(result, x, y);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable the requested precision", QUDA_PRECISION);
        }
      }

      // Repeat instantiations only; names and bodies are the templates above.
#define INSTANTIATE_MULTI_REDUCE_STORE(store_t)                                                                        \
  template void reDotProduct_t<store_t>(std::vector<real_t> &, cvector_ref<const ColorSpinorField> &,                  \
                                        cvector_ref<const ColorSpinorField> &);                                        \
  template void cDotProduct_t<store_t>(std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,                \
                                       cvector_ref<const ColorSpinorField> &);                                         \
  template void hDotProduct_t<store_t>(std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,                \
                                       cvector_ref<const ColorSpinorField> &);                                         \
  template void hDotProduct_Anorm_t<store_t>(std::vector<complex_t> &, cvector_ref<const ColorSpinorField> &,          \
                                             cvector_ref<const ColorSpinorField> &);

    } // namespace block

  } // namespace blas

} // namespace quda
