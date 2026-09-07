#pragma once

#include <blas_quda.h>

/**
   Host-visible declarations of the per-store multi-reduce entry points.
   Definitions live in multi_reduce_quda_inst.hpp and are explicitly
   instantiated in multi_reduce_quda_{double,single,half,quarter}.cu.
   This header must not include multi_reduce_quda.hpp (CUDA templates).
*/

namespace quda
{

  namespace blas
  {

    namespace block
    {

      template <typename store_t>
      void reDotProduct_t(std::vector<real_t> &result, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<const ColorSpinorField> &y);

      template <typename store_t>
      void cDotProduct_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                         cvector_ref<const ColorSpinorField> &y);

      template <typename store_t>
      void hDotProduct_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                         cvector_ref<const ColorSpinorField> &y);

      template <typename store_t>
      void hDotProduct_Anorm_t(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                               cvector_ref<const ColorSpinorField> &y);

    } // namespace block

  } // namespace blas

} // namespace quda
