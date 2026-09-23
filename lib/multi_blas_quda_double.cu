#include "multi_blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_BLAS_STORE(double)
#ifdef QUDA_FPMP_FLOATFLOAT
      INSTANTIATE_MULTI_BLAS_STORE(floatfloat)
#endif

    } // namespace block

  } // namespace blas

} // namespace quda
