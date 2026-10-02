#include "blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_BLAS_STORE(double)
#ifdef QUDA_FPMP_FLOATFLOAT
    INSTANTIATE_BLAS_STORE(floatfloat)
#endif

  } // namespace blas

} // namespace quda
