#if QUDA_PRECISION & 4
#include "blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_BLAS_STORE(float)

  } // namespace blas

} // namespace quda
#endif
