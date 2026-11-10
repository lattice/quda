#if QUDA_PRECISION & 2
#include "blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_BLAS_STORE(short)

  } // namespace blas

} // namespace quda
#endif
