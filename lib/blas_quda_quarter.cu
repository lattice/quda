#if QUDA_PRECISION & 1
#include "blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_BLAS_STORE(int8_t)

  } // namespace blas

} // namespace quda
#endif
