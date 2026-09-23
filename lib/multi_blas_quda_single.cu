#if QUDA_PRECISION & 4
#include "multi_blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_BLAS_STORE(float)

    } // namespace block

  } // namespace blas

} // namespace quda
#endif
