#if QUDA_PRECISION & 2
#include "multi_blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_BLAS_STORE(short)

    } // namespace block

  } // namespace blas

} // namespace quda
#endif
