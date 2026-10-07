#if QUDA_PRECISION & 1
#include "multi_blas_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_BLAS_STORE(int8_t)

    } // namespace block

  } // namespace blas

} // namespace quda
#endif
