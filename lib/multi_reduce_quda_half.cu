#if QUDA_PRECISION & 2
#include "multi_reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_REDUCE_STORE(short)

    } // namespace block

  } // namespace blas

} // namespace quda
#endif
