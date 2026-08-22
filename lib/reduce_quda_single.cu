#if QUDA_PRECISION & 4
#include "reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_REDUCE_STORE(float)

  } // namespace blas

} // namespace quda
#endif
