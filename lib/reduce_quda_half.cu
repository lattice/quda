#if QUDA_PRECISION & 2
#include "reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_REDUCE_STORE(short)

  } // namespace blas

} // namespace quda
#endif
