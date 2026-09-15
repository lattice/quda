#if QUDA_PRECISION & 1
#include "reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_REDUCE_STORE(int8_t)

  } // namespace blas

} // namespace quda
#endif
