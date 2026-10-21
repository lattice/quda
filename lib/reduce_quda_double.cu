#include "reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    INSTANTIATE_REDUCE_STORE(double)
#ifdef QUDA_FPMP_FLOATFLOAT
    INSTANTIATE_REDUCE_STORE(floatfloat)
#endif

  } // namespace blas

} // namespace quda
