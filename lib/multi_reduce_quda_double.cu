#include "multi_reduce_quda_inst.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      INSTANTIATE_MULTI_REDUCE_STORE(double)
#ifdef QUDA_FPMP_FLOATFLOAT
      INSTANTIATE_MULTI_REDUCE_STORE(floatfloat)
#endif

    } // namespace block

  } // namespace blas

} // namespace quda
