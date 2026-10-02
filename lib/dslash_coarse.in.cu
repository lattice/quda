@QUDA_COARSE_FILE_BEGIN@
#include "dslash_coarse_inst.hpp"

namespace quda
{

  INSTANTIATE_DSLASH_COARSE(@QUDA_MULTIGRID_DAGGER@, @QUDA_MULTIGRID_NVEC@, @QUDA_COARSE_FLOAT@, @QUDA_COARSE_YFLOAT@,
                            @QUDA_COARSE_GHOSTFLOAT@)

} // namespace quda
@QUDA_COARSE_FILE_END@
