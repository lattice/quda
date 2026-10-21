@QUDA_BLOCK_ORTHO_FILE_BEGIN@
#include "block_orthogonalize_inst.hpp"

namespace quda
{

  INSTANTIATE_BLOCK_ORTHOGONALIZE(@QUDA_MULTIGRID_NC_NVEC@, @QUDA_MULTIGRID_NVEC2@, @QUDA_BLOCK_ORTHO_VFLOAT@,
                                  @QUDA_BLOCK_ORTHO_BFLOAT@)

} // namespace quda
@QUDA_BLOCK_ORTHO_FILE_END@
