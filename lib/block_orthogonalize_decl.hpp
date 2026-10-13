#pragma once

#include <color_spinor_field.h>
#include <vector>

/**
   Host-visible declarations of the per-precision block-orthogonalize
   entry points. Definitions live in the precision .cu files.
   This header must not include block_orthogonalize.hpp (CUDA templates).
*/

namespace quda
{

  template <int fineColor, int coarseColor, typename vFloat, typename bFloat>
  void BlockOrthogonalize_t(ColorSpinorField &V, const std::vector<ColorSpinorField> &B, const int *fine_to_coarse,
                            const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass);

} // namespace quda
