#pragma once

#include "block_orthogonalize_decl.hpp"
#include <multigrid.h>

namespace quda
{

  template <int fineColor, int coarseColor>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass)
  {
    if (!is_enabled(V.Precision()) || !is_enabled(B[0].Precision()))
      errorQuda("QUDA_PRECISION=%d does not enable required precision combination (V = %d B = %d)", QUDA_PRECISION,
                V.Precision(), B[0].Precision());

    if constexpr (is_enabled_multigrid()) {
      if (B[0].data() == nullptr) {
        warningQuda("Trying to BlockOrthogonalize staggered transform, skipping...");
        return;
      }
      if (V.Precision() == QUDA_DOUBLE_PRECISION && B[0].Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          BlockOrthogonalize_t<fineColor, coarseColor, double, double>(V, B, fine_to_coarse, coarse_to_fine, geo_bs,
                                                                       spin_bs, n_block_ortho, two_pass);
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (V.Precision() == QUDA_SINGLE_PRECISION && B[0].Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          BlockOrthogonalize_t<fineColor, coarseColor, float, float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs,
                                                                     spin_bs, n_block_ortho, two_pass);
      } else if (V.Precision() == QUDA_HALF_PRECISION && B[0].Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION) && is_enabled(QUDA_SINGLE_PRECISION))
          BlockOrthogonalize_t<fineColor, coarseColor, short, float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs,
                                                                     spin_bs, n_block_ortho, two_pass);
      } else if (V.Precision() == QUDA_HALF_PRECISION && B[0].Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          BlockOrthogonalize_t<fineColor, coarseColor, short, short>(V, B, fine_to_coarse, coarse_to_fine, geo_bs,
                                                                     spin_bs, n_block_ortho, two_pass);
      } else {
        errorQuda("Unsupported precision combination V=%d B=%d\n", V.Precision(), B[0].Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
