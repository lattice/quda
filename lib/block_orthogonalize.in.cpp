#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void BlockOrthogonalize2(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                           const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass,
                           IntList<coarseColor, N...>)
  {
    if (B.size() == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        BlockOrthogonalize<fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho,
                                                   two_pass);
      } else {
        errorQuda("Invalid coarseColor = %d, cannot be less than fineColor = %d", coarseColor, fineColor);
      }
    } else {
      if constexpr (sizeof...(N) > 0) {
        BlockOrthogonalize2<fineColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass,
                                       IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %lu has not been instantiated", B.size());
      }
    }
  }

  template <int fineColor, int... N>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass,
                          IntList<fineColor, N...>)
  {
    if (V.Ncolor() / B.size() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      BlockOrthogonalize2<fineColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass,
                                     coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        BlockOrthogonalize(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass,
                           IntList<N...>());
      } else {
        errorQuda("Fine Nc = %lu has not been instantiated", V.Ncolor() / B.size());
      }
    }
  }

  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass)
  {
    if constexpr (is_enabled_multigrid()) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
      // clang-format on
      BlockOrthogonalize(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass, fineColors);
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
