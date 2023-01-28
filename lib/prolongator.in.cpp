#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void Prolongate2(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<coarseColor, N...>)
  {
    if (in.Ncolor() == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        Prolongate<fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Invalid coarseColor = %d, cannot be less than fineColor = %d", coarseColor, fineColor);
      }
    } else {
      if constexpr (sizeof...(N) > 0) {
        Prolongate2<fineColor>(out, in, v, fine_to_coarse, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", in.Ncolor());
      }
    }
  }

  template <int fineColor, int... N>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<fineColor, N...>)
  {
    if (out.Ncolor() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      Prolongate2<fineColor>(out, in, v, fine_to_coarse, spin_map, parity, coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        Prolongate(out, in, v, fine_to_coarse, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Fine Nc = %d has not been instantiated", out.Ncolor());
      }
    }
  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
      // clang-format on
      Prolongate(out, in, v, fine_to_coarse, spin_map, parity, fineColors);
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
