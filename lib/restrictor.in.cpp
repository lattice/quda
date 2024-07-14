#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void Restrict2(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                 const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity, IntList<coarseColor, N...>)
  {
    if (out[0].Ncolor() == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        Restrict<fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else {
        errorQuda("Invalid coarseColor = %d, cannot be less than fineColor = %d", coarseColor, fineColor);
      }
    } else {
      if constexpr (sizeof...(N) > 0) {
        Restrict2<fineColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", out[0].Ncolor());
      }
    }
  }

  template <int fineColor, int... N>
  void Restrict(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity, IntList<fineColor, N...>)
  {
    if (in[0].Ncolor() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      Restrict2<fineColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity, coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        Restrict(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Fine Nc = %d has not been instantiated", in[0].Ncolor());
      }
    }
  }

  void Restrict(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      if (v.Nspin() != 1 && out[0].GammaBasis() != v.GammaBasis())
        errorQuda("Cannot apply restrictor using fields in a different basis from the null space (%d,%d) != %d",
                  out[0].GammaBasis(), in[0].GammaBasis(), v.GammaBasis());

      // clang-format off
      IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
      // clang-format on

      Restrict(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity, fineColors);
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
