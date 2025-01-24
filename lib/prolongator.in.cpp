#include "multigrid.h"
#include <multigrid.hpp>
#include <blas_quda.h>

namespace quda
{

  template <int fineColor, int coarseColor, int nVec, int... N>
  void ProlongateMma2(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                      const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<nVec, N...>)
  {
    if (out.Nvec() == nVec) {
      ProlongateMma<fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, spin_map, parity);
    } else {
      if constexpr (sizeof...(N) > 0) {
        ProlongateMma2<fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("nVec = %d has not been instantiated", out.Nvec());
      }
    }
  }

  template <bool use_mma, int fineColor, int coarseColor, int... N>
  void Prolongate2(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<coarseColor, N...>)
  {
    if (in[0].Ncolor() == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        if constexpr (use_mma) {
          constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          auto V = create_color_spinor_copy(v, csOrder);
          blas::copy(V, v);

          auto op = [&](cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int nVec) {
            auto v_in = create_color_spinor_copy(in, nVec, csOrder);
            auto v_out = create_color_spinor_copy(out, nVec, csOrder);
            BlockTransposeForward(v_in, in);

            // clang-format off
            IntList<@QUDA_MULTIGRID_MRHS_LIST@> nVecs;
            // clang-format on
            ProlongateMma2<fineColor, coarseColor>(v_out, v_in, V, fine_to_coarse, spin_map, parity, nVecs);

            bool to_non_rel = (out.Nspin() == 4) && (out[0].GammaBasis() == QUDA_UKQCD_GAMMA_BASIS);
            BlockTransposeBackward(v_out, out, to_non_rel);
          };

          divide_and_conquer(op, out, in);
        } else {
          Prolongate<fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
        }
      } else {
        errorQuda("Invalid coarseColor = %d, cannot be less than fineColor = %d", coarseColor, fineColor);
      }
    } else {
      if constexpr (sizeof...(N) > 0) {
        Prolongate2<use_mma, fineColor>(out, in, v, fine_to_coarse, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", in[0].Ncolor());
      }
    }
  }

  template <bool use_mma, int fineColor, int... N>
  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<fineColor, N...>)
  {
    if (out[0].Ncolor() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      Prolongate2<use_mma, fineColor>(out, in, v, fine_to_coarse, spin_map, parity, coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        Prolongate<use_mma>(out, in, v, fine_to_coarse, spin_map, parity, IntList<N...>());
      } else {
        errorQuda("Fine Nc = %d has not been instantiated", out[0].Ncolor());
      }
    }
  }

  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, bool use_mma, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      if (v.Nspin() != 1 && in[0].GammaBasis() != v.GammaBasis())
        errorQuda("Cannot apply prolongator using fields in a different basis from the null space (%d,%d) != %d",
                  out[0].GammaBasis(), in[0].GammaBasis(), v.GammaBasis());

      // clang-format off
      IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
      // clang-format on
      if (use_mma) {
        // use MMA
        Prolongate<true>(out, in, v, fine_to_coarse, spin_map, parity, fineColors);
      } else {
        Prolongate<false>(out, in, v, fine_to_coarse, spin_map, parity, fineColors);
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
