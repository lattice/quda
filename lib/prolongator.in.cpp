#include "multigrid.h"
#include <blas_quda.h>

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int nVec, int... N>
  void ProlongateMma2(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
      const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<nVec, N...>) {
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

  template <class F> auto create_color_spinor_copy(cvector_ref<F> &fs, QudaFieldOrder order)
  {
    ColorSpinorParam param(fs[0]);
    int nVec = (fs.size() + 7) / 8 * 8; // Make a multiple of 8
    param.nColor = fs[0].Ncolor() * nVec;
    param.nVec = nVec;
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return ColorSpinorField(param);
  }

  auto create_color_spinor_copy(const ColorSpinorField &f, QudaFieldOrder order)
  {
    ColorSpinorParam param(f);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return ColorSpinorField(param);
  }

  template <bool use_mma, int fineColor, int coarseColor, int... N>
  void Prolongate2(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *const *spin_map, int parity, IntList<coarseColor, N...>)
  {
    if (in[0].Ncolor() == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        if constexpr (use_mma) {

          constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          ColorSpinorField v_in = create_color_spinor_copy(in, csOrder);
          ColorSpinorField v_out = create_color_spinor_copy(out, csOrder);
          ColorSpinorField V = create_color_spinor_copy(v, csOrder);
          BlockTransposeForward(v_in, in);
          V.copy(v);

          IntList<@QUDA_MULTIGRID_MRHS_LIST@> nvecs;
          ProlongateMma2<fineColor, coarseColor>(v_out, v_in, V, fine_to_coarse, spin_map, parity, nvecs);

          bool to_non_rel = (out.Nspin() == 4) && (out[0].GammaBasis() == QUDA_UKQCD_GAMMA_BASIS);
          BlockTransposeBackward(v_out, out, to_non_rel);
#if 0
          std::vector<ColorSpinorField> v_cmp(out.size());
          for (size_t i = 0; i < out.size(); i++) {
            ColorSpinorParam param(out[i]);
            param.create = QUDA_NULL_FIELD_CREATE;
            v_cmp[i] = ColorSpinorField(param);
          }
          auto vv_cmp = make_set(v_cmp);
          Prolongate<fineColor, coarseColor>(vv_cmp, in, v, fine_to_coarse, spin_map, parity);

          blas::mxpy(out, v_cmp);
          auto vn = blas::norm2(vv_cmp);
          printf("n = ");
          for (size_t i = 0; i < vn.size(); i++) {
            printf("%f ", vn[i]);
          }
          printf("\n");
#endif
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
                  const int *fine_to_coarse, const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      if (v.Nspin() != 1 && in[0].GammaBasis() != v.GammaBasis())
        errorQuda("Cannot apply prolongator using fields in a different basis from the null space (%d,%d) != %d",
                  out[0].GammaBasis(), in[0].GammaBasis(), v.GammaBasis());

      // clang-format off
      IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
      // clang-format on
      if (1) {
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
