#include "multigrid.h"
#include <multigrid.hpp>
#include <dirac_quda.h>

namespace quda
{

#if defined(QUDA_MMA_AVAILABLE)
  template <bool dagger, int Nc, int nVec, int... N>
  void ApplyCoarseMma(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                      cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, double kappa,
                      int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision,
                      IntList<nVec, N...>)
  {
    if (out[0].Nvec() == nVec) {
      ApplyCoarseMma<dagger, Nc, nVec>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
    } else {
      if constexpr (sizeof...(N) > 0) {
        ApplyCoarseMma<dagger, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision,
                                   IntList<N...>());
      } else {
        errorQuda("nVec = %d has not been instantiated", out[0].Nvec());
      }
    }
  }
 #else
  template <bool dagger, int Nc, int nVec, int... N>
  void ApplyCoarseMma(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                      cvector_ref<const ColorSpinorField> &, const GaugeField &, const GaugeField &, double,
                      int, bool, bool, const int *, QudaPrecision,
                      IntList<nVec, N...>)
  {
    errorQuda("MMA not instantiated");
  }
 #endif

  template <bool use_mma, int Nc, int... N>
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, double kappa,
                   int parity, bool dslash, bool clover, bool dagger, const int *commDim, QudaPrecision halo_precision,
                   IntList<Nc, N...>)
  {
    int nColor = use_mma ? inA[0].Ncolor() / inA[0].Nvec() : inA[0].Ncolor();
    if (nColor == Nc) {
      if (dagger)
        if constexpr (use_mma)
          ApplyCoarseMma<true, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision,
                                   IntList<@QUDA_MULTIGRID_MRHS_LIST@>());
        else
          ApplyCoarse<true, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
      else if constexpr (use_mma)
        ApplyCoarseMma<false, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision,
                                  IntList<@QUDA_MULTIGRID_MRHS_LIST@>());
      else
        ApplyCoarse<false, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
    } else {
      if constexpr (sizeof...(N) > 0) {
        ApplyCoarse<use_mma>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision,
                             IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", inA[0].Ncolor());
      }
    }
  }

  auto create_gauge_copy(const GaugeField &X, QudaGaugeFieldOrder order, bool copy_content)
  {
    GaugeField *output;
    if (X.Order() == order) {
      output = const_cast<GaugeField *>(&X);
    } else {
      GaugeFieldParam param(X);
      param.order = order;
      param.location = QUDA_CUDA_FIELD_LOCATION;
      output = static_cast<GaugeField *>(GaugeField::Create(param));
      if (copy_content) { output->copy(X); }
    }
    return output;
  }

  // Apply the coarse Dirac matrix to a coarse grid vector
  // out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //   or
  //  out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  //  Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, double kappa,
                   int parity, bool dslash, bool clover, bool dagger, const int *commDim, QudaPrecision halo_precision,
                   bool use_mma)
  {
    if constexpr (is_enabled_multigrid()) {
      if (!DiracCoarse::apply_mma(out, use_mma) || checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
        ApplyCoarse<false>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision,
                           IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
      } else {

        constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;
        auto X_ = create_gauge_copy(X, gOrder, clover);
        auto Y_ = create_gauge_copy(Y, gOrder, dslash);

        if (Y_ != &Y) { Y_->exchangeGhost(QUDA_LINK_BIDIRECTIONAL); }

        if (out.size() != inA.size() || out.size() != inB.size()) {
          errorQuda("out.size(), inA.size(), inB.size(): %lu, %lu, %lu", out.size(), inA.size(), inB.size());
        }
        int instantiated_nVec = instantiated_nVec_to_use(out.size());
        size_t size = out.size();
        logQuda(QUDA_DEBUG_VERBOSE, "Dslash coarse nVec/out.size() = %d/%lu\n", instantiated_nVec, size);
        for (size_t offset = 0; offset < size; offset += instantiated_nVec) {
          cvector_ref<ColorSpinorField> out_offseted {out.begin() + offset,
                                                      out.begin() + std::min(offset + instantiated_nVec, size)};
          cvector_ref<const ColorSpinorField> inA_offseted {inA.begin() + offset,
                                                            inA.begin() + std::min(offset + instantiated_nVec, size)};
          cvector_ref<const ColorSpinorField> inB_offseted {inB.begin() + offset,
                                                            inB.begin() + std::min(offset + instantiated_nVec, size)};

          constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          auto v_inA = create_color_spinor_copy(inA_offseted, instantiated_nVec, csOrder);
          auto v_inB = create_color_spinor_copy(inB_offseted, instantiated_nVec, csOrder);
          auto v_out = create_color_spinor_copy(out_offseted, instantiated_nVec, csOrder);

          if (dslash) { BlockTransposeForward(v_inA, inA_offseted); }
          if (clover) { BlockTransposeForward(v_inB, inB_offseted); }

          // clang-format off
          IntList<@QUDA_MULTIGRID_NVEC_LIST@> nColors;
          // clang-format on
          ApplyCoarse<true>(v_out, v_inA, v_inB, *Y_, *X_, kappa, parity, dslash, clover, dagger, commDim,
                            halo_precision, nColors);

          BlockTransposeBackward(v_out, out_offseted);
        }

        if (X_ != &X) { delete X_; }
        if (Y_ != &Y) { delete Y_; }
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
