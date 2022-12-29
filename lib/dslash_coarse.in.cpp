#include "multigrid.h"

namespace quda {

  template <int...> struct IntList { };

  template <int Nc, int...N>
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                   double kappa, int parity, bool dslash, bool clover, bool dagger,
                   const int *commDim, QudaPrecision halo_precision, bool use_mma, IntList<Nc, N...>)
  {
    if (inA[0].Ncolor() / inA[0].Nvec() == Nc) {
      if (dagger)
        ApplyCoarse<true, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision, use_mma);
      else
        ApplyCoarse<false, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision, use_mma);
    } else {
      if constexpr (sizeof...(N) > 0) {
        ApplyCoarse(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision, use_mma, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", inA[0].Ncolor());
      }
    }
  }

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //  or
  //out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                   double kappa, int parity, bool dslash, bool clover, bool dagger,
                   const int *commDim, QudaPrecision halo_precision, bool use_mma)
  {
    if constexpr (is_enabled_multigrid()) {
      ApplyCoarse(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision, use_mma, IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
