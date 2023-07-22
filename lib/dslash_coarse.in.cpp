#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int Nc, int... N>
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                   int parity, bool dslash, bool clover, bool dagger, const int *commDim, QudaPrecision halo_precision,
                   IntList<Nc, N...>)
  {
    if (inA[0].Ncolor() == Nc) {
      if (dagger)
        ApplyCoarse<true, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
      else
        ApplyCoarse<false, Nc>(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
    } else {
      if constexpr (sizeof...(N) > 0) {
        ApplyCoarse(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", inA[0].Ncolor());
      }
    }
  }

  // Apply the coarse Dirac matrix to a coarse grid vector
  // out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //  or
  // out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                   int parity, bool dslash, bool clover, bool dagger, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      // clang-format off
      ApplyCoarse(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, commDim, halo_precision,
                  IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
      // clang-format on
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
