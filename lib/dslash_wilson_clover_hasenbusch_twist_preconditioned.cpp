#include <instantiate_dslash.h>

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistPCNoClovInvApply {
    WilsonCloverHasenbuschTwistPCNoClovInvApply(cvector_ref<ColorSpinorField> &out,
                                                cvector_ref<const ColorSpinorField> &in,
                                                cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                                                const CloverField &A, double a, double b, int parity, bool dagger,
                                                const int *comm_override, TimeProfile &profile);
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistPCClovInvApply {
    WilsonCloverHasenbuschTwistPCClovInvApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                              cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                                              const CloverField &A, double kappa, double mu, int parity, bool dagger,
                                              const int *comm_override, TimeProfile &profile);
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwistPCNoClovInv(cvector_ref<ColorSpinorField> &out,
                                                   cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                                                   const CloverField &A, double a, double b,
                                                   cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                                   const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH>()) {
      instantiate<WilsonCloverHasenbuschTwistPCNoClovInvApply>(out, in, x, U, A, a, b, parity, dagger, comm_override,
                                                               profile);
    } else {
      errorQuda("Clover Hasenbusch Twist operator has not been built");
    }
  }

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwistPCClovInv(cvector_ref<ColorSpinorField> &out,
                                                 cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                                                 const CloverField &A, double a, double b,
                                                 cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                                 const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH>()) {
      instantiate<WilsonCloverHasenbuschTwistPCClovInvApply>(out, in, x, U, A, a, b, parity, dagger, comm_override,
                                                             profile);
    } else {
      errorQuda("Clover Hasenbusch Twist operator has not been built");
    }
  }

} // namespace quda
