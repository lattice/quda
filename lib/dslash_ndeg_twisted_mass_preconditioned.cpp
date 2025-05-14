#include <instantiate_dslash.h>

/**
   This is the preconditioned twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct NdegTwistedMassPreconditionedApply {
    template <bool distance_pc>
    NdegTwistedMassPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                       cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b,
                                       double c, bool xpay, int parity, bool dagger, bool asymmetric,
                                       const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile);
  };

  // Apply the non-degenerate twisted-mass Dslash operator
  // out(x) = M*in = a*(1 + i*b*gamma_5*tau_3 + c*tau_1)*D + x
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyNdegTwistedMassPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                          const GaugeField &U, double a, double b, double c, bool xpay,
                                          cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                          bool asymmetric, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<NdegTwistedMassPreconditionedApply>(out, in, x, U, a, b, c, xpay, parity, dagger, asymmetric,
                                                      comm_override, dummy, profile);
    } else {
      errorQuda("Non-degenerate preconditioned twisted-mass dslash has not been built");
    }
  }

} // namespace quda
