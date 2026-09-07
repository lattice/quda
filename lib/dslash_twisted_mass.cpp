#include <instantiate_dslash.h>

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct TwistedMassApply {
    TwistedMassApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                     cvector_ref<const ColorSpinorField> &x, const GaugeField &U, real_t a, real_t b, int parity,
                     bool dagger, const int *comm_override, TimeProfile &profile);
  };

  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (1 + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedMass(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                        const GaugeField &U, real_t a, real_t b, cvector_ref<const ColorSpinorField> &x, int parity,
                        bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      instantiate<TwistedMassApply>(out, in, x, U, a, b, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Twisted-mass operator has not been built");
    }
  }

} // namespace quda
