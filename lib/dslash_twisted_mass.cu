#define SIGNATURE_ONLY
#include <dslash_twisted_mass.hpp>
#undef SIGNATURE_ONLY

namespace quda
{


  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (1 + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedMass(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                        const GaugeField &U, double a, double b, cvector_ref<const ColorSpinorField> &x, int parity,
                        bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<TwistedMassApply>(out, in, x, U, a, b, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Twisted-mass operator has not been built");
    }
  }

} // namespace quda
