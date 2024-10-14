#define SIGNATURE_ONLY
#include <dslash_twisted_clover_preconditioned.hpp>
#undef SIGNATURE_ONLY

/**
   This is the preconditioned gauged twisted-mass operator
*/

namespace quda
{

  /*
    Apply the preconditioned twisted-mass Dslash operator

    out = x + a*A^{-1} D * in = x + a*(C + i*b*gamma_5)^{-1}*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  */
  void ApplyTwistedCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                        const GaugeField &U, const CloverField &C, double a, double b, bool xpay,
                                        cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                        const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<TwistedCloverPreconditionedApply>(out, in, x, U, C, a, b, xpay, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Twisted-clover operator has not been built");
    }
  }

} // namespace quda
