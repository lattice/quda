#include <instantiate_dslash.h>

/**
   This is the preconditioned gauged twisted-mass operator
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct TwistedMassPreconditionedApply {
    TwistedMassPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, real_t a, real_t b,
                                   bool xpay, int parity, bool dagger, bool asymmetric, const int *comm_override,
                                   TimeProfile &profile);
  };

  /*
    Apply the preconditioned twisted-mass Dslash operator

    out = x + A^{-1} D * in = x + a*(1 + i*b*gamma_5)*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  */
  void ApplyTwistedMassPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      const GaugeField &U, real_t a, real_t b, bool xpay,
                                      cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, bool asymmetric,
                                      const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      instantiate<TwistedMassPreconditionedApply>(out, in, x, U, a, b, xpay, parity, dagger, asymmetric, comm_override,
                                                  profile);
    } else {
      errorQuda("Twisted-mass operator has not been built");
    }
  }

} // namespace quda
