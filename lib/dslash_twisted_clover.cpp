#include <instantiate_dslash.h>

/**
   This is the basic gauged twisted-clover operator
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct TwistedCloverApply {
    TwistedCloverApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                       cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &C, real_t a,
                       real_t b, int parity, bool dagger, const int *comm_override, TimeProfile &profile);
  };

  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (A + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                          const GaugeField &U, const CloverField &C, real_t a, real_t b,
                          cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                          TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<TwistedCloverApply>(out, in, x, U, C, a, b, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Twisted-clover operator has not been built");
    }
  }

} // namespace quda
