#define SIGNATURE_ONLY
#include <dslash_wilson_clover.hpp>
#undef SIGNATURE_ONLY

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x)*in(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, const CloverField &A, double a, cvector_ref<const ColorSpinorField> &x,
                         int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<WilsonCloverApply>(out, in, x, U, A, a, 0, -1, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Wilson-clover operator has not been built");
    }
  }

} // namespace quda
