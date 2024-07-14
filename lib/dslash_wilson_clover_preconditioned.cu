#include <dslash_wilson_clover_preconditioned.hpp>

/**
   This is the Wilson-clover preconditioned linear operator
*/

namespace quda
{

  // Apply the preconditioned Wilson-clover operator
  // out(x) = M*in = a * A(x)^{-1} (\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                       const GaugeField &U, const CloverField &A, double a,
                                       cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                       const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<WilsonCloverPreconditionedApply>(out, in, x, U, A, a, 0.0, -1, parity, dagger, comm_override, dummy,
                                                   profile);
    } else {
      errorQuda("Wilson-clover operator has not been built");
    }
  }

} // namespace quda
