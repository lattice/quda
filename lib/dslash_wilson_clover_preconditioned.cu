#include <dslash_wilson_clover_preconditioned.hpp>

/**
   This is the Wilson-clover preconditioned linear operator
*/

namespace quda
{


  // Apply the preconditioned Wilson-clover operator
  // out(x) = M*in = a * A(x)^{-1} (\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_CLOVER_DIRAC
  void ApplyWilsonCloverPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                       const CloverField &A, double a, const ColorSpinorField &x, int parity,
                                       bool dagger, const int *comm_override, TimeProfile &profile)
  {
    auto dummy = DistanceType<false>();
    instantiate<WilsonCloverPreconditionedApply>(out, in, U, A, a, 0, -1, x, parity, dagger, comm_override, dummy,
                                                 profile);
  }
#else
  void ApplyWilsonCloverPreconditioned(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                       const CloverField &, double, const ColorSpinorField &, int, bool, const int *,
                                       TimeProfile &)
  {
    errorQuda("Clover dslash has not been built");
  }
#endif

} // namespace quda
