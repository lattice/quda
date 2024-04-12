#include <dslash_wilson.hpp>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
#ifdef GPU_WILSON_DIRAC
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    auto dummy = DistanceType<false>();
    instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, 0, 0, x, parity, dagger, comm_override, dummy, profile);
  }
#else
  void ApplyWilson(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, const ColorSpinorField &,
                   int, bool, const int *, TimeProfile &)
  {
    errorQuda("Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC

} // namespace quda
