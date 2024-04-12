#include <dslash_wilson.hpp>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

#ifdef GPU_WILSON_DIRAC
  void ApplyWilsonDistance(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                           double alpha0, int t0, const ColorSpinorField &x, int parity, bool dagger,
                           const int *comm_override, TimeProfile &profile)
  {
    auto dummy = DistanceType<true>();
    instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, alpha0, t0, x, parity, dagger, comm_override, dummy,
                                                profile);
  }
#else
  void ApplyWilsonDistance(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, double, int,
                           const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC

} // namespace quda
