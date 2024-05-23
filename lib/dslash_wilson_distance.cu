#include <dslash_wilson.hpp>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  // Apply the distance preconditioned Wilson operator
  // out(x) = M*in = - a*[ \sum_i U_i(x)in(x+\hat{i}) + U^\dagger_i(x-\hat{i})in(x-\hat{i})
  //                     + fwd(x_4)*U_4(x)in(x+\hat{4}) + bwd(x_4)*U^\dagger_4(x-\hat{4})in(x-\hat{4}) ]
  // with fwd(t)=\alpha(t+1)/\alpha(t), bwd(t)=\alpha(t+1)/\alpha(t), \alpha(t)=\cosh(\alpha_0*((t-t_0)%L_t-L_t/2))
  // Uses the a normalization for the Wilson operator.
#if defined(GPU_WILSON_DIRAC) && defined(GPU_DISTANCE_PRECONDITIONING)
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
    errorQuda("Distance preconditioned Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC && GPU_DISTANCE_PRECONDITIONING

} // namespace quda
