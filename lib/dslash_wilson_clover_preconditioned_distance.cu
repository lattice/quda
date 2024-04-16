#include <dslash_wilson_clover_preconditioned.hpp>

/**
   This is the Wilson-clover preconditioned linear operator
*/

namespace quda
{

  // Apply the distance and even-odd preconditioned Wilson-clover operator
  // out(x) = M*in = a * A(x)^{-1} [ \sum_i U_i(x)in(x+\hat{i}) + U^\dagger_i(x-\hat{i})in(x-\hat{i})
  //                               + fwd(x_4)*U_4(x)in(x+\hat{4}) + bwd(x_4)*U^\dagger_4(x-\hat{4})in(x-\hat{4}) ]
  // with fwd(t)=\alpha(t+1)/\alpha(t), bwd(t)=\alpha(t+1)/\alpha(t), \alpha(t)=\cosh(\alpha_0*((t-t_0)%L_t-L_t/2))
  // Uses the kappa normalization for the Wilson operator.
#if defined(GPU_CLOVER_DIRAC) && defined(GPU_DIRAC_DISTANCE_PRECONDITIONING)
  void ApplyWilsonCloverPreconditionedDistance(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                               const CloverField &A, double a, double alpha0, int t0,
                                               const ColorSpinorField &x, int parity, bool dagger,
                                               const int *comm_override, TimeProfile &profile)
  {
    auto dummy = DistanceType<true>();
    instantiate<WilsonCloverPreconditionedApply>(out, in, U, A, a, alpha0, t0, x, parity, dagger, comm_override, dummy,
                                                 profile);
  }
#else
  void ApplyWilsonCloverPreconditionedDistance(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                               const CloverField &, double, double, int, const ColorSpinorField &, int,
                                               bool, const int *, TimeProfile &)
  {
    errorQuda("Distance preconditioned clover dslash has not been built");
  }
#endif // GPU_CLOVER_DIRAC && GPU_DIRAC_DISTANCE_PRECONDITIONING

} // namespace quda
