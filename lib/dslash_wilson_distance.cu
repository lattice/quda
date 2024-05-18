#include <dslash_wilson.hpp>

namespace quda
{

  // Apply the distance preconditioned Wilson operator
  // out(x) = M*in = - a*[ \sum_i U_i(x)in(x+\hat{i}) + U^\dagger_i(x-\hat{i})in(x-\hat{i})
  //                     + fwd(x_4)*U_4(x)in(x+\hat{4}) + bwd(x_4)*U^\dagger_4(x-\hat{4})in(x-\hat{4}) ]
  // with fwd(t)=\alpha(t+1)/\alpha(t), bwd(t)=\alpha(t+1)/\alpha(t), \alpha(t)=\cosh(\alpha_0*((t-t_0)%L_t-L_t/2))
  // Uses the a normalization for the Wilson operator.
  void ApplyWilsonDistance(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           const GaugeField &U, double a, double alpha0, int t0, cvector_ref<const ColorSpinorField> &x,
                           int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if (in[0].Ndim() == 5) errorQuda("Unexpected nDim = 5");
    if constexpr (is_enabled<QUDA_WILSON_DSLASH>() && is_enabled_distance_precondition()) {
      auto dummy = DistanceType<true>();
      instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, alpha0, t0, x, parity, dagger, comm_override, dummy,
                                                  profile);
    } else {
      errorQuda("Wilson operator with distance preconditioning has not been built");
    }
  }

} // namespace quda
