#include <instantiate_dslash.h>

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct WilsonApply {
    template <bool distance_pc>
    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double alpha0, int t0, int parity,
                bool dagger, const int *comm_override, std::integral_constant<bool, distance_pc>, TimeProfile &profile);
  };

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
  void ApplyWilson(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                   double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                   TimeProfile &profile)
  {
    if (in.Ndim() == 5) errorQuda("Unexpected nDim = 5");
    if constexpr (is_enabled<QUDA_WILSON_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<WilsonApply>(out, in, x, U, a, 0, -1, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Wilson operator has not been built");
    }
  }

} // namespace quda
