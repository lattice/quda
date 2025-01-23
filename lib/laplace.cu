#define SIGNATURE_ONLY
#include <laplace.hpp>
#undef SIGNATURE_ONLY

namespace quda
{

  // Apply the Laplace operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu) + b*in(x)
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                    int dir, double a, double b, cvector_ref<const ColorSpinorField> &x, int parity,
                    const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_LAPLACE_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<LaplaceApply>(out, in, x, U, dir, a, b, parity, comm_override, dummy, profile);
    } else {
      errorQuda("Laplace operator has not been enabled");
    }
  }
} // namespace quda
