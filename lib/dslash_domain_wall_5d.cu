#define SIGNATURE_ONLY
#include <dslash_domain_wall_5d.hpp>
#undef SIGNATURE_ONLY

/**
   This is the gauged domain-wall 5-d preconditioned operator.
*/

namespace quda
{


  // Apply the 5-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall5D(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, double a, double m_f, cvector_ref<const ColorSpinorField> &x, int parity,
                         bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<DomainWall5DApply>(out, in, x, U, a, m_f, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Domain-wall operator has not been built");
    }
  }

} // namespace quda
