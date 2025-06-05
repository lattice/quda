#include <instantiate_dslash.h>
/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct DomainWall4DApply {
    DomainWall4DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                      cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double m_5,
                      const Complex *b_5, const Complex *c_5, int parity, bool dagger, const int *comm_override,
                      TimeProfile &profile);
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall4D(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, double a, double m_5, const Complex *b_5, const Complex *c_5,
                         cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                         TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>() || is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<DomainWall4DApply>(out, in, x, U, a, m_5, b_5, c_5, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Domain-wall dslash has not been built");
    }
  }

} // namespace quda
