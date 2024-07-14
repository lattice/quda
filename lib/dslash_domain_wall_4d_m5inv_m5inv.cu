#include <dslash_domain_wall_4d_fused_m5.hpp>

/**
   This is the gauged domain-wall 4-d preconditioned operator, fused with immediately followed fifth dimension operators.
*/

namespace quda
{

  // Apply the 4-d preconditioned domain-wall Dslash operator
  //   i.e. out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // ... and then m5inv + m5inv-dagger
  void ApplyDomainWall4DM5invM5inv(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                   const GaugeField &U, double a, double m_5, const Complex *b_5, const Complex *c_5,
                                   cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, int parity,
                                   bool dagger, const int *comm_override, double m_f, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>()) {
      auto dummy_list = Dslash5TypeList<Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG>();
      instantiate<DomainWall4DApplyFusedM5>(out, in, x, y, U, b_5, c_5, a, m_5, parity, dagger, comm_override, m_f,
                                            dummy_list, profile);
    } else {
      errorQuda("Domain-wall operator has not been built");
    }
  }

} // namespace quda
