#include <instantiate_dslash.h>

/**
   This is the gauged domain-wall 4-d preconditioned operator, fused with immediately followed fifth dimension operators.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct DomainWall4DApplyFusedM5 {
    template <Dslash5Type dslash5_type_impl, Dslash5Type... N>
    DomainWall4DApplyFusedM5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                             cvector_ref<ColorSpinorField> &y, const complex_t *b_5, const complex_t *c_5, real_t a,
                             real_t m_5, int parity, bool dagger, const int *comm_override, real_t m_f,
                             Dslash5TypeList<dslash5_type_impl, N...>, TimeProfile &profile);
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  //   i.e. out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // ... and then m5pre + m5
  void ApplyDomainWall4DM5preM5mob(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                   const GaugeField &U, real_t a, real_t m_5, const complex_t *b_5, const complex_t *c_5,
                                   cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, int parity,
                                   bool dagger, const int *comm_override, real_t m_f, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>()) {
      auto dummy_list = Dslash5TypeList<Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB>();
      instantiate<DomainWall4DApplyFusedM5>(out, in, x, y, U, b_5, c_5, a, m_5, parity, dagger, comm_override, m_f,
                                            dummy_list, profile);
    } else {
      errorQuda("Domain-wall operator has not been built");
    }
  }

} // namespace quda
