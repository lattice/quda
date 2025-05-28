#include <instantiate_dslash.h>

/**
   This is the gauged domain-wall 4-d preconditioned operator, fused with immediately followed fifth dimension operators.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct DomainWall4DApplyFusedM5 {
    template <bool distance_pc, Dslash5Type dslash5_type_impl, Dslash5Type... N>
    DomainWall4DApplyFusedM5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                             cvector_ref<ColorSpinorField> &y, const Complex *b_5, const Complex *c_5, double a,
                             double m_5, int parity, bool dagger, const int *comm_override, double m_f,
                             DistanceType<distance_pc>, Dslash5TypeList<dslash5_type_impl, N...>, TimeProfile &profile);
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  //   i.e. out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // ... and then m5pre
  void ApplyDomainWall4DM5pre(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                              const GaugeField &U, double a, double m_5, const Complex *b_5, const Complex *c_5,
                              cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, int parity,
                              bool dagger, const int *comm_override, double m_f, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>()) {
      auto dummy = DistanceType<false>();
      auto dummy_list = Dslash5TypeList<Dslash5Type::DSLASH5_MOBIUS_PRE>();
      instantiate<DomainWall4DApplyFusedM5>(out, in, x, y, U, b_5, c_5, a, m_5, parity, dagger, comm_override, m_f,
                                            dummy, dummy_list, profile);
    } else {
      errorQuda("Domain-wall operator has not been built");
    }
  }

} // namespace quda
