#include <dslash_domain_wall_4d_fused_m5.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  constexpr Dslash5Type dslash5_type = Dslash5Type::@QUDA_DSLASH5_TYPE@;

  using DDArg = @QUDA_DSLASH_DDARG@;
  using Float = precision_type_mapper<precision>::type;

  template struct DomainWall4DApplyFusedM5<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template DomainWall4DApplyFusedM5<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::DomainWall4DApplyFusedM5(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, cvector_ref<ColorSpinorField> &y, const complex_t *b_5, const complex_t *c_5, real_t a,
    real_t m_5, int parity, bool dagger, const int *comm_override, real_t m_f, Dslash5TypeList<dslash5_type>,
    TimeProfile &profile);

} // namespace quda
