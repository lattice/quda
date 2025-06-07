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
    const GaugeField &U, cvector_ref<ColorSpinorField> &y, const Complex *b_5, const Complex *c_5, double a, double m_5,
    int parity, bool dagger, const int *comm_override, double m_f, Dslash5TypeList<dslash5_type>, TimeProfile &profile);

} // namespace quda
