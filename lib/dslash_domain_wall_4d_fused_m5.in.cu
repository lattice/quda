#include <dslash_wilson_clover_preconditioned.hpp>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  constexpr Dslash5Type dslash5_type = Dslash5Type::@QUDA_DSLASH_DWTYPE@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;

  template struct DomainWall4DApplyFusedM5<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template DomainWall4DApplyFusedM5<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::DomainWall4DFusedM5Apply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, cvector_ref<ColorSpinorField> &y, const Complex *b_5, const Complex *c_5, double a, double m_5,
    int parity, bool dagger, const int *comm_override, double m_f, Dslash5TypeList<dslash5_type>,
    DistanceType<distance_pc>, TimeProfile &profile);

} // namespace quda
