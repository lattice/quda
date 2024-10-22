#include <dslash_domain_wall_4d.hpp>


namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;

  template struct DomainWall4DApply<Float, nColor, DDArg, ReconstructStaggered::recon[reconI]>;
   
  template DomainWall4DApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>:: 
    DomainWall4DApply<distance_pc>(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                      cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double m_5,
                      const Complex *b_5, const Complex *c_5, int parity, bool dagger, const int *comm_override,DistanceType<distance_pc>,
                      TimeProfile &profile);
} // namespace quda
