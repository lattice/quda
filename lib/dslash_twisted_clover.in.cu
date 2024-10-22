#include <dslash_twisted_clover.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;

  template struct TwistedCloverApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;
  
  template TwistedCloverApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::TwistedCloverApply<distance_pc>(
		  cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, 
		  cvector_ref<const ColorSpinorField> &x,const GaugeField &U, 
		  const CloverField &C, double a, double b, int parity, 
		  bool dagger, const int *comm_override,
		  DistanceType<distance_pc>, TimeProfile &profile);

} // namespace quda