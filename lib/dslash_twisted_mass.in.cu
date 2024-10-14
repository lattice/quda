#include <dslash_twisted_mass.hpp>

namespace quda
{
  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  template struct TwistedMassApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>; 

  template TwistedMassApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::TwistedMassApply<distance_pc>(
		  cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                  cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b, int parity,
                  bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile);

} // namespace quda
