#include <dslash_twisted_mass_preconditioned.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;

  template struct TwistedMassPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template TwistedMassPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::TwistedMassPreconditionedApply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, double a, double b, bool xpay, int parity, bool dagger, bool asymmetric,
    const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile);
} // namespace quda
