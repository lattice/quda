#include <dslash_wilson.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  using DDArg = @QUDA_DSLASH_DDARG@;
  using Float = precision_type_mapper<precision>::type;

  template struct WilsonApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template WilsonApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::WilsonApply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, double a, double alpha0, int t0, int parity, bool dagger, const int *comm_override,
    DistanceType<true>, TimeProfile &profile);

} // namespace quda
