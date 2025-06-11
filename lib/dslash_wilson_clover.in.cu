#include <dslash_wilson_clover.hpp>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  using DDArg = @QUDA_DSLASH_DDARG@;
  using Float = precision_type_mapper<precision>::type;

  template struct WilsonCloverApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template WilsonCloverApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::WilsonCloverApply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, const CloverField &A, double a, double alpha0, int t0, int parity, bool dagger,
    const int *comm_override, DistanceType<false>, TimeProfile &profile);

} // namespace quda
