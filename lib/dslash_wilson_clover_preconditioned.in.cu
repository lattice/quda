#include <dslash_wilson_clover_preconditioned.hpp>

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

  template struct WilsonCloverPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

  template WilsonCloverPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>::WilsonCloverPreconditionedApply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &U, const CloverField &A, real_t a, real_t alpha0, int t0, int parity, bool dagger,
    const int *comm_override, DistanceType<false>, TimeProfile &profile);

} // namespace quda
