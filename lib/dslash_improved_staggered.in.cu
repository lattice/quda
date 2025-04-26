#include <dslash_improved_staggered.hpp>

namespace quda
{
  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;
  constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

  using DDArg = @QUDA_DSLASH_DDARG@;
  using Float = precision_type_mapper<precision>::type;

  template struct ImprovedStaggeredApply<Float, nColor, DDArg, ReconstructStaggered::recon[reconI]>;

  template ImprovedStaggeredApply<Float, nColor, DDArg, ReconstructStaggered::recon[reconI]>::ImprovedStaggeredApply(
    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
    const GaugeField &L, const GaugeField &U, double a, int parity, bool dagger, const int *comm_override,
    DistanceType<distance_pc>, TimeProfile &profile);

} // namespace quda
