#include <llfat.hpp>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision precision = QUDA_@QUDA_GAUGE_PREC@_PRECISION;
  constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_@QUDA_GAUGE_RECON@;
  // clang-format on
  using Float = precision_type_mapper<precision>::type;

  template void applyLongLink<Float, reconstruct>(const GaugeField &, GaugeField &, double);
  template void applyOneLink<Float, reconstruct>(const GaugeField &, GaugeField &, double);
  template void applyStaple<Float, reconstruct>(const GaugeField &, GaugeField &, GaugeField &, const GaugeField &, int,
                                                int, int, double, bool);

} // namespace quda
