#include <gauge_hyp.hpp>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision precision = QUDA_@QUDA_GAUGE_PREC@_PRECISION;
  constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_@QUDA_GAUGE_RECON@;
  // clang-format on
  using Float = precision_type_mapper<precision>::type;

  template void applyGaugeHYP<Float, reconstruct>(GaugeField &, GaugeField *[4], const GaugeField &, real_t, int, int);

} // namespace quda
