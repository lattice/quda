#include <gauge_ape.hpp>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision precision = QUDA_@QUDA_GAUGE_PREC@_PRECISION;
  constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_@QUDA_GAUGE_RECON@;
  // clang-format on
  using Float = precision_type_mapper<precision>::type;

  template void applyGaugeAPE<Float, reconstruct>(GaugeField &, const GaugeField &, real_t, int, real_t);

} // namespace quda
