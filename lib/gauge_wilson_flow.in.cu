#include <gauge_wilson_flow.hpp>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision precision = QUDA_@QUDA_GAUGE_PREC@_PRECISION;
  constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_@QUDA_GAUGE_RECON@;
  // clang-format on
  using Float = precision_type_mapper<precision>::type;

  template void applyGaugeWFlowStep<Float, reconstruct>(GaugeField &, GaugeField &, const GaugeField &, real_t, real_t,
                                                        QudaGaugeSmearType, QudaWFlowStepType);

} // namespace quda
