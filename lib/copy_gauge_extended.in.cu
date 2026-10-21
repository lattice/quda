#include <copy_gauge_extended.hpp>
#include <instantiate.h>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision prec_in = QUDA_@QUDA_COPY_GAUGE_PREC_IN@_PRECISION;
  constexpr QudaPrecision prec_out = QUDA_@QUDA_COPY_GAUGE_PREC_OUT@_PRECISION;
  // clang-format on
  using store_in_t = typename precision_type_mapper<prec_in>::type;
  using store_out_t = typename precision_type_mapper<prec_out>::type;

  // clang-format off
  void copyExtendedGauge_@QUDA_COPY_GAUGE_PREC_IN_LOWER@_@QUDA_COPY_GAUGE_PREC_OUT_LOWER@(
    GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out, void *In)
  // clang-format on
  {
    copyGaugeEx(out, in, location, scale, reinterpret_cast<store_out_t *>(Out), reinterpret_cast<store_in_t *>(In));
  }

} // namespace quda
