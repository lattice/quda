#include <copy_gauge_offset.hpp>
#include <instantiate.h>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision prec = QUDA_@QUDA_COPY_GAUGE_PREC@_PRECISION;
  // clang-format on
  using store_t = typename precision_type_mapper<prec>::type;

  // clang-format off
  void copyGaugeOffset_@QUDA_COPY_GAUGE_PREC_LOWER@(GaugeField &out, const GaugeField &in, CommKey offset)
  // clang-format on
  {
    CopyGaugeOffset<store_t, 3>(out, in, offset);
  }

} // namespace quda
