#include <gauge_field.h>
#include <instantiate.h>

namespace quda
{

  void copyGaugeOffset_double(GaugeField &out, const GaugeField &in, CommKey offset);
  void copyGaugeOffset_single(GaugeField &out, const GaugeField &in, CommKey offset);
  void copyGaugeOffset_half(GaugeField &out, const GaugeField &in, CommKey offset);
  void copyGaugeOffset_quarter(GaugeField &out, const GaugeField &in, CommKey offset);

  void copyFieldOffset(GaugeField &out, const GaugeField &in, CommKey offset, QudaPCType pc_type)
  {
    checkPrecision(out, in);
    checkLocation(out, in); // check all locations match
    checkReconstruct(out, in);

    if (pc_type != QUDA_4D_PC) { errorQuda("Gauge field copy must use 4d even-odd preconditioning."); }

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      copyGaugeOffset_double(out, in, offset);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
        copyGaugeOffset_single(out, in, offset);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION))
        copyGaugeOffset_half(out, in, offset);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
        copyGaugeOffset_quarter(out, in, offset);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d", in.Precision());
    }
  }

} // namespace quda
