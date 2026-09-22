#include <gauge_field.h>

namespace quda
{

  void copyGaugeOffset_double(GaugeField &out, const GaugeField &in, CommKey offset);
#if QUDA_PRECISION & 4
  void copyGaugeOffset_single(GaugeField &out, const GaugeField &in, CommKey offset);
#endif
#if QUDA_PRECISION & 2
  void copyGaugeOffset_half(GaugeField &out, const GaugeField &in, CommKey offset);
#endif
#if QUDA_PRECISION & 1
  void copyGaugeOffset_quarter(GaugeField &out, const GaugeField &in, CommKey offset);
#endif

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
#if QUDA_PRECISION & 4
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      copyGaugeOffset_single(out, in, offset);
#endif
#if QUDA_PRECISION & 2
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      copyGaugeOffset_half(out, in, offset);
#endif
#if QUDA_PRECISION & 1
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      copyGaugeOffset_quarter(out, in, offset);
#endif
    } else {
      errorQuda("Unsupported precision %d", in.Precision());
    }
  }

} // namespace quda
