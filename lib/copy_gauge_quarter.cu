#include "copy_gauge_inc.cu"
namespace quda {

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeQuarterOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                                  void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    if (out.Precision() == QUDA_QUARTER_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
        copyGauge(out, in, location, (char*)Out, (double*)In, (char**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
        copyGauge(out, in, location, (char*)Out, (float*)In, (char**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
        copyGauge(out, in, location, (char*)Out, (short*)In, (char**)ghostOut, (short**)ghostIn, type);
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
        copyGauge(out, in, location, (char*)Out, (char*)In, (char**)ghostOut, (char**)ghostIn, type);
      } else {
        errorQuda("Unsupported precision %d", in.Precision());
      }
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
  }

} // namespace quda
