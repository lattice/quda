#include "copy_gauge_inc.cu"
namespace quda {

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeHalfOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                               void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
        copyGauge(out, in, location, (short*)Out, (double*)In, (short**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
        copyGauge(out, in, location, (short*)Out, (float*)In, (short**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
        copyGauge(out, in, location, (short*)Out, (short*)In, (short**)ghostOut, (short**)ghostIn, type);
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
        copyGauge(out, in, location, (short*)Out, (char*)In, (short**)ghostOut, (char**)ghostIn, type);
      } else {
        errorQuda("Unsupported precision %d", in.Precision());
      }
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
  }

} // namespace quda
