#include "copy_gauge_inc.cu"
namespace quda {

    // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeHalfOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
      void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        errorQuda("Double Precision for output not supported");
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        errorQuda("Single Precision for output not supported");
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
  copyGauge(out, in, location, (short*)Out, (double*)In, (short**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
  copyGauge(out, in, location, (short*)Out, (float*)In, (short**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
  copyGauge(out, in, location, (short*)Out, (short*)In, (short**)ghostOut, (short**)ghostIn, type);
      }
    } 
  }

} // namespace quda
