#include "copy_gauge_inc.cu"
namespace quda {
 
  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeDoubleOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                                 void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
#if QUDA_PRECISION & 8
    copyGenericGauge<double>(out, in, location, Out, In, ghostOut, ghostIn, type);
#else
    errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
  }

} // namespace quda
