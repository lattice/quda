#include "copy_gauge_inc.cu"
namespace quda {
 
  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeDoubleIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale,
                                void *Out, void *In, void **ghostOut, void **ghostIn, int type)
  {
    copyGenericGauge<double>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
  }

} // namespace quda
