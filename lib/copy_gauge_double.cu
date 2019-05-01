#include "copy_gauge_inc.cu"
namespace quda {
 
  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeDoubleOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
      void **ghostOut, void **ghostIn, int type)
  {
    copyGenericGauge<double>(out, in, location, Out, In, ghostOut, ghostIn, type);
  }

} // namespace quda
