#include "copy_gauge_inc.cu"
namespace quda
{

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeQuarterOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out,
      void *In, void **ghostOut, void **ghostIn, int type)
  {
#if QUDA_PRECISION & 1
    copyGenericGauge<char>(out, in, location, Out, In, ghostOut, ghostIn, type);
#else
    errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
  }

} // namespace quda
