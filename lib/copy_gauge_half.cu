#include "copy_gauge_inc.cu"
namespace quda {

  // this is the function that is actually called, from here on down we instantiate all required templates
#if QUDA_PRECISION & 2
  void copyGenericGaugeHalfIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale,
                              void *Out, void *In, void **ghostOut, void **ghostIn, int type)
  {
    copyGenericGauge<short>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
  }
#else
  void copyGenericGaugeHalfIn(GaugeField &, const GaugeField &, QudaFieldLocation, double, void *, void *, void **,
                              void **, int)
  {
    errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
  }
#endif

} // namespace quda
