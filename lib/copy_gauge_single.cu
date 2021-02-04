#include "copy_gauge_inc.cu"
namespace quda {

#if QUDA_PRECISION & 4
  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeSingleIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                                void **ghostOut, void **ghostIn, int type)
  {
    copyGenericGauge<float>(out, in, location, Out, In, ghostOut, ghostIn, type);
  }
#else
  void copyGenericGaugeSingleIn(GaugeField &, const GaugeField &, QudaFieldLocation, void *, void *, void **, void **, int)
  {
    errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
  }
#endif

} // namespace quda
