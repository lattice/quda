#include <gauge_field_order.h>

namespace quda {
 
  void copyGenericGaugeDoubleOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
      void *Out, void *In, void **ghostOut, void **ghostIn, int type);

  void copyGenericGaugeHalfOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
      void *Out, void *In, void **ghostOut, void **ghostIn, int type);

  void copyGenericGaugeSingleOut(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
      void *Out, void *In, void **ghostOut, void **ghostIn, int type);

  void checkMomOrder(const GaugeField &u) {
    if (u.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
  errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO)
  errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
  errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else {
      errorQuda("Unsupported gauge field order %d", u.Order());
    }
  }

    // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
      void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    // do not copy the ghost zone if it does not exist
    if (type == 0 && (in.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD || 
          out.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD)) type = 2;

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      copyGenericGaugeDoubleOut(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      copyGenericGaugeSingleOut(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      copyGenericGaugeHalfOut(out, in, location, Out, In, ghostOut, ghostIn, type);
    }
  } 
 

} // namespace quda
