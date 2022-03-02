#include <gauge_field.h>

namespace quda {

  void copyGenericGaugeDoubleIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                                void **ghostOut, void **ghostIn, int type);

  void copyGenericGaugeSingleIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                                void **ghostOut, void **ghostIn, int type);

  void copyGenericGaugeHalfIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                              void **ghostOut, void **ghostIn, int type);

  void copyGenericGaugeQuarterIn(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                                 void **ghostOut, void **ghostIn, int type);

  // specialized variation where we restrict different field orders supported but instantiate different colors
  // this, as with all of the above are hacks until JIT is supported
  void copyGenericGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
			  void *Out, void *In, void **ghostOut, void **ghostIn, int type);

  void checkMomOrder(const GaugeField &u) {
    if (u.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10 and u.Reconstruct() != QUDA_RECONSTRUCT_NO)
	errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER || u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO)
	errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
	errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
        errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
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

    if (in.Ncolor() != out.Ncolor())
      errorQuda("Colors (%d,%d) do not match", out.Ncolor(), in.Ncolor());

    if (out.Geometry() != in.Geometry())
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());

    if (in.Ncolor() != 3) {
      copyGenericGaugeMG(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      copyGenericGaugeDoubleIn(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      copyGenericGaugeSingleIn(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      copyGenericGaugeHalfIn(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      copyGenericGaugeQuarterIn(out, in, location, Out, In, ghostOut, ghostIn, type);
    } else {
      errorQuda("Unknown precision %d", out.Precision());
    }
  } 
 

} // namespace quda
