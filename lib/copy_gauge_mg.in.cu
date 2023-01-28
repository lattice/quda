#include "gauge_field_order.h"
#include "copy_gauge_helper.hpp"
#include "multigrid.h"

namespace quda {

  constexpr bool fine_grain() { return true; }

  template <typename sFloatOut, typename FloatIn, int Nc, typename InOrder>
  void copyGaugeMG(const InOrder &inOrder, GaugeField &out, const GaugeField &in,
		   QudaFieldLocation location, sFloatOut *Out, sFloatOut **outGhost, int type)
  {
    typedef typename mapper<sFloatOut>::type FloatOut;
    constexpr int length = 2*Nc*Nc;

    if (out.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Reconstruct type %d not supported", out.Reconstruct());

    if constexpr (fine_grain()) {
      if (out.Precision() == QUDA_HALF_PRECISION) {
        if (in.Precision() == QUDA_HALF_PRECISION) {
          out.Scale(in.Scale());
        } else {
          InOrder in_(const_cast<GaugeField &>(in));
          out.Scale(in.abs_max());
        }
      }
    }

    if (out.isNative()) {

      if constexpr (fine_grain()) {
        if (outGhost) {
          typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_FLOAT2_GAUGE_ORDER, false, sFloatOut> G;
          copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
        } else {
          typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_FLOAT2_GAUGE_ORDER, true, sFloatOut> G;
          copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
        }
      } else {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO, length>::type G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
      }

    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_QDP_GAUGE_ORDER, true, sFloatOut> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
      } else {
        typedef typename gauge::QDPOrder<FloatOut, length> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
      }

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, sFloatOut> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
      } else {
        using G = typename gauge::MILCOrder<FloatOut, length>;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, type);
      }

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <int Nc, typename sFloatOut, typename sFloatIn>
  void copyGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location, sFloatOut *Out, sFloatIn *In,
                   sFloatOut **outGhost, sFloatIn **inGhost, int type)
  {
    using FloatIn = typename mapper<sFloatIn>::type;

    if (in.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Reconstruct type %d not supported", in.Reconstruct());

    if (in.isNative()) {
      if constexpr (fine_grain()) {
        if (inGhost) {
          typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_FLOAT2_GAUGE_ORDER, false, sFloatIn> G;
          copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, Out,
                                              outGhost, type);
        } else {
          typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_FLOAT2_GAUGE_ORDER, true, sFloatIn> G;
          copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, Out,
                                              outGhost, type);
        }
      } else {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO, 2 * Nc * Nc>::type G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_QDP_GAUGE_ORDER, true, sFloatIn> G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, Out,
                                            outGhost, type);
      } else {
        using G = typename gauge::QDPOrder<FloatIn, 2 * Nc * Nc>;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
      }

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, sFloatIn> G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, Out,
                                            outGhost, type);
      } else {
        using G = typename gauge::MILCOrder<FloatIn, 2 * Nc * Nc>;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
      }

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }
  }

  template <int nColor>
  void copyGenericGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                          void *Out, void *In, void **ghostOut, void **ghostIn, int type);

  constexpr int nColor = @QUDA_MULTIGRID_NVEC@;

  template <>
  void copyGenericGaugeMG<nColor>(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                                  void *Out, void *In, void **ghostOut, void **ghostIn, int type)
  {
    if (!fine_grain() && (out.Precision() < QUDA_SINGLE_PRECISION || in.Precision() < QUDA_SINGLE_PRECISION))
      errorQuda("Precision format not supported");

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled_multigrid_double()) {
        if (in.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGaugeMG<2 * nColor>(out, in, location, (double *)Out, (double *)In, (double **)ghostOut, (double **)ghostIn, type);
        } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
          copyGaugeMG<2 * nColor>(out, in, location, (double *)Out, (float *)In, (double **)ghostOut, (float **)ghostIn, type);
        } else if (in.Precision() == QUDA_HALF_PRECISION) {
          copyGaugeMG<2 * nColor>(out, in, location, (double *)Out, (short *)In, (double **)ghostOut, (short **)ghostIn, type);
        } else {
          errorQuda("Precision %d not supported", in.Precision());
        }
      } else {
        errorQuda("Double precision multigrid has not been enabled");
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) {
          copyGaugeMG<2 * nColor>(out, in, location, (float *)Out, (double *)In, (float **)ghostOut, (double **)ghostIn, type);
        } else {
          errorQuda("Double precision multigrid has not been enabled");
        }
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeMG<2 * nColor>(out, in, location, (float*)Out, (float*)In, (float**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyGaugeMG<2 * nColor>(out, in, location, (float*)Out, (short*)In, (float**)ghostOut, (short**)ghostIn, type);
      } else {
	errorQuda("Precision %d not supported", in.Precision());
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) {
          copyGaugeMG<2 * nColor>(out, in, location, (short *)Out, (double *)In, (short **)ghostOut, (double **)ghostIn, type);
        } else {
          errorQuda("Double precision multigrid has not been enabled");
        }
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeMG<2 * nColor>(out, in, location, (short*)Out, (float*)In, (short**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyGaugeMG<2 * nColor>(out, in, location, (short*)Out, (short*)In, (short**)ghostOut, (short**)ghostIn, type);
      } else {
	errorQuda("Precision %d not supported", in.Precision());
      }
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    } 
  } 

} // namespace quda
