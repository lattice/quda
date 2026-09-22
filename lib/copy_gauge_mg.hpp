#include "gauge_field_order.h"
#include "copy_gauge_helper.hpp"
#include "multigrid.h"

namespace quda {

  constexpr bool fine_grain() { return true; }

  template <typename sFloatOut, typename FloatIn, int Nc, typename InOrder>
  void copyGaugeMG(const InOrder &inOrder, GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                   double scale, sFloatOut *Out, sFloatOut **outGhost, int type)
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
          typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_NATIVE_GAUGE_ORDER, false, sFloatOut> G;
          copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                             type);
        } else {
          typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_NATIVE_GAUGE_ORDER, true, sFloatOut> G;
          copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                             type);
        }
      } else {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO, length>::type G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                           type);
      }

    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_QDP_GAUGE_ORDER, true, sFloatOut> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                           type);
      } else {
        typedef typename gauge::QDPOrder<FloatOut, length> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                           type);
      }

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatOut, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, sFloatOut> G;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                           type);
      } else {
        using G = typename gauge::MILCOrder<FloatOut, length>;
        copyGauge<FloatOut, FloatIn, length, fine_grain()>(G(out, Out, outGhost), inOrder, out, in, location, scale,
                                                           type);
      }

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <int Nc, typename sFloatOut, typename sFloatIn>
  void copyGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, sFloatOut *Out,
                   sFloatIn *In, sFloatOut **outGhost, sFloatIn **inGhost, int type)
  {
    using FloatIn = typename mapper<sFloatIn>::type;

    if (in.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Reconstruct type %d not supported", in.Reconstruct());

    if (in.isNative()) {
      if constexpr (fine_grain()) {
        if (inGhost) {
          typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_NATIVE_GAUGE_ORDER, false, sFloatIn> G;
          copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, scale,
                                              Out, outGhost, type);
        } else {
          typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_NATIVE_GAUGE_ORDER, true, sFloatIn> G;
          copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, scale,
                                              Out, outGhost, type);
        }
      } else {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO, 2 * Nc * Nc>::type G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, scale, Out, outGhost, type);
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_QDP_GAUGE_ORDER, true, sFloatIn> G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, scale, Out,
                                            outGhost, type);
      } else {
        using G = typename gauge::QDPOrder<FloatIn, 2 * Nc * Nc>;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, scale, Out, outGhost, type);
      }

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (fine_grain()) {
        typedef typename gauge::FieldOrder<FloatIn, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, sFloatIn> G;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(const_cast<GaugeField &>(in), In, inGhost), out, in, location, scale, Out,
                                            outGhost, type);
      } else {
        using G = typename gauge::MILCOrder<FloatIn, 2 * Nc * Nc>;
        copyGaugeMG<sFloatOut, FloatIn, Nc>(G(in, In, inGhost), out, in, location, scale, Out, outGhost, type);
      }

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }
  }

} // namespace quda
