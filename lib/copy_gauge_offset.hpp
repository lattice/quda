#include <gauge_field.h>
#include <gauge_field_order.h>
#include <instantiate.h>
#include "copy_field_offset.hpp"

namespace quda
{

  template <class Field, class Element, class G> void copy_gauge_offset(Field &out, const Field &in, CommKey offset)
  {
    G out_accessor(out);
    G in_accessor(in);
    CopyFieldOffsetArg<Field, Element, G> arg(out_accessor, out, in_accessor, in, offset);
    CopyFieldOffset<decltype(arg)> copier(arg, in);
  }

  template <typename Float, int nColor> struct CopyGaugeOffset {
    CopyGaugeOffset(GaugeField &out, const GaugeField &in, CommKey offset)
    {
      using Field = GaugeField;
      using real = typename mapper<Float>::type;
      using Element = Matrix<complex<real>, nColor>;

      if (in.isNative()) {
        if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_12>()) {
            using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type;
            copy_gauge_offset<Field, Element, G>(out, in, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
          }
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_8>()) {
            using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type;
            copy_gauge_offset<Field, Element, G>(out, in, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
          }
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
            using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_13>::type;
            copy_gauge_offset<Field, Element, G>(out, in, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
          }
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_9>()) {
            using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type;
            copy_gauge_offset<Field, Element, G>(out, in, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
          }
        } else {
          errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
        }
      } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_QDP_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("QDP interface has not been built\n");
        }
      } else if (in.Order() == QUDA_QDPJIT_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_QDPJIT_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("QDP interface has not been built\n");
        }

      } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_MILC_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("MILC interface has not been built\n");
        }
      } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_CPS_WILSON_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_CPS_WILSON_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("CPS interface has not been built\n");
        }
      } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_BQCD_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_BQCD_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("BQCD interface has not been built\n");
        }
      } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_TIFR_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("TIFR interface has not been built\n");
        }
      } else if (in.Order() == QUDA_OPENQCD_GAUGE_ORDER) {
        if constexpr (is_enabled<QUDA_OPENQCD_GAUGE_ORDER>()) {
          using G = typename gauge_order_mapper<Float, QUDA_OPENQCD_GAUGE_ORDER, nColor>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else {
          errorQuda("OpenQCD interface has not been built\n");
        }
      } else {
        errorQuda("Gauge field %d order not supported", in.Order());
      }
    }
  };

} // namespace quda
