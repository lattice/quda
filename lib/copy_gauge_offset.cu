#include <gauge_field.h>
#include <gauge_field_order.h>

#include <kernels/copy_field_offset.cuh>
#include <instantiate.h>

namespace quda
{

  template <class Field, class Element, class G>
  void copy_gauge_offset(Field &out, const Field &in, const int offset[4])
  {
    using Arg = CopyFieldOffsetArg<Field, Element, G>;
    Arg arg(out, in, offset);
    CopyFieldOffset<Arg> copier(arg, in);
  }

  template <typename Float, int nColor> struct CopyGaugeOffset {
    CopyGaugeOffset(GaugeField &out, const GaugeField &in, const int offset[4])
    {
      using Field = GaugeField;
      using real = typename mapper<Float>::type;
      using Element = Matrix<complex<real>, nColor>;

      if (in.isNative()) {
        if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_13>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
        } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type;
          copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
        } else {
          errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
        }
      } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
#ifdef BUILD_QDP_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_QDP_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("QDP interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_QDPJIT_GAUGE_ORDER) {
#ifdef BUILD_QDPJIT_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_QDPJIT_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("QDP interface has not been built\n");
#endif

      } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_MILC_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("MILC interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
#ifdef BUILD_CPS_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_CPS_WILSON_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("CPS interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {
#ifdef BUILD_BQCD_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_BQCD_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("BQCD interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
        using G = typename gauge_order_mapper<Float, QUDA_TIFR_GAUGE_ORDER, nColor>::type;
        copy_gauge_offset<Field, Element, G>(out, in, offset);
#else
        errorQuda("TIFR interface has not been built\n");
#endif
      } else {
        errorQuda("Gauge field %d order not supported", in.Order());
      }
    }
  };

  void copyFieldOffset(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    instantiate<CopyGaugeOffset>(out, in, offset);
  }

} // namespace quda
