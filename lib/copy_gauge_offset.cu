#include <gauge_field.h>
#include <gauge_field_order.h>

#include <kernels/copy_field_offset.cuh>

namespace quda
{

  template <typename Float, int nColor> void copy_gauge_offset(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    using Field = GaugeField;
    using real = typename mapper<Float>::type;
    using Element = Matrix<complex<real>, nColor>;

    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, G>;
        Arg arg(out, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, G>;
        Arg arg(out, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, G>;
        Arg arg(out, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_13>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, G>;
        Arg arg(out, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, G>;
        Arg arg(out, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
        errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) { // TODO: Add other gauge field orders.
#ifdef BUILD_QDP_INTERFACE
      using G = typename gauge_order_mapper<Float, QUDA_QDP_GAUGE_ORDER, nColor>::type;
      using Arg = CopyFieldOffsetArg<Field, Element, G>;
      Arg arg(out, in, offset);
      CopyFieldOffset<Arg> copier(arg, in);
#else
      errorQuda("QDP interface has not been built\n");
#endif
    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
      using G = typename gauge_order_mapper<Float, QUDA_MILC_GAUGE_ORDER, nColor>::type;
      using Arg = CopyFieldOffsetArg<Field, Element, G>;
      Arg arg(out, in, offset);
      CopyFieldOffset<Arg> copier(arg, in);
#else
      errorQuda("MILC interface has not been built\n");
#endif
    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }
  }

  template <typename Float> void copy_gauge_offset(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc = %d, in.Nc = %d", out.Ncolor(), in.Ncolor());
    }

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    copy_gauge_offset<Float, 3>(out, in, offset);
  }

  void copyFieldOffset(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);

    switch (in.Precision()) {
    case QUDA_DOUBLE_PRECISION: copy_gauge_offset<double>(out, in, offset); break;
    case QUDA_SINGLE_PRECISION: copy_gauge_offset<float>(out, in, offset); break;
    case QUDA_HALF_PRECISION: copy_gauge_offset<short>(out, in, offset); break;
    case QUDA_QUARTER_PRECISION: copy_gauge_offset<int8_t>(out, in, offset); break;
    default: errorQuda("unknown precision.");
    }
  }

} // namespace quda
