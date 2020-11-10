#include <clover_field_order.h>
#include <instantiate.h>
#include <kernels/copy_field_offset.cuh>

namespace quda
{

  using namespace clover;

  template <typename Float> struct CopyCloverOffset {
    CopyCloverOffset(CloverField &out, const CloverField &in, const int offset[4], bool inverse)
    {
      constexpr int length = 72;
      using Field = CloverField;
      using real = typename mapper<Float>::type;
      using Element = typename mapper<Float>::type;

      if (in.isNative()) {
        using C = typename clover_mapper<Float>::type;
        using Arg = CopyFieldOffsetArg<Field, Element, C>;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        Arg arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
      } else if (in.Order() == QUDA_PACKED_CLOVER_ORDER) {
        using C = QDPOrder<Float, length>;
        using Arg = CopyFieldOffsetArg<Field, Element, C>;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        Arg arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<Arg> copier(arg, in);
      } else if (in.Order() == QUDA_QDPJIT_CLOVER_ORDER) {
#ifdef BUILD_QDPJIT_INTERFACE
        using C = QDPJITOrder<Float, length>;
        using Arg = CopyFieldOffsetArg<Field, Element, C>;
        C out_accessor(out, inverse, Out);
        C in_accessor(in, inverse, In);
        Arg arg(out_accessor, out, in_accessor, in, offset);
#else
        errorQuda("QDPJIT interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_BQCD_CLOVER_ORDER) {
#ifdef BUILD_BQCD_INTERFACE
        using C = BQCDOrder<Float, length>;
        using Arg = CopyFieldOffsetArg<Field, Element, C>;
        C out_accessor(out, inverse, Out);
        C in_accessor(in, inverse, In);
        Arg arg(out_accessor, out, in_accessor, in, offset);
#else
        errorQuda("BQCD interface has not been built\n");
#endif
      } else {
        errorQuda("Clover field %d order not supported", in.Order());
      }
    }
  };

  void copyFieldOffset(CloverField &out, const CloverField &in, const int offset[4], QudaPCType pc_type)
  {
#ifdef GPU_CLOVER_DIRAC
    if (out.Precision() < QUDA_SINGLE_PRECISION && out.Order() > 4) {
      errorQuda("Fixed-point precision not supported for order %d", out.Order());
    }

    if (in.Precision() < QUDA_SINGLE_PRECISION && in.Order() > 4) {
      errorQuda("Fixed-point precision not supported for order %d", in.Order());
    }

    if (out.Precision() != in.Precision()) {
      errorQuda("Precision mismatch: %d (out) vs %d (in)", out.Precision(), in.Precision());
    }
    if (out.Order() != in.Order()) { errorQuda("Order mismatch: %d (out) vs %d (in)", out.Order(), in.Order()); }

    if (pc_type != QUDA_4D_PC) { errorQuda("Gauge field copy must use 4d even-odd preconditioning."); }

    if (in.V(true)) { instantiate<CopyCloverOffset>(out, in, offset, true); }
    if (in.V(false)) { instantiate<CopyCloverOffset>(out, in, offset, false); }
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
