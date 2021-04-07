#include <clover_field_order.h>
#include <instantiate.h>
#include "copy_field_offset.hpp"

namespace quda
{

  using namespace clover;

  template <typename Float> struct CopyCloverOffset {
    CopyCloverOffset(CloverField &out, const CloverField &in, CommKey offset, bool inverse)
    {
      constexpr int length = 72;
      using Field = CloverField;
      using Element = typename mapper<Float>::type;

      if (in.isNative()) {
        using C = typename clover_mapper<Float>::type;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        CopyFieldOffsetArg<Field, Element, C> arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<decltype(arg)> copier(arg, in);
      } else if (in.Order() == QUDA_PACKED_CLOVER_ORDER) {
        using C = QDPOrder<Float, length>;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        CopyFieldOffsetArg<Field, Element, C> arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<decltype(arg)> copier(arg, in);
      } else if (in.Order() == QUDA_QDPJIT_CLOVER_ORDER) {
#ifdef BUILD_QDPJIT_INTERFACE
        using C = QDPJITOrder<Float, length>;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        CopyFieldOffsetArg<Field, Element, C> arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<decltype(arg)> copier(arg, in);
#else
        errorQuda("QDPJIT interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_BQCD_CLOVER_ORDER) {
#ifdef BUILD_BQCD_INTERFACE
        using C = BQCDOrder<Float, length>;
        C out_accessor(out, inverse);
        C in_accessor(in, inverse);
        CopyFieldOffsetArg<Field, Element, C> arg(out_accessor, out, in_accessor, in, offset);
        CopyFieldOffset<decltype(arg)> copier(arg, in);
#else
        errorQuda("BQCD interface has not been built\n");
#endif
      } else {
        errorQuda("Clover field %d order not supported", in.Order());
      }
    }
  };

#ifdef GPU_CLOVER_DIRAC
  void copyFieldOffset(CloverField &out, const CloverField &in, CommKey offset, QudaPCType pc_type)
  {
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
  }
#else
  void copyFieldOffset(CloverField &, const CloverField &, CommKey, QudaPCType)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda
