#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <kernels/copy_field_offset.cuh>

#include <instantiate.h>

namespace quda
{

  template <class Float, int nColor> struct CopyColorSpinorOffset {
    CopyColorSpinorOffset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
    {
      using Field = ColorSpinorField;
      using real = typename mapper<Float>::type;
      using Element = ColorSpinor<real, nColor, 4>;

      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
          using Accessor = typename colorspinor_order_mapper<Float, QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, 4, nColor>::type;
          using Arg = CopyFieldOffsetArg<Field, Element, Accessor>;
          Arg arg(out, in, offset);
          CopyFieldOffset<Arg> dummy(arg, in);
        } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          using Accessor = typename colorspinor_order_mapper<Float, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, 4, nColor>::type;
          using ArgType = CopyFieldOffsetArg<Field, Element, Accessor>;
          ArgType arg(out, in, offset);
          CopyFieldOffset<ArgType> dummy(arg, in);
        } else {
          errorQuda("Unsupported field order = %d.", in.FieldOrder());
        }
      } else {

        if (!in.isNative() || !out.isNative()) { errorQuda("CUDA field has be in native order."); }

        using Accessor = typename colorspinor_mapper<Float, 4, nColor>::type;
        using ArgType = CopyFieldOffsetArg<Field, Element, Accessor>;
        ArgType arg(out, in, offset);
        CopyFieldOffset<ArgType> dummy(arg, in);
      }
    }
  };

  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    checkLocation(out, in); // check all locations match
    instantiate<CopyColorSpinorOffset>(out, in, offset);
  }

} // namespace quda
