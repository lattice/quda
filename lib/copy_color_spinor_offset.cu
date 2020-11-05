#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <kernels/copy_field_offset.cuh>

namespace quda
{

  template <typename Float, int nColor>
  void copy_color_spinor_offset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
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

  // template on the number of colors
  template <typename Float>
  void copy_color_spinor_offset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    switch (in.Ncolor()) {
    case 3: copy_color_spinor_offset<Float, 3>(out, in, offset); break;
    default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    checkLocation(out, in); // check all locations match

    switch (checkPrecision(out, in)) {
    case QUDA_DOUBLE_PRECISION: copy_color_spinor_offset<double>(out, in, offset); break;
    case QUDA_SINGLE_PRECISION: copy_color_spinor_offset<float>(out, in, offset); break;
    case QUDA_HALF_PRECISION: copy_color_spinor_offset<short>(out, in, offset); break;
    case QUDA_QUARTER_PRECISION: copy_color_spinor_offset<int8_t>(out, in, offset); break;
    default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
  }

} // namespace quda
