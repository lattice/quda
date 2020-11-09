#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <kernels/copy_field_offset.cuh>

#include <instantiate.h>

namespace quda
{

  template <class Field, class Element, class F>
  void copy_color_spinor_offset(Field &out, const Field &in, const int offset[4])
  {
    using Arg = CopyFieldOffsetArg<Field, Element, F>;
    F out_accessor(out);
    F in_accessor(in);
    Arg arg(out_accessor, out, in_accessor, in, offset);
    CopyFieldOffset<Arg> copier(arg, in);
  }

  template <class Float, int nColor> struct CopyColorSpinorOffset {
    CopyColorSpinorOffset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
    {
      using Field = ColorSpinorField;
      using real = typename mapper<Float>::type;
      using Element = ColorSpinor<real, nColor, 4>;

      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
          using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, 4, nColor>::type;
          copy_color_spinor_offset<Field, Element, F>(out, in, offset);
        } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, 4, nColor>::type;
          copy_color_spinor_offset<Field, Element, F>(out, in, offset);
        } else {
          errorQuda("Unsupported field order = %d.", in.FieldOrder());
        }
      } else {

        if (!in.isNative() || !out.isNative()) { errorQuda("CUDA field has be in native order."); }

        using F = typename colorspinor_mapper<Float, 4, nColor>::type;
        copy_color_spinor_offset<Field, Element, F>(out, in, offset);
      }
    }
  };

  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    checkPrecision(out, in);
    checkLocation(out, in); // check all locations match
    instantiate<CopyColorSpinorOffset>(out, in, offset);
  }

} // namespace quda
