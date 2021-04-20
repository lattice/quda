#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <instantiate.h>
#include "copy_field_offset.hpp"

namespace quda
{

  template <class Field, class Element, class F>
  void copy_color_spinor_offset(Field &out, const Field &in, CommKey offset, QudaPCType pc_type)
  {
    F out_accessor(out);
    F in_accessor(in);
    if (pc_type == QUDA_4D_PC) {
      CopyFieldOffsetArg<Field, Element, F, QUDA_4D_PC> arg(out_accessor, out, in_accessor, in, offset);
      CopyFieldOffset<decltype(arg)> copier(arg, in);
    } else if (pc_type == QUDA_5D_PC) {
      CopyFieldOffsetArg<Field, Element, F, QUDA_5D_PC> arg(out_accessor, out, in_accessor, in, offset);
      CopyFieldOffset<decltype(arg)> copier(arg, in);
    } else {
      errorQuda("pc_type should either be QUDA_4D_PC or QUDA_5D_PC.\n");
    }
  }

  template <class Float, int nColor, int nSpin>
  void copy_color_spinor_offset(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type)
  {
    using Field = ColorSpinorField;
    using real = typename mapper<Float>::type;
    using Element = ColorSpinor<real, nColor, nSpin>;

    if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
      if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
        using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, nSpin, nColor>::type;
        copy_color_spinor_offset<Field, Element, F>(out, in, offset, pc_type);
      } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
        using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, nSpin, nColor>::type;
        copy_color_spinor_offset<Field, Element, F>(out, in, offset, pc_type);
      } else {
        errorQuda("Unsupported field order = %d.", in.FieldOrder());
      }
    } else {
      if (!in.isNative() || !out.isNative()) { errorQuda("CUDA field has be in native order."); }

      using F = typename colorspinor_mapper<Float, nSpin, nColor>::type;
      copy_color_spinor_offset<Field, Element, F>(out, in, offset, pc_type);
    }
  }

  template <class Float, int nColor> struct CopyColorSpinorOffset {
    CopyColorSpinorOffset(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type)
    {
      if (in.Nspin() == 4) {
        copy_color_spinor_offset<Float, nColor, 4>(out, in, offset, pc_type);
      } else if (in.Nspin() == 1) {
        copy_color_spinor_offset<Float, nColor, 1>(out, in, offset, pc_type);
      } else {
        errorQuda("Unsupported spin = %d.\n", in.Nspin());
      }
    }
  };

  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type)
  {
    checkPrecision(out, in);
    checkLocation(out, in); // check all locations match
    instantiate<CopyColorSpinorOffset>(out, in, offset, pc_type);
  }

} // namespace quda
