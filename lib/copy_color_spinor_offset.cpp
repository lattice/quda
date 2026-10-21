#include <color_spinor_field.h>
#include <instantiate.h>

namespace quda
{

  void copyColorSpinorOffset_double(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type);
  void copyColorSpinorOffset_single(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type);
  void copyColorSpinorOffset_half(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type);
  void copyColorSpinorOffset_quarter(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset,
                                     QudaPCType pc_type);

  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type)
  {
    checkPrecision(out, in);
    checkLocation(out, in); // check all locations match

    if (in.Ncolor() != 3) errorQuda("Unsupported number of colors %d", in.Ncolor());

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      copyColorSpinorOffset_double(out, in, offset, pc_type);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
        copyColorSpinorOffset_single(out, in, offset, pc_type);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION))
        copyColorSpinorOffset_half(out, in, offset, pc_type);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
        copyColorSpinorOffset_quarter(out, in, offset, pc_type);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d", in.Precision());
    }
  }

} // namespace quda
