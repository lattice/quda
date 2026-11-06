#include <copy_color_spinor_offset.hpp>
#include <instantiate.h>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision prec = QUDA_@QUDA_COPY_GAUGE_PREC@_PRECISION;
  // clang-format on
  using store_t = typename precision_type_mapper<prec>::type;

  // clang-format off
  void copyColorSpinorOffset_@QUDA_COPY_GAUGE_PREC_LOWER@(ColorSpinorField &out, const ColorSpinorField &in,
                                                          CommKey offset, QudaPCType pc_type)
  // clang-format on
  {
    CopyColorSpinorOffset<store_t, 3>(out, in, offset, pc_type);
  }

} // namespace quda
