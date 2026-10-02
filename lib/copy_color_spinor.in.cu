#include <copy_color_spinor.cuh>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision prec_in = QUDA_@QUDA_COPY_GAUGE_PREC_IN@_PRECISION;
  constexpr QudaPrecision prec_out = QUDA_@QUDA_COPY_GAUGE_PREC_OUT@_PRECISION;
  // clang-format on
  using store_in_t = typename precision_type_mapper<prec_in>::type;
  using store_out_t = typename precision_type_mapper<prec_out>::type;

  // clang-format off
  void copyGenericColorSpinor_@QUDA_COPY_GAUGE_PREC_IN_LOWER@_@QUDA_COPY_GAUGE_PREC_OUT_LOWER@(const copy_pack_t &pack)
  // clang-format on
  {
    CopyGenericColorSpinor<3, store_out_t, store_in_t>(pack);
  }

} // namespace quda
