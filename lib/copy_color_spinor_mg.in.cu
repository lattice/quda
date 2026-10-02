#include <copy_color_spinor_mg.hpp>
#include <instantiate.h>

namespace quda
{

  // clang-format off
  constexpr QudaPrecision prec_in = QUDA_@QUDA_COPY_GAUGE_PREC_IN@_PRECISION;
  constexpr QudaPrecision prec_out = QUDA_@QUDA_COPY_GAUGE_PREC_OUT@_PRECISION;
  // clang-format on
  using store_in_t = typename precision_type_mapper<prec_in>::type;
  using store_out_t = typename precision_type_mapper<prec_out>::type;

  // clang-format off
  void copyGenericColorSpinorMG_@QUDA_COPY_GAUGE_PREC_IN_LOWER@_@QUDA_COPY_GAUGE_PREC_OUT_LOWER@(const copy_pack_t &pack)
  // clang-format on
  {
    constexpr bool uses_double = (prec_in == QUDA_DOUBLE_PRECISION || prec_out == QUDA_DOUBLE_PRECISION);
    if constexpr (uses_double && !is_enabled_multigrid_double()) {
      errorQuda("Double precision multigrid has not been enabled");
    } else {
      instantiateColor<store_out_t, store_in_t>(std::get<0>(pack), pack);
    }
  }

} // namespace quda
