#include <copy_gauge_mg.hpp>
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
  template <int nColor>
  void copyGenericGaugeMG_@QUDA_COPY_GAUGE_PREC_IN_LOWER@_@QUDA_COPY_GAUGE_PREC_OUT_LOWER@(
    GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out, void *In,
    void **ghostOut, void **ghostIn, int type);
  // clang-format on

  // clang-format off
  constexpr int nColor = @QUDA_MULTIGRID_NVEC@;
  // clang-format on

  // clang-format off
  template <>
  void copyGenericGaugeMG_@QUDA_COPY_GAUGE_PREC_IN_LOWER@_@QUDA_COPY_GAUGE_PREC_OUT_LOWER@<nColor>(
    GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out, void *In,
    void **ghostOut, void **ghostIn, int type)
  // clang-format on
  {
    if (!fine_grain() && (out.Precision() < QUDA_SINGLE_PRECISION || in.Precision() < QUDA_SINGLE_PRECISION))
      errorQuda("Precision format not supported");

    constexpr bool uses_double = (prec_in == QUDA_DOUBLE_PRECISION || prec_out == QUDA_DOUBLE_PRECISION);
    if constexpr (uses_double && !is_enabled_multigrid_double()) {
      errorQuda("Double precision multigrid has not been enabled");
    } else {
      copyGaugeMG<2 * nColor>(out, in, location, scale, (store_out_t *)Out, (store_in_t *)In, (store_out_t **)ghostOut,
                              (store_in_t **)ghostIn, type);
    }
  }

} // namespace quda
