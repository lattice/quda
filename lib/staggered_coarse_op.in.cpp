#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void StaggeredCoarseOp2(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                          const cudaGaugeField &longGauge, const GaugeField &XinvKD, double mass, bool allow_truncation,
                          QudaDiracType dirac, QudaMatPCType matpc, IntList<coarseColor, N...>)
  {
    if (Y.Ncolor() / 2 == coarseColor) {
      StaggeredCoarseOp<fineColor, coarseColor>(Y, X, T, gauge, longGauge, XinvKD, mass, allow_truncation, dirac, matpc);
    } else {
      if constexpr (sizeof...(N) > 0) {
        StaggeredCoarseOp2<fineColor>(Y, X, T, gauge, longGauge, XinvKD, mass, allow_truncation, dirac, matpc,
                                      IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", Y.Ncolor() / 2);
      }
    }
  }

  template <int fineColor, int... N>
  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                         const cudaGaugeField &longGauge, const GaugeField &XinvKD, double mass, bool allow_truncation,
                         QudaDiracType dirac, QudaMatPCType matpc, IntList<fineColor, N...>)
  {
    if (gauge.Ncolor() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      StaggeredCoarseOp2<fineColor>(Y, X, T, gauge, longGauge, XinvKD, mass, allow_truncation, dirac, matpc,
                                    coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        StaggeredCoarseOp<fineColor>(Y, X, T, gauge, longGauge, XinvKD, mass, allow_truncation, dirac, matpc,
                                     IntList<N...>());
        errorQuda("Fine Nc = %d has not been instantiated", gauge.Ncolor());
      }
    }
  }

  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                         const cudaGaugeField &longGauge, const GaugeField &XinvKD, double mass, bool allow_truncation,
                         QudaDiracType dirac, QudaMatPCType matpc)
  {
    if constexpr (is_enabled_spin(1) && is_enabled_multigrid()) {
      if (gauge.Ncolor() != 3 && longGauge.Ncolor() != 3 && XinvKD.Ncolor() != 3)
        errorQuda("Unsupported number of colors %d %d %d\n", gauge.Ncolor(), longGauge.Ncolor(), XinvKD.Ncolor());
      IntList<3> fineColors;
      StaggeredCoarseOp(Y, X, T, gauge, longGauge, XinvKD, mass, allow_truncation, dirac, matpc, fineColors);
    } else {
      errorQuda("Staggered Multigrid has not been built");
    }
  }

} // namespace quda
