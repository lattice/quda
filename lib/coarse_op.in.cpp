#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void CoarseOp2(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge, const CloverField *clover,
                 double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                 IntList<coarseColor, N...>)
  {
    if (Y.Ncolor() / 2 == coarseColor) {
      CoarseOp<fineColor, coarseColor>(Y, X, T, gauge, clover, kappa, mass, mu, mu_factor, dirac, matpc);
    } else {
      if constexpr (sizeof...(N) > 0) {
        CoarseOp2<fineColor>(Y, X, T, gauge, clover, kappa, mass, mu, mu_factor, dirac, matpc, IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", Y.Ncolor() / 2);
      }
    }
  }

  template <int fineColor, int... N>
  void CoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge, const CloverField *clover,
                double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                IntList<fineColor, N...>)
  {
    if (gauge.Ncolor() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      CoarseOp2<fineColor>(Y, X, T, gauge, clover, kappa, mass, mu, mu_factor, dirac, matpc, coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        CoarseOp(Y, X, T, gauge, clover, kappa, mass, mu, mu_factor, dirac, matpc, IntList<N...>());
      } else {
        errorQuda("Fine Nc = %d has not been instantiated", gauge.Ncolor() / T.Vectors().Nspin());
      }
    }
  }

  void CoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge, const CloverField *clover,
                double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if constexpr (is_enabled_multigrid()) {
      IntList<3> fineColors;
      CoarseOp(Y, X, T, gauge, clover, kappa, mass, mu, mu_factor, dirac, matpc, fineColors);
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
