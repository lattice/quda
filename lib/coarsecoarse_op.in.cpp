#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int fineColor, int coarseColor, int... N>
  void CoarseCoarseOp2(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                       const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass, double mu,
                       double mu_factor, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional,
                       bool use_mma, IntList<coarseColor, N...>)
  {
    if (Y.Ncolor() / 2 == coarseColor) {
      if constexpr (coarseColor >= fineColor) {
        if (use_mma) {
          CoarseCoarseOp<fineColor, coarseColor, true>(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor,
                                                       dirac, matpc, need_bidirectional);
        } else {
          CoarseCoarseOp<fineColor, coarseColor, false>(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor,
                                                        dirac, matpc, need_bidirectional);
        }
      } else {
        errorQuda("Invalid coarseColor = %d, cannot be less than fineColor = %d", coarseColor, fineColor);
      }
    } else {
      if constexpr (sizeof...(N) > 0) {
        CoarseCoarseOp2<fineColor>(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                   need_bidirectional, use_mma, IntList<N...>());
      } else {
        errorQuda("Coarse Nc = %d has not been instantiated", Y.Ncolor() / 2);
      }
    }
  }

  template <int fineColor, int... N>
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                      const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass, double mu,
                      double mu_factor, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional, bool use_mma,
                      IntList<fineColor, N...>)
  {
    if (gauge.Ncolor() / T.Vectors().Nspin() == fineColor) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
      // clang-format on
      CoarseCoarseOp2<fineColor>(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                 need_bidirectional, use_mma, coarseColors);
    } else {
      if constexpr (sizeof...(N) > 0) {
        CoarseCoarseOp(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor, dirac, matpc, need_bidirectional,
                       use_mma, IntList<N...>());
      } else {
        errorQuda("Fine Nc = %d has not been instantiated", gauge.Ncolor() / T.Vectors().Nspin());
      }
    }
  }

  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                      const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass, double mu,
                      double mu_factor, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional, bool use_mma)
  {
    if constexpr (is_enabled_multigrid()) {
      // clang-format off
      IntList<@QUDA_MULTIGRID_NVEC_LIST@> fineColors;
      // clang-format on
      CoarseCoarseOp(Y, X, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor, dirac, matpc, need_bidirectional,
                     use_mma, fineColors);
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
