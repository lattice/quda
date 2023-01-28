#include "multigrid.h"

namespace quda
{

  template <int...> struct IntList {
  };

  template <int Nc, int... N>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma,
                     IntList<Nc, N...>)
  {
    if (Y.Ncolor() / 2 == Nc) {
      calculateYhat<Nc>(Yhat, Xinv, Y, X, use_mma);
    } else {
      if constexpr (sizeof...(N) > 0) {
        calculateYhat(Yhat, Xinv, Y, X, use_mma, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", Y.Ncolor() / 2);
      }
    }
  }
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    if constexpr (is_enabled_multigrid()) {
      // clang-format off
      calculateYhat(Yhat, Xinv, Y, X, use_mma, IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
      // clang-format on
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
