#pragma once
#include <gauge_field.h>

namespace quda {

  /**
     @brief Compute and return global the momentum action 1/2 mom^2
     @param mom Momentum field
     @return Momentum action contribution
   */
  double computeMomAction(const GaugeField &mom);

  /**
     Update the momentum field from the force field

     mom = mom - coeff * [force]_TA

     where [A]_TA means the traceless anti-hermitian projection of A

     @param mom Momentum field
     @param force Force field
   */
  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force);

  /**
     Left multiply the force field by the gauge field

     force = U * force

     @param force Force field
     @param U Gauge field
   */
  void applyU(GaugeField &force, GaugeField &U);

} // namespace quda
