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

     mom = mom - [force]_TA

     where [A]_TA means the traceless anti-hermitian projection of A

     @param mom Momentum field
     @param force Force field
   */
  void updateMomentum(cudaGaugeField &mom, double coeff, cudaGaugeField &force);

} // namespace quda
