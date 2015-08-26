#pragma once
#include <gauge_field.h>

namespace quda {

  /**
     @brief Compute and return global the momentum action 1/2 mom^2
     @param mom Momentum field
     @return Momentum action contribution
   */
  double computeMomAction(const GaugeField &mom);

}
