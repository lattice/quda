#pragma once

#include <quda_internal.h>
#include <gauge_field.h>

namespace quda {

  /**
     @brief Compute the 2-link field for the smearing operation
     @param[out] newTwoLink The computed 2-link output
     @param[in] link Thin-link gauge field
  */
  void computeTwoLink(GaugeField &newTwoLink, const GaugeField &link);


}  // namespace quda
