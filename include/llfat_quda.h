#pragma once

#include "quda.h"
#include "quda_internal.h"

namespace quda {

  /**
     @brief Compute the fat links for an improved staggered (Kogut-Susskind) fermions.
     @param fat[out] The computed fat link
     @param u[in] The input gauge field
     @param coeff[in] Array of path coefficients
  */
  void fatKSLink(GaugeField *fat, const GaugeField &u, const double *coeff);

  /**
     @brief Compute the long links for an improved staggered (Kogut-Susskind) fermions.
     @param lng[out] The computed long link (only computed if lng!=0)
     @param u[in] The input gauge field
     @param coeff[in] Array of path coefficients
  */
  void longKSLink(GaugeField *lng, const GaugeField &u, const double *coeff);

} // namespace quda
