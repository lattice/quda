#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"
#include "quda_internal.h"

namespace quda {

  /**
     @brief Compute the fat and long links for an improved staggered
     (Kogut-Susskind) fermions.
     @param fat[out] The computed fat link
     @param lng[out] The computed long link (only computed if lng!=0)
     @param u[in] The input gauge field
     @param coeff[in] Array of path coefficients
  */
  void fatLongKSLink(GaugeField *fat, GaugeField *lng, const GaugeField &u, const double *coeff);

} // namespace quda

#endif // _LLFAT_QUDA_H
