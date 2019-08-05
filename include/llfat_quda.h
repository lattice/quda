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
  void fatLongKSLink(cudaGaugeField* fat,
		     cudaGaugeField* lng,
		     const cudaGaugeField& gauge,
		     const double* coeff);
  
} // namespace quda

#endif // _LLFAT_QUDA_H
