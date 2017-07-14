#pragma once
#include <gauge_field.h>
#include <color_spinor_field.h>

namespace quda {

  /**
     @brief Compute the outer-product field between the staggered quark
     field's one and (for HISQ and ASQTAD) three hop sites.  E.g.,

     out[0][d](x) = (in(x+1_d) x conj(in(x)))
     out[1][d](x) = (in(x+3_d) x conj(in(x)))

     where 1_d and 3_d represent a relative shift of magnitude 1 and 3 in dimension d, respectively

     Note out[1] is only computed if nFace=3

     @param[out] out Array of nFace outer-product matrix fields
     @param[in] in Input quark field
     @param[in] coeff Coefficient
     @param[in] nFace Number of faces (1 or 3)
  */
  void computeStaggeredOprod(GaugeField *out[], ColorSpinorField& in, const double coeff[], int nFace);

} // namespace quda
