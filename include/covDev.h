#pragma once

namespace quda {

  /**
     @brief Driver for applying the covariant derivative

     out = U * in

     where U is the gauge field in a particular direction.

     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the covariant derivative
     @param[in] mu Direction of the derivative. For mu > 3 it goes backwards
  */
  void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int parity, int mu);

} // namespace quda
