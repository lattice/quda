#pragma once

namespace quda {

  /**
     @brief Driver for applying the Laplace stencil

     out = - kappa * A * in

     where A is the gauge laplace linear operator.

     If x is defined, the operation is given by out = x - kappa * A in.
     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the gauge Laplace
     @param[in] kappa Scale factor applied
     @param[in] x Vector field we accumulate onto to
  */
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		    double kappa, const ColorSpinorField *x, int parity);

} // namespace quda
