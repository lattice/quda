#include "clover_field.h"
#include "gauge_field.h"
#include "color_spinor_field.h"
#include "momentum.h"

namespace quda {

  void clover_force(GaugeField &mom, const GaugeField &gaugeEx, const GaugeField &gauge, const CloverField &clover,
                    cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &p,
                    const std::vector<double> &coeff, const std::vector<array<double, 2>> &epsilon,
                    double sigma_coeff, bool detratio, int parity)
  {
    // create oprod and trace field
    GaugeFieldParam param(mom);
    param.link_type = QUDA_GENERAL_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    GaugeField force(param);
    param.geometry = QUDA_TENSOR_GEOMETRY;
    GaugeField oprod(param);

    // derivative of the wilson operator it correspond to deriv_Sb(OE,...) plus  deriv_Sb(EO,...) in tmLQCD
    computeCloverForce(force, gauge, x, p, coeff);
    // derivative of the determinant of the sw term, second term of (A12) in hep-lat/0112051,  sw_deriv(EE, mnl->mu) in tmLQCD
    if (!detratio) computeCloverSigmaTrace(oprod, clover, sigma_coeff, parity);

    // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus sw_spinor_eo(OO,..)  in tmLQCD
    computeCloverSigmaOprod(oprod, p, x, epsilon);

    // oprod = (A12) of hep-lat/0112051 
    // compute the insertion of oprod in Fig.27 of hep-lat/0112051 
    cloverDerivative(force, gaugeEx, oprod, 1.0);

    updateMomentum(mom, -1.0, force, "clover");
  }

}
