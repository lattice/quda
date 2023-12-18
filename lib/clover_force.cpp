#include "clover_field.h"
#include "gauge_field.h"
#include "color_spinor_field.h"
#include "momentum.h"
#include "blas_quda.h"
#include "dirac_quda.h"

namespace quda {

  void computeCloverForce(GaugeField &mom, const GaugeField &gaugeEx, const GaugeField &gauge,
                          const CloverField &clover, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &x0,
                          const std::vector<double> &coeff, const std::vector<array<double, 2>> &epsilon,
                          double sigma_coeff, bool detratio, QudaInvertParam &inv_param)
  {
    if (!inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC & !inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("MatPC type %d not supported", inv_param.matpc_type);

    QudaParity parity = inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
    QudaParity other_parity = static_cast<QudaParity>(1 - parity);
    bool dagger = inv_param.dagger;
    bool not_dagger = static_cast<QudaDagType>(1 - inv_param.dagger);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, true);
    Dirac *dirac = Dirac::create(diracParam);

    ColorSpinorParam csParam(x[0]);
    csParam.create = QUDA_NULL_FIELD_CREATE;
    std::vector<ColorSpinorField> p(x.size());
    for (auto i = 0u; i < p.size(); i++) p[i] = ColorSpinorField(csParam);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    for (auto i = 0u; i < x.size(); i++) {
      gamma5(p[i][parity], x[i][parity]);

      if (dagger) dirac->Dagger(QUDA_DAG_YES);
      dirac->Dslash(x[i][other_parity], p[i][parity], other_parity);
      // want to apply \hat Q_{-} = \hat M_{+}^\dagger \gamma_5 to get Y_o
      dirac->M(p[i][parity], p[i][parity]); // this is the odd part of Y
      if (dagger) dirac->Dagger(QUDA_DAG_NO);

      gamma5(x[i][other_parity], x[i][other_parity]);
      if (detratio) blas::xpy(x0[i][parity], p[i][parity]);

      if (not_dagger) dirac->Dagger(QUDA_DAG_YES);
      dirac->Dslash(p[i][other_parity], p[i][parity], other_parity); // and now the even part of Y
      if (not_dagger) dirac->Dagger(QUDA_DAG_NO);
      // up to here x.odd match X.odd in tmLQCD and p.odd=-Y.odd of tmLQCD
      // x.Even= X.Even.tmLQCD*kappa and p.Even=-Y.Even.tmLQCD*kappa

      // the gamma5 application in tmLQCD is done inside deriv_Sb
      gamma5(p[i], p[i]);
    }

    delete dirac;

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
    if (!detratio) computeCloverSigmaTrace(oprod, clover, sigma_coeff, other_parity);

    // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus sw_spinor_eo(OO,..)  in tmLQCD
    computeCloverSigmaOprod(oprod, inv_param.dagger == QUDA_DAG_YES ? p : x, inv_param.dagger == QUDA_DAG_YES ? x : p,
                            epsilon);

    // oprod = (A12) of hep-lat/0112051 
    // compute the insertion of oprod in Fig.27 of hep-lat/0112051 
    cloverDerivative(force, gaugeEx, oprod, 1.0);

    updateMomentum(mom, -1.0, force, "clover");

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

}
