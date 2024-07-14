#include "clover_field.h"
#include "gauge_field.h"
#include "color_spinor_field.h"
#include "momentum.h"
#include "blas_quda.h"
#include "dirac_quda.h"

namespace quda
{

  void computeCloverForce(GaugeField &mom, const GaugeField &gaugeEx, const GaugeField &gauge,
                          const CloverField &clover, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &x0,
                          const std::vector<double> &coeff, const std::vector<array<double, 2>> &epsilon,
                          double sigma_coeff, bool detratio, QudaInvertParam &inv_param)
  {
    if (inv_param.matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC && inv_param.matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC)
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
    std::vector<ColorSpinorField> p_(x.size());
    for (auto i = 0u; i < x.size(); i++) p_[i] = ColorSpinorField(csParam);
    auto p = vector_ref<ColorSpinorField>(p_);

    // create oprod and trace field
    GaugeFieldParam param(mom);
    param.link_type = QUDA_GENERAL_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    GaugeField force(param);
    param.geometry = QUDA_TENSOR_GEOMETRY;
    GaugeField oprod(param);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    gamma5(p(parity), x(parity));
    if (dagger) dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(x(other_parity), p(parity), other_parity);
    // want to apply \hat Q_{-} = \hat M_{+}^\dagger \gamma_5 to get Y_o
    dirac->M(p(parity), p(parity)); // this is the odd part of Y
    if (dagger) dirac->Dagger(QUDA_DAG_NO);

    if (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      blas::ax(1.0 / inv_param.evmax, p(parity));
      ApplyTau(x(other_parity), x(other_parity), 1);
      ApplyTau(p(parity), p(parity), 1);
      std::vector<Complex> a(x.size());
      for (auto i = 0u; i < x.size(); i++) a[i] = {0.0, -inv_param.offset[i]};
      blas::caxpy(a, x(parity), p(parity));
    }

    gamma5(x(other_parity), x(other_parity));
    if (detratio && inv_param.twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) blas::xpy(x0(parity), p(parity));

    if (not_dagger || inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) dirac->Dagger(QUDA_DAG_YES);
    if (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) gamma5(p(parity), p(parity));
    dirac->Dslash(p(other_parity), p(parity), other_parity); // and now the even part of Y
    if (not_dagger || inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) dirac->Dagger(QUDA_DAG_NO);

    if (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      ApplyTau(p(other_parity), p(other_parity), 1);
      // up to here x.odd match X.odd in tmLQCD and p.odd=- gamma5 Y.odd of tmLQCD
      // x.Even= X.Even.tmLQCD/kappa and p.Even=- gamma5 Y.Even.tmLQCD/kappa
      // the gamma5 application in tmLQCD inside deriv_Sb is otimized away in here
    } else {
      // up to here x.odd match X.odd in tmLQCD and p.odd=-Y.odd of tmLQCD
      // x.Even= X.Even.tmLQCD/kappa and p.Even=-Y.Even.tmLQCD/kappa
      // the gamma5 application in tmLQCD is done inside deriv_Sb
      gamma5(p, p);
    }

    // derivative of the wilson operator it correspond to deriv_Sb(OE,...) plus  deriv_Sb(EO,...) in tmLQCD
    computeCloverOprod(force, gauge, inv_param.dagger == QUDA_DAG_YES ? p : x, inv_param.dagger == QUDA_DAG_YES ? x : p,
                       coeff);
    // derivative of the determinant of the sw term, second term of (A12) in hep-lat/0112051,  sw_deriv(EE, mnl->mu) in tmLQCD
    if (!detratio) computeCloverSigmaTrace(oprod, clover, sigma_coeff, other_parity);

    // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus
    // sw_spinor_eo(OO,..)  in tmLQCD
    if (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      ApplyTau(p(parity), p(parity), 1);
      ApplyTau(p(other_parity), p(other_parity), 1);
    }
    computeCloverSigmaOprod(oprod, inv_param.dagger == QUDA_DAG_YES ? p : x, inv_param.dagger == QUDA_DAG_YES ? x : p,
                            epsilon);
    p_.clear(); // deallocate the p vectors prior to cloverDerivative to reduce footprint

    // oprod = (A12) of hep-lat/0112051
    // compute the insertion of oprod in Fig.27 of hep-lat/0112051
    cloverDerivative(force, gaugeEx, oprod, 1.0);

    updateMomentum(mom, -1.0, force, "clover");

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    delete dirac;
  }

} // namespace quda
