#include "invert_quda.h"

namespace quda
{

  CGNR::CGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
             const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, mdagmSloppy, mdagmPrecon, mdagmEig, param),
    mdagm(mat.Expose()),
    mdagmSloppy(matSloppy.Expose()),
    mdagmPrecon(matPrecon.Expose()),
    mdagmEig(matEig.Expose())
  {
    switch (param.inv_type) {
    case QUDA_CGNR_INVERTER: cg = std::make_unique<CG>(mdagm, mdagmSloppy, mdagmPrecon, mdagmEig, param); break;
    case QUDA_CA_CGNR_INVERTER: cg = std::make_unique<CACG>(mdagm, mdagmSloppy, mdagmPrecon, mdagmEig, param); break;
    case QUDA_CG3NR_INVERTER: cg = std::make_unique<CG3>(mdagm, mdagmSloppy, mdagmPrecon, param); break;
    default: errorQuda("Unexpected CG solver type %d", param.inv_type);
    }
  }

  void CGNR::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);
    if (!init || br.size() != b.size()) {
      ColorSpinorParam csParam(b[0]);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      resize(br, b.size(), csParam);
      init = true;
    }
  }

  cvector_ref<const ColorSpinorField> CGNR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return br;
  }

  // CGNR: Mdag M x = Mdag b is solved.
  void CGNR::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    vector<double> b2(b.size(), 0.0);
    if (param.compute_true_res) {
      b2 = blas::norm2(b);
      bool is_zero = true;
      for (auto i = 0u; i < b2.size(); i++) {
        is_zero = is_zero && b2[i] == 0.0;
        if (b2[i] == 0.0 && !is_zero) errorQuda("Mixture of zero and non-zero sources not supported");
      }
      if (is_zero) { // compute initial residual vector
        mdagm.Expose()->M(br, x);
        b2 = blas::norm2(br);
      }
    }

    mdagm.Expose()->Mdag(br, b);
    cg->operator()(x, br);

    if (param.compute_true_res || param.return_residual) {
      // compute the true residual
      mdagm.Expose()->M(br, x);
      blas::xpay(b, -1.0, br); // br now holds the residual

      if (param.compute_true_res) {
        vector<double> r2(b.size());
        if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          auto hq = blas::HeavyQuarkResidualNorm(x, br);
          for (auto i = 0u; i < b.size(); i++) {
            param.true_res_hq[i] = sqrt(hq[i].z);
            r2[i] = hq[i].y;
          }
        } else {
          r2 = blas::norm2(br);
        }
        for (auto i = 0u; i < b.size(); i++) param.true_res[i] = sqrt(r2[i] / b2[i]);
        PrintSummary("CGNR", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
      }
    }
  }

} // namespace quda
