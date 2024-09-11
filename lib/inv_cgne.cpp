#include "invert_quda.h"

namespace quda
{

  CGNE::CGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
             const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param),
    mmdag(mat.Expose()),
    mmdagSloppy(matSloppy.Expose()),
    mmdagPrecon(matPrecon.Expose()),
    mmdagEig(matEig.Expose())
  {
    switch (param.inv_type) {
    case QUDA_CGNE_INVERTER: cg = std::make_unique<CG>(mmdag, mmdagSloppy, mmdagPrecon, mmdagEig, param); break;
    case QUDA_CA_CGNE_INVERTER: cg = std::make_unique<CACG>(mmdag, mmdagSloppy, mmdagPrecon, mmdagEig, param); break;
    case QUDA_CG3NE_INVERTER: cg = std::make_unique<CG3>(mmdag, mmdagSloppy, mmdagPrecon, param); break;
    default: errorQuda("Unexpected CG solver type %d", param.inv_type);
    }
  }

  void CGNE::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);
    if (!init || xe.size() != b.size()) {
      ColorSpinorParam csParam(x[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      resize(xe, b.size(), csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      resize(ye, b.size(), csParam);
      init = true;
    }
  }

  cvector_ref<const ColorSpinorField> CGNE::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    // CG residual will match the CGNE residual (FIXME: but only with zero initial guess?)
    return param.use_init_guess ? xe : cg->get_residual();
  }

  // CGNE: M Mdag y = b is solved; x = Mdag y is returned as solution.
  void CGNE::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    auto b2 = param.compute_true_res ? blas::norm2(b) : vector<double>(b.size(), 0.0);

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // compute initial residual
      mmdag.Expose()->M(xe, x);

      if (param.compute_true_res) {
        bool is_zero = true;
        for (auto i = 0u; i < b2.size(); i++) {
          is_zero = is_zero || b2[i] == 0.0;
          if (b2[i] == 0.0 && !is_zero) errorQuda("Mixture of zero and non-zero sources not supported");
        }
        if (is_zero) b2 = blas::xmyNorm(b, xe);
      } else {
        blas::xpay(b, -1.0, xe);
      }

      // compute solution to residual equation
      cg->operator()(ye, xe);

      mmdag.Expose()->Mdag(xe, ye);

      // compute full solution
      blas::xpy(xe, x);
    } else {
      cg->operator()(ye, b);
      mmdag.Expose()->Mdag(x, ye);
    }

    if (param.compute_true_res || (param.use_init_guess && param.return_residual)) {
      // compute the true residual
      mmdag.Expose()->M(xe, x);
      blas::xpay(b, -1.0, xe); // xe now holds the residual

      vector<double> r2(b2.size());
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        auto hq = blas::HeavyQuarkResidualNorm(x, xe);
        for (auto i = 0u; i < b.size(); i++) {
          param.true_res_hq[i] = sqrt(hq[i].z);
          r2[i] = hq[i].y;
        }
      } else {
        r2 = blas::norm2(xe);
      }
      for (auto i = 0u; i < b.size(); i++) param.true_res[i] = sqrt(r2[i] / b2[i]);
      PrintSummary("CGNE", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
    }
  }

} // namespace quda
