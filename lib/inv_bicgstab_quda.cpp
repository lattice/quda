#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda {

  BiCGstab::BiCGstab(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), matMdagM(matEig.Expose())
  {
  }

  BiCGstab::~BiCGstab() { destroyDeflationSpace(); }

  void BiCGstab::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      ColorSpinorParam csParam(x[0]);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      resize(y, b.size(), csParam);
      resize(r, b.size(), csParam);
      csParam.setPrecision(param.precision_sloppy);
      resize(p, b.size(), csParam);
      resize(v, b.size(), csParam);
      resize(t, b.size(), csParam);

      if (param.precision_sloppy == x.Precision()) {
        create_alias(r_sloppy, r);
        if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
          create_alias(r0, b);
        } else {
          csParam.create = QUDA_NULL_FIELD_CREATE;
          resize(r0, b.size(), csParam);
          blas::copy(r0, r);
        }
      } else {
        csParam.create = QUDA_NULL_FIELD_CREATE;
        resize(r_sloppy, b.size(), csParam);
        resize(r0, b.size(), csParam);
      }

      if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) {
        create_alias(x_sloppy, x);
      } else {
        resize(x_sloppy, b.size(), csParam);
      }

      init = true;
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
    } // init
  }

  cvector_ref<const ColorSpinorField> BiCGstab::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return r;
  }

  int reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta) {
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    return updateR;
  }

  void BiCGstab::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    create(x, b);

    getProfile().TPSTART(QUDA_PROFILE_INIT);

    auto b2 = blas::norm2(b);         // norm sq of source
    vector<double> r2(b.size(), 0.0); // norm sq of residual

    // Check to see that we're not trying to invert on a zero-field source
    if (is_zero_src(x, b, b2)) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      if (param.eig_param.eig_type == QUDA_EIG_TR_LANCZOS || param.eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) {
        constructDeflationSpace(b[0], matMdagM);
      } else {
        // Use Arnoldi to inspect the space only and turn off deflation
        constructDeflationSpace(b[0], mat);
        param.deflate = false;
      }
      if (deflate_compute) {
        // compute the deflation space.
        (*eig_solve)(evecs, evals);
        if (param.deflate) {
          // double the size of the Krylov space
          extendSVDDeflationSpace();
          // populate extra memory with L/R singular vectors
          eig_solve->computeSVD(evecs, evals);
        }
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        eig_solve->computeSVD(evecs, evals);
        recompute_evals = false;
      }
    }

    // Compute initial residual depending on whether we have an initial guess or not.
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      for (auto i = 0u; i < x.size(); i++) std::swap(y[i], x[i]);
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(x);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate: Hardcoded to SVD. If maxiter == 1, this is a dummy solve
      eig_solve->deflateSVD(x, r, evecs, evals, true);

      // Compute r_defl = RHS - A * LHS
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
    }

    if (param.precision != param.precision_sloppy) {
      blas::copy(r_sloppy, r);
      blas::copy(r0, param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO ? b : r);
    }

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    auto stop_hq = std::vector(b.size(), param.tol_hq);

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    vector<double> heavy_quark_res(b.size(), 0.0);
    if (use_heavy_quark_res) {
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double delta = param.delta;

    int k = 0;
    int rUpdate = 0;

    vector<Complex> rho(b.size(), {1.0, 0.0});
    vector<Complex> rho0 = rho;
    vector<Complex> alpha(b.size(), {1.0, 0.0});
    vector<Complex> omega(b.size(), {1.0, 0.0});
    vector<Complex> beta(b.size());

    vector<double3> rho_r2(b.size());

    double rNorm = sqrt(r2[0]);
    //double r0Norm = rNorm;
    double maxrr = rNorm;
    double maxrx = rNorm;

    PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
    blas::copy(p, r_sloppy);

    bool converged = convergence(r2, heavy_quark_res, stop, stop_hq);

    // track if we just performed an exact recalculation of y, r, r2
    bool just_updated = false;

    while (!converged && k < param.maxiter) {
      just_updated = false;

      matSloppy(v, p);

      vector<Complex> r0v;
      if (param.pipeline) {
        r0v = blas::cDotProduct(r0, v);
        if (k > 0) rho = blas::cDotProduct(r0, r);
      } else {
        r0v = blas::cDotProduct(r0, v);
      }
      for (auto i = 0u; i < b.size(); i++) {
        if (abs(rho[i]) == 0.0)
          alpha[i] = 0.0;
        else
          alpha[i] = rho[i] / r0v[i];
      }

      // r -= alpha*v
      blas::caxpy(-alpha, v, r_sloppy);

      matSloppy(t, r_sloppy);

      int updateR = 0;
      if (param.pipeline) {
        // omega = (t, r) / (t, t)
        auto omega_t2_s2 = blas::cDotProductNormAB(t, r_sloppy);
        auto r0t = blas::cDotProduct(r0, t);

        for (auto i = 0u; i < b.size(); i++) {
          omega[i] = Complex {omega_t2_s2[i].x, omega_t2_s2[i].y} / omega_t2_s2[i].z;
          beta[i] = -r0t[i] / r0v[i];
          r2[i] = omega_t2_s2[i].w - real(omega[i] * conj(Complex {omega_t2_s2[i].x, omega_t2_s2[i].y}));
        }
        // now we can work out if we need to do a reliable update
        updateR = reliable(rNorm, maxrx, maxrr, r2[0], delta);
      } else {
        // omega = (t, r) / (t, t)
        auto omega_t2 = blas::cDotProductNormA(t, r_sloppy);
        for (auto i = 0u; i < b.size(); i++)
          omega[i] = Complex(omega_t2[i].x / omega_t2[i].z, omega_t2[i].y / omega_t2[i].z);
      }

      if (param.pipeline && !updateR) {
        // x += alpha*p + omega*r, r -= omega*t, p = r - beta*omega*v + beta*p
        blas::caxpbypzYmbw(alpha, p, omega, r_sloppy, x_sloppy, t);
        vector<Complex> beta_omega(b.size());
        for (auto i = 0u; i < b.size(); i++) beta_omega[i] = -beta[i] * omega[i];
        blas::cxpaypbz(r_sloppy, beta_omega, v, beta, p);
        // tripleBiCGstabUpdate(alpha, p, omega, r_sloppy, x_sloppy, t, -beta*omega, v, beta, p
      } else {
        // x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
        rho_r2 = blas::caxpbypzYmbwcDotProductUYNormY(alpha, p, omega, r_sloppy, x_sloppy, t, r0);
        rho0 = rho;
        for (auto i = 0u; i < b.size(); i++) {
          rho[i] = Complex(rho_r2[i].x, rho_r2[i].y);
          r2[i] = rho_r2[i].z;
        }
      }

      if (use_heavy_quark_res && k % heavy_quark_check == 0) {
        vector<double3> hq;

        if (x.Precision() != x_sloppy[0].Precision()) {
          hq = blas::HeavyQuarkResidualNorm(x_sloppy, r_sloppy);
        } else {
          blas::copy(r, r_sloppy);
          hq = blas::xpyHeavyQuarkResidualNorm(x, y, r);
        }
        for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
      }

      if (!param.pipeline) updateR = reliable(rNorm, maxrx, maxrr, r2[0], delta);

      if (updateR) {
        if (x.Precision() != x_sloppy[0].Precision()) blas::copy(x, x_sloppy);

        blas::xpy(x, y);

        mat(r, y);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y);
          r2 = blas::xmyNorm(b, r);
        }

        if (param.precision != param.precision_sloppy) blas::copy(r_sloppy, r);
        blas::zero(x_sloppy);

        rNorm = sqrt(r2[0]);
        maxrr = rNorm;
        maxrx = rNorm;
        // r0Norm = rNorm;
        rUpdate++;

        just_updated = true;
      }

      k++;

      PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);
      converged = convergence(r2, heavy_quark_res, stop, stop_hq);

      if (converged) {
        // make sure we've truly converged
        if (!just_updated) {
          if (x.Precision() != x_sloppy[0].Precision()) blas::copy(x, x_sloppy);
          blas::xpy(x, y);
          mat(r, y);
          r2 = blas::xmyNorm(b, r);

          if (param.deflate && sqrt(r2[0]) < param.tol_restart) {
            // Deflate and accumulate to solution vector
            eig_solve->deflate(y, r, evecs, evals, true);
            // Compute r_defl = RHS - A * LHS
            mat(r, y);
            r2 = blas::xmyNorm(b, r);
          }

          if (r[0].Precision() != r_sloppy[0].Precision()) blas::copy(r_sloppy, r);
          blas::zero(x_sloppy);

          rNorm = sqrt(r2[0]);
          maxrr = rNorm;
          maxrx = rNorm;
          // r0Norm = rNorm;
          rUpdate++;

          just_updated = true;
        }

        // explicitly compute the HQ residual if need be
        if (use_heavy_quark_res) {
          auto hq = blas::HeavyQuarkResidualNorm(y, r);
          for (auto i = 0u; i < b.size(); i++) heavy_quark_res = sqrt(hq[i].z);
        }

        // Update convergence check
        converged = convergence(r2, heavy_quark_res, stop, stop_hq);
      }

      // update p
      if ((!param.pipeline || updateR) && !converged) { // need to update if not pipeline or did a reliable update
        vector<Complex> beta_omega(b.size());
        for (auto i = 0u; i < b.size(); i++) {
          if (abs(rho[i] * alpha[i]) == 0.0)
            beta[i] = 0.0;
          else
            beta[i] = (rho[i] / rho0[i]) * (alpha[i] / omega[i]);
          beta_omega[i] = -beta[i] * omega[i];
        }
        blas::cxpaypbz(r_sloppy, beta_omega, v, beta, p);
      }
    }

    // We have a guarantee that we just converged via the true residual
    // y has already been updated
    blas::copy(x, y);

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    if (!param.is_preconditioner) {
      param.iter += k;
      if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);
    }

    logQuda(QUDA_VERBOSE, "BiCGstab: Reliable updates = %d\n", rUpdate);

    if (!param.is_preconditioner) { // do not do the below if we this is an inner solver
      // r2 was freshly computed
      auto hq = use_heavy_quark_res ? blas::HeavyQuarkResidualNorm(x, r) : vector<double3>(b.size(), {});
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(r2[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
      PrintSummary("BiCGstab", k, r2, b2, stop, stop_hq);
    }

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

} // namespace quda
