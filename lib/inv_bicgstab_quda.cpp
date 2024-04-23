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

  void BiCGstab::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);

    if (!init) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      y = ColorSpinorField(csParam);
      r = ColorSpinorField(csParam);
      csParam.setPrecision(param.precision_sloppy);
      p = ColorSpinorField(csParam);
      v = ColorSpinorField(csParam);
      t = ColorSpinorField(csParam);

      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      init = true;
    } // init
  }

  ColorSpinorField &BiCGstab::get_residual()
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

    //printf("reliable %d %e %e %e %e\n", updateR, rNorm, maxrx, maxrr, r2);

    return updateR;
  }

  void BiCGstab::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    create(x, b);

    getProfile().TPSTART(QUDA_PROFILE_INIT);

    double b2 = blas::norm2(b); // norm sq of source
    double r2 = 0.0;            // norm sq of residual

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      if (param.eig_param.eig_type == QUDA_EIG_TR_LANCZOS || param.eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) {
        constructDeflationSpace(b, matMdagM);
      } else {
        // Use Arnoldi to inspect the space only and turn off deflation
        constructDeflationSpace(b, mat);
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
      blas::copy(y, x);
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

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
        warningQuda("inverting on zero-field source");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        getProfile().TPSTOP(QUDA_PROFILE_INIT);
        return;
      } else if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
        b2 = r2;
      } else {
        errorQuda("Null vector computing requires non-zero guess!");
      }
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = r.create_alias();

      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
        r0 = b.create_alias();
      } else {
        ColorSpinorParam csParam(r);
        csParam.create = QUDA_NULL_FIELD_CREATE;
        r0 = ColorSpinorField(csParam);
        blas::copy(r0, r);
      }
    } else {
      ColorSpinorParam csParam(x);
      csParam.setPrecision(param.precision_sloppy);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      r_sloppy = ColorSpinorField(csParam);
      blas::copy(r_sloppy, r);
      r0 = ColorSpinorField(csParam);
      blas::copy(r0, r);
    }

    if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) {
      x_sloppy = x.create_alias();
      blas::zero(x_sloppy);
    } else {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy = ColorSpinorField(csParam);
    }

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double delta = param.delta;

    int k = 0;
    int rUpdate = 0;

    Complex rho(1.0, 0.0);
    Complex rho0 = rho;
    Complex alpha(1.0, 0.0);
    Complex omega(1.0, 0.0);
    Complex beta;

    double3 rho_r2;
    double3 omega_t2;

    double rNorm = sqrt(r2);
    //double r0Norm = rNorm;
    double maxrr = rNorm;
    double maxrx = rNorm;

    PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
    blas::copy(p, r_sloppy);

    bool converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

    logQuda(QUDA_DEBUG_VERBOSE, "BiCGstab debug: x2=%e, r2=%e, v2=%e, p2=%e, r0=%e, t2=%e\n", blas::norm2(x),
            blas::norm2(r_sloppy), blas::norm2(v), blas::norm2(p), blas::norm2(r0), blas::norm2(t));

    // track if we just performed an exact recalculation of y, r, r2
    bool just_updated = false;

    while (!converged && k < param.maxiter) {
      just_updated = false;

      matSloppy(v, p);

      Complex r0v;
      if (param.pipeline) {
        r0v = blas::cDotProduct(r0, v);
        if (k > 0) rho = blas::cDotProduct(r0, r);
      } else {
        r0v = blas::cDotProduct(r0, v);
      }
      if (abs(rho) == 0.0) alpha = 0.0;
      else alpha = rho / r0v;

      // r -= alpha*v
      blas::caxpy(-alpha, v, r_sloppy);

      matSloppy(t, r_sloppy);

      int updateR = 0;
      if (param.pipeline) {
        // omega = (t, r) / (t, t)
        omega_t2 = blas::cDotProductNormA(t, r_sloppy);
        Complex tr = Complex(omega_t2.x, omega_t2.y);
        double t2 = omega_t2.z;
        omega = tr / t2;
        double s2 = blas::norm2(r_sloppy);
        Complex r0t = blas::cDotProduct(r0, t);
        beta = -r0t / r0v;
        r2 = s2 - real(omega * conj(tr));
        // now we can work out if we need to do a reliable update
        updateR = reliable(rNorm, maxrx, maxrr, r2, delta);
      } else {
        // omega = (t, r) / (t, t)
        omega_t2 = blas::cDotProductNormA(t, r_sloppy);
        omega = Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);
      }

      if (param.pipeline && !updateR) {
        // x += alpha*p + omega*r, r -= omega*t, p = r - beta*omega*v + beta*p
        blas::caxpbypzYmbw(alpha, p, omega, r_sloppy, x_sloppy, t);
        blas::cxpaypbz(r_sloppy, -beta * omega, v, beta, p);
        // tripleBiCGstabUpdate(alpha, p, omega, r_sloppy, x_sloppy, t, -beta*omega, v, beta, p
      } else {
        // x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
        rho_r2 = blas::caxpbypzYmbwcDotProductUYNormY(alpha, p, omega, r_sloppy, x_sloppy, t, r0);
        rho0 = rho;
        rho = Complex(rho_r2.x, rho_r2.y);
        r2 = rho_r2.z;
      }

      if (use_heavy_quark_res && k % heavy_quark_check == 0) {
        if (&x != &x_sloppy) {
          heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x_sloppy, r_sloppy).z);
        } else {
          blas::copy(r, r_sloppy);
          heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
        }
      }

      if (!param.pipeline) updateR = reliable(rNorm, maxrx, maxrr, r2, delta);

      if (updateR) {
        if (x.Precision() != x_sloppy.Precision()) blas::copy(x, x_sloppy);

        blas::xpy(x, y);

        mat(r, y);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y);
          r2 = blas::xmyNorm(b, r);
        }

        if (x.Precision() != r_sloppy.Precision()) blas::copy(r_sloppy, r);
        blas::zero(x_sloppy);

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        // r0Norm = rNorm;
        rUpdate++;

        just_updated = true;
      }

      k++;

      PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);
      logQuda(QUDA_DEBUG_VERBOSE, "BiCGstab debug: x2=%e, r2=%e, v2=%e, p2=%e, r0=%e, t2=%e\n", blas::norm2(x),
              blas::norm2(r_sloppy), blas::norm2(v), blas::norm2(p), blas::norm2(r0), blas::norm2(t));

      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

      if (converged) {
        // make sure we've truly converged
        if (!just_updated) {
          if (x.Precision() != x_sloppy.Precision()) blas::copy(x, x_sloppy);
          blas::xpy(x, y);
          mat(r, y);
          r2 = blas::xmyNorm(b, r);

          if (param.deflate && sqrt(r2) < param.tol_restart) {
            // Deflate and accumulate to solution vector
            eig_solve->deflate(y, r, evecs, evals, true);
            // Compute r_defl = RHS - A * LHS
            mat(r, y);
            r2 = blas::xmyNorm(b, r);
          }

          if (x.Precision() != r_sloppy.Precision()) blas::copy(r_sloppy, r);
          blas::zero(x_sloppy);

          rNorm = sqrt(r2);
          maxrr = rNorm;
          maxrx = rNorm;
          // r0Norm = rNorm;
          rUpdate++;

          just_updated = true;
        }

        // explicitly compute the HQ residual if need be
        heavy_quark_res = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(y, r).z) : 0.0;

        // Update convergence check
        converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);
      }

      // update p
      if ((!param.pipeline || updateR) && !converged) { // need to update if not pipeline or did a reliable update
        if (abs(rho * alpha) == 0.0)
          beta = 0.0;
        else
          beta = (rho / rho0) * (alpha / omega);
        blas::cxpaypbz(r_sloppy, -beta * omega, v, beta, p);
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
      param.true_res = sqrt(r2 / b2);
      param.true_res_hq = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;

      PrintSummary("BiCGstab", k, r2, b2, stop, param.tol_hq);
    }

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

} // namespace quda
