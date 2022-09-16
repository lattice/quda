#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

namespace quda {

  CG3::CG3(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
           TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matPrecon, param, profile)
  {
  }

  CG3::~CG3()
  {
    if ( init ) {
      delete rp;
      delete yp;
      delete tmpp;
      delete ArSp;
      delete rS_oldp;
      if (param.precision != param.precision_sloppy) {
        delete rSp;
        delete xSp;
        delete xS_oldp;
        delete tmpSp;
      }
    }
  }

  CG3NE::CG3NE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
               TimeProfile &profile) :
    CG3(mmdag, mmdagSloppy, mmdagPrecon, param, profile),
    mmdag(mat.Expose()),
    mmdagSloppy(matSloppy.Expose()),
    mmdagPrecon(matPrecon.Expose())
  {
  }

  void CG3NE::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);
    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      xp = ColorSpinorField(csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField(csParam);
      init = true;
    }
  }

  ColorSpinorField &CG3NE::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    // CG3 residual will match the CG3NE residual (FIXME: but only with zero initial guess?)
    return param.use_init_guess ? xp : CG3::get_residual();
  }

  // CG3NE: M Mdag y = b is solved; x = Mdag y is returned as solution.
  void CG3NE::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    double b2 = param.compute_true_res ? blas::norm2(b) : 0.0;

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // compute initial residual
      mmdag.Expose()->M(xp, x);
      if (param.compute_true_res && b2 == 0.0)
        b2 = blas::xmyNorm(b, xp);
      else
        blas::xpay(b, -1.0, xp);

      // compute solution to residual equation
      CG3::operator()(yp, xp);

      mmdag.Expose()->Mdag(xp, yp);

      // compute full solution
      blas::xpy(xp, x);
    } else {
      CG3::operator()(yp, b);
      mmdag.Expose()->Mdag(x, yp);
    }

    if (param.compute_true_res || (param.use_init_guess && param.return_residual)) {
      // compute the true residual
      mmdag.Expose()->M(xp, x);
      blas::xpay(b, -1.0, xp); // xp now holds the residual

      double r2;
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        double3 h3 = blas::HeavyQuarkResidualNorm(x, xp);
        r2 = h3.y;
        param.true_res_hq = sqrt(h3.z);
      } else {
        r2 = blas::norm2(xp);
      }
      param.true_res = sqrt(r2 / b2);
      PrintSummary("CG3NE", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
    }
  }

  CG3NR::CG3NR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
               SolverParam &param, TimeProfile &profile) :
    CG3(mdagm, mdagmSloppy, mdagmPrecon, param, profile),
    mdagm(mat.Expose()),
    mdagmSloppy(matSloppy.Expose()),
    mdagmPrecon(matPrecon.Expose())
  {
  }

  void CG3NR::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);
    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      br = ColorSpinorField(csParam);
      init = true;
    }
  }

  ColorSpinorField &CG3NR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return br;
  }

  // CG3NR: Mdag M x = Mdag b is solved.
  void CG3NR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    double b2 = 0.0;
    if (param.compute_true_res) {
      b2 = blas::norm2(b);
      if (b2 == 0.0) { // compute initial residual vector
        mdagm.Expose()->M(br, x);
        b2 = blas::norm2(br);
      }
    }

    mdagm.Expose()->Mdag(br, b);
    CG3::operator()(x, br);

    if (param.compute_true_res || param.return_residual) {
      // compute the true residual
      mdagm.Expose()->M(br, x);
      blas::xpay(b, -1.0, br); // br now holds the residual

      if (param.compute_true_res) {
        double r2;
        if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          double3 h3 = blas::HeavyQuarkResidualNorm(x, br);
          r2 = h3.y;
          param.true_res_hq = sqrt(h3.z);
        } else {
          r2 = blas::norm2(br);
        }
        param.true_res = sqrt(r2 / b2);
        PrintSummary("CG3NR", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
      }
    }
  }

  void CG3::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");
    if (x.Precision() != param.precision || b.Precision() != param.precision)
      errorQuda("Precision mismatch");

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    double b2 = blas::norm2(b);
    if(b2 == 0 &&
       (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO || param.use_init_guess == QUDA_USE_INIT_GUESS_NO)){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    const bool mixed_precision = (param.precision != param.precision_sloppy);
    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      yp = ColorSpinorField::Create(csParam);

      // Sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      ArSp = ColorSpinorField::Create(csParam);
      rS_oldp = ColorSpinorField::Create(csParam);
      if (mixed_precision) {
        rSp = ColorSpinorField::Create(csParam);
        xSp = ColorSpinorField::Create(csParam);
        xS_oldp = ColorSpinorField::Create(csParam);
        tmpSp = ColorSpinorField::Create(csParam);
      } else {
        xS_oldp = yp;
        tmpSp = tmpp;
      }

      init = true;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &rS = mixed_precision ? *rSp : r;
    ColorSpinorField &xS = mixed_precision ? *xSp : x;
    ColorSpinorField &ArS = *ArSp;
    ColorSpinorField &rS_old = *rS_oldp;
    ColorSpinorField &xS_old = *xS_oldp;
    ColorSpinorField &tmpS = *tmpSp;

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    int resIncrease = 0;
    int resIncreaseTotal = 0;

    // these are only used if we use the heavy_quark_res
    const int hqmaxresIncrease = maxResIncrease + 1;
    int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual
    double heavy_quark_res = 0.0; // heavy quark residual
    double heavy_quark_res_old = 0.0;  // heavy quark residual
    int hqresIncrease = 0;
    bool L2breakdown = false;

    int pipeline = param.pipeline;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    blas::flops = 0;

    // compute initial residual depending on whether we have an initial guess or not
    double r2;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      if(b2==0) b2 = r2;
      if (mixed_precision) {
	blas::copy(y, x);
	blas::zero(xS);
      }
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(x);
      if (mixed_precision) {
        blas::zero(y);
        blas::zero(xS);
      }
    }
    blas::copy(rS, r);

    if (use_heavy_quark_res) {
      heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
      heavy_quark_res_old = heavy_quark_res;
    }

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    if (convergence(r2, heavy_quark_res, stop, param.tol_hq)) return;
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    double r2_old = r2;
    double rNorm  = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx  = rNorm;
    double maxrr  = rNorm;
    double delta  = param.delta;
    bool restart = false;

    int k = 0;
    PrintStats("CG3", k, r2, b2, heavy_quark_res);
    double rho = 1.0, gamma = 1.0;

    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter) {

      matSloppy(ArS, rS);
      double gamma_old = gamma;
      double rAr = blas::reDotProduct(rS,ArS);
      gamma = r2/rAr;
      
      // CG3 step
      if (k == 0 || restart) { // First iteration
        if (pipeline) {
          r2 = blas::quadrupleCG3InitNorm(gamma, xS, rS, xS_old, rS_old, ArS);
        } else {
          blas::copy(xS_old, xS);
          blas::copy(rS_old, rS);

          blas::axpy(gamma, rS, xS);  // x += gamma*r
          r2 = blas::axpyNorm(-gamma, ArS, rS); // r -= gamma*w
        }
        restart = false;
      } else {
        rho = rho/(rho-(gamma/gamma_old)*(r2/r2_old));
        r2_old = r2;

        if (pipeline) {
          r2 = blas::quadrupleCG3UpdateNorm(gamma, rho, xS, rS, xS_old, rS_old, ArS);
        } else {
          blas::axpbyz(gamma * rho, rS, rho, xS, tmpS);
          blas::axpy(1. - rho, xS_old, tmpS);
          std::swap(xS_old, xS);
          std::swap(xS, tmpS);

          blas::axpbyz(-gamma * rho, ArS, rho, rS, tmpS);
          r2 = blas::axpyNorm(1. - rho, rS_old, tmpS);
          std::swap(rS_old, rS);
          std::swap(rS, tmpS);
        }
      }

      k++;

      if (use_heavy_quark_res && k%heavy_quark_check==0) {
        heavy_quark_res_old = heavy_quark_res;
        if (mixed_precision) {
          blas::copy(tmpS,y);
          heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xS, tmpS, rS).z);
        } else {
          heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(xS, rS).z);
        }
      }

      // reliable update conditions
      if (mixed_precision) {
        rNorm = sqrt(r2);
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;
        bool update = (rNorm < delta*r0Norm && r0Norm <= maxrx); // condition for x
        update = ( update || (rNorm < delta*maxrr && r0Norm <= maxrr)); // condition for r

        // force a reliable update if we are within target tolerance (only if doing reliable updates)
        if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol ) update = true;

        // For heavy-quark inversion force a reliable update if we continue after
        if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq) and param.delta >= param.tol ) {
          update = true;
        }

        if (update) {
          // updating the "new" vectors
          blas::copy(x, xS);
          blas::xpy(x, y);
          mat(r, y);
          r2 = blas::xmyNorm(b, r);
          param.true_res = sqrt(r2 / b2);
          if (use_heavy_quark_res) {
            heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);
            param.true_res_hq = heavy_quark_res;
          }
          rNorm = sqrt(r2);
          r0Norm = sqrt(r2);
          maxrr = rNorm;
          maxrx = rNorm;
          // we update sloppy and old fields
          if (!convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
            blas::copy(rS, r);
            blas::axpy(-1., xS, xS_old);
            // we preserve the orthogonality between the previous residual and the new
            Complex rr_old = blas::cDotProduct(rS, rS_old);
            r2_old = blas::caxpyNorm(-rr_old/r2, rS, rS_old);
            blas::zero(xS);
          }
        }

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CG3: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2), r0Norm, resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("CG3: solver exiting due to too many true residual norm increases");
              break;
            }
          }
        } else {
          resIncrease = 0;
        }

        // if L2 broke down we turn off reliable updates and restart the CG
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          heavy_quark_check = 1;
          warningQuda("CG3: Restarting without reliable updates for heavy-quark residual");
          restart = true;
          L2breakdown = false;
          if (heavy_quark_res > heavy_quark_res_old) {
            hqresIncrease++;
            warningQuda("CG3: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res, heavy_quark_res_old);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG3: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }
      } else {
        if (convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
          mat(r, x);
          r2 = blas::xmyNorm(b, r);
          r0Norm = sqrt(r2);
          // we update sloppy and old fields
          if (!convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
            // we preserve the orthogonality between the previous residual and the new
            Complex rr_old = blas::cDotProduct(rS, rS_old);
            r2_old = blas::caxpyNorm(-rr_old/r2, rS, rS_old);
          }
        }

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CG3: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2), r0Norm, resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
              warningQuda("CG3: solver exiting due to too many true residual norm increases");
              break;
          }
        }
      }

      PrintStats("CG3", k, r2, b2, heavy_quark_res);
    }

    if (mixed_precision) blas::copy(x, y);
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residuals
    if (!mixed_precision && param.compute_true_res) {
      mat(r, x);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
      if (use_heavy_quark_res) param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
    }

    PrintSummary("CG3", k, r2, b2, stop, param.tol_hq);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

} // namespace quda
