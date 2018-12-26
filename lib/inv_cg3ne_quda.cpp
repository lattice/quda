#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda {

  CG3NE::CG3NE(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matDagSloppy(matSloppy), init(false)
  {
  }

  CG3NE::~CG3NE() {
    if ( init ) {
      delete rp;
      delete yp;
      delete AdagrSp;
      delete AAdagrSp;
      delete tmpSp;
      if(param.precision != param.precision_sloppy) {
        delete rSp;
        delete xSp;
        delete xS_oldp;
        delete rS_oldp;
      }

      init = false;
    }
  }

  void CG3NE::operator()(ColorSpinorField &x, ColorSpinorField &b)
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

    const bool is_cg3ne = (param.inv_type == QUDA_CG3NE_INVERTER) ? true:false;
    char name[6];
    strcpy(name, is_cg3ne ? "CG3NE" : "CG3NR");

    const bool mixed_precision = (param.precision != param.precision_sloppy);
    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);
      yp = ColorSpinorField::Create(csParam);

      // Sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      AdagrSp = ColorSpinorField::Create(csParam);
      AAdagrSp = ColorSpinorField::Create(csParam);
      rS_oldp = ColorSpinorField::Create(csParam);
      tmpSp = ColorSpinorField::Create(csParam);
      if(mixed_precision) {
        rSp = ColorSpinorField::Create(csParam);
        xSp = ColorSpinorField::Create(csParam);
        xS_oldp = ColorSpinorField::Create(csParam);
      } else {
        xS_oldp = yp;
      }

      init = true;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &rS = mixed_precision ? *rSp : r;
    ColorSpinorField &xS = mixed_precision ? *xSp : x;
    ColorSpinorField &AdagrS = *AdagrSp;
    ColorSpinorField &AAdagrS = *AAdagrSp;
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
      mat(r, x, y);
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
    if(convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
      if(param.preserve_source == QUDA_PRESERVE_SOURCE_NO) {
        blas::copy(b, r);
      }
      return;
    }
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    double r2_old = r2;
    double rNorm  = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx  = rNorm;
    double maxrr  = rNorm;
    double delta  = param.delta;
    bool restart = false;

    int k = 0;
    double rho = 1.0, gamma = 1.0, Ar2 = 1.0;
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter) {

      matDagSloppy(AdagrS, rS, tmpS);
      matSloppy(AAdagrS, AdagrS, tmpS);
      double gamma_old = gamma;
      double Ar2_old = Ar2;
      Ar2 = blas::norm2(AdagrS);
      if(is_cg3ne) {
        gamma = r2/Ar2;
      } else {
        double AAr2 = blas::norm2(AAdagrS);
        gamma = Ar2/AAr2;
      }
      // CG3NE step
      if(k==0 || restart) { // First iteration
        if(pipeline) {
          blas::doubleCG3Init(gamma, xS, xS_old, AdagrS);
          r2 = blas::doubleCG3InitNorm(gamma, rS, rS_old, AAdagrS);
        } else {
          blas::copy(xS_old, xS);
          blas::copy(rS_old, rS);

          blas::axpy(gamma, AdagrS, xS);  // x += gamma*Adag*r            
          r2 = blas::axpyNorm(-gamma, AAdagrS, rS); // r -= gamma*w
        }
        restart = false;
      } else {
        if(is_cg3ne) {
          rho = rho/(rho-(gamma/gamma_old)*(r2/r2_old));
        } else {
          rho = rho/(rho-(gamma/gamma_old)*(Ar2/Ar2_old));
        }
        r2_old = r2;

        if(pipeline) {
          blas::doubleCG3Update(gamma, rho, xS, xS_old, AdagrS);
          r2 = blas::doubleCG3UpdateNorm(gamma, rho, rS, rS_old, AAdagrS);
        } else {
          blas::copy(tmpS, xS);
          blas::axpby(gamma*rho, AdagrS, rho, xS);
          blas::axpy(1.-rho, xS_old, xS);
          blas::copy(xS_old, tmpS);

          blas::copy(tmpS, rS);
          blas::axpby(-gamma*rho, AAdagrS, rho, rS);
          r2 = blas::axpyNorm(1.-rho, rS_old, rS);
          blas::copy(rS_old, tmpS);
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
          mat(r, y, x); //  here we can use x as tmp
          r2 = blas::xmyNorm(b, r);
          param.true_res = sqrt(r2 / b2);
          if (use_heavy_quark_res) {
            heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);
            param.true_res_hq = heavy_quark_res;
          }
          // we update sloppy and old fields
          if (!convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
            blas::copy(rS, r);
            blas::axpy(-1., xS, xS_old);
            if(is_cg3ne) {
              // we preserve the orthogonality between the previous residual and the new
              // This is possible only for cg3ne. For cg3nr ArS and ArS_old should be orthogonal
              Complex rr_old = blas::cDotProduct(rS, rS_old);
              r2_old = blas::caxpyNorm(-rr_old/r2, rS, rS_old);
              blas::zero(xS);
            }
          }
        }

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("%s: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
                      name, sqrt(r2), sqrt(r2_old), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("%s: solver exiting due to too many true residual norm increases", name);
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
          warningQuda("%s: Restarting without reliable updates for heavy-quark residual", name);
          restart = true;
          L2breakdown = false;
          if (heavy_quark_res > heavy_quark_res_old) {
            hqresIncrease++;
            warningQuda("%s: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", name, heavy_quark_res, heavy_quark_res_old);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("%s: solver exiting due to too many heavy quark residual norm increases", name);
              break;
            }
          }
        }
      } else {
        if (convergence(r2, heavy_quark_res, stop, param.tol_hq)) {
          mat(r, x, tmpS);
          r2 = blas::xmyNorm(b, r);
          // we update sloppy and old fields
          if (!convergence(r2, heavy_quark_res, stop, param.tol_hq) and is_cg3ne) {
            // we preserve the orthogonality between the previous residual and the new
              // This is possible only for cg3ne. For cg3nr ArS and ArS_old should be orthogonal
            Complex rr_old = blas::cDotProduct(rS, rS_old);
            r2_old = blas::caxpyNorm(-rr_old/r2, rS, rS_old);
          }
        }

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG3: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
                      sqrt(r2), sqrt(r2_old), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
              warningQuda("CG3: solver exiting due to too many true residual norm increases");
              break;
          }
        }
      }

      PrintStats(name, k, r2, b2, heavy_quark_res);
    }

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
      mat(r, x, y);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
      if (use_heavy_quark_res) param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
    }

    if(param.preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      blas::copy(b, r);
    }

    PrintSummary(name, k, r2, b2, stop, param.tol_hq);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    return;
  }

} // namespace quda
