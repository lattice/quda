#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>

#include <iostream>

namespace quda {

  CG::CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), init(false) {
  }

  CG::~CG() {
    if ( init ) {
      delete rp;
      delete yp;
      delete App;
      delete tmpp;
      init = false;
    }
  }

  void CG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = blas::norm2(b);
    if (b2 == 0) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;

    mat(r, x, y);
    double r2 = blas::xmyNorm(b, r);
    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
    }

    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
        !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p =
      (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
      ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField* pp = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &p = *pp;

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    bool heavy_quark_restart = false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double r2_old;

    double stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

    double heavy_quark_res = 0.0;  // heavy quark res idual
    double heavy_quark_res_old = 0.0;  // heavy quark residual

    if (use_heavy_quark_res) {
      heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
      heavy_quark_res_old = heavy_quark_res;   // heavy quark residual
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double alpha = 0.0;
    double beta = 0.0;
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    // 0 means we have no tolerance
    // maybe we should expose this as a parameter
    const int hqmaxresIncrease = maxResIncrease + 1;

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;

    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    int k = 0;

    PrintStats("CG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;
    bool converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

    while ( !converged && k < param.maxiter ) {
      matSloppy(Ap, p, tmp, tmp2);  // tmp as tmp

      double sigma;

      bool breakdown = false;
      if (param.pipeline) {
        double3 triplet = blas::tripleCGReduction(rSloppy, Ap, p);
        r2 = triplet.x; double Ap2 = triplet.y; pAp = triplet.z;
        r2_old = r2;
        alpha = r2 / pAp;
        sigma = alpha*(alpha * Ap2 - pAp);
        if (sigma < 0.0 || steps_since_reliable == 0) { // sigma condition has broken down
          r2 = blas::axpyNorm(-alpha, Ap, rSloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;
        pAp = blas::reDotProduct(p, Ap);
        alpha = r2 / pAp;

        // here we are deploying the alternative beta computation
        Complex cg_norm = blas::axpyCGNorm(-alpha, Ap, rSloppy);
        r2 = real(cg_norm);  // (r_new, r_new)
        sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2;  // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol ) updateX = 1;

      // For heavy-quark inversion force a reliable update if we continue after
      if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq) and param.delta >= param.tol ) {
        updateX = 1;
      }

      if ( !(updateR || updateX )) {
        beta = sigma / r2_old;  // use the alternative beta computation

        if (param.pipeline && !breakdown)
          blas::tripleCGUpdate(alpha, beta, Ap, rSloppy, xSloppy, p);
        else
          blas::axpyZpbx(alpha, p, xSloppy, rSloppy, beta);


        if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
          if (&x != &xSloppy) {
            blas::copy(tmp, y);
            heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
          } else {
            blas::copy(r, rSloppy);
            heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
          }
        }

        steps_since_reliable++;
      } else {
        blas::axpy(alpha, p, xSloppy);
        blas::copy(x, xSloppy); // nop when these pointers alias

        blas::xpy(x, y); // swap these around?
        mat(r, y, x, tmp3); //  here we can use x as tmp
        r2 = blas::xmyNorm(b, r);

        blas::copy(rSloppy, r); //nop when these pointers alias
        blas::zero(xSloppy);

        // calculate new reliable HQ resididual
        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2), r0Norm, resIncreaseTotal);
          if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("CG: solver exiting due to too many true residual norm increases");
              break;
            }
          }
        } else {
          resIncrease = 0;
        }
        // if L2 broke down already we turn off reliable updates and restart the CG
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res > heavy_quark_res_old) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res, heavy_quark_res_old);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        rUpdate++;

        if (use_heavy_quark_res and heavy_quark_restart) {
          // perform a restart
          blas::copy(p, rSloppy);
          heavy_quark_restart = false;
        } else {
          // explicitly restore the orthogonality of the gradient vector
          double rp = blas::reDotProduct(rSloppy, p) / (r2);
          blas::axpy(-rp, rSloppy, p);

          beta = r2 / r2_old;
          blas::xpay(rSloppy, beta, p);
        }


        steps_since_reliable = 0;
        heavy_quark_res_old = heavy_quark_res;
      }

      breakdown = false;
      k++;

      PrintStats("CG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        // L2 is concverged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2, heavy_quark_res, stop, param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq);
        converged = L2done and HQdone;
      }

    }

    blas::copy(x, xSloppy);
    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, x, y, tmp3);
    param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    PrintSummary("CG", k, r2, b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
    if (xSloppy.Precision() != x.Precision()) delete x_sloppy;

    delete pp;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

  void CG::solve(ColorSpinorField& x, ColorSpinorField& b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
    errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);


    // Check to see that we're not trying to invert on a zero-field source
    //MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_MULTI_SHIFT];
    for(int i=0; i< param.num_src; i++){
      b2[i]=blas::norm2(b.Component(i));
      if(b2[i] == 0){
        profile.TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source\n");
        x=b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;


  //  const int i = 0;  // MW: hack to be able to write Component(i) instead and try with i=0 for now

    for(int i=0; i<param.num_src; i++)
      mat(r.Component(i), x.Component(i), y.Component(i));
    double r2[QUDA_MAX_MULTI_SHIFT];
    for(int i=0; i<param.num_src; i++){
      r2[i] = blas::xmyNorm(b.Component(i), r.Component(i));
      printfQuda("r2[%i] %e\n", i, r2[i]);
    }

    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      // will that work ?
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
    }

    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
    !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p =
    (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField* pp = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &p = *pp;

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res =
    (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    bool heavy_quark_restart = false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double r2_old[QUDA_MAX_MULTI_SHIFT];
    double heavy_quark_res[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark res idual
    double heavy_quark_res_old[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark residual
    double stop[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
      if (use_heavy_quark_res) {
        heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
        heavy_quark_res_old[i] = heavy_quark_res[i];   // heavy quark residual
      }
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double alpha[QUDA_MAX_MULTI_SHIFT] = {0.0};
    double beta[QUDA_MAX_MULTI_SHIFT] = {0.0};
    double pAp[QUDA_MAX_MULTI_SHIFT];;
    int rUpdate = 0;

    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      rNorm[i] = sqrt(r2[i]);
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }

    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    // 0 means we have no tolerance
    // maybe we should expose this as a parameter
    const int hqmaxresIncrease = maxResIncrease + 1;

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;

    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    int k = 0;

    for(int i=0; i<param.num_src; i++){
      PrintStats("CG", k, r2[i], b2[i], heavy_quark_res[i]);
    }

    int steps_since_reliable = 1;
    bool allconverged = true;
    bool converged[QUDA_MAX_MULTI_SHIFT];
    for(int i=0; i<param.num_src; i++){
      converged[i] = convergence(r2[i], heavy_quark_res[i], stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }
    double sigma[QUDA_MAX_MULTI_SHIFT];

    while ( !allconverged && k < param.maxiter ) {
      for(int i=0; i<param.num_src; i++){
        matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
      }

      bool breakdown = false;
      for(int i=0; i<param.num_src; i++){
        if (param.pipeline) {
          double3 triplet = blas::tripleCGReduction(rSloppy.Component(i), Ap.Component(i), p.Component(i));
          r2[i] = triplet.x; double Ap2 = triplet.y; pAp[i] = triplet.z;
          r2_old[i] = r2[i];
          alpha[i] = r2[i] / pAp[i];
          sigma[i] = alpha[i]*(alpha[i] * Ap2 - pAp[i]);
          if (sigma[i] < 0.0 || steps_since_reliable == 0) { // sigma condition has broken down
            r2[i] = blas::axpyNorm(-alpha[i], Ap.Component(i), rSloppy.Component(i));
            sigma[i] = r2[i];
            breakdown = true;
          }

          r2[i] = sigma[i];
        } else {
          r2_old[i] = r2[i];
          pAp[i] = blas::reDotProduct(p.Component(i), Ap.Component(i));
          alpha[i] = r2[i] / pAp[i];
          // here we are deploying the alternative beta computation
          Complex cg_norm = blas::axpyCGNorm(-alpha[i], Ap.Component(i), rSloppy.Component(i));
          r2[i] = real(cg_norm);  // (r_new, r_new)
          sigma[i] = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2[i];  // use r2 if (r_k+1, r_k+1-r_k) breaks
        }
      }
      int updateX;
      int updateR;
      // reliable update conditions
      for(int i=0; i<param.num_src; i++){
        rNorm[i] = sqrt(r2[i]);
        if (rNorm[i] > maxrx[i]) maxrx[i] = rNorm[i];
        if (rNorm[i] > maxrr[i]) maxrr[i] = rNorm[i];
        updateX = (rNorm[i] < delta * r0Norm[i] && r0Norm[i] <= maxrx[i]) ? 1 : 0;
        updateR = ((rNorm[i] < delta * maxrr[i] && r0Norm[i] <= maxrr[i]) || updateX) ? 1 : 0;

        // force a reliable update if we are within target tolerance (only if doing reliable updates)
        if ( convergence(r2[i], heavy_quark_res[i], stop[i], param.tol_hq) && param.delta >= param.tol ) updateX = 1;

        // For heavy-quark inversion force a reliable update if we continue after
        if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2[i], heavy_quark_res[i], stop[i], param.tol_hq) and param.delta >= param.tol ) {
          updateX = 1;
        }
      }

      if ( !(updateR || updateX )) {
        for(int i=0; i<param.num_src; i++){
          beta[i] = sigma[i] / r2_old[i];  // use the alternative beta computation
        }

        if (param.pipeline && !breakdown)
        for(int i=0; i<param.num_src; i++){
          blas::tripleCGUpdate(alpha[i], beta[i], Ap.Component(i), rSloppy.Component(i), xSloppy.Component(i), p.Component(i));
        }
        else
        for(int i=0; i<param.num_src; i++){
          blas::axpyZpbx(alpha[i], p.Component(i), xSloppy.Component(i), rSloppy.Component(i), beta[i]);
        }

        if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
          if (&x != &xSloppy) {
            blas::copy(tmp, y);   //  FIXME: check whether copy works here
            for(int i=0; i<param.num_src; i++){
              heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy.Component(i), tmp.Component(i), rSloppy.Component(i)).z);
            }
          } else {
            blas::copy(r, rSloppy);  //  FIXME: check whether copy works here
            for(int i=0; i<param.num_src; i++){
              heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(x.Component(i), y.Component(i), r.Component(i)).z);
            }
          }
        }

        steps_since_reliable++;
      } else {
        for(int i=0; i<param.num_src; i++){
          blas::axpy(alpha[i], p.Component(i), xSloppy.Component(i));
        }
        blas::copy(x, xSloppy); // nop when these pointers alias

        for(int i=0; i<param.num_src; i++){
          blas::xpy(x.Component(i), y.Component(i)); // swap these around?
        }
        for(int i=0; i<param.num_src; i++){
          mat(r.Component(i), y.Component(i), x.Component(i), tmp3.Component(i)); //  here we can use x as tmp
        }
        for(int i=0; i<param.num_src; i++){
          r2[i] = blas::xmyNorm(b.Component(i), r.Component(i));
        }

        blas::copy(rSloppy, r); //nop when these pointers alias
        blas::zero(xSloppy);

        // calculate new reliable HQ resididual
        if (use_heavy_quark_res){
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(y.Component(i), r.Component(i)).z);
          }
        }

        // MW: FIXME as this probably goes terribly wrong right now
        for(int i = 0; i<param.num_src; i++){
          // break-out check if we have reached the limit of the precision
          if (sqrt(r2[i]) > r0Norm[i] && updateX) { // reuse r0Norm for this
            resIncrease++;
            resIncreaseTotal++;
            warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2[i]), r0Norm[i], resIncreaseTotal);
            if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
              if (use_heavy_quark_res) {
                L2breakdown = true;
              } else {
                warningQuda("CG: solver exiting due to too many true residual norm increases");
                break;
              }
            }
          } else {
            resIncrease = 0;
          }
        }
      // if L2 broke down already we turn off reliable updates and restart the CG
      for(int i = 0; i<param.num_src; i++){
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res[i] > heavy_quark_res_old[i]) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res[i], heavy_quark_res_old[i]);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }
      }

        for(int i=0; i<param.num_src; i++){
          rNorm[i] = sqrt(r2[i]);
          maxrr[i] = rNorm[i];
          maxrx[i] = rNorm[i];
          r0Norm[i] = rNorm[i];
          heavy_quark_res_old[i] = heavy_quark_res[i];
        }
        rUpdate++;

        if (use_heavy_quark_res and heavy_quark_restart) {
          // perform a restart
          blas::copy(p, rSloppy);
          heavy_quark_restart = false;
        } else {
          // explicitly restore the orthogonality of the gradient vector
          for(int i=0; i<param.num_src; i++){
            double rp = blas::reDotProduct(rSloppy.Component(i), p.Component(i)) / (r2[i]);
            blas::axpy(-rp, rSloppy.Component(i), p.Component(i));

            beta[i] = r2[i] / r2_old[i];
            blas::xpay(rSloppy.Component(i), beta[i], p.Component(i));
          }
        }

        steps_since_reliable = 0;
      }

      breakdown = false;
      k++;

      allconverged = true;
      for(int i=0; i<param.num_src; i++){
        PrintStats("CG", k, r2[i], b2[i], heavy_quark_res[i]);
        // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
        converged[i] = convergence(r2[i], heavy_quark_res[i], stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
      }

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        for(int i=0; i<param.num_src; i++){
          // L2 is concverged or precision maxed out for L2
          bool L2done = L2breakdown or convergenceL2(r2[i], heavy_quark_res[i], stop[i], param.tol_hq);
          // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
          bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2[i], heavy_quark_res[i], stop[i], param.tol_hq);
          converged[i] = L2done and HQdone;
        }
      }

    }

    blas::copy(x, xSloppy);
    for(int i=0; i<param.num_src; i++){
      blas::xpy(y.Component(i), x.Component(i));
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
    warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    for(int i=0; i<param.num_src; i++){
      mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
      param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;

      PrintSummary("CG", k, r2[i], b2[i]);
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
    if (xSloppy.Precision() != x.Precision()) delete x_sloppy;

    delete pp;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;



  }



}  // namespace quda
