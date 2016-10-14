#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>
#include <iostream>

/***
* IPCG algorithm (nKrylov = 0):
* G.H. Golub, Q. Ye, "Inexact Preconditioned Conjugate Gradient Method with Inner-Outer Iteration", SIAM J. Sci. Comput., 21(4), 1305–1320
*
* FCG algorithm (nKrylov > 0):
* Y. Notay, "Flexible Conjugate Gradients", SIAM J. Sci. Comput., 22(4), 1444–1460
***/

namespace quda {

  using namespace blas;
  
  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam &inner, const SolverParam &outer)
  {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver

    if(outer.inv_type == QUDA_PCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  PreconCG::PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param), nKrylov(param.Nkrylov), init(false)
  {
    fillInnerSolverParam(Kparam, param);

    use_ipcg_iters = nKrylov == 0 ? true : false;

    if(use_ipcg_iters) printfQuda("Running Inexact Preconditioned CG.\n");
    //printfQuda("Running flexible CG with restart length m_max = %d\n", nKrylov);

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = new MR(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = new SD(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
    //
    p.reserve(nKrylov+1);
    Ap.reserve(nKrylov+1);
    //if we actually want to run a regular CG:
    if(param.inv_type_precondition == QUDA_INVALID_INVERTER)  K = new CG(mat, matSloppy, param, profile);
  }
  
  //Flexible version of CG
  PreconCG::PreconCG(DiracMatrix &mat, Solver &K, DiracMatrix &matSloppy, DiracMatrix &matPrecon, 
	   SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(&K), Kparam(param),
    nKrylov(param.Nkrylov), init(false)
  {
    use_ipcg_iters = nKrylov == 0 ? true : false;

    if(use_ipcg_iters) printfQuda("Running Inexact Preconditioned CG.\n");
    printfQuda("Running flexible CG with restart length m_max = %d\n", nKrylov);
    //
    p.reserve(nKrylov+1);
    Ap.reserve(nKrylov+1);
  }

  PreconCG::~PreconCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {
      if (param.precision_sloppy != param.precision) {
        delete x_sloppy;
        delete r_sloppy;
      }

      if (param.precision_precondition != param.precision_sloppy) {
        delete p_pre;
        delete r_pre;
      }

      for (int i=0; i<=nKrylov; i++) {
        delete p[i];
        delete Ap[i];
      }

      delete tmpp;
      delete rp;
      delete yp;

      if(use_ipcg_iters) delete wp;

      delete [] pAp;
    }

    if(K) delete K;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void PreconCG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    //this is a hack, just in case if we want to run unpreconditioned solver:
    if(param.inv_type_precondition == QUDA_INVALID_INVERTER)
    {
      printfQuda("\nRunning regular CG.\n");
      (*K)(x, b);//this is just a regular CG
      return;
    }
    //for the remaining part,  we don't allow for empty preconditioner:
    if(K == nullptr) errorQuda("\nInvalid inner solver\n");

    profile.TPSTART(QUDA_PROFILE_INIT);

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      // high precision residual:
      rp = ColorSpinorField::Create(b, csParam);
      // high precision accumulator:
      yp = ColorSpinorField::Create(b, csParam);

      // create sloppy fields used for orthogonalization
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);

      for (int i = 0; i <= nKrylov; i++) {
	p.push_back(ColorSpinorField::Create(csParam));
	Ap.push_back(ColorSpinorField::Create(csParam));
      }

      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      if (param.precision_sloppy != param.precision) {
	x_sloppy = ColorSpinorField::Create(csParam);
	r_sloppy = ColorSpinorField::Create(csParam);
      } else {
	x_sloppy = &x;
	r_sloppy = rp;
      }

      if(use_ipcg_iters) wp = ColorSpinorField::Create(csParam);
      else               wp = tmpp;//pointer alias

      // these low precision fields are used by the inner solver
      if (param.precision_precondition != param.precision_sloppy) {
        csParam.setPrecision(param.precision_precondition);
	p_pre = ColorSpinorField::Create(csParam);
	r_pre = ColorSpinorField::Create(csParam);
      } else {
	p_pre = wp;
	r_pre = r_sloppy;
      }
      pAp = new double[nKrylov+1];

      init = true;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;
    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;
    ColorSpinorField &tmp  = *tmpp;

    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = blas::norm2(b);
    if(b2 == 0){
      //profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      blas::copy(x, b);
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
    }

    mat(r, x, y); // => r = A*x;

    double r2 = blas::xmyNorm(b,r);

    if(&x != &xSloppy){
      blas::copy(y, x); // copy x to y
      blas::zero(xSloppy);
    }else{
      blas::zero(y); // no reliable updates // NB: check this
    }

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    const bool precMatch           = (param.precision_precondition != param.precision_sloppy) ? false : true;

    blas::copy(rSloppy, r);

    if( !precMatch )  blas::copy(rPre, rSloppy);
    commGlobalReductionSet(false);
    (*K)(pPre, rPre);
    commGlobalReductionSet(true);
    if( !precMatch ) blas::copy(*wp, pPre);

    blas::copy(*p[0], *wp);

  
    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    double heavy_quark_res = 0.0; // heavy quark residual 
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNorm(x,r).z);

    double alpha = 0.0, beta=0.0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
 
    double rMinvr = blas::reDotProduct(rSloppy,*wp);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    blas::flops = 0;

    const int maxResIncrease      = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    
    int resIncrease = 0;
    int resIncreaseTotal = 0;

    int k = 0, rUpdate = 0;

    Complex cg_norm = 0.0;

    while(!convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter){

      int j = 0;//internal index
      bool refresh_cycle = (((k+1) % (nKrylov+1)) == 0) && !use_ipcg_iters;
      //run FCG(nKrylov) cycles:
      do {
        matSloppy(*Ap[j], *p[j], tmp);
        //
        pAp[j]   = blas::reDotProduct( *p[j], *Ap[j]);
        alpha    = rMinvr/pAp[j];
        //update residual:
        cg_norm = blas::axpyCGNorm(-alpha, *Ap[j], rSloppy); 
        //
        r2     = real(cg_norm);
        //stop here if we run IPCG algorithm:
        if(use_ipcg_iters) break;

        //apply preconditioner 
        if( !precMatch )  blas::copy(rPre, rSloppy);
        commGlobalReductionSet(false);
        blas::zero(pPre);
        (*K)(pPre, rPre);
        commGlobalReductionSet(true);
        if( !precMatch ) blas::copy(*wp, pPre);

        //update solution, and increment loop index:
        blas::axpy(+alpha, *p[j++], xSloppy);

        //orthogonalize agains previously computed search vectors
        blas::copy(*p[j], *wp);//j was incremented

        for(int l = 0; l < j; l++)
        {
          beta = blas::reDotProduct(*Ap[l],*wp) / pAp[l];
          blas::axpy(- beta, *p[l], *p[j]);//!
        }

        //stop internal iterarions if this is the last cycle
      } while( j < nKrylov || refresh_cycle );

      if( refresh_cycle ) blas::copy(*p[0], *p[j]);

      rNorm = sqrt(r2);
      if(rNorm > maxrx) maxrx = rNorm;
      if(rNorm > maxrr) maxrr = rNorm;

      int updateX = (rNorm < param.delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < param.delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if( convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol) updateX = 1;

      if( !(updateR || updateX) ){

        if( use_ipcg_iters ){
          double rMinvr_old      = rMinvr;
          double r_new_Minvr_old = blas::reDotProduct(rSloppy,*wp);

          //apply preconditioner 
          if( !precMatch )  blas::copy(rPre, rSloppy);
          commGlobalReductionSet(false);

          blas::zero(pPre); 
          (*K)(pPre, rPre);

          commGlobalReductionSet(true);
          if( !precMatch ) blas::copy(*wp, pPre);

          rMinvr      = reDotProduct(rSloppy,*wp);
          beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; 
          blas::axpyZpbx(alpha, *p[0], xSloppy, *wp, beta);
        }

      } else { // reliable update
        //we need to recompute alpha for FCG part:
        if(!use_ipcg_iters)
        {
          matSloppy(*Ap[0], *p[0], tmp);
          pAp[j]   = blas::reDotProduct(*p[0],*Ap[0]);

          rMinvr = blas::reDotProduct(rSloppy,*p[0]);
          alpha  = rMinvr/pAp[0];
        }

        blas::axpy(alpha, *p[0], xSloppy); // xSloppy += alpha*p
        blas::copy(x, xSloppy);
        blas::xpy(x, y); // y += x
        // Now compute r 
        mat(r, y, x); // x is just a temporary here
        r2 = blas::xmyNorm(b, r);
        blas::copy(rSloppy, r); // copy r to rSloppy
        blas::zero(xSloppy);
        // break-out check if we have reached the limit of the precision

        if(sqrt(r2) > r0Norm && updateX) {
 
          resIncrease++;
          resIncreaseTotal++;
          // reuse r0Norm for this 
          warningQuda("PCG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)", sqrt(r2), r0Norm, resIncreaseTotal);
	  if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) break;

        } else {
	  resIncrease = 0;
	}

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        ++rUpdate;

        if( !precMatch )  blas::copy(rPre, rSloppy);
        commGlobalReductionSet(false);
        (*K)(pPre, rPre);
        commGlobalReductionSet(true);
        if( !precMatch ) blas::copy(*wp, pPre);

        if( use_ipcg_iters ){ 
          double rMinvr_old  = rMinvr;
          rMinvr = blas::reDotProduct(rSloppy,*wp);
          //Fletcher-Reeves formula:
          beta = rMinvr/rMinvr_old;        
          blas::xpay(*wp, beta, *p[0]); // p = *wp + beta*p
        }
        else blas::copy(*p[0], *wp);
      }
     
      k += (use_ipcg_iters)? 1 : (j);//+1 for the regular pcg
      PrintStats( use_ipcg_iters ? "IPCG" : "FCG", k, r2, b2, heavy_quark_res);
    }


    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if(x.Precision() != param.precision_sloppy) blas::copy(x, xSloppy);
    blas::xpy(y, x); // x += y

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("PCG: Reliable updates = %d\n", rUpdate);

    // compute the true residual 
    mat(r, x, y);
    double true_res = blas::xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }


} // namespace quda
