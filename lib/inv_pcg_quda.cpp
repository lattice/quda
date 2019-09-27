#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

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
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param)
  {

    fillInnerSolverParam(Kparam, param);

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = new MR(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = new SD(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
  }

  PreconCG::~PreconCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(K) delete K;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void PreconCG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {

    profile.TPSTART(QUDA_PROFILE_INIT);
    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
    }

    int k=0;
    int rUpdate=0;

    cudaColorSpinorField* minvrPre = NULL;
    cudaColorSpinorField* rPre = NULL;
    cudaColorSpinorField* minvr = NULL;
    cudaColorSpinorField* minvrSloppy = NULL;
    cudaColorSpinorField* p = NULL;


    ColorSpinorParam csParam(b);
    cudaColorSpinorField r(b);
    if(K) minvr = new cudaColorSpinorField(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b,csParam);

    mat(r, x, y); // => r = A*x;
    double r2 = xmyNorm(b,r);

    csParam.setPrecision(param.precision_sloppy);
    cudaColorSpinorField tmpSloppy(x,csParam);
    cudaColorSpinorField Ap(x,csParam);

    cudaColorSpinorField *r_sloppy;
    if(param.precision_sloppy == x.Precision())
    {
      r_sloppy = &r;
      minvrSloppy = minvr;
    }else{
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(r,csParam);
      if(K) minvrSloppy = new cudaColorSpinorField(*minvr,csParam);
    }
  

    cudaColorSpinorField *x_sloppy;
    if(param.precision_sloppy == x.Precision() ||
        !param.use_sloppy_partial_accumulator) {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      x_sloppy = &static_cast<cudaColorSpinorField&>(x);
    }else{
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x,csParam);
    }


    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;

    if(&x != &xSloppy){
      copy(y, x); // copy x to y
      zero(xSloppy);
    }else{
      zero(y); // no reliable updates // NB: check this
    }

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if(K){
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.setPrecision(param.precision_precondition);
      rPre = new cudaColorSpinorField(rSloppy,csParam);
      // Create minvrPre 
      minvrPre = new cudaColorSpinorField(*rPre);
      commGlobalReductionSet(false);
      (*K)(*minvrPre, *rPre);  
      commGlobalReductionSet(true);
      *minvrSloppy = *minvrPre;
      p = new cudaColorSpinorField(*minvrSloppy);
    }else{
      p = new cudaColorSpinorField(rSloppy);
    }

  
    profile.TPSTOP(QUDA_PROFILE_INIT);


    profile.TPSTART(QUDA_PROFILE_PREAMBLE);



    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    double heavy_quark_res = 0.0; // heavy quark residual 
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNorm(x,r).z);

    double alpha = 0.0, beta=0.0;
    double pAp;
    double rMinvr  = 0;
    double rMinvr_old = 0.0;
    double r_new_Minvr_old = 0.0;
    double r2_old = 0;
    r2 = norm2(r);

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;


    if(K) rMinvr = reDotProduct(rSloppy,*minvrSloppy);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);


    blas::flops = 0;

    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    
    int resIncrease = 0;
    int resIncreaseTotal = 0;
    
    while(!convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter){

      matSloppy(Ap, *p, tmpSloppy);

      double sigma;
      pAp   = reDotProduct(*p,Ap);

      alpha = (K) ? rMinvr/pAp : r2/pAp;
      Complex cg_norm = axpyCGNorm(-alpha, Ap, rSloppy); 
      // r --> r - alpha*A*p
      r2_old = r2;
      r2 = real(cg_norm);
  
      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k-1 - r_k) breaks

      if(K) rMinvr_old = rMinvr;

      rNorm = sqrt(r2);
      if(rNorm > maxrx) maxrx = rNorm;
      if(rNorm > maxrr) maxrr = rNorm;


      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

  
      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if( convergence(r2, heavy_quark_res, stop, param.tol_hq) && delta >= param.tol) updateX = 1;
    

      if( !(updateR || updateX) ){

        if(K){
          r_new_Minvr_old = reDotProduct(rSloppy,*minvrSloppy);
          *rPre = rSloppy;
	  commGlobalReductionSet(false);
          (*K)(*minvrPre, *rPre);
	  commGlobalReductionSet(true);
      

          *minvrSloppy = *minvrPre;

          rMinvr = reDotProduct(rSloppy,*minvrSloppy);
          beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; 
          axpyZpbx(alpha, *p, xSloppy, *minvrSloppy, beta);
        }else{
          beta = sigma/r2_old; // use the alternative beta computation
          axpyZpbx(alpha, *p, xSloppy, rSloppy, beta);
        }
      } else { // reliable update

        axpy(alpha, *p, xSloppy); // xSloppy += alpha*p
        copy(x, xSloppy);
        xpy(x, y); // y += x
        // Now compute r 
        mat(r, y, x); // x is just a temporary here
        r2 = xmyNorm(b, r);
        copy(rSloppy, r); // copy r to rSloppy
        zero(xSloppy);


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

        if(K){
          *rPre = rSloppy;
	  commGlobalReductionSet(false);
          (*K)(*minvrPre, *rPre);
	  commGlobalReductionSet(true);

          *minvrSloppy = *minvrPre;

          rMinvr = reDotProduct(rSloppy,*minvrSloppy);
          beta = rMinvr/rMinvr_old;        

          xpay(*minvrSloppy, beta, *p); // p = minvrSloppy + beta*p
        }else{ // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          double rp = reDotProduct(rSloppy, *p)/(r2);
          axpy(-rp, rSloppy, *p);

          beta = r2/r2_old;
          xpay(rSloppy, beta, *p);
        }
      }      
      ++k;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);
    }


    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if(x.Precision() != param.precision_sloppy) copy(x, xSloppy);
    xpy(y, x); // x += y


    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);





    // compute the true residual 
    mat(r, x, y);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(K){ // These are only needed if preconditioning is used
      delete minvrPre;
      delete rPre;
      delete minvr;
      if(x.Precision() != param.precision_sloppy)  delete minvrSloppy;
    }
    delete p;

    if(x.Precision() != param.precision_sloppy){
      delete x_sloppy;
      delete r_sloppy;
    }

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }


} // namespace quda
