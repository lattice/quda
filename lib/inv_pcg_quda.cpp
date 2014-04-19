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

namespace quda {


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

    inner.inv_type_precondition = QUDA_PCG_INVERTER; // used to tell the inner solver it is an inner solver 

    if(outer.inv_type == QUDA_PCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  PreconCG::PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param)
  {

    fillInnerSolverParam(Kparam, param);

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = new MR(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      printf("Creating SD\n");
      fflush(stdout);
      K = new SD(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
  }

  PreconCG::~PreconCG(){
    profile.Start(QUDA_PROFILE_FREE);

    if(K) delete K;

    profile.Stop(QUDA_PROFILE_FREE);
  }


  void PreconCG::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
  {

    if(K){
      printf("Preconditioner has been set\n");
    }else{
      printf("Oh no! Preconditioner not set\n");
    }

    profile.Start(QUDA_PROFILE_INIT);
    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
    }

    int k=0;
    int rUpdate=0;

    cudaColorSpinorField* minvrPre;
    cudaColorSpinorField* rPre;
    cudaColorSpinorField* minvr;
    cudaColorSpinorField* minvrSloppy;
    cudaColorSpinorField* p;


    ColorSpinorParam csParam(b);
    cudaColorSpinorField r(b);
    if(K) minvr = new cudaColorSpinorField(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b,csParam);

    mat(r, x, y); // => r = A*x;
    double r2 = xmyNormCuda(b,r);

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
      x_sloppy = &x;
    }else{
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x,csParam);
    }


    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;

    if(&x != &xSloppy){
      copyCuda(y, x); // copy x to y
      zeroCuda(xSloppy);
    }else{
      zeroCuda(y); // no reliable updates // NB: check this
    }

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if(K){
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.setPrecision(param.precision_precondition);
      rPre = new cudaColorSpinorField(rSloppy,csParam);
      // Create minvrPre 
      minvrPre = new cudaColorSpinorField(*rPre);
      globalReduce = false;
      (*K)(*minvrPre, *rPre);  
      globalReduce = true;
      *minvrSloppy = *minvrPre;
      p = new cudaColorSpinorField(*minvrSloppy);
    }else{
      p = new cudaColorSpinorField(rSloppy);
    }

  
    profile.Stop(QUDA_PROFILE_INIT);


    profile.Start(QUDA_PROFILE_PREAMBLE);



    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    double heavy_quark_res = 0.0; // heavy quark residual 
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    int heavy_quark_check = 10; // how often to check the heavy quark residual


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


    if(K) rMinvr = reDotProductCuda(rSloppy,*minvrSloppy);

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);


    quda::blas_flops = 0;

    int steps_since_reliable = 1;

    const int maxResIncrease = 0;

    while(!convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter){

      matSloppy(Ap, *p, tmpSloppy);

      double sigma;
      bool breakdown = false;
      pAp   = reDotProductCuda(*p,Ap);

      alpha = (K) ? rMinvr/pAp : r2/pAp;
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy); 
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
          r_new_Minvr_old = reDotProductCuda(rSloppy,*minvrSloppy);
          *rPre = rSloppy;
          globalReduce = false;
          (*K)(*minvrPre, *rPre);
          globalReduce = true;
      

          *minvrSloppy = *minvrPre;

          rMinvr = reDotProductCuda(rSloppy,*minvrSloppy);
          beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; 
          axpyZpbxCuda(alpha, *p, xSloppy, *minvrSloppy, beta);
        }else{
          beta = sigma/r2_old; // use the alternative beta computation
          axpyZpbxCuda(alpha, *p, xSloppy, rSloppy, beta);
        }
      } else { // reliable update

        axpyCuda(alpha, *p, xSloppy); // xSloppy += alpha*p
        copyCuda(x, xSloppy);
        xpyCuda(x, y); // y += x
        // Now compute r 
        mat(r, y, x); // x is just a temporary here
        r2 = xmyNormCuda(b, r);
        copyCuda(rSloppy, r); // copy r to rSloppy
        zeroCuda(xSloppy);


        // break-out check if we have reached the limit of the precision
        static int resIncrease = 0;
        if(sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this 
          warningQuda("PCG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);

          k++;
          rUpdate++;
          if(++resIncrease > maxResIncrease) break;
        }else{
          resIncrease = 0;
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        ++rUpdate;

        if(K){
          *rPre = rSloppy;
          globalReduce = false;
          (*K)(*minvrPre, *rPre);
          globalReduce = true;

          *minvrSloppy = *minvrPre;

          rMinvr = reDotProductCuda(rSloppy,*minvrSloppy);
          beta = rMinvr/rMinvr_old;        

          xpayCuda(*minvrSloppy, beta, *p); // p = minvrSloppy + beta*p
        }else{ // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          double rp = reDotProductCuda(rSloppy, *p)/(r2);
          axpyCuda(-rp, rSloppy, *p);

          beta = r2/r2_old;
          xpayCuda(rSloppy, beta, *p);

          steps_since_reliable = 0;
        }
      }      
      breakdown = false;
      ++k;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);
    }


    profile.Stop(QUDA_PROFILE_COMPUTE);

    profile.Start(QUDA_PROFILE_EPILOGUE);

    if(x.Precision() != param.precision_sloppy) copyCuda(x, xSloppy);
    xpyCuda(y, x); // x += y


    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);





    // compute the true residual 
    mat(r, x, y);
    double true_res = xmyNormCuda(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

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

    profile.Stop(QUDA_PROFILE_FREE);
    return;
  }


} // namespace quda
