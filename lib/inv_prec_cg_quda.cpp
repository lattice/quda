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
#include <sys/time.h>
#include <face_quda.h>

// I need to add functionality to the conjugate-gradient solver.
// Should I do this by simple inheritance, or should I use a decorator? 
// Is it even possible to use a decorator?

namespace quda {

   // set the required parameters for the inner solver
  void localFillInnerInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer) {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.reliable_delta = 1e-20; // no reliable updates within the inner solver

    inner.cuda_prec = outer.cuda_prec_precondition; // preconditioners are uni-precision solvers
    inner.cuda_prec_sloppy = outer.cuda_prec_precondition;

    inner.verbosity = outer.verbosity;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_GCR_INVERTER; // used to tell the inner solver it is an inner solver

    if (outer.inv_type == QUDA_CG_INVERTER && outer.cuda_prec_sloppy != outer.cuda_prec_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES; // What does this mean?

  } // copied from inv_gcr_quda.cpp



  PreconCG::PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrec, QudaInvertParam &invParam, TimeProfile &profile) :
  Solver(invParam, profile), mat(mat), matSloppy(matSloppy), matPrec(matPrec), K(NULL)
  {
    Kparam = newQudaInvertParam();
    invParam.maxiter_precondition = 2;
    localFillInnerInvertParam(Kparam, invParam);

    
    Kparam.dslash_type   = invParam.dslash_type;
    Kparam.inv_type      = invParam.inv_type;
    Kparam.solution_type = invParam.solution_type;
    Kparam.solve_type    = invParam.solve_type;
    Kparam.matpc_type    = invParam.matpc_type;
    Kparam.dirac_order   = invParam.dirac_order;


    Kparam.input_location  = invParam.input_location;
    Kparam.output_location = invParam.output_location;
    Kparam.mass = invParam.mass;
    Kparam.dagger = invParam.dagger;
    Kparam.mass_normalization = invParam.mass_normalization;
    Kparam.preserve_source = invParam.preserve_source;
    
    Kparam.cpu_prec = invParam.cpu_prec;
    Kparam.cuda_prec = invParam.cuda_prec_precondition;
    Kparam.cuda_prec_sloppy = invParam.cuda_prec_precondition;
  
  //  Kparam = invParam;
  //  Kparam.maxiter = 2;


	  K = new CG(matPrec, matPrec, Kparam, profile);
  }


  PreconCG::~PreconCG() {
    if (K) delete K;
  }

  void PreconCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b)
  {
    printfQuda("Calling preconditioned solver\n");
    int k=0;
    int rUpdate;
    QudaInvertParam newInvParam = invParam;
    newInvParam.maxiter = 2;
   // CG cg(matPrec, matPrec, newInvParam, profile);

    cudaColorSpinorField r(b);
    ColorSpinorParam param(x);
    param.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b,param);

    mat(r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
		  //		cudaColorSpinorField& tmp);
		  //
		  // => r = A*x;
    double r2 = xmyNormCuda(b,r);
    rUpdate++;

    param.precision = invParam.cuda_prec_sloppy;
    cudaColorSpinorField Ap(x,param);
    cudaColorSpinorField tmp(x,param);


    ColorSpinorParam prec_param(x);
    prec_param.create = QUDA_COPY_FIELD_CREATE;
    prec_param.precision = invParam.cuda_prec_precondition;

    cudaColorSpinorField minvr_pre(r,prec_param);
    cudaColorSpinorField r_pre(r,prec_param); // I trust this copies r, but changes the precision
    cudaColorSpinorField minv_r(r);

    // p_{0} = M^{-1} r_{0}
    K->operator()(minvr_pre, r_pre);
    minv_r = minvr_pre;

    cudaColorSpinorField p(minv_r);
    
    double src_norm = norm2(b);
    double stop = src_norm*invParam.tol*invParam.tol; // stopping condition
    double alpha = 0.0, beta=0.0;
    double pAp;
    double rMinvr  = reDotProductCuda(r,minv_r);
    double rMinvr_old = 0.0;
    r2 = normCuda(r);
    printfQuda("r2 = %e\n",r2);

 
    while(r2 > stop && k < invParam.maxiter){
      mat(Ap, p, tmp);
      pAp   = reDotProductCuda(p,Ap);
      alpha = rMinvr/pAp;
      printfQuda("alpha = %e\n",alpha);

      
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, r); // disregard cg_norm
      //axpyCuda(-alpha, Ap, r);
						       // r --> r - alpha*A*p
      rMinvr_old = rMinvr;
      double r_new_Minvr_old = reDotProductCuda(r,minv_r);
      r_pre = r;
      minvr_pre = r_pre;
      K->operator()(minvr_pre, r_pre);
      minv_r = minvr_pre;

      rMinvr = reDotProductCuda(r,minv_r);

      double r2_old = r2;
      r2 = real(cg_norm);
      printfQuda("r2 = %e\n",r2);
      fflush(stdout);

      beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; // Not flexible!
      if(beta < 0){ beta = 0.0; }
      else{
        if(beta > (rMinvr/rMinvr_old)) beta = rMinvr/rMinvr_old;
      }
      // update x and p
      // x = x + alpha*p
      // p = Minv_r + beta*p
      axpyZpbxCuda(alpha, p, x, minv_r, beta);
      //axpyCuda(alpha,p,x);
      //axpyCuda(beta,p,minv_r);
      ++k;
    }

    printfQuda("Number of iterations = %d\n",k);

    // compute the true residual 
    mat(r, x, y);
    double true_res = xmyNormCuda(b, r);
    invParam.true_res = sqrt(true_res / src_norm);
    printfQuda("true_res = %e\n", invParam.true_res);
   
    return;
  }


} // namespace quda
