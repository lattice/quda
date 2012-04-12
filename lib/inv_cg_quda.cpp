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

CG::CG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam) :
  Solver(invParam), mat(mat), matSloppy(matSloppy)
{

}

CG::~CG() {

}

void CG::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
{
  int k=0;
  int rUpdate = 0;
    
  cudaColorSpinorField r(b);

  ColorSpinorParam param(x);
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField y(b, param); 
  
  mat(r, x, y);
  zeroCuda(y);

  double r2 = xmyNormCuda(b, r);
  rUpdate ++;
  
  param.precision = invParam.cuda_prec_sloppy;
  cudaColorSpinorField Ap(x, param);
  cudaColorSpinorField tmp(x, param);

  cudaColorSpinorField *tmp2_p = &tmp;
  // tmp only needed for multi-gpu Wilson-like kernels
  if (mat.Type() != typeid(DiracStaggeredPC).name() && 
      mat.Type() != typeid(DiracStaggered).name()) {
    tmp2_p = new cudaColorSpinorField(x, param);
  }
  cudaColorSpinorField &tmp2 = *tmp2_p;

  cudaColorSpinorField *x_sloppy, *r_sloppy;
  if (invParam.cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
  } else {
    param.create = QUDA_COPY_FIELD_CREATE;
    x_sloppy = new cudaColorSpinorField(x, param);
    r_sloppy = new cudaColorSpinorField(r, param);
  }

  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &rSloppy = *r_sloppy;

  cudaColorSpinorField p(rSloppy);

  double r2_old;
  double src_norm = norm2(b);
  double stop = src_norm*invParam.tol*invParam.tol; // stopping condition of solver

  double alpha=0.0, beta=0.0;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invParam.reliable_delta;

  if (invParam.verbosity == QUDA_DEBUG_VERBOSE) {
    double x2 = norm2(x);
    double p2 = norm2(p);
    printf("CG: %d iterations, r2 = %e, x2 = %e, p2 = %e, alpha = %e, beta = %e\n", 
	   k, r2, x2, p2, alpha, beta);
  } else if (invParam.verbosity >= QUDA_VERBOSE) {
    printfQuda("CG: %d iterations, r2 = %e\n", k, r2);
  }

  quda::blas_flops = 0;

  stopwatchStart();
  while (r2 > stop && k<invParam.maxiter) {

    matSloppy(Ap, p, tmp, tmp2); // tmp as tmp
    
    pAp = reDotProductCuda(p, Ap);
    alpha = r2 / pAp;        
    r2_old = r2;
    r2 = axpyNormCuda(-alpha, Ap, rSloppy);

    // reliable update conditions
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
    if ( !(updateR || updateX)) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
    } else {
      axpyCuda(alpha, p, xSloppy);
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
      xpyCuda(x, y); // swap these around?
      mat(r, y, x); // here we can use x as tmp
      r2 = xmyNormCuda(b, r);
      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;

      beta = r2 / r2_old; 
      xpayCuda(rSloppy, beta, p);
    }

    k++;
    if (invParam.verbosity == QUDA_DEBUG_VERBOSE) {
      double x2 = norm2(x);
      double p2 = norm2(p);
      printf("CG: %d iterations, r2 = %e, x2 = %e, p2 = %e, alpha = %e, beta = %e\n", 
	     k, r2, x2, p2, alpha, beta);
    } else if (invParam.verbosity >= QUDA_VERBOSE) {
      printfQuda("CG: %d iterations, r2 = %e\n", k, r2);
    }
  }

  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);

  invParam.secs = stopwatchReadSeconds();

  
  if (k==invParam.maxiter) 
    warningQuda("Exceeded maximum iterations %d", invParam.maxiter);

  if (invParam.verbosity >= QUDA_SUMMARIZE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

  double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
  reduceDouble(gflops);

  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invParam.gflops = gflops;
  invParam.iter = k;

  quda::blas_flops = 0;

  if (invParam.verbosity >= QUDA_SUMMARIZE){
    mat(r, x, y);
    double true_res = xmyNormCuda(b, r);
    printfQuda("CG: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
	       k, sqrt(r2/src_norm), sqrt(true_res / src_norm));    
  }

  if (&tmp2 != &tmp) delete tmp2_p;

  if (invParam.cuda_prec_sloppy != x.Precision()) {
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}
