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

void invertCgCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, 
		  cudaColorSpinorField &y, QudaInvertParam *invert_param)
{
  int k=0;
  int rUpdate = 0;
    
  cudaColorSpinorField r(b);
  
  dirac.MdagM(r, x);
  double r2 = xmyNormCuda(b, r);
  rUpdate ++;
  
  ColorSpinorParam param;
  param.create = QUDA_ZERO_CREATE;
  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField Ap(x, param);
  cudaColorSpinorField tmp(x, param);
  
  cudaColorSpinorField *x_sloppy, *r_sloppy;
  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
  } else {
    param.create = QUDA_COPY_CREATE;
    x_sloppy = new cudaColorSpinorField(x, param);
    r_sloppy = new cudaColorSpinorField(r, param);
  }

  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &rSloppy = *r_sloppy;

  cudaColorSpinorField p(rSloppy);
  zeroCuda(y);

  double r2_old;
  double src_norm = norm2(b);;
  double stop = src_norm*invert_param->tol*invert_param->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CG: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    diracSloppy.MdagM(Ap, p);
    //MatVec(Ap, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, invert_param, tmp);
    
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
    
    if (!(updateR || updateX)) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
    } else {
      axpyCuda(alpha, p, xSloppy);
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
      xpyCuda(x, y); // swap these around?
      dirac.MdagM(r, y);
      //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, y, invert_param, x);
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
    if (invert_param->verbosity >= QUDA_VERBOSE)
      printfQuda("CG: %d iterations, r2 = %e\n", k, r2);
  }

  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);

  invert_param->secs = stopwatchReadSeconds();

  
  if (k==invert_param->maxiter) 
    warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

  float gflops = (blas_quda_flops + dirac.Flops() + diracSloppy.Flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops = gflops;
  invert_param->iter = k;

  blas_quda_flops = 0;

  //#if 0
  // Calculate the true residual
  dirac.MdagM(r, x);
  //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, y);
  double true_res = xmyNormCuda(b, r);
  if (invert_param->verbosity >= QUDA_SUMMARIZE){
    printfQuda("Converged after %d iterations, r2 = %e, relative true_r2 = %e\n", 
	       k, r2, true_res / src_norm);
  }
  //#endif

  if (invert_param->cuda_prec_sloppy != x.Precision()) {
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}
