#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <quda_internal.h>
#include <util_quda.h>
#include <spinor_quda.h>

void MatVec(ParitySpinor out, FullGauge gauge,  FullClover clover, FullClover cloverInv, ParitySpinor in, 
	    QudaInvertParam *invert_param, ParitySpinor tmp) {
  double kappa = invert_param->kappa;
  if (invert_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;
  
  if (invert_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCDagMatPCCuda(out, gauge, in, kappa, tmp, invert_param->matpc_type);
  } else {
    cloverMatPCDagMatPCCuda(out, gauge, clover, cloverInv, in, kappa, tmp, invert_param->matpc_type);
  }
}

void invertCgCuda(ParitySpinor x, ParitySpinor b, ParitySpinor y, QudaInvertParam *invert_param)
{
  ParitySpinor r = allocateParitySpinor(x.X, x.precision, x.pad);

  ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  ParitySpinor Ap = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  ParitySpinor tmp = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);

  ParitySpinor x_sloppy, r_sloppy;
  if (invert_param->cuda_prec_sloppy == x.precision) {
    x_sloppy = x;
    r_sloppy = r;
  } else {
    x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
    r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  }

  copyCuda(r, b);
  if (r_sloppy.precision != r.precision) copyCuda(r_sloppy, r);
  copyCuda(p, r_sloppy);
  zeroCuda(x_sloppy);
  zeroCuda(y);

  double b2 = 0.0;
  b2 = normCuda(b);

  double r2 = b2;
  double r2_old;
  double stop = r2*invert_param->tol*invert_param->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  int k=0;
  int rUpdate = 0;

  if (invert_param->verbosity >= QUDA_VERBOSE) printf("%d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    MatVec(Ap, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, invert_param, tmp);

    pAp = reDotProductCuda(p, Ap);

    alpha = r2 / pAp;        
    r2_old = r2;
    r2 = axpyNormCuda(-alpha, Ap, r_sloppy);

    // reliable update conditions
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

    if (!(updateR || updateX)) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, p, x_sloppy, r_sloppy, beta);
    } else {
      axpyCuda(alpha, p, x_sloppy);
      
      if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
      
      xpyCuda(x, y);
      MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, y, invert_param, x);
      r2 = xmyNormCuda(b, r);
      if (x.precision != r_sloppy.precision) copyCuda(r_sloppy, r);            
      zeroCuda(x_sloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;

      beta = r2 / r2_old;
      xpayCuda(r_sloppy, beta, p);
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE)
      printf("%d iterations, r2 = %e\n", k, r2);
  }

  if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
  xpyCuda(y, x);

  invert_param->secs = stopwatchReadSeconds();
  

  if (k==invert_param->maxiter) 
    printf("Exceeded maximum iterations %d\n", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printf("Reliable updates = %d\n", rUpdate);

  float gflops = (blas_quda_flops + dslash_quda_flops)*1e-9;
  //  printf("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops = gflops;
  invert_param->iter = k;

  blas_quda_flops = 0;
  dslash_quda_flops = 0;

#if 0
  // Calculate the true residual
  MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, y);
  double true_res = xmyNormCuda(b, r);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  if (invert_param->cuda_prec_sloppy != x.precision) {
    freeParitySpinor(r_sloppy);
    freeParitySpinor(x_sloppy);
  }

  freeParitySpinor(p);
  freeParitySpinor(Ap);
  freeParitySpinor(tmp);

  freeParitySpinor(r);

  return;
}
