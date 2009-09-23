#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

void invertCgCuda(ParitySpinor x, ParitySpinor source, ParitySpinor tmp, QudaInvertParam *perf)
{
  ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
  ParitySpinor Ap = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
  ParitySpinor y = allocateParitySpinor(x.X, x.precision);
  ParitySpinor r = allocateParitySpinor(x.X, x.precision);

  ParitySpinor b;
  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
    b = allocateParitySpinor(x.X, x.precision);
    copyCuda(b, source);
  } else {
    b = source;
  }

  ParitySpinor x_sloppy, r_sloppy, tmp_sloppy;
  if (invert_param->cuda_prec_sloppy == x.precision) {
    x_sloppy = x;
    r_sloppy = r;
    tmp_sloppy = tmp;
  } else {
    x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    tmp_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
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
  double stop = r2*perf->tol*perf->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  int k=0;
  int xUpdate = 0, rUpdate = 0;

  if (invert_param->verbosity >= QUDA_VERBOSE)
    printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<perf->maxiter) {

    if (invert_param->dslash_type == QUDA_WILSON_DSLASH) {
      MatPCDagMatPCCuda(Ap, cudaGaugeSloppy, p, perf->kappa, tmp_sloppy, perf->matpc_type);
    } else {
      cloverMatPCDagMatPCCuda(Ap, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, perf->kappa,
			      tmp_sloppy, perf->matpc_type);
    }

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

    if (!updateR) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, p, x_sloppy, r_sloppy, beta);
    } else {
      axpyCuda(alpha, p, x_sloppy);
      
      if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);

      if (invert_param->dslash_type == QUDA_WILSON_DSLASH) {
      	MatPCDagMatPCCuda(r, cudaGaugePrecise, x, invert_param->kappa, tmp, invert_param->matpc_type);
      } else {
	cloverMatPCDagMatPCCuda(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, invert_param->kappa,
				tmp, invert_param->matpc_type);
      }

      r2 = xmyNormCuda(b, r);
      if (x.precision != r_sloppy.precision) copyCuda(r_sloppy, r);
      rNorm = sqrt(r2);

      maxrr = rNorm;
      rUpdate++;
      
      if (updateX) {
	xpyCuda(x, y);
	zeroCuda(x_sloppy);
	copyCuda(b, r);
	r0Norm = rNorm;

	maxrx = rNorm;
	xUpdate++;
      }

      beta = r2 / r2_old;
      xpayCuda(r_sloppy, beta, p);
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE)
      printf("%d iterations, r2 = %e\n", k, r2);
  }

  if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
  xpyCuda(y, x);

  perf->secs = stopwatchReadSeconds();

  if (k==invert_param->maxiter) 
    printf("Exceeded maximum iterations %d\n", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printf("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);

  float gflops = k*(1.0e-9*x.volume)*(2*(2*1320+48) + 10*spinorSiteSize);
  if (invert_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) gflops += k*(1.0e-9*x.volume)*4*504;
  //printf("%f gflops\n", k*gflops / stopwatchReadSeconds());
  perf->gflops = gflops;
  perf->iter = k;

#if 0
  // Calculate the true residual
  if (invert_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCDagMatPCCuda(Ap, cudaGaugePrecise, x, perf->kappa, tmp, perf->matpc_type);
  } else {
    cloverMatPCDagMatPCCuda(Ap, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, perf->kappa,
			    tmp, perf->matpc_type);
  }
  copyCuda(r, b);
  mxpyCuda(Ap, r);
  double true_res = normCuda(r);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  if (invert_param->cuda_prec_sloppy != x.precision) {
    freeParitySpinor(tmp_sloppy);
    freeParitySpinor(r_sloppy);
    freeParitySpinor(x_sloppy);
  }

  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(b);
  freeParitySpinor(r);
  freeParitySpinor(p);
  freeParitySpinor(Ap);

  freeParitySpinor(b);
  freeParitySpinor(y);

  return;
}
