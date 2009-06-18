#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

void invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
		  ParitySpinor tmp, QudaInvertParam *perf)
{
  ParitySpinor r;
  ParitySpinor p = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
  ParitySpinor Ap = allocateParitySpinor(x.length/spinorSiteSize, x.precision);

  double b2 = 0.0;
  b2 = normQuda(b);

  double r2 = b2;
  double r2_old;
  double stop = r2*perf->tol*perf->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
    r = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
    copyQuda(r, b);
  } else {
    r = b;
  }
  copyQuda(p, r);
  zeroQuda(x);

  int k=0;
  printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<perf->maxiter) {
    MatPCDagMatPCCuda(Ap, gauge, p, perf->kappa, tmp, perf->matpc_type);

    pAp = reDotProductQuda(p, Ap);

    alpha = r2 / pAp;        
    r2_old = r2;
    r2 = axpyNormQuda(-alpha, Ap, r);

    beta = r2 / r2_old;

    axpyZpbxQuda(alpha, p, x, r, beta);

    k++;
    printf("%d iterations, r2 = %e\n", k, r2);
  }
  perf->secs = stopwatchReadSeconds();

  //if (k==maxiters)
  //printf("Exceeded maximum iterations %d\n", maxiters);

  float gflops = k*(1.0e-9*Nh)*(2*(2*1320+48) + 10*spinorSiteSize);
  //printf("%f gflops\n", k*gflops / stopwatchReadSeconds());
  perf->gflops = gflops;
  perf->iter = k;

#if 0
  // Calculate the true residual
  MatPCDagMatPCCuda(Ap, gauge, x, perf->kappa, tmp, perf->matpc_type);
  copyQuda(r, b);
  mxpyQuda(Ap, r);
  double true_res = normQuda(r);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(r);
  freeParitySpinor(p);
  freeParitySpinor(Ap);

  return;
}
