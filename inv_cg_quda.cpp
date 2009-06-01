#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <util_quda.h>
#include <field_quda.h>

void invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
		  ParitySpinor tmp, QudaInvertParam *perf)
{
  int len = Nh*spinorSiteSize;
  Precision prec = QUDA_SINGLE_PRECISION;

  ParitySpinor r;
  ParitySpinor p = allocateParitySpinor(prec);
  ParitySpinor Ap = allocateParitySpinor(prec);

  float b2 = normCuda((float *)b.spinor, len);
  float r2 = b2;
  float r2_old;
  float stop = r2*perf->tol*perf->tol; // stopping condition of solver

  float alpha, beta;
  QudaSumFloat pAp;

  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
    r = allocateParitySpinor(prec);
    copyCuda((float *)r.spinor, (float *)b.spinor, len);
  } else {
    r = b;
  }
  copyCuda((float *)p.spinor, (float *)r.spinor, len);
  zeroCuda((float *)x.spinor, len);

  int k=0;
  //printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<perf->maxiter) {
    MatPCDagMatPCCuda(Ap, gauge, p, perf->kappa, tmp, perf->matpc_type);

    pAp = reDotProductCuda((float *)p.spinor, (float *)Ap.spinor, len);

    alpha = r2 / (float)pAp;        
    r2_old = r2;
    r2 = axpyNormCuda(-alpha, (float *)Ap.spinor, (float *)r.spinor, len);

    beta = r2 / r2_old;

    axpyZpbxCuda(alpha, (float *)p.spinor, (float *)x.spinor, (float *)r.spinor, beta, len);

    k++;
    //printf("%d iterations, r2 = %e\n", k, r2);
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
  copyCuda((float *)r.spinor, (float *)b.spinor, len);
  mxpyCuda((float *)Ap.spinor, (float *)r.spinor, len);
  double true_res = normCuda((float *)r.spinor, len);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(r);
  freeParitySpinor(p);
  freeParitySpinor(Ap);

  return;
}
