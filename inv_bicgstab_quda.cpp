#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#include <quda.h>
#include <util_quda.h>
#include <field_quda.h>

void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
			ParitySpinor tmp, QudaInvertParam *perf, DagType dag_type)
{
  int len = Nh*spinorSiteSize;

  ParitySpinor r = allocateParitySpinor();
  ParitySpinor p = allocateParitySpinor();
  ParitySpinor v = allocateParitySpinor();
  ParitySpinor t = allocateParitySpinor();

  copyCuda((float *)r, (float *)b, len);
  zeroCuda((float *)x, len);

  float b2 = normCuda((float *)b, len);
  float r2 = b2;
  float stop = b2*perf->tol*perf->tol; // stopping condition of solver

  cuComplex rho = make_cuFloatComplex(1.0f, 0.0f);
  cuComplex rho0 = rho;
  cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
  cuComplex omega = make_cuFloatComplex(1.0f, 0.0f);
  cuComplex beta;

  cuComplex rv;
  cuComplex rho_rho0;
  cuComplex alpha_omega;
  cuComplex beta_omega;
  cuComplex one = make_cuFloatComplex(1.0f, 0.0f);

  float3 rho_r2;
  float3 omega_t2;

  int k=0;
  //printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<perf->maxiter) {

    if (k==0) {
      rho = make_cuFloatComplex(r2, 0.0);
      copyCuda((float *)p, (float *)r, len);
    } else {
      alpha_omega = cuCdivf(alpha, omega);
      rho_rho0 = cuCdivf(rho, rho0);
      beta = cuCmulf(rho_rho0, alpha_omega);

      // p = r - beta*omega*v + beta*(p)
      beta_omega = cuCmulf(beta, omega); beta_omega.x *= -1.0f; beta_omega.y *= -1.0f;
      cxpaypbzCuda((float2*)r, beta_omega, (float2*)v, beta, (float2*)p, len/2); // 8
    }

    if (dag_type == QUDA_DAG_NO) 
      rv = MatPCcDotWXCuda(v, gauge, p, perf->kappa, tmp, b, perf->matpc_type);
    else 
      rv = MatPCDagcDotWXCuda(v, gauge, p, perf->kappa, tmp, b, perf->matpc_type);

    alpha = cuCdivf(rho, rv);

    // r -= alpha*v
    alpha.x *= -1.0f; alpha.y *= -1.0f;
    caxpyCuda(alpha, (float2*)v, (float2*)r, len/2); // 4
    alpha.x *= -1.0f; alpha.y *= -1.0f;

    if (dag_type == QUDA_DAG_NO) MatPCCuda(t, gauge, r, perf->kappa, tmp, perf->matpc_type);
    else  MatPCDagCuda(t, gauge, r, perf->kappa, tmp, perf->matpc_type);

    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda((float2*)t, (float2*)r, len/2); // 6
    omega.x = omega_t2.x / omega_t2.z; omega.y = omega_t2.y/omega_t2.z;

    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYCuda(alpha, (float2*)p, omega, (float2*)r, 
						(float2*)x, (float2*)t, (float2*)b, len/2);
    rho0 = rho; rho.x = rho_r2.x; rho.y = rho_r2.y; r2 = rho_r2.z;

    k++;
    //printf("%d iterations, r2 = %e, x2 = %e\n", k, r2, normCuda((float*)x, len));
  }
  perf->secs += stopwatchReadSeconds();

  //if (k==maxiters) printf("Exceeded maximum iterations %d\n", maxiters);

  float gflops = (1.0e-9*Nh)*(2*(2*1320+48)*k + (32*k + 8*(k-1))*spinorSiteSize);
  //printf("%f gflops\n", k*gflops / stopwatchReadSeconds());
  perf->gflops += gflops;
  perf->iter += k;

#if 0
  // Calculate the true residual
  if (dag_type == QUDA_DAG_NO) MatPCCuda(t, gauge, x, perf->kappa, tmp, perf->matpc_type);
  else MatPCDagCuda(t, gauge, x, perf->kappa, tmp, perf->matpc_type);
  copyCuda((float *)r, (float *)b, len);
  mxpyCuda((float *)t, (float *)r, len);
  double true_res = normCuda((float *)r, len);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  freeParitySpinor(r);
  freeParitySpinor(v);
  freeParitySpinor(t);
  freeParitySpinor(p);

  return;
}
