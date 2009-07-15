#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

void invertBiCGstabCuda(ParitySpinor x, ParitySpinor src, FullGauge gaugePrecise, 
			FullGauge gaugeSloppy, ParitySpinor tmp, 
			QudaInvertParam *invert_param, DagType dag_type)
{
  ParitySpinor r = allocateParitySpinor(x.X, x.precision);
  ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
  ParitySpinor v = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
  ParitySpinor t = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);

  ParitySpinor y = allocateParitySpinor(x.X, x.precision);
  ParitySpinor b = allocateParitySpinor(x.X, x.precision);

  ParitySpinor x_sloppy, r_sloppy, tmp_sloppy, src_sloppy;
  if (invert_param->cuda_prec_sloppy == x.precision) {
    x_sloppy = x;
    r_sloppy = r;
    tmp_sloppy = tmp;
    src_sloppy = src;
  } else {
    x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    src_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    tmp_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    copyCuda(src_sloppy, src);
  }

  zeroCuda(x_sloppy);
  copyCuda(b, src);
  copyCuda(r_sloppy, src);

  /*MatPCDagCuda(y, gaugePrecise, src, invert_param->kappa, tmp, invert_param->matpc_type);
    copyCuda(src_sloppy, y);*/ // uncomment for BiCRstab

  zeroCuda(y);

  double b2 = normCuda(b);
  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver

  cuDoubleComplex rho = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex rho0 = rho;
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex omega = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex beta;

  cuDoubleComplex rv;

  cuDoubleComplex rho_rho0;
  cuDoubleComplex alpha_omega;
  cuDoubleComplex beta_omega;
  cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

  double3 rho_r2;
  double3 omega_t2;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  int k=0;
  int xUpdate = 0, rUpdate = 0;

  printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    if (k==0) {
      rho = make_cuDoubleComplex(r2, 0.0); // cDotProductCuda(src_sloppy, r_sloppy); // BiCRstab
      copyCuda(p, r_sloppy);
    } else {
      alpha_omega = cuCdiv(alpha, omega);
      rho_rho0 = cuCdiv(rho, rho0);
      beta = cuCmul(rho_rho0, alpha_omega);

      // p = r - beta*omega*v + beta*(p)
      beta_omega = cuCmul(beta, omega); beta_omega.x *= -1.0; beta_omega.y *= -1.0;
      cxpaypbzCuda(r_sloppy, beta_omega, v, beta, p); // 8
    }

    MatPCCuda(v, gaugeSloppy, p, invert_param->kappa, tmp_sloppy, invert_param->matpc_type, dag_type);
    
    // rv = (r0,v)
    rv = cDotProductCuda(src_sloppy, v);

    alpha = cuCdiv(rho, rv);

    // r -= alpha*v
    alpha.x *= -1.0; alpha.y *= -1.0;
    caxpyCuda(alpha, v, r_sloppy); // 4
    alpha.x *= -1.0; alpha.y *= -1.0;

    MatPCCuda(t, gaugeSloppy, r_sloppy, invert_param->kappa, tmp_sloppy, invert_param->matpc_type, dag_type);

    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(t, r_sloppy); // 6
    omega.x = omega_t2.x / omega_t2.z; omega.y = omega_t2.y/omega_t2.z;

    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYQuda(alpha, p, omega, r_sloppy, x_sloppy, t, src_sloppy);
    rho0 = rho; rho.x = rho_r2.x; rho.y = rho_r2.y; r2 = rho_r2.z;

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

    if (updateR) {
      if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);

      MatPCCuda(r, gaugePrecise, x, invert_param->kappa, tmp, invert_param->matpc_type, dag_type);

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

    }

    k++;
    printf("%d iterations, r2 = %e\n", k, r2);
  }

  if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
  xpyCuda(y, x);

  invert_param->secs += stopwatchReadSeconds();

  if (k==invert_param->maxiter) 
    printf("Exceeded maximum iterations %d\n", invert_param->maxiter);

  printf("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);

  float gflops = (1.0e-9*x.volume)*(2*(2*1320+48)*k + (32*k + 8*(k-1))*spinorSiteSize);
  gflops += 1.0e-9*x.volume*rUpdate*((2*1320+48) + 3*spinorSiteSize);
  gflops += 1.0e-9*x.volume*xUpdate*spinorSiteSize;
  //printf("%f gflops\n", k*gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;

#if 0
  // Calculate the true residual
  MatPCCuda(r, gaugePrecise, x, invert_param->kappa, tmp, invert_param->matpc_type, dag_type);
  double true_res = xmyNormCuda(src, r);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", k, sqrt(r2/b2), sqrt(true_res / b2));
#endif

  if (invert_param->cuda_prec_sloppy != x.precision) {
    freeParitySpinor(src_sloppy);
    freeParitySpinor(tmp_sloppy);
    freeParitySpinor(r_sloppy);
    freeParitySpinor(x_sloppy);
  }

  freeParitySpinor(b);
  freeParitySpinor(y);
  freeParitySpinor(r);
  freeParitySpinor(v);
  freeParitySpinor(t);
  freeParitySpinor(p);

  return;
}
