#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

void invertBiCGstabCuda(ParitySpinor x, ParitySpinor source, FullGauge gaugeSloppy, 
			FullGauge gaugePrecise, ParitySpinor tmp, 
			QudaInvertParam *invert_param, DagType dag_type)
{
  ParitySpinor r = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
  ParitySpinor p = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
  ParitySpinor v = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
  ParitySpinor t = allocateParitySpinor(x.length/spinorSiteSize, x.precision);

  ParitySpinor y = allocateParitySpinor(x.length/spinorSiteSize, x.precision);
  ParitySpinor b = allocateParitySpinor(x.length/spinorSiteSize, x.precision);

  copyQuda(b, source);
  copyQuda(r, b);
  zeroQuda(y);
  zeroQuda(x);

  double b2 = normQuda(b);
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

  //printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    if (k==0) {
      rho = make_cuDoubleComplex(r2, 0.0);
      copyQuda(p, r);
    } else {
      alpha_omega = cuCdiv(alpha, omega);
      rho_rho0 = cuCdiv(rho, rho0);
      beta = cuCmul(rho_rho0, alpha_omega);

      // p = r - beta*omega*v + beta*(p)
      beta_omega = cuCmul(beta, omega); beta_omega.x *= -1.0; beta_omega.y *= -1.0;
      cxpaypbzQuda(r, beta_omega, v, beta, p); // 8
    }

    if (dag_type == QUDA_DAG_NO) 
      //rv = MatPCcDotWXCuda(v, gauge, p, invert_param->kappa, tmp, b, invert_param->matpc_type);
      MatPCCuda(v, gaugeSloppy, p, invert_param->kappa, tmp, invert_param->matpc_type);
    else 
      //rv = MatPCDagcDotWXCuda(v, gauge, p, invert_param->kappa, tmp, b, invert_param->matpc_type);
      MatPCDagCuda(v, gaugeSloppy, p, invert_param->kappa, tmp, invert_param->matpc_type);

    rv = cDotProductQuda(source, v);
    alpha = cuCdiv(rho, rv);

    // r -= alpha*v
    alpha.x *= -1.0; alpha.y *= -1.0;
    caxpyQuda(alpha, v, r); // 4
    alpha.x *= -1.0; alpha.y *= -1.0;

    if (dag_type == QUDA_DAG_NO) 
      MatPCCuda(t, gaugeSloppy, r, invert_param->kappa, tmp, invert_param->matpc_type);
    else  
      MatPCDagCuda(t, gaugeSloppy, r, invert_param->kappa, tmp, invert_param->matpc_type);

    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormAQuda(t, r); // 6
    omega.x = omega_t2.x / omega_t2.z; omega.y = omega_t2.y/omega_t2.z;

    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYQuda(alpha, p, omega, r, x, t, source);
    rho0 = rho; rho.x = rho_r2.x; rho.y = rho_r2.y; r2 = rho_r2.z;
    
    // reliable updates (ideally should be double precision)
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

    if (updateR) {
      QudaPrecision spinorPrec = invert_param->cuda_prec;
      invert_param -> cuda_prec = QUDA_SINGLE_PRECISION;

      if (dag_type == QUDA_DAG_NO) 
	MatPCCuda(t, gaugePrecise, x, invert_param->kappa, tmp, invert_param->matpc_type);
      else 
	MatPCDagCuda(t, gaugePrecise, x, invert_param->kappa, tmp, invert_param->matpc_type);

      invert_param -> cuda_prec = spinorPrec;

      copyQuda(r, b);
      mxpyQuda(t, r);
      r2 = normQuda(r);
      rNorm = sqrt(r2);

      maxrr = rNorm;
      rUpdate++;

      if (updateX) {
	axpyQuda(1.0, x, y);
	zeroQuda(x);
	copyQuda(b, r);
	r0Norm = rNorm;

	maxrx = rNorm;
	xUpdate++;
      }
      
    }
      

    k++;
    printf("%d iterations, r2 = %e, x2 = %e\n", k, r2, normQuda(x));
  }
  axpyQuda(1.0f, y, x);

  invert_param->secs += stopwatchReadSeconds();

  //if (k==maxiters) printf("Exceeded maximum iterations %d\n", maxiters);

  printf("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);

  float gflops = (1.0e-9*Nh)*(2*(2*1320+48)*k + (32*k + 8*(k-1))*spinorSiteSize);
  gflops += 1.0e-9*Nh*rUpdate*((2*1320+48) + 3*spinorSiteSize);
  gflops += 1.0e-9*Nh*xUpdate*spinorSiteSize;
  //printf("%f gflops\n", k*gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;

#if 0
  // Calculate the true residual
  if (dag_type == QUDA_DAG_NO) 
    MatPCCuda(t.spinor, gauge, x.spinor, invert_param->kappa, tmp.spinor, invert_param->matpc_type);
  else 
    MatPCDagCuda(t.spinor, gauge, x.spinor, invert_param->kappa, tmp.spinor, invert_param->matpc_type);
  copyQuda(r, b);
  mxpyQuda(t, r);
  double true_res = normQuda(r);
  
  printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	 k, r2, true_res / b2);
#endif

  freeParitySpinor(b);
  freeParitySpinor(y);
  freeParitySpinor(r);
  freeParitySpinor(v);
  freeParitySpinor(t);
  freeParitySpinor(p);

  return;
}
