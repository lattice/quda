#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#include <quda_internal.h>
#include <spinor_quda.h>

#include <util_quda.h>

void MatVec(ParitySpinor out, FullGauge gauge,  FullClover clover, FullClover cloverInv, ParitySpinor in, 
	    QudaInvertParam *invert_param, ParitySpinor tmp, DagType dag_type) {
  double kappa = invert_param->kappa;
  if (invert_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;
  
  if (invert_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCCuda(out, gauge, in, kappa, tmp, invert_param->matpc_type, dag_type);
  } else {
    cloverMatPCCuda(out, gauge, clover, cloverInv, in, kappa, tmp,
		    invert_param->matpc_type, dag_type);
  }
}

void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, ParitySpinor r, 
			QudaInvertParam *invert_param, DagType dag_type)
{
  ParitySpinor y = allocateParitySpinor(x.X, x.precision, x.pad);

  ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  ParitySpinor v = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  ParitySpinor tmp = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
  ParitySpinor t = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);

  ParitySpinor x_sloppy, r_sloppy, r0;
  if (invert_param->cuda_prec_sloppy == x.precision) {
    x_sloppy = x;
    r_sloppy = r;
    r0 = b;
  } else {
    x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
    r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
    r0 = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy, x.pad);
    copyCuda(r0, b);
  }

  zeroCuda(x_sloppy);
  copyCuda(r_sloppy, b);

  /*{
  cuDoubleComplex rv;
  MatVec(v, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, r_sloppy, invert_param, tmp, dag_type);
  // rv = (r0,v)
  rv = cDotProductCuda(r0, v);    
  printf("%e %e %e %e %e\n", rv.x, rv.y, normCuda(r_sloppy), normCuda(v), normCuda(tmp)); exit(0);
  } */

  zeroCuda(y);

  double b2 = normCuda(b);

  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver
  double delta = invert_param->reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
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
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invert_param->verbosity >= QUDA_VERBOSE) printf("%d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;
  dslash_quda_flops = 0;

  stopwatchStart();

  while (r2 > stop && k<invert_param->maxiter) {
    
    if (k==0) {
      rho = make_cuDoubleComplex(r2, 0.0); // cDotProductCuda(r0, r_sloppy); // BiCRstab
      copyCuda(p, r_sloppy);
    } else {
      alpha_omega = cuCdiv(alpha, omega);
      rho_rho0 = cuCdiv(rho, rho0);
      beta = cuCmul(rho_rho0, alpha_omega);
      
      // p = r - beta*omega*v + beta*(p)
      beta_omega = cuCmul(beta, omega); beta_omega.x *= -1.0; beta_omega.y *= -1.0;
      cxpaypbzCuda(r_sloppy, beta_omega, v, beta, p); // 8
    }
    
    MatVec(v, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, invert_param, tmp, dag_type);
    // rv = (r0,v)
    rv = cDotProductCuda(r0, v);    

    alpha = cuCdiv(rho, rv);
    
    // r -= alpha*v
    alpha.x *= -1.0; alpha.y *= -1.0;
    caxpyCuda(alpha, v, r_sloppy); // 4
    alpha.x *= -1.0; alpha.y *= -1.0;

    MatVec(t, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, r_sloppy, invert_param, tmp, dag_type);
    
    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(t, r_sloppy); // 6
    omega.x = omega_t2.x / omega_t2.z; omega.y = omega_t2.y/omega_t2.z;
    
    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYQuda(alpha, p, omega, r_sloppy, x_sloppy, t, r0);
    rho0 = rho; rho.x = rho_r2.x; rho.y = rho_r2.y; r2 = rho_r2.z;

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
      
      xpyCuda(x, y);
      MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, y, invert_param, x, dag_type);
      r2 = xmyNormCuda(b, r);

      if (x.precision != r_sloppy.precision) copyCuda(r_sloppy, r);            
      zeroCuda(x_sloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }
    
    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) printf("%d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
  xpyCuda(y, x);
    
  if (k==invert_param->maxiter) printf("Exceeded maximum iterations %d\n", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printf("Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  float gflops = (blas_quda_flops + dslash_quda_flops)*1e-9;
  //  printf("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
#if 0
  // Calculate the true residual
  MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, invert_param, y, dag_type);
  double true_res = xmyNormCuda(src, r);
  copyCuda(b, src);
    
  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printf("Converged after %d iterations, r2 = %e, true_r2 = %e\n", k, sqrt(r2/b2), sqrt(true_res / b2));    
#endif

  if (invert_param->cuda_prec_sloppy != x.precision) {
    freeParitySpinor(r0);
    freeParitySpinor(r_sloppy);
    freeParitySpinor(x_sloppy);
  }

  freeParitySpinor(tmp);
  freeParitySpinor(v);
  freeParitySpinor(t);
  freeParitySpinor(p);

  freeParitySpinor(y);

  return;
}
