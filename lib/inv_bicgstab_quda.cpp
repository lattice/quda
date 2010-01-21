#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <color_spinor_field.h>

void invertBiCGstabCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, 
			cudaColorSpinorField &r, QudaInvertParam *invert_param, DagType dag_type)
{
  ColorSpinorParam param;
  param.create = QUDA_ZERO_CREATE;
  cudaColorSpinorField y(x, param);

  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField p(x, param);
  cudaColorSpinorField v(x, param);
  cudaColorSpinorField tmp(x, param);
  cudaColorSpinorField t(x, param);

  cudaColorSpinorField x_sloppy, r_sloppy, r0;
  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_CREATE;
    x_sloppy = cudaColorSpinorField(x, param);
    r_sloppy = cudaColorSpinorField(r, param);
    r0 = cudaColorSpinorField(b, param);
    zeroCuda(x_sloppy);
    copyCuda(r_sloppy, b);
  } else {
    x_sloppy = cudaColorSpinorField(x, param);
    param.create = QUDA_COPY_CREATE;
    r_sloppy = cudaColorSpinorField(b, param);
    r0 = cudaColorSpinorField(b, param);
  }

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

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("%d iterations, r2 = %e\n", k, r2);

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
    
    diracSloppy.M(v, p, dag_type);
    //MatVec(v, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, invert_param, tmp, dag_type);

    // rv = (r0,v)
    rv = cDotProductCuda(r0, v);    

    alpha = cuCdiv(rho, rv);
    
    // r -= alpha*v
    alpha.x *= -1.0; alpha.y *= -1.0;
    caxpyCuda(alpha, v, r_sloppy); // 4
    alpha.x *= -1.0; alpha.y *= -1.0;

    diracSloppy.M(t, r_sloppy, dag_type);
    //MatVec(t, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, r_sloppy, invert_param, tmp, dag_type);
    
    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(t, r_sloppy); // 6
    omega.x = omega_t2.x / omega_t2.z; omega.y = omega_t2.y/omega_t2.z;
    
    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYCuda(alpha, p, omega, r_sloppy, x_sloppy, t, r0);
    rho0 = rho; rho.x = rho_r2.x; rho.y = rho_r2.y; r2 = rho_r2.z;

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (x.Precision() != x_sloppy.Precision()) copyCuda(x, x_sloppy);
      
      xpyCuda(x, y); // swap these around?
      dirac.M(r, y, dag_type);
      //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, y, invert_param, x, dag_type);
      r2 = xmyNormCuda(b, r);

      if (x.Precision() != r_sloppy.Precision()) copyCuda(r_sloppy, r);            
      zeroCuda(x_sloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }
    
    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("%d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.Precision() != x_sloppy.Precision()) copyCuda(x, x_sloppy);
  xpyCuda(y, x);
    
  if (k==invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  float gflops = (blas_quda_flops + dslash_quda_flops)*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
#if 0
  // Calculate the true residual
  dirac.M(r, x, dag_type);
  //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, invert_param, y, dag_type);
  double true_res = xmyNormCuda(src, r);
  copyCuda(b, src);
    
  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("Converged after %d iterations, r2 = %e, true_r2 = %e\n", k, sqrt(r2/b2), sqrt(true_res / b2));    
#endif

  return;
}
