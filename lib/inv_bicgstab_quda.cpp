#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include<face_quda.h>

#include <color_spinor_field.h>

void invertBiCGstabCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x, 
			cudaColorSpinorField &b, QudaInvertParam *invert_param)
{
  typedef std::complex<double> Complex;

  ColorSpinorParam param(x);
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField y(x, param);
  cudaColorSpinorField r(x, param); 
  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField p(x, param);
  cudaColorSpinorField v(x, param);
  cudaColorSpinorField tmp(x, param);
  cudaColorSpinorField t(x, param);

  cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;

  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
    r_0 = &b;
    zeroCuda(*x_sloppy);
    copyCuda(*r_sloppy, b);
  } else {
    x_sloppy = new cudaColorSpinorField(x, param);
    param.create = QUDA_COPY_FIELD_CREATE;
    r_sloppy = new cudaColorSpinorField(b, param);
    r_0 = new cudaColorSpinorField(b, param);
  }

  // Syntatic sugar
  cudaColorSpinorField &rSloppy = *r_sloppy;
  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &r0 = *r_0;

  double b2 = normCuda(b);

  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver
  double delta = invert_param->reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
  Complex rho(1.0, 0.0);
  Complex rho0 = rho;
  Complex alpha(1.0, 0.0);
  Complex omega(1.0, 0.0);
  Complex beta;

  double3 rho_r2;
  double3 omega_t2;
  
  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();

  while (r2 > stop && k<invert_param->maxiter) {
    
    if (k==0) {
      rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
      copyCuda(p, rSloppy);
    } else {
      if (abs(rho*alpha) == 0.0) beta = 0.0;
      else beta = (rho/rho0) * (alpha/omega);

      cxpaypbzCuda(rSloppy, -beta*omega, v, beta, p);
    }
    
    matSloppy(v, p, tmp);

    if (abs(rho) == 0.0) alpha = 0.0;
    else alpha = rho / cDotProductCuda(r0, v);

    // r -= alpha*v
    caxpyCuda(-alpha, v, rSloppy);

    matSloppy(t, rSloppy, tmp);
    
    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(t, rSloppy);
    omega = Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);

    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductUYNormYCuda(alpha, p, omega, rSloppy, xSloppy, t, r0);

    rho0 = rho;
    rho = Complex(rho_r2.x, rho_r2.y);
    r2 = rho_r2.z;

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
      xpyCuda(x, y); // swap these around?
      mat(r, y, x);
      r2 = xmyNormCuda(b, r);

      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }
    
    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);
    
  if (k==invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  double gflops = (blas_quda_flops + mat.flops() + matSloppy.flops())*1e-9;
  reduceDouble(gflops);

  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
  if (invert_param->verbosity >= QUDA_SUMMARIZE) {
    // Calculate the true residual
    mat(r, x);
    double true_res = xmyNormCuda(b, r);
    
    printfQuda("BiCGstab: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
	       k, sqrt(r2/b2), sqrt(true_res / b2));    
  }

  if (invert_param->cuda_prec_sloppy != x.Precision()) {
    delete r_0;
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}
