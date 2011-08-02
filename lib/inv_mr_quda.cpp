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

cudaColorSpinorField *rp_mr = 0;
cudaColorSpinorField *Arp_mr = 0;
cudaColorSpinorField *tmpp_mr = 0;

bool initMR = false;

void freeMR() {
  if (initMR) {
    if (rp_mr) delete rp_mr;
    delete Arp_mr;
    delete tmpp_mr;

    initMR = false;
  }
}

void invertMRCuda(const DiracMatrix &mat, cudaColorSpinorField &x, cudaColorSpinorField &b, 
		  QudaInvertParam *invert_param)
{

  globalReduce = false; // use local reductions for DD solver

  typedef std::complex<double> Complex;

  if (!initMR) {
    ColorSpinorParam param(x);
    param.create = QUDA_ZERO_FIELD_CREATE;
    if (invert_param->preserve_source == QUDA_PRESERVE_SOURCE_YES)
      rp_mr = new cudaColorSpinorField(x, param); 
    Arp_mr = new cudaColorSpinorField(x);
    tmpp_mr = new cudaColorSpinorField(x, param); //temporary for mat-vec

    initMR = true;
  }
  cudaColorSpinorField &r = 
    (invert_param->preserve_source == QUDA_PRESERVE_SOURCE_YES) ? *rp_mr : b;
  cudaColorSpinorField &Ar = *Arp_mr;
  cudaColorSpinorField &tmp = *tmpp_mr;

  if (&r != &b) copyCuda(r, b);

  double b2 = normCuda(b);
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver

  // calculate initial residual
  //mat(Ar, x, tmp);
  //double r2 = xmyNormCuda(b, Ar);  
  //r = Ar;

  zeroCuda(x);
  double r2 = b2;

  if (invert_param->inv_type_precondition != QUDA_GCR_INVERTER) {
    blas_quda_flops = 0;
    stopwatchStart();
  }

  double omega = 1.0;

  int k = 0;
  if (invert_param->verbosity >= QUDA_VERBOSE) 
    printfQuda("MR: %d iterations, r2 = %e\n", k, r2);

  while (r2 > stop && k < invert_param->maxiter) {
    
    mat(Ar, r, tmp);
    
    double3 Ar3 = cDotProductNormACuda(Ar, r);
    Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

    //printfQuda("%d MR %e %e %e\n", k, Ar3.x, Ar3.y, Ar3.z);

    // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = norm2(r)
    r2 = caxpyXmazNormXCuda(omega*alpha, r, x, Ar);

    k++;

    if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("MR: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (k>=invert_param->maxiter && invert_param->verbosity >= QUDA_SUMMARIZE) 
    warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);
  
  if (invert_param->inv_type_precondition != QUDA_GCR_INVERTER) {
    invert_param->secs += stopwatchReadSeconds();
  
    double gflops = (blas_quda_flops + mat.flops())*1e-9;
    reduceDouble(gflops);

    //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
    invert_param->gflops += gflops;
    invert_param->iter += k;
    
    if (invert_param->verbosity >= QUDA_SUMMARIZE) {
      // Calculate the true residual
      mat(r, x);
      double true_res = xmyNormCuda(b, r);
      
      printfQuda("MR: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		 k, sqrt(r2/b2), sqrt(true_res / b2));    
    }
  }

  globalReduce = true; // renable global reductions for outer solver

  return;
}
