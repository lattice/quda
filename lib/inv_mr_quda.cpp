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

MR::MR(DiracMatrix &mat, QudaInvertParam &invParam) :
  Solver(invParam), mat(mat), init(false)
{
 
}

MR::~MR() {
  if (init) {
    if (rp) delete rp;
    delete Arp;
    delete tmpp;
  }
}

void MR::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
{

  globalReduce = false; // use local reductions for DD solver

  if (!init) {
    ColorSpinorParam param(x);
    param.create = QUDA_ZERO_FIELD_CREATE;
    if (invParam.preserve_source == QUDA_PRESERVE_SOURCE_YES)
      rp = new cudaColorSpinorField(x, param); 
    Arp = new cudaColorSpinorField(x);
    tmpp = new cudaColorSpinorField(x, param); //temporary for mat-vec

    init = true;
  }
  cudaColorSpinorField &r = 
    (invParam.preserve_source == QUDA_PRESERVE_SOURCE_YES) ? *rp : b;
  cudaColorSpinorField &Ar = *Arp;
  cudaColorSpinorField &tmp = *tmpp;

  if (&r != &b) copyCuda(r, b);

  double b2 = normCuda(b);
  double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver

  // calculate initial residual
  //mat(Ar, x, tmp);
  //double r2 = xmyNormCuda(b, Ar);  
  //r = Ar;

  zeroCuda(x);
  double r2 = b2;

  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) {
    quda::blas_flops = 0;
    stopwatchStart();
  }

  double omega = 1.0;

  int k = 0;
  if (invParam.verbosity >= QUDA_VERBOSE) 
    printfQuda("MR: %d iterations, r2 = %e\n", k, r2);

  while (r2 > stop && k < invParam.maxiter) {
    
    mat(Ar, r, tmp);
    
    double3 Ar3 = cDotProductNormACuda(Ar, r);
    quda::Complex alpha = quda::Complex(Ar3.x, Ar3.y) / Ar3.z;

    //printfQuda("%d MR %e %e %e\n", k, Ar3.x, Ar3.y, Ar3.z);

    // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = norm2(r)
    r2 = caxpyXmazNormXCuda(omega*alpha, r, x, Ar);

    k++;

    if (invParam.verbosity >= QUDA_VERBOSE) printfQuda("MR: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (k>=invParam.maxiter && invParam.verbosity >= QUDA_SUMMARIZE) 
    warningQuda("Exceeded maximum iterations %d", invParam.maxiter);
  
  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) {
    invParam.secs += stopwatchReadSeconds();
  
    double gflops = (quda::blas_flops + mat.flops())*1e-9;
    reduceDouble(gflops);

    //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
    invParam.gflops += gflops;
    invParam.iter += k;
    
    if (invParam.verbosity >= QUDA_SUMMARIZE) {
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
