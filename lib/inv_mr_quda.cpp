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

  // set initial guess to zero and thus the residual is just the source
  zeroCuda(x);  // can get rid of this for a special first update kernel  
  double b2 = normCuda(b);
  if (&r != &b) copyCuda(r, b);

  // domain-wise normalization of the initial residual to prevent underflow
  double r2=0.0; // if zero source then we will exit immediately doing no work
  if (b2 > 0.0) {
    axCuda(1/sqrt(b2), r); // can merge this with the prior copy
    r2 = 1.0; // by definition by this is now true
  }
  double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver

  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) {
    quda::blas_flops = 0;
    stopwatchStart();
  }

  double omega = 1.0;

  int k = 0;
  if (invParam.verbosity >= QUDA_DEBUG_VERBOSE) {
    double x2 = norm2(x);
    double3 Ar3 = cDotProductNormBCuda(Ar, r);
    printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e, %e), x2 = %e\n", 
	       k, Ar3.x, Ar3.y, Ar3.z, x2);
  }

  while (r2 > stop && k < invParam.maxiter) {
    
    mat(Ar, r, tmp);
    
    double3 Ar3 = cDotProductNormACuda(Ar, r);
    quda::Complex alpha = quda::Complex(Ar3.x, Ar3.y) / Ar3.z;

    // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = norm2(r)
    //r2 = caxpyXmazNormXCuda(omega*alpha, r, x, Ar);
    caxpyXmazCuda(omega*alpha, r, x, Ar);

    if (invParam.verbosity >= QUDA_DEBUG_VERBOSE) {
      double x2 = norm2(x);
      double r2 = norm2(r);
      printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e,%e) x2 = %e\n", 
		 k+1, r2, Ar3.x, Ar3.y, x2);
    } else if (invParam.verbosity >= QUDA_VERBOSE) {
      printfQuda("MR: %d iterations, <r|A|r> = (%e, %e)\n", k, Ar3.x, Ar3.y);
    }

    k++;
  }
  
  if (invParam.verbosity >= QUDA_VERBOSE) {
    mat(Ar, r, tmp);    
    quda::Complex Ar2 = cDotProductCuda(Ar, r);
    printfQuda("MR: %d iterations, <r|A|r> = (%e, %e)\n", k, real(Ar2), imag(Ar2));
  }

  // Obtain global solution by rescaling
  if (b2 > 0.0) axCuda(sqrt(b2), x);

  if (k>=invParam.maxiter && invParam.verbosity >= QUDA_SUMMARIZE) 
    warningQuda("Exceeded maximum iterations %d", invParam.maxiter);
  
  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) {
    invParam.secs += stopwatchReadSeconds();
  
    double gflops = (quda::blas_flops + mat.flops())*1e-9;
    reduceDouble(gflops);

    invParam.gflops += gflops;
    invParam.iter += k;
    
    if (invParam.verbosity >= QUDA_SUMMARIZE) {
      // Calculate the true residual
      r2 = norm2(r);
      mat(r, x);
      double true_res = xmyNormCuda(b, r);
      
      printfQuda("MR: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		 k, sqrt(r2/b2), sqrt(true_res / b2));    
    }
  }

  globalReduce = true; // renable global reductions for outer solver

  return;
}
