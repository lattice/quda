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

namespace quda {

  MR::MR(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), init(false), allocate_r(false)
  {
 
  }

  MR::~MR() {
    if (param.inv_type_precondition != QUDA_GCR_INVERTER) profile.Start(QUDA_PROFILE_FREE);
    if (init) {
      if (allocate_r) delete rp;
      delete Arp;
      delete tmpp;
    }
    if (param.inv_type_precondition != QUDA_GCR_INVERTER) profile.Stop(QUDA_PROFILE_FREE);
  }

  void MR::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
  {

    globalReduce = false; // use local reductions for DD solver

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      if (param.preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	rp = new cudaColorSpinorField(x, csParam); 
	allocate_r = true;
      }
      Arp = new cudaColorSpinorField(x);
      tmpp = new cudaColorSpinorField(x, csParam); //temporary for mat-vec

      init = true;
    }
    cudaColorSpinorField &r = 
      (param.preserve_source == QUDA_PRESERVE_SOURCE_YES) ? *rp : b;
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

    if (param.inv_type_precondition != QUDA_GCR_INVERTER) {
      quda::blas_flops = 0;
      profile.Start(QUDA_PROFILE_COMPUTE);
    }

    double omega = 1.0;

    int k = 0;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      double x2 = norm2(x);
      double3 Ar3 = cDotProductNormBCuda(Ar, r);
      printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e, %e), x2 = %e\n", 
		 k, Ar3.z, Ar3.x, Ar3.y, x2);
    }

    while (k < param.maxiter && r2 > 0.0) {
    
      mat(Ar, r, tmp);

      double3 Ar3 = cDotProductNormACuda(Ar, r);
      Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

      // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = norm2(r)
      //r2 = caxpyXmazNormXCuda(omega*alpha, r, x, Ar);
      caxpyXmazCuda(omega*alpha, r, x, Ar);

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	double x2 = norm2(x);
	double r2 = norm2(r);
	printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e,%e) x2 = %e\n", 
		   k+1, r2, Ar3.x, Ar3.y, x2);
      } else if (getVerbosity() >= QUDA_VERBOSE) {
	printfQuda("MR: %d iterations, <r|A|r> = (%e, %e)\n", k, Ar3.x, Ar3.y);
      }

      k++;
    }
  
    if (getVerbosity() >= QUDA_VERBOSE) {
      mat(Ar, r, tmp);    
      Complex Ar2 = cDotProductCuda(Ar, r);
      printfQuda("MR: %d iterations, <r|A|r> = (%e, %e)\n", k, real(Ar2), imag(Ar2));
    }

    // Obtain global solution by rescaling
    if (b2 > 0.0) axCuda(sqrt(b2), x);

    if (param.inv_type_precondition != QUDA_GCR_INVERTER) {
        profile.Stop(QUDA_PROFILE_COMPUTE);
        profile.Start(QUDA_PROFILE_EPILOGUE);
	param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
  
	double gflops = (quda::blas_flops + mat.flops())*1e-9;
	reduceDouble(gflops);
	
	param.gflops += gflops;
	param.iter += k;
	
	// Calculate the true residual
	r2 = norm2(r);
	mat(r, x);
	double true_res = xmyNormCuda(b, r);
	param.true_res = sqrt(true_res / b2);

	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("MR: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		     k, sqrt(r2/b2), param.true_res);    
	}

	// reset the flops counters
	quda::blas_flops = 0;
	mat.flops();
        profile.Stop(QUDA_PROFILE_EPILOGUE);
    }

    globalReduce = true; // renable global reductions for outer solver

    return;
  }

} // namespace quda
