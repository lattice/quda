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
    Solver(param, profile), mat(mat), init(false), allocate_r(false), allocate_y(false)
  {
 
  }

  MR::~MR() {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    if (init) {
      if (allocate_r) delete rp;
      delete Arp;
      delete tmpp;
      if (allocate_y) delete yp;

    }
    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void MR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    globalReduce = false; // use local reductions for DD solver

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      Arp = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam); //temporary for mat-vec
      init = true;
    }

      //Source needs to be preserved if initial guess is used.
    if(!allocate_r && ((param.preserve_source == QUDA_PRESERVE_SOURCE_YES) || (param.use_init_guess == QUDA_USE_INIT_GUESS_YES))) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);
      allocate_r = true;
    }

    if (!allocate_y && (param.use_init_guess == QUDA_USE_INIT_GUESS_YES)) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(csParam);
      allocate_y = true;
    }
    ColorSpinorField &r = 
      ((param.preserve_source == QUDA_PRESERVE_SOURCE_YES) || (param.use_init_guess == QUDA_USE_INIT_GUESS_YES)) ? *rp : b;
    ColorSpinorField &Ar = *Arp;
    ColorSpinorField &tmp = *tmpp;
    //y is used to store initial guess, otherwise it is unused.
    ColorSpinorField &y = (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) ? *yp : *tmpp;  
    double r2=0.0; // if zero source then we will exit immediately doing no work
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(y, x, tmp);    
      r2 = blas::xmyNorm(b, y);   //y = b - Ax
      blas::copy(r, y);            //Copy y to r
      blas::copy(y, x);           //Save initial guess
    } else {
      if (&r != &b) blas::copy(r, b);
      r2 = blas::norm2(r);
    }
    // set initial guess to zero and thus the residual is just the source
    blas::zero(x);  // can get rid of this for a special first update kernel  
    double b2 = blas::norm2(b);  //Save norm of b
    double c2 = r2;  //c2 holds the initial r2 after (possible) subtraction of initial guess
   
    // domain-wise normalization of the initial residual to prevent underflow
    if (c2 > 0.0) {
      blas::ax(1/sqrt(c2), r); // can merge this with the prior copy
      r2 = 1.0; // by definition by this is now true
    }

    if (!param.is_preconditioner) {
      blas::flops = 0;
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }

    double omega = param.omega;

    int k = 0;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      double x2 = blas::norm2(x);
      double3 Ar3 = blas::cDotProductNormB(Ar, r);
      printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e, %e), x2 = %e\n", 
		 k, Ar3.z, Ar3.x, Ar3.y, x2);
    } else if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("MR: %d iterations, r2 = %e\n", k, r2);
    }

    while (k < param.maxiter && r2 > 0.0) {
    
      mat(Ar, r, tmp);

      double3 Ar3 = blas::cDotProductNormA(Ar, r);
      Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

      // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
      //r2 = blas::caxpyXmazNormX(omega*alpha, r, x, Ar);
      blas::caxpyXmaz(omega*alpha, r, x, Ar);

      k++;

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	double x2 = blas::norm2(x);
	double r2 = blas::norm2(r);
	printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e,%e) x2 = %e\n", 
		   k+1, r2, Ar3.x, Ar3.y, x2);
      } else if (getVerbosity() >= QUDA_VERBOSE) {
	printfQuda("MR: %d iterations, <r|A|r> = (%e, %e)\n", k, Ar3.x, Ar3.y);
      }
    }
  
    //Add back initial guess (if appropriate) and scale if necessary
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      double scale = c2 > 0.0 ? sqrt(c2) : 1.0;
      blas::xpay(y,scale,x);
    } else {
      if (c2 > 0.0) blas::ax(sqrt(c2), x);
    }
    if (c2 > 0.0) blas::ax(sqrt(c2), r);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      double gflops = (blas::flops + mat.flops())*1e-9;
      reduceDouble(gflops);

      param.gflops += gflops;
      param.iter += k;

      // compute the iterated relative residual
      if (getVerbosity() >= QUDA_SUMMARIZE) r2 = blas::norm2(r) / b2;

      // calculate the true residual
      if (param.preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	mat(r, x, tmp);
	double true_res = blas::xmyNorm(b, r);
	param.true_res = sqrt(true_res / b2);

	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("MR: Converged after %d iterations, relative residual: iterated = %e, true = %e\n",
		     k, sqrt(r2), param.true_res);
	}
      } else {
	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("MR: Converged after %d iterations, relative residual: iterated = %e\n", k, sqrt(r2));
	}
      }

      // reset the flops counters
      blas::flops = 0;
      mat.flops();
      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    globalReduce = true; // renable global reductions for outer solver
    return;
  }

} // namespace quda
