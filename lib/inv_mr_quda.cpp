#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda {

  MR::MR(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), init(false), allocate_r(false), allocate_y(false)
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
    commGlobalReductionSet(param.global_reduction); // use local reductions for DD solver

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.precision = param.precision_sloppy;
      Arp = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam); //temporary for mat-vec
      init = true;
    }

      //Source needs to be preserved if initial guess is used or if different precision is requested
    if(!allocate_r &&
       ((param.preserve_source == QUDA_PRESERVE_SOURCE_YES) || (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) || (param.precision_sloppy != b.Precision()) )) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.precision = param.precision_sloppy;
      rp = ColorSpinorField::Create(csParam);
      allocate_r = true;
    }

    // y is the (sloppy) iterated solution vector
    if (!allocate_y) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.precision = param.precision_sloppy;
      yp = ColorSpinorField::Create(csParam);
      allocate_y = true;
    }

    ColorSpinorField &r = allocate_r ? *rp : b;
    ColorSpinorField &Ar = *Arp;
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &y = *yp;

    double r2=0.0; // if zero source then we will exit immediately doing no work
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      blas::copy(tmp, x);
      matSloppy(r, tmp, Ar);
      blas::copy(y, b);
      r2 = blas::xmyNorm(y, r);   //r = b - Ax0
    } else {
      if (&r != &b) blas::copy(r, b);
      r2 = blas::norm2(r);
      blas::zero(x);
    }

    // set initial guess to zero and thus the residual is just the source
    blas::zero(y);  // can get rid of this for a special first update kernel
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
      double x2 = blas::norm2(y);
      double3 Ar3 = blas::cDotProductNormB(Ar, r);
      printfQuda("MR: %d iterations, r2 = %e, <r|A|r> = (%e, %e), x2 = %e\n", 
		 k, Ar3.z, Ar3.x, Ar3.y, x2);
    } else if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("MR: %d iterations, r2 = %e\n", k, r2);
    }

    double3 Ar3;
    while (k < param.maxiter && r2 > 0.0) {
    
      matSloppy(Ar, r, tmp);

      if (param.global_reduction) {
        Ar3 = blas::cDotProductNormA(Ar, r);
	Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

	// x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
	//r2 = blas::caxpyXmazNormX(omega*alpha, r, x, Ar);
	blas::caxpyXmaz(omega*alpha, r, y, Ar);
      } else {
	// doing local reductions so can make it asynchronous
	commAsyncReductionSet(true);
	Ar3 = blas::cDotProductNormA(Ar, r);

	// omega*alpha is done in the kernel
	blas::caxpyXmazMR(omega, r, y, Ar);
	commAsyncReductionSet(false);
      }
      k++;

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	double x2 = blas::norm2(y);
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
      blas::axpy(scale,y,x);
    } else {
      if (c2 > 0.0) blas::axpby(sqrt(c2), y, 0.0, x); // FIXME: if x contains a Nan then this will fail: hence zero of x above
    }
    // if not preserving source then overide source with residual
    if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO && &r != &b) {
      if (c2 > 0.0) blas::axpby(sqrt(c2), r, 0.0, b);
    } else {
      if (c2 > 0.0) blas::ax(sqrt(c2), r);
    }

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;

      param.gflops += gflops;
      param.iter += k;

      // compute the iterated relative residual
      if (getVerbosity() >= QUDA_SUMMARIZE) r2 = blas::norm2(r) / b2;

      // calculate the true sloppy residual
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

    commGlobalReductionSet(true); // renable global reductions for outer solver
    return;
  }

} // namespace quda
