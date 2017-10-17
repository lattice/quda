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
    Solver(param, profile), mat(mat), matSloppy(matSloppy), rp(nullptr), r_sloppy(nullptr),
    Arp(nullptr), tmpp(nullptr), tmp_sloppy(nullptr), x_sloppy(nullptr), init(false)
  {
    if (param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ && param.Nsteps % 2 == 1) {
      errorQuda("For multiplicative Schwarz, number of solver steps %d must be even", param.Nsteps);
    }
  }

  MR::~MR() {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    if (init) {
      if (x_sloppy) delete x_sloppy;
      if (tmp_sloppy) delete tmp_sloppy;
      if (tmpp) delete tmpp;
      if (Arp) delete Arp;
      if (r_sloppy) delete r_sloppy;
      if (rp) delete rp;
    }
    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void MR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (checkPrecision(x,b) != param.precision) errorQuda("Precision mismatch %d %d", checkPrecision(x,b), param.precision);

    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    if (!init) {
      bool mixed = param.precision != param.precision_sloppy;

      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      // Source needs to be preserved if we're computing the true residual
      rp = (param.use_init_guess == QUDA_USE_INIT_GUESS_YES || param.preserve_source == QUDA_PRESERVE_SOURCE_YES
	    || param.Nsteps > 1 || param.compute_true_res == 1) ?
	ColorSpinorField::Create(csParam) : nullptr;

      tmpp = (param.use_init_guess == QUDA_USE_INIT_GUESS_YES || param.Nsteps > 1 || param.compute_true_res) ?
	ColorSpinorField::Create(csParam) : nullptr;

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);

      r_sloppy = mixed ? ColorSpinorField::Create(csParam) : nullptr;  // we need a separate sloppy residual vector
      Arp = ColorSpinorField::Create(csParam);

      //sloppy temporary for mat-vec
      tmp_sloppy = (!tmpp || mixed) ? ColorSpinorField::Create(csParam) : nullptr;

      //  iterated sloppy solution vector
      x_sloppy = ColorSpinorField::Create(csParam);

      init = true;
    } // init

    ColorSpinorField &r = rp ? *rp : b;
    ColorSpinorField &rSloppy = r_sloppy ? *r_sloppy : r;
    ColorSpinorField &Ar = *Arp;
    ColorSpinorField &tmp = tmpp ? *tmpp : b;
    ColorSpinorField &tmpSloppy = tmp_sloppy ? *tmp_sloppy : tmp;
    ColorSpinorField &xSloppy = *x_sloppy;

    if (!param.is_preconditioner) {
      blas::flops = 0;
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }

    double b2 = blas::norm2(b);  //Save norm of b
    double r2 = 0.0; // if zero source then we will exit immediately doing no work
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, tmp);
      r2 = blas::xmyNorm(b, r);   //r = b - Ax0
    } else {
      r2 = b2;
      blas::copy(r, b);
      blas::zero(x); // needed?
    }
    blas::copy(rSloppy, r);

    // if invalid residual then convergence is set by iteration count only
    double stop = param.residual_type == QUDA_INVALID_RESIDUAL ? 0.0 : b2*param.tol*param.tol;
    int step = 0;

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("MR: Initial residual = %e\n", sqrt(r2));

    bool converged = false;
    while (!converged) {

      double scale = 1.0;
      if ((node_parity+step)%2 == 0 && param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ) {
	// for multiplicative Schwarz we alternate updates depending on node parity
      } else {

	commGlobalReductionSet(param.global_reduction); // use local reductions for DD solver

	blas::zero(xSloppy);  // can get rid of this for a special first update kernel
	double c2 = param.global_reduction == QUDA_BOOLEAN_YES ? r2 : blas::norm2(r);  // c2 holds the initial r2
	scale = c2 > 0.0 ? sqrt(c2) : 1.0;

	// domain-wise normalization of the initial residual to prevent underflow
	if (c2 > 0.0) {
	  blas::ax(1/scale, rSloppy); // can merge this with the prior copy
	  r2 = 1.0; // by definition by this is now true
	}

	int k = 0;
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("MR: %d cycle, %d iterations, r2 = %e\n", step, k, r2);

	double3 Ar3;
	while (k < param.maxiter && r2 > 0.0) {
    
	  matSloppy(Ar, rSloppy, tmpSloppy);

	  if (param.global_reduction) {
	    Ar3 = blas::cDotProductNormA(Ar, rSloppy);
	    Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

	    // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
	    //r2 = blas::caxpyXmazNormX(omega*alpha, r, x, Ar);
	    blas::caxpyXmaz(param.omega*alpha, rSloppy, xSloppy, Ar);

	    if (getVerbosity() >= QUDA_VERBOSE)
	      printfQuda("MR: %d cycle, %d iterations, <r|A|r> = (%e, %e)\n", step, k+1, Ar3.x, Ar3.y);
	  } else {
	    // doing local reductions so can make it asynchronous
	    commAsyncReductionSet(true);
	    Ar3 = blas::cDotProductNormA(Ar, rSloppy);

	    // omega*alpha is done in the kernel
	    blas::caxpyXmazMR(param.omega, rSloppy, xSloppy, Ar);
	    commAsyncReductionSet(false);
	  }
	  k++;

	}

	// Scale and sum to accumulator
	blas::axpy(scale,xSloppy,x);

	commGlobalReductionSet(true); // renable global reductions for outer solver

      }
      step++;

      // FIXME - add over/under relaxation in outer loop
      if (param.compute_true_res || param.Nsteps > 1) {
	mat(r, x, tmp);
	r2 = blas::xmyNorm(b, r);
	param.true_res = sqrt(r2 / b2);

	converged = (step < param.Nsteps && r2 > stop) ? false : true;

	// if not preserving source and finished then overide source with residual
	if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO && converged) blas::copy(b, r);
	else blas::copy(rSloppy, r);

	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("MR: %d cycle, Converged after %d iterations, relative residual: true = %e\n",
		     step, param.maxiter, sqrt(r2));
	}
      } else {

	blas::ax(scale, rSloppy);
	r2 = blas::norm2(rSloppy);

	converged = (step < param.Nsteps) ? false : true;

	// if not preserving source and finished then overide source with residual
	if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO && converged) blas::copy(b, rSloppy);
	else blas::copy(r, rSloppy);

	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("MR: %d cycle, Converged after %d iterations, relative residual: iterated = %e\n",
		     step, param.maxiter, sqrt(r2));
	}
      }

    }

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      // store flops and reset counters
      double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;

      param.gflops += gflops;
      param.iter += param.Nsteps * param.maxiter;
      blas::flops = 0;

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    return;
  }

} // namespace quda
