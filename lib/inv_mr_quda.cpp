#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda {

  MR::MR(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matSloppy, matSloppy, param, profile),
    init(false)
  {
    if (param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ && param.Nsteps % 2 == 1) {
      errorQuda("For multiplicative Schwarz, number of solver steps %d must be even", param.Nsteps);
    }
  }

  void MR::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      r = ColorSpinorField(csParam);
      tmp = ColorSpinorField(csParam);

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      Ar = ColorSpinorField(csParam);
      x_sloppy = ColorSpinorField(csParam);

      bool mixed = param.precision != param.precision_sloppy;

      if (!mixed) csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      csParam.v = r.V();
      r_sloppy = ColorSpinorField(csParam);

      csParam.v = tmp.V();
      tmp_sloppy = ColorSpinorField(csParam);

      init = true;
    } // init
  }

  void MR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    pushOutputPrefix("MR: ");

    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b); // allocate fields

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
    blas::copy(r_sloppy, r);

    // if invalid residual then convergence is set by iteration count only
    double stop = param.residual_type == QUDA_INVALID_RESIDUAL ? 0.0 : b2*param.tol*param.tol;
    int step = 0;

    logQuda(QUDA_VERBOSE, "Initial residual = %e\n", sqrt(r2));

    bool converged = false;
    while (!converged) {

      double scale = 1.0;
      if ((node_parity+step)%2 == 0 && param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ) {
	// for multiplicative Schwarz we alternate updates depending on node parity
      } else {

        commGlobalReductionPush(param.global_reduction); // use local reductions for DD solver

        blas::zero(x_sloppy);  // can get rid of this for a special first update kernel
	double c2 = param.global_reduction == QUDA_BOOLEAN_TRUE ? r2 : blas::norm2(r);  // c2 holds the initial r2
	scale = c2 > 0.0 ? sqrt(c2) : 1.0;

	// domain-wise normalization of the initial residual to prevent underflow
	if (c2 > 0.0) {
	  blas::ax(1/scale, r_sloppy); // can merge this with the prior copy
	  r2 = 1.0; // by definition by this is now true
	}

	int k = 0;
	logQuda(QUDA_VERBOSE, "%d cycle, %d iterations, r2 = %e\n", step, k, r2);

	double3 Ar3;
	while (k < param.maxiter && r2 > 0.0) {
    
	  matSloppy(Ar, r_sloppy, tmp_sloppy);

	  if (param.global_reduction) {
	    Ar3 = blas::cDotProductNormA(Ar, r_sloppy);
	    Complex alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;

	    // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
	    //r2 = blas::caxpyXmazNormX(omega*alpha, r, x, Ar);
	    blas::caxpyXmaz(param.omega*alpha, r_sloppy, x_sloppy, Ar);

            logQuda(QUDA_VERBOSE, "%d cycle, %d iterations, <r|A|r> = (%e, %e)\n", step, k+1, Ar3.x, Ar3.y);
	  } else {
	    // doing local reductions so can make it asynchronous
	    commAsyncReductionSet(true);
	    Ar3 = blas::cDotProductNormA(Ar, r_sloppy);

	    // omega*alpha is done in the kernel
	    blas::caxpyXmazMR(param.omega, r_sloppy, x_sloppy, Ar);
	    commAsyncReductionSet(false);
	  }
	  k++;

	}

	// Scale and sum to accumulator
	blas::axpy(scale, x_sloppy, x);

        commGlobalReductionPop(); // renable global reductions for outer solver
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
	else blas::copy(r_sloppy, r);

        logQuda(QUDA_SUMMARIZE, "%d cycle, Converged after %d iterations, relative residual: true = %e\n",
                step, param.maxiter, sqrt(r2));
      } else {

	blas::ax(scale, r_sloppy);
	r2 = blas::norm2(r_sloppy);

	converged = (step < param.Nsteps) ? false : true;

	// if not preserving source and finished then overide source with residual
	if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO && converged) blas::copy(b, r_sloppy);
	else blas::copy(r, r_sloppy);

        logQuda(QUDA_SUMMARIZE, "%d cycle, Converged after %d iterations, relative residual: iterated = %e\n",
                step, param.maxiter, sqrt(r2));
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

    popOutputPrefix();
  }

} // namespace quda
