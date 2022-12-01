#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda
{

  MR::MR(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matSloppy, matSloppy, param, profile)
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

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      Ar = ColorSpinorField(csParam);
      x_sloppy = ColorSpinorField(csParam);

      bool mixed = param.precision != param.precision_sloppy;

      if (!mixed) csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      csParam.v = r.V();
      r_sloppy = ColorSpinorField(csParam);

      init = true;
    } // init
  }

  ColorSpinorField &MR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    if (param.precision != param.precision_sloppy && !param.compute_true_res) blas::copy(r, r_sloppy);
    return r;
  }

  void MR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b); // allocate fields

    if (!param.is_preconditioner) {
      blas::flops = 0;
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }

    double b2 = blas::norm2(b); // Save norm of b
    double r2 = 0.0;            // if zero source then we will exit immediately doing no work
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      r2 = blas::xmyNorm(b, r); // r = b - Ax0
    } else {
      r2 = b2;
      blas::copy(r, b);
      blas::zero(x); // needed?
    }
    blas::copy(r_sloppy, r);

    // if invalid residual then convergence is set by iteration count only
    double stop = param.residual_type == QUDA_INVALID_RESIDUAL ? 0.0 : b2 * param.tol * param.tol;

    int iter = 0;
    int step = 0;
    bool converged = false;

    PrintStats("MR", iter, r2, b2, 0.0);
    while (!converged) {

      int k = 0;
      double scale = 1.0;
      if ((node_parity + step) % 2 == 0 && param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ) {
        // for multiplicative Schwarz we alternate updates depending on node parity
      } else {

        commGlobalReductionPush(param.global_reduction); // use local reductions for DD solver

        blas::zero(x_sloppy); // can get rid of this for a special first update kernel
        double c2 = param.global_reduction == QUDA_BOOLEAN_TRUE ? r2 : blas::norm2(r); // c2 holds the initial r2
        scale = c2 > 0.0 ? sqrt(c2) : 1.0;

        // domain-wise normalization of the initial residual to prevent underflow
        if (c2 > 0.0) {
          blas::ax(1 / scale, r_sloppy); // can merge this with the prior copy
          r2 = 1.0;                      // by definition by this is now true
        }

        while (k < param.maxiter && r2 > param.delta * param.delta) {

          matSloppy(Ar, r_sloppy);

          if (param.global_reduction) {
            auto Ar4 = blas::cDotProductNormAB(Ar, r_sloppy);
            Complex alpha = Complex(Ar4.x, Ar4.y) / Ar4.z;
            r2 = Ar4.w;
            PrintStats("MR (inner)", iter, r2, b2, 0.0);

            // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
            blas::caxpyXmaz(param.omega * alpha, r_sloppy, x_sloppy, Ar);
          } else {
            // doing local reductions so can make it asynchronous
            commAsyncReductionSet(true);
            blas::cDotProductNormA(Ar, r_sloppy);

            // omega*alpha is done in the kernel
            blas::caxpyXmazMR(param.omega, r_sloppy, x_sloppy, Ar);
            commAsyncReductionSet(false);
          }

          k++;
          iter++;
        }

        blas::axpy(scale, x_sloppy, x); // Scale and sum to accumulator

        commGlobalReductionPop(); // renable global reductions for outer solver
      }

      // FIXME - add over/under relaxation in outer loop
      bool compute_true_res = param.compute_true_res || param.Nsteps > 1;
      if (compute_true_res) {
        mat(r, x);
        r2 = blas::xmyNorm(b, r);
        param.true_res = sqrt(r2 / b2);
        converged = (step < param.Nsteps && r2 > stop) ? false : true;
        if (!converged) blas::copy(r_sloppy, r);
        PrintStats("MR (restart)", iter, r2, b2, 0.0);
      } else {
        blas::ax(scale, r_sloppy);
        r2 = blas::norm2(r_sloppy);
        converged = (step < param.Nsteps && r2 > stop) ? false : true;
        if (!converged) blas::copy(r, r_sloppy);
      }
      step++;
    }

    PrintSummary("MR", iter, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      // store flops and reset counters
      double gflops = (blas::flops + mat.flops() + matSloppy.flops()) * 1e-9;

      param.gflops += gflops;
      param.iter += iter;
      blas::flops = 0;

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }
  }

} // namespace quda
