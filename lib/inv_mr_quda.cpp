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

  MR::MR(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param) :
    Solver(mat, matSloppy, matSloppy, matSloppy, param)
  {
    if (param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ && param.Nsteps % 2 == 1) {
      errorQuda("For multiplicative Schwarz, number of solver steps %d must be even", param.Nsteps);
    }
  }

  void MR::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);

      // now allocate sloppy fields
      ColorSpinorParam csParam(b[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      resize(Ar, b.size(), csParam);
      resize(x_sloppy, b.size(), csParam);

      if (param.precision != param.precision_sloppy) { // mixed precision
        resize(r_sloppy, b.size(), csParam);
      } else {
        create_alias(r_sloppy, r);
      }

      init = true;
    } // init
  }

  cvector_ref<const ColorSpinorField> MR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    if (param.precision != param.precision_sloppy && !param.compute_true_res) blas::copy(r, r_sloppy);
    return r;
  }

  void MR::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b); // allocate fields

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    vector<double> b2 = blas::norm2(b); // Save norm of b
    vector<double> r2;

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      r2 = blas::xmyNorm(b, r); // r = b - Ax0
      for (auto i = 0u; i < b.size(); i++)
        if (b2[i] == 0) b2[i] = r2[i];
    } else {
      r2 = b2;
      blas::copy(r, b);
      blas::zero(x); // needed?
    }
    blas::copy(r_sloppy, r);

    auto stop = stopping(param.tol, b2, param.residual_type);

    int iter = 0;
    int step = 0;
    bool converged = false;

    PrintStats("MR", iter, r2, b2);
    while (!converged) {

      int k = 0;
      vector<double> scale(b.size(), 1.0);
      vector<double> scale_inv(b.size(), 1.0);
      vector<double> delta2(b.size(), param.delta * param.delta);

      if ((node_parity + step) % 2 == 0 && param.schwarz_type == QUDA_MULTIPLICATIVE_SCHWARZ) {
        // for multiplicative Schwarz we alternate updates depending on node parity
      } else {

        commGlobalReductionPush(param.global_reduction); // use local reductions for DD solver

        blas::zero(x_sloppy); // can get rid of this for a special first update kernel
        auto c2 = param.global_reduction == QUDA_BOOLEAN_TRUE ? r2 : blas::norm2(r); // c2 holds the initial r2
        for (auto i = 0u; i < b.size(); i++) {
          scale[i] = c2[i] > 0.0 ? sqrt(c2[i]) : 1.0;
          scale_inv[i] = 1.0 / scale[i];
          // domain-wise normalization of the initial residual to prevent underflow
          if (c2[i] > 0.0) r2[i] = 1.0; // by definition by this is now true
        }
        blas::ax(scale_inv, r_sloppy); // can merge this with the prior copy

        while (k < param.maxiter && r2 > delta2) {

          matSloppy(Ar, r_sloppy);

          if (param.global_reduction) {
            auto Ar4 = blas::cDotProductNormAB(Ar, r_sloppy);
            vector<Complex> alpha(b.size());
            for (auto i = 0u; i < b.size(); i++) {
              alpha[i] = Complex(Ar4[i].x, Ar4[i].y) / Ar4[i].z;
              r2[i] = Ar4[i].w;
            }
            PrintStats("MR (inner)", iter, r2, b2);

            // x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)
            blas::caxpyXmaz(param.omega * alpha, r_sloppy, x_sloppy, Ar);
          } else {
            // doing local reductions so can make it asynchronous
            commAsyncReductionSet(true);
            blas::cDotProductNormAB(Ar, r_sloppy);

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
        for (auto i = 0u; i < b2.size(); i++) param.true_res[i] = sqrt(r2[i] / b2[i]);
        converged = (step > param.Nsteps || r2 < stop);
        if (!converged) blas::copy(r_sloppy, r);
        PrintStats("MR (restart)", iter, r2, b2);
      } else {
        blas::ax(scale, r_sloppy);
        r2 = blas::norm2(r_sloppy);
        converged = (step > param.Nsteps || r2 < stop);
        if (!converged) blas::copy(r, r_sloppy);
      }
      step++;
    }

    PrintSummary("MR", iter, r2, b2, stopping(param.tol, b2, param.residual_type));

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      param.iter += iter;
    }
  }

} // namespace quda
