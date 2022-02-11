#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <reliable_updates.h>

#include <invert_x_update.h>

namespace quda
{

  using namespace blas;

  PreconCG::PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);
    // Preconditioners do not need a deflation space,
    // so we explicily set this here.
    Kparam.deflate = false;

    K = createPreconditioner(matPrecon, matPrecon, matPrecon, matEig, param, Kparam, profile);
  }

  PreconCG::PreconCG(const DiracMatrix &mat, Solver& K_, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);

    K = wrapExternalPreconditioner(K_);
  }

  PreconCG::~PreconCG()
  {
    profile.TPSTART(QUDA_PROFILE_FREE);
    extractInnerSolverParam(param, Kparam);

    destroyDeflationSpace();

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  // set the required parameters for the inner solver
  void PreconCG::fillInnerSolverParam(SolverParam &inner, const SolverParam &outer)
  {
    Solver::fillInnerSolverParam(inner, outer);

    // custom behavior
    if (outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;

  }

  void PreconCG::solve_and_collect(ColorSpinorField &x, ColorSpinorField &b, std::vector<ColorSpinorField *> &v_r,
                                   int collect_miniter, double collect_tol)
  {
    K->train_param(*this, b);

    profile.TPSTART(QUDA_PROFILE_INIT);

    // whether to select alternative reliable updates
    bool alternative_reliable = param.use_alternative_reliable;

    double b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0 && param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    int k = 0;

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b, matEig);
      if (deflate_compute) {
        // compute the deflation space.
        (*eig_solve)(evecs, evals);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(matEig, evecs, evals);
        recompute_evals = false;
      }
    }

    ColorSpinorField *minvrPre = nullptr;
    ColorSpinorField *rPre = nullptr;
    ColorSpinorField *minvr = nullptr;
    ColorSpinorField *minvrSloppy = nullptr;

    const int Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
    if (Np < 0 || Np > 16) { errorQuda("Invalid value %d for solution_accumulator_pipeline", Np); }

    ColorSpinorParam csParam(b);
    ColorSpinorField r(b);
    if (K) minvr = new ColorSpinorField(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField y(csParam);

    csParam.setPrecision(param.precision_sloppy);

    // temporary fields
    ColorSpinorField *tmpp = ColorSpinorField::Create(csParam);
    ColorSpinorField *tmp2p = nullptr;
    ColorSpinorField *tmp3p = nullptr;
    if (!mat.isStaggered()) {
      // tmp2 only needed for multi-gpu Wilson-like kernels
      tmp2p = ColorSpinorField::Create(csParam);
      // additional high-precision temporary if Wilson and mixed-precision
      csParam.setPrecision(param.precision);
      tmp3p = (param.precision != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : tmpp;
      csParam.setPrecision(param.precision_sloppy);
    } else {
      tmp3p = tmp2p = tmpp;
    }

    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmp2 = *tmp2p;
    ColorSpinorField &tmp3 = *tmp3p;

    double Anorm = 0;

    // for alternative reliable updates
    if (alternative_reliable) {
      // estimate norm for reliable updates
      mat(r, b, y, tmp3);
      Anorm = sqrt(blas::norm2(r) / b2);
    }

    // compute initial residual
    double r2 = 0.0;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x, y, tmp3);
      r2 = blas::xmyNorm(b, r);
      if (b2 == 0) b2 = r2;
      // y contains the original guess.
      blas::copy(y, x);
    } else {
      if (&r != &b) blas::copy(r, b);
      r2 = b2;
      blas::zero(y);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate and accumulate to solution vector
      eig_solve->deflate(y, r, evecs, evals, true);
      mat(r, y, x, tmp3);
      r2 = blas::xmyNorm(b, r);
    }

    ColorSpinorField Ap(csParam);

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
      minvrSloppy = minvr;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.field = &r;
      r_sloppy = new ColorSpinorField(csParam);
      if (K) {
        csParam.field = minvr;
        minvrSloppy = new ColorSpinorField(csParam);
      }
    }

    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.field = &x;
      x_sloppy = new ColorSpinorField(csParam);
    }

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    blas::zero(x);
    if (&x != &xSloppy) blas::zero(xSloppy);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if (K) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.setPrecision(Kparam.precision);
      csParam.field = r_sloppy;
      rPre = new ColorSpinorField(csParam);
      // Create minvrPre
      minvrPre = new ColorSpinorField(*rPre);
      pushVerbosity(param.verbosity_precondition);
      (*K)(*minvrPre, *rPre);
      popVerbosity();
      *minvrSloppy = *minvrPre;
    }

    csParam.create = QUDA_NULL_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    XUpdateBatch x_update_batch(Np, K ? *minvrSloppy : rSloppy, csParam);

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    double heavy_quark_res = 0.0;                               // heavy quark residual
    if (use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNorm(x, r).z);

    double beta = 0.0;
    double pAp;
    double rMinvr = 0;
    double rMinvr_old = 0.0;
    double r_new_Minvr_old = 0.0;
    double r2_old = 0;
    r2 = norm2(r);

    if (K) rMinvr = reDotProduct(rSloppy, *minvrSloppy);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    blas::flops = 0;

    PrintStats("PCG", k, r2, b2, heavy_quark_res);

    int collect = v_r.size();

    ReliableUpdatesParams ru_params;

    ru_params.alternative_reliable = alternative_reliable;
    ru_params.u = precisionEpsilon(param.precision_sloppy);
    ru_params.uhigh = precisionEpsilon(); // solver precision
    ru_params.Anorm = Anorm;
    ru_params.delta = param.delta;

    ru_params.maxResIncrease = param.max_res_increase;
    ru_params.maxResIncreaseTotal = param.max_res_increase_total;
    ru_params.use_heavy_quark_res = use_heavy_quark_res;
    ru_params.hqmaxresIncrease = param.max_hq_res_increase;
    ru_params.hqmaxresRestartTotal = param.max_hq_res_restart_total;

    ReliableUpdates ru(ru_params, r2);

    bool converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

    while (!converged && k < param.maxiter) {

      matSloppy(Ap, x_update_batch.get_current_field(), tmp, tmp2);

      double sigma;
      // alternative reliable updates,
      if (alternative_reliable) {
        double3 pAppp = blas::cDotProductNormA(x_update_batch.get_current_field(), Ap);
        pAp = pAppp.x;
        ru.update_ppnorm(pAppp.z);
      } else {
        pAp = reDotProduct(x_update_batch.get_current_field(), Ap);
      }

      x_update_batch.get_current_alpha() = (K) ? rMinvr / pAp : r2 / pAp;
      Complex cg_norm = axpyCGNorm(-x_update_batch.get_current_alpha(), Ap, rSloppy);
      // r --> r - alpha*A*p
      r2_old = r2;
      r2 = real(cg_norm);

      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k-1 - r_k) breaks

      if (K) rMinvr_old = rMinvr;

      ru.update_rNorm(sqrt(r2));

      ru.evaluate(r2_old);

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if (convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol) ru.set_updateX();

      if (collect > 0 && k > collect_miniter && r2 < collect_tol * collect_tol * b2) {
        *v_r[v_r.size() - collect] = rSloppy;
        printfQuda("Collecting r %2d: r2 / b2 = %12.8e, k = %5d.\n", collect, sqrt(r2 / b2), k);
        collect--;
      }

      if (!ru.trigger()) {

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(rSloppy, *minvrSloppy);
          *rPre = rSloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(*minvrPre, *rPre);
          popVerbosity();

          // can fuse these two kernels
          *minvrSloppy = *minvrPre;
          rMinvr = reDotProduct(rSloppy, *minvrSloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
        } else {
          beta = sigma / r2_old; // use the alternative beta computation
        }

        if (Np == 1) {
          axpyZpbx(x_update_batch.get_current_alpha(), x_update_batch.get_current_field(), xSloppy,
                   K ? *minvrSloppy : rSloppy, beta);
        } else {
          if (x_update_batch.is_container_full()) { x_update_batch.accumulate_x(xSloppy); }
          blas::xpayz(K ? *minvrSloppy : rSloppy, beta, x_update_batch.get_current_field(),
                      x_update_batch.get_next_field());
        }

        ru.accumulate_norm(x_update_batch.get_current_alpha());

      } else { // reliable update

        // Now that we are performing reliable update, need to update x with the p's that have
        // not been used yet
        x_update_batch.accumulate_x(xSloppy);
        x_update_batch.reset_next();

        xpy(xSloppy, y);          // y += x
        // Now compute r
        mat(r, y, x, tmp3); // x is just a temporary here
        r2 = xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < ru.maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y, x, tmp3);
          r2 = blas::xmyNorm(b, r);

          ru.update_maxr_deflate(r2);
        }

        copy(rSloppy, r); // copy r to rSloppy
        zero(xSloppy);

        bool L2breakdown = false;
        double L2breakdown_eps = 0;
        if (ru.reliable_break(r2, stop, L2breakdown, L2breakdown_eps)) { break; }

        ru.update_norm(r2, y);

        ru.reset(r2);

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(rSloppy, *minvrSloppy);
          *rPre = rSloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(*minvrPre, *rPre);
          popVerbosity();

          // can fuse these two kernels
          *minvrSloppy = *minvrPre;
          rMinvr = reDotProduct(rSloppy, *minvrSloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
        } else {                        // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          double rp = reDotProduct(rSloppy, x_update_batch.get_current_field()) / (r2);
          axpy(-rp, rSloppy, x_update_batch.get_current_field());

          beta = r2 / r2_old;
        }
        xpayz(K ? *minvrSloppy : rSloppy, beta, x_update_batch.get_current_field(), x_update_batch.get_next_field());
      }

      ++k;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);

      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);
      // if we have converged and need to update any trailing solutions
      if ((converged || k == param.maxiter) && ru.steps_since_reliable > 0 && !x_update_batch.is_container_full()) {
        x_update_batch.accumulate_x(xSloppy);
      }

      if (ru.steps_since_reliable == 0) {
        x_update_batch.reset();
      } else {
        ++x_update_batch;
      }
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if (x.Precision() != param.precision_sloppy) copy(x, xSloppy);
    xpy(y, x); // x += y

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops() + matEig.flops()) * 1e-9;
    if (K) gflops += K->flops()*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (collect > 0) { warningQuda("%d r vectors still to be collected ...\n", collect); }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("PCG: Reliable updates = %d\n", ru.rUpdate);

    // compute the true residual
    mat(r, x, y, tmp3);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();
    matEig.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (tmpp) delete tmpp;
    if (!mat.isStaggered()) {
      if (tmp2p && tmpp != tmp2p) delete tmp2p;
      if (tmp3p && tmpp != tmp3p && param.precision != param.precision_sloppy) delete tmp3p;
    }

    if (K) { // These are only needed if preconditioning is used
      delete minvrPre;
      delete rPre;
      delete minvr;
      if (x.Precision() != param.precision_sloppy) delete minvrSloppy;
    }

    if (param.precision_sloppy != x.Precision()) {
      delete r_sloppy;
      if (param.use_sloppy_partial_accumulator) { delete x_sloppy; }
    }

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }

} // namespace quda
