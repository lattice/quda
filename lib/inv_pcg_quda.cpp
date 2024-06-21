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
                     const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);
    // Preconditioners do not need a deflation space,
    // so we explicily set this here.
    Kparam.deflate = false;

    K = createPreconditioner(matPrecon, matPrecon, matPrecon, matEig, param, Kparam);
  }

  PreconCG::PreconCG(const DiracMatrix &mat, Solver &K_, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);

    K = wrapExternalPreconditioner(K_);
  }

  PreconCG::~PreconCG()
  {
    getProfile().TPSTART(QUDA_PROFILE_FREE);

    extractInnerSolverParam(param, Kparam);
    destroyDeflationSpace();

    getProfile().TPSTOP(QUDA_PROFILE_FREE);
  }

  void PreconCG::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);

    if (!init) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      ColorSpinorParam csParam(b);

      r = ColorSpinorField(b);
      if (K) minvr = ColorSpinorField(b);

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      y = ColorSpinorField(csParam);

      // create sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      Ap = ColorSpinorField(csParam);

      x_sloppy = (!mixed() || !param.use_sloppy_partial_accumulator) ?
        x.create_alias() : ColorSpinorField(csParam);

      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.field = &r;
      r_sloppy = !mixed() ? r.create_alias() : ColorSpinorField(csParam);

      if (K) {
        csParam.field = &minvr;
        minvr_sloppy = !mixed() ? minvr.create_alias() : ColorSpinorField(csParam);

        // create preconditioner intermediates
        csParam.create = QUDA_NULL_FIELD_CREATE;
        csParam.setPrecision(Kparam.precision);
        r_pre = ColorSpinorField(csParam);
        // Create minvr_pre
        minvr_pre = ColorSpinorField(csParam);
      }

      Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
      if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline", Np);

      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      x_update_batch = XUpdateBatch(Np, K ? minvr_sloppy : r_sloppy, csParam);

      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      init = true;
    }
  }

  void PreconCG::solve_and_collect(ColorSpinorField &x, const ColorSpinorField &b, cvector_ref<ColorSpinorField> &v_r,
                                   int collect_miniter, double collect_tol)
  {
    if (K) K->train_param(*this, b);

    create(x, b);

    getProfile().TPSTART(QUDA_PROFILE_INIT);

    // whether to select alternative reliable updates
    bool alternative_reliable = param.use_alternative_reliable;

    double b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0 && param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      warningQuda("Warning: inverting on zero-field source");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b, matEig);
      if (deflate_compute) {
        // compute the deflation space.
        (*eig_solve)(evecs, evals);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        recompute_evals = false;
      }
    }

    double Anorm = 0;

    // for alternative reliable updates
    if (alternative_reliable) {
      // estimate norm for reliable updates
      mat(r, b);
      Anorm = sqrt(blas::norm2(r) / b2);
    }

    // compute initial residual
    double r2 = 0.0;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      if (b2 == 0) b2 = r2;
      // y contains the original guess.
      blas::copy(y, x);
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(y);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate and accumulate to solution vector
      eig_solve->deflate(y, r, evecs, evals, true);
      mat(r, y);
      r2 = blas::xmyNorm(b, r);
    }

    blas::zero(x);
    if (&x != &x_sloppy) blas::zero(x_sloppy);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if (K) {
      r_pre = r_sloppy;
      pushVerbosity(param.verbosity_precondition);
      (*K)(minvr_pre, r_pre);
      popVerbosity();
      minvr_sloppy = minvr_pre;
    }

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

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

    if (K) rMinvr = reDotProduct(r_sloppy, minvr_sloppy);

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;
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

      matSloppy(Ap, x_update_batch.get_current_field());

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
      double2 cg_norm = axpyCGNorm(-x_update_batch.get_current_alpha(), Ap, r_sloppy);
      // r --> r - alpha*A*p
      r2_old = r2;
      r2 = cg_norm.x;

      sigma = cg_norm.y >= 0.0 ? cg_norm.y : r2; // use r2 if (r_k+1, r_k-1 - r_k) breaks

      if (K) rMinvr_old = rMinvr;

      ru.update_rNorm(sqrt(r2));

      ru.evaluate(r2_old);

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if (convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol) ru.set_updateX();

      if (collect > 0 && k > collect_miniter && r2 < collect_tol * collect_tol * b2) {
        v_r[v_r.size() - collect] = r_sloppy;
        logQuda(QUDA_VERBOSE, "Collecting r %2d: r2 / b2 = %12.8e, k = %5d\n", collect, sqrt(r2 / b2), k);
        collect--;
      }

      if (!ru.trigger()) {

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(r_sloppy, minvr_sloppy);
          r_pre = r_sloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(minvr_pre, r_pre);
          popVerbosity();

          // can fuse these two kernels
          minvr_sloppy = minvr_pre;
          rMinvr = reDotProduct(r_sloppy, minvr_sloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
        } else {
          beta = sigma / r2_old; // use the alternative beta computation
        }

        if (Np == 1) {
          axpyZpbx(x_update_batch.get_current_alpha(), x_update_batch.get_current_field(), x_sloppy,
                   K ? minvr_sloppy : r_sloppy, beta);
        } else {
          if (x_update_batch.is_container_full()) { x_update_batch.accumulate_x(x_sloppy); }
          blas::xpayz(K ? minvr_sloppy : r_sloppy, beta, x_update_batch.get_current_field(),
                      x_update_batch.get_next_field());
        }

        ru.accumulate_norm(x_update_batch.get_current_alpha());

      } else { // reliable update

        // Now that we are performing reliable update, need to update x with the p's that have
        // not been used yet
        x_update_batch.accumulate_x(x_sloppy);
        x_update_batch.reset_next();

        xpy(x_sloppy, y);          // y += x
        // Now compute r
        mat(r, y);
        r2 = xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < ru.maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y);
          r2 = blas::xmyNorm(b, r);

          ru.update_maxr_deflate(r2);
        }

        copy(r_sloppy, r);
        zero(x_sloppy);

        bool L2breakdown = false;
        double L2breakdown_eps = 0;
        if (ru.reliable_break(r2, stop, L2breakdown, L2breakdown_eps)) { break; }

        ru.update_norm(r2, y);

        ru.reset(r2);

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(r_sloppy, minvr_sloppy);
          r_pre = r_sloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(minvr_pre, r_pre);
          popVerbosity();

          // can fuse these two kernels
          minvr_sloppy = minvr_pre;
          rMinvr = reDotProduct(r_sloppy, minvr_sloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
        } else {                        // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          double rp = reDotProduct(r_sloppy, x_update_batch.get_current_field()) / (r2);
          axpy(-rp, r_sloppy, x_update_batch.get_current_field());

          beta = r2 / r2_old;
        }
        xpayz(K ? minvr_sloppy : r_sloppy, beta, x_update_batch.get_current_field(), x_update_batch.get_next_field());
      }

      ++k;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);

      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);
      // if we have converged and need to update any trailing solutions
      if ((converged || k == param.maxiter) && ru.steps_since_reliable > 0 && !x_update_batch.is_container_full()) {
        x_update_batch.accumulate_x(x_sloppy);
      }

      if (ru.steps_since_reliable == 0) {
        x_update_batch.reset();
      } else {
        ++x_update_batch;
      }
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    if (mixed()) copy(x, x_sloppy);
    xpy(y, x); // x += y

    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);
    if (collect > 0) { warningQuda("%d r vectors still to be collected...", collect); }
    logQuda(QUDA_VERBOSE, "PCG: Reliable updates = %d\n", ru.rUpdate);

    // compute the true residual
    mat(r, x);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

} // namespace quda
