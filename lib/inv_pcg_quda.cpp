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

  PCG::PCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);
    // Preconditioners do not need a deflation space,
    // so we explicily set this here.
    Kparam.deflate = false;

    K = createPreconditioner(matPrecon, matPrecon, matPrecon, matEig, param, Kparam);
  }

  PCG::PCG(const DiracMatrix &mat, Solver &K_, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), K(nullptr), Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);

    K = wrapExternalPreconditioner(K_);
  }

  PCG::~PCG()
  {
    getProfile().TPSTART(QUDA_PROFILE_FREE);

    extractInnerSolverParam(param, Kparam);
    destroyDeflationSpace();

    getProfile().TPSTOP(QUDA_PROFILE_FREE);
  }

  void PCG::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      ColorSpinorParam csParam(b[0]);

      resize(r, b.size(), csParam);

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      resize(y, b.size(), csParam);

      // create sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      resize(Ap, b.size(), csParam);

      if (!mixed() || !param.use_sloppy_partial_accumulator) {
        create_alias(x_sloppy, x);
      } else {
        resize(x_sloppy, b.size(), csParam);
      }

      if (!mixed()) {
        create_alias(r_sloppy, r);
      } else {
        resize(r_sloppy, b.size(), csParam);
      }

      if (K) {
        resize(minvr_sloppy, b.size(), csParam);

        // create preconditioner intermediates
        csParam.setPrecision(Kparam.precision);
        resize(r_pre, b.size(), csParam);
        // Create minvr_pre
        resize(minvr_pre, b.size(), csParam);
      }

      Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
      if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline", Np);

      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      init = true;
    }
  }

  void PCG::solve_and_collect(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                              cvector_ref<ColorSpinorField> &v_r, int collect_miniter, double collect_tol)
  {
    if (K) K->train_param(*this, b[0]);

    if (v_r.size() && x.size() > 1) errorQuda("Collect not supported for multi-RHS PCG");

    getProfile().TPSTART(QUDA_PROFILE_INIT);

    // whether to select alternative reliable updates
    bool alternative_reliable = param.use_alternative_reliable;

    auto b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (is_zero_src(x, b, b2)) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    create(x, b);

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b[0], matEig);
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

    double Anorm = 0.0;

    // for alternative reliable updates
    if (alternative_reliable) {
      // estimate norm for reliable updates
      mat(r[0], b[0]);
      Anorm = sqrt(norm2(r[0]) / b2[0]);
    }

    // compute initial residual
    vector<double> r2(b.size(), 0.0);
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      for (auto i = 0u; i < b.size(); i++)
        if (b2[i] == 0) b2[i] = r2[i];
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
    if (param.use_sloppy_partial_accumulator) blas::zero(x_sloppy);
    if (r_sloppy[0].Precision() != r[0].Precision()) blas::copy(r_sloppy, r);

    auto csParam(r_sloppy[0]);
    std::vector<XUpdateBatch> x_update_batch(b.size());
    for (auto i = 0u; i < b.size(); i++)
      x_update_batch[i] = XUpdateBatch(Np, K ? minvr_sloppy[i] : r_sloppy[i], csParam);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if (K) {
      blas::copy(r_pre, r_sloppy);
      pushVerbosity(param.verbosity_precondition);
      (*K)(minvr_pre, r_pre);
      popVerbosity();
      blas::copy(minvr_sloppy, minvr_pre);
    }

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    auto stop_hq = std::vector(b.size(), param.tol_hq);

    std::vector<double> heavy_quark_res(b.size(), 0.0); // heavy quark residual
    if (use_heavy_quark_res) {
      auto hq = HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
    }

    std::vector<double> beta(b.size(), 0.0);
    std::vector<double> pAp(b.size(), 0.0);
    std::vector<double> rMinvr(b.size(), 0.0);
    std::vector<double> rMinvr_old(b.size(), 0.0);
    std::vector<double> r_new_Minvr_old(b.size(), 0.0);
    std::vector<double> r2_old(b.size(), 0.0);

    if (K) { rMinvr = reDotProduct(r_sloppy, minvr_sloppy); }

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

    ReliableUpdates ru(ru_params, r2[0]);

    bool converged = convergence(r2, heavy_quark_res, stop, stop_hq);

    auto get_p = [](std::vector<XUpdateBatch> &x_update_batch, bool next = false) {
      vector_ref<ColorSpinorField> p;
      p.reserve(x_update_batch.size());
      for (auto &x : x_update_batch) p.push_back(next ? x.get_next_field() : x.get_current_field());
      return p;
    };

    auto get_alpha = [](std::vector<XUpdateBatch> &x_update_batch) {
      vector<double> alpha;
      alpha.reserve(x_update_batch.size());
      for (auto &x : x_update_batch) alpha.push_back(x.get_current_alpha());
      return alpha;
    };

    while (!converged && k < param.maxiter) {
      auto p = get_p(x_update_batch);
      auto p_next = get_p(x_update_batch, true);
      matSloppy(Ap, p);

      // alternative reliable updates,
      if (alternative_reliable) {
        auto pAppp = blas::cDotProductNormA(p, Ap);
        for (auto i = 0u; i < b.size(); i++) pAp[i] = pAppp[i].x;
        ru.update_ppnorm(pAppp[0].z);
      } else {
        pAp = reDotProduct(p, Ap);
      }

      for (auto i = 0u; i < b.size(); i++)
        x_update_batch[i].get_current_alpha() = K ? rMinvr[i] / pAp[i] : r2[i] / pAp[i];

      auto cg_norm = axpyCGNorm(-get_alpha(x_update_batch), Ap, r_sloppy);
      // r --> r - alpha*A*p
      r2_old = r2;

      vector<double> sigma(b.size());
      for (auto i = 0u; i < b.size(); i++) {
        r2[i] = cg_norm[i].x;
        sigma[i] = cg_norm[i].y >= 0.0 ? cg_norm[i].y : r2[i]; // use r2 if (r_k+1, r_k-1 - r_k) breaks
      }

      if (K) rMinvr_old = rMinvr;

      ru.update_rNorm(sqrt(r2[0]));
      ru.evaluate(r2_old[0]);

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if (convergence(r2, heavy_quark_res, stop, stop_hq) && param.delta >= param.tol) ru.set_updateX();

      if (collect > 0 && k > collect_miniter && r2[0] < collect_tol * collect_tol * b2[0]) {
        blas::copy(v_r[v_r.size() - collect], r_sloppy);
        logQuda(QUDA_VERBOSE, "Collecting r %2d: r2 / b2 = %12.8e, k = %5d\n", collect, sqrt(r2[0] / b2[0]), k);
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

          for (auto i = 0u; i < b.size(); i++) beta[i] = (rMinvr[i] - r_new_Minvr_old[i]) / rMinvr_old[i];
        } else {
          for (auto i = 0u; i < b.size(); i++) beta[i] = sigma[i] / r2_old[i]; // use the alternative beta computation
        }

        if (Np == 1) {
          axpyZpbx(get_alpha(x_update_batch), p, x_sloppy, K ? minvr_sloppy : r_sloppy, beta);
        } else {
          for (auto i = 0u; i < b.size(); i++) {
            if (x_update_batch[i].is_container_full()) x_update_batch[i].accumulate_x(x_sloppy[i]);
          }
          blas::xpayz(K ? minvr_sloppy : r_sloppy, beta, p, p_next);
        }

        ru.accumulate_norm(get_alpha(x_update_batch)[0]);

      } else { // reliable update

        // Now that we are performing reliable update, need to update x with the p's that have
        // not been used yet
        for (auto i = 0u; i < b.size(); i++) {
          x_update_batch[i].accumulate_x(x_sloppy[i]);
          x_update_batch[i].reset_next();
        }
        xpy(x_sloppy, y);          // y += x
        // Now compute r
        mat(r, y);
        r2 = xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < ru.maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y);
          r2 = blas::xmyNorm(b, r);

          ru.update_maxr_deflate(r2[0]);
        }

        copy(r_sloppy, r);
        zero(x_sloppy);

        bool L2breakdown = false;
        double L2breakdown_eps = 0;
        if (ru.reliable_break(r2[0], stop[0], L2breakdown, L2breakdown_eps)) { break; }

        ru.update_norm(r2[0], y[0]);
        ru.reset(r2[0]);

        auto p = get_p(x_update_batch);
        auto p_next = get_p(x_update_batch, true);

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

          for (auto i = 0u; i < b.size(); i++) beta[i] = (rMinvr[i] - r_new_Minvr_old[i]) / rMinvr_old[i];
        } else {                        // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          auto rp = cDotProduct(r_sloppy, p);
          for (auto i = 0u; i < b.size(); i++) rp[i] /= r2[i];
          caxpy(-rp, r_sloppy, p);

          for (auto i = 0u; i < b.size(); i++) beta[i] = r2[i] / r2_old[i];
        }
        xpayz(K ? minvr_sloppy : r_sloppy, beta, p, p_next);
      }

      k++;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);

      converged = convergence(r2, heavy_quark_res, stop, stop_hq);

      // if we have converged and need to update any trailing solutions
      for (auto i = 0u; i < b.size(); i++) {
        if ((converged || k == param.maxiter) && ru.steps_since_reliable > 0 && !x_update_batch[i].is_container_full()) {
          x_update_batch[i].accumulate_x(x_sloppy[i]);
        }

        if (ru.steps_since_reliable == 0) {
          x_update_batch[i].reset();
        } else {
          ++x_update_batch[i];
        }
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
    auto true_res = xmyNorm(b, r);
    for (auto i = 0u; i < b.size(); i++) param.true_res[i] = sqrt(true_res[i] / b2[i]);

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

} // namespace quda
