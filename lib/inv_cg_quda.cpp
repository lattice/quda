#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <eigensolve_quda.h>
#include <eigen_helper.h>

#include <reliable_updates.h>
#include <invert_x_update.h>

namespace quda {

  CG::CG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
         SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param)
  {
  }

  CG::~CG() { destroyDeflationSpace(); }

  void CG::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      resize(y, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);

      // sloppy fields
      ColorSpinorParam csParam(x[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      resize(p, b.size(), csParam);
      resize(Ap, b.size(), csParam);

      if (param.precision != param.precision_sloppy) {
        resize(r_sloppy, b.size(), csParam);
      } else {
        create_alias(r_sloppy, r);
      }
      param.use_sloppy_partial_accumulator = false; // hard-code precise accumulation
      if (param.use_sloppy_partial_accumulator) resize(x_sloppy, b.size(), csParam);

      init = true;
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
    }

    // need to reset x_sloppy every solve
    if (!param.use_sloppy_partial_accumulator) create_alias(x_sloppy, x);
  }

  void CG::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                      cvector_ref<const ColorSpinorField> &p_init, cvector<double> &r2_old_init)
  {
    if (param.is_preconditioner) commGlobalReductionPush(param.global_reduction);

    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    const int Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
    if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline", Np);

    // Determine whether or not we're doing a heavy quark residual
    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if (use_heavy_quark_res) {
      hqsolve(x, b);
      if (param.is_preconditioner) commGlobalReductionPop();
      return;
    }

    // This check is pointless in the current version of the code, but it's being proactively added
    // just in case HQ residual solves are split into a separate file
    if (use_heavy_quark_res) errorQuda("The \"vanilla\" CG solver does not support HQ residual solves");

    /**
      When CG is used as a preconditioner, and we disable the `advanced features`, these features are turned off:
      - Reliable updates
      - Pipelining
      - Always use zero as the initial guess
      - Heavy quark residual
    */
    bool advanced_feature = !(param.precondition_no_advanced_feature && param.is_preconditioner);

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);

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
        if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_INIT);
        (*eig_solve)(evecs, evals);
        if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        recompute_evals = false;
      }
    }

    const double u = precisionEpsilon(param.precision_sloppy);
    const double uhigh = precisionEpsilon(); // solver precision

    double Anorm = 0.0;
    vector<double> beta(b.size(), 0.0);

    // for alternative reliable updates
    if (advanced_feature && alternative_reliable) {
      // estimate norm for reliable updates
      mat(r[0], b[0]);
      Anorm = sqrt(blas::norm2(r[0]) / b2[0]);
    }

    // compute initial residual
    vector<double> r2(b2.size(), 0.0);
    if (advanced_feature && param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
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
    blas::copy(r_sloppy, r);

    auto csParam(r_sloppy[0]);
    std::vector<XUpdateBatch> x_update_batch(b.size());
    for (auto i = 0u; i < b.size(); i++)
      x_update_batch[i] = XUpdateBatch(Np, !p_init[i].empty() ? p_init[i] : r_sloppy[i], csParam);

    vector<double> r2_old(r2.size(), 0.0);
    for (auto i = 0u; i < b.size(); i++) {
      if (r2_old_init[i] != 0.0 and !p_init[i].empty()) {
        // FIXME vectorize this
        r2_old[i] = r2_old_init[i];
        Complex rp = blas::cDotProduct(r_sloppy[i], x_update_batch[i].get_current_field()) / (r2[i]);
        blas::caxpy(-rp, r_sloppy[i], x_update_batch[i].get_current_field());
        beta[i] = r2[i] / r2_old[i];
        blas::xpayz(r_sloppy[i], beta[i], x_update_batch[i].get_current_field(), x_update_batch[i].get_current_field());
      }
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
    }

    auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    vector<double> pAp(b.size());

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }

    int k = 0;

    PrintStats("CG", k, r2, b2);

    bool converged = convergenceL2(r2, stop);

    ReliableUpdatesParams ru_params;

    ru_params.alternative_reliable = alternative_reliable;
    ru_params.u = u;
    ru_params.uhigh = uhigh; // solver precision
    ru_params.Anorm = Anorm;
    ru_params.delta = param.delta;

    ru_params.maxResIncrease = param.max_res_increase;
    ru_params.maxResIncreaseTotal = param.max_res_increase_total;
    ru_params.use_heavy_quark_res = false; // since we've removed HQ residual support

    ReliableUpdates ru(ru_params, r2[0]);

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

    while ( !converged && k < param.maxiter ) {
      auto p = get_p(x_update_batch);
      auto p_next = get_p(x_update_batch, true);
      matSloppy(Ap, p);

      vector<double> sigma(b.size());

      bool breakdown = false;
      if (advanced_feature && param.pipeline) {
        vector<double> Ap2(b.size());
        if (alternative_reliable) {
          auto quadruple = blas::quadrupleCGReduction(r_sloppy, Ap, p);
          for (auto i = 0u; i < b.size(); i++) {
            r2[i] = quadruple[i].x;
            Ap2[i] = quadruple[i].y;
            pAp[i] = quadruple[i].z;
          }
          ru.update_ppnorm(quadruple[0].w); // using 0th system for RU
        } else {
          auto triplet = blas::tripleCGReduction(r_sloppy, Ap, p);
          for (auto i = 0u; i < b.size(); i++) {
            r2[i] = triplet[i].x;
            Ap2[i] = triplet[i].y;
            pAp[i] = triplet[i].z;
          }
        }
        r2_old = r2;
        for (auto i = 0u; i < b.size(); i++) {
          x_update_batch[i].get_current_alpha() = r2[i] / pAp[i];
          sigma[i] = x_update_batch[i].get_current_alpha() * (x_update_batch[i].get_current_alpha() * Ap2[i] - pAp[i]);
        }
        if (sigma[0] < 0.0 || ru.steps_since_reliable == 0) { // sigma condition has broken down
          r2 = blas::axpyNorm(-get_alpha(x_update_batch), Ap, r_sloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;

        // alternative reliable updates,
        if (advanced_feature && alternative_reliable) {
          auto pAppp = blas::cDotProductNormA(p, Ap);
          for (auto i = 0u; i < b.size(); i++) pAp[i] = pAppp[i].x;
          ru.update_ppnorm(pAppp[0].z); // using 0th system for RU
        } else {
          pAp = blas::reDotProduct(p, Ap);
        }

        for (auto i = 0u; i < b.size(); i++) x_update_batch[i].get_current_alpha() = r2[i] / pAp[i];

        // here we are deploying the alternative beta computation
        auto cg_norm = blas::axpyCGNorm(-get_alpha(x_update_batch), Ap, r_sloppy);
        for (auto i = 0u; i < b.size(); i++) {
          r2[i] = cg_norm[i].x;                                  // (r_new, r_new)
          sigma[i] = cg_norm[i].y >= 0.0 ? cg_norm[i].y : r2[i]; // use r2 if (r_k+1, r_k+1-r_k) breaks
        }
      }

      // reliable update conditions
      ru.update_rNorm(sqrt(r2[0]));

      if (advanced_feature) {
        ru.evaluate(r2_old[0]);
        // force a reliable update if we are within target tolerance (only if doing reliable updates)
        if (convergenceL2(r2, stop) && param.delta >= param.tol) ru.set_updateX();
      }

      if (!ru.trigger()) {
        for (auto i = 0u; i < beta.size(); i++) beta[i] = sigma[i] / r2_old[i]; // use the alternative beta computation

        if (advanced_feature && param.pipeline && !breakdown) {

          if (Np == 1) {
            blas::tripleCGUpdate(get_alpha(x_update_batch), beta, Ap, x_sloppy, r_sloppy, p);
          } else {
            errorQuda("Not implemented pipelined CG with Np > 1");
          }
        } else {

          if (Np == 1) {
            // with Np=1 we just run regular fusion between x and p updates
            blas::axpyZpbx(get_alpha(x_update_batch), p, x_sloppy, r_sloppy, beta);
          } else {

            for (auto i = 0u; i < b.size(); i++) {
              if (x_update_batch[i].is_container_full()) x_update_batch[i].accumulate_x(x_sloppy[i]);
            }

            // p[(k+1)%Np] = r + beta * p[k%Np]
            blas::xpayz(r_sloppy, beta, p, p_next);
          }
        }

        // alternative reliable updates
        if (advanced_feature) { ru.accumulate_norm(x_update_batch[0].get_current_alpha()); }
      } else {

        for (auto i = 0u; i < b.size(); i++) {
          x_update_batch[i].accumulate_x(x_sloppy[i]);
          x_update_batch[i].reset_next();
        }
        blas::xpy(x_sloppy, y); // swap these around?

        mat(r, y);       //  here we can use x as tmp
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < ru.maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y);
          r2 = blas::xmyNorm(b, r);

          ru.update_maxr_deflate(r2[0]);
        }

        blas::copy(r_sloppy, r); // nop when these pointers alias
        blas::zero(x_sloppy);

        if (advanced_feature) { ru.update_norm(r2[0], y[0]); }

        if (advanced_feature) {
          // needed as a "dummy parameter" to reliable_break.
          bool L2breakdown = false;
          if (ru.reliable_break(r2[0], stop[0], L2breakdown, 0)) { break; }
        }

        // explicitly restore the orthogonality of the gradient vector
        auto p = get_p(x_update_batch);
        auto p_next = get_p(x_update_batch, true);

        auto rp = blas::cDotProduct(r_sloppy, p);
        for (auto i = 0u; i < b.size(); i++) rp[i] /= r2[i];
        blas::caxpy(-rp, r_sloppy, p);

        for (auto i = 0u; i < beta.size(); i++) beta[i] = r2[i] / r2_old[i];
        blas::xpayz(r_sloppy, beta, p, p_next);

        ru.reset(r2[0]);
      }

      breakdown = false;
      k++;

      PrintStats("CG", k, r2, b2);
      // check convergence
      converged = convergenceL2(r2, stop);

      // if we have converged and need to update any trailing solutions
      for (auto i = 0u; i < b.size(); i++) {
        if (converged && ru.steps_since_reliable > 0 && !x_update_batch[i].is_container_full()) {
          x_update_batch[i].accumulate_x(x_sloppy[i]);
        }

        if (ru.steps_since_reliable == 0) {
          x_update_batch[i].reset();
        } else {
          ++x_update_batch[i];
        }
      }
    }

    blas::copy(x, x_sloppy);
    blas::xpy(y, x);

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

      param.iter += k;

      if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);
    }

    logQuda(QUDA_VERBOSE, "CG: Reliable updates = %d\n", ru.rUpdate);

    if (advanced_feature && param.compute_true_res) {
      // compute the true residuals
      mat(r, x);
      auto true_r2 = blas::xmyNorm(b, r);
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(true_r2[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
    }

    PrintSummary("CG", k, r2, b2, stop);

    if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);

    if (param.is_preconditioner) commGlobalReductionPop();
  }

  cvector_ref<const ColorSpinorField> CG::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    return r;
  }

  // Separate HQ residual codepath
  void CG::hqsolve(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    logQuda(QUDA_VERBOSE, "Performing a HQ CG solve\n");

    // Verbose errors: HQ solves won't support deflation, pipelining
    if (param.deflate) errorQuda("HQ solves don't support deflation");
    if (param.is_preconditioner) errorQuda("HQ solves cannot be preconditioners");

    // Non-terminal errors: HQ solves don't support advanced reliable updates
    if (param.use_alternative_reliable)
      logQuda(QUDA_SUMMARIZE,
              "HQ solves don't support alternative reliable updates, reverting to traditional reliable updates\n");
    if (param.pipeline) logQuda(QUDA_SUMMARIZE, "HQ solves don't support pipelining, disabling...");

    getProfile().TPSTART(QUDA_PROFILE_INIT);

    vector<double> b2 = blas::norm2(b);

    // Detect whether this is a pure double solve or not; informs the necessity of some stability checks
    bool is_pure_double = (param.precision == QUDA_DOUBLE_PRECISION && param.precision_sloppy == QUDA_DOUBLE_PRECISION);

    bool heavy_quark_restart = false;

    // Check to see that we're not trying to invert on a zero-field source
    if (is_zero_src(x, b, b2)) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    create(x, b);

    // for detecting HQ residual stalls
    // let |r2/b2| drop to epsilon tolerance * 1e-30, semi-arbitrarily, but
    // with the intent of letting the solve grind as long as possible before
    // triggering a `NaN`. Ignored for pure double solves because if
    // pure double has stability issues, bigger problems are at hand.
    const double uhigh = precisionEpsilon(); // solver precision
    const double hq_res_stall_check = is_pure_double ? 0. : uhigh * uhigh * 1e-60;

    // compute initial residual
    vector<double> r2(b.size());
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

    blas::zero(x);
    if (param.use_sloppy_partial_accumulator) blas::zero(x_sloppy);
    blas::copy(r_sloppy, r);
    blas::copy(p, r_sloppy);

    vector<double> r2_old(b.size(), 0.0);

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    auto stop_hq = std::vector(b.size(), param.tol_hq);

    auto get_hq_res = [](cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &r) {
      auto hq_nrm = blas::HeavyQuarkResidualNorm(x, r);
      vector<double> hq_res(hq_nrm.size());
      for (auto i = 0u; i < hq_nrm.size(); i++) hq_res[i] = sqrt(hq_nrm[i].z);
      return hq_res;
    };

    // compute the initial heavy quark residual
    vector<double> hq_res = get_hq_res(x, r);

    vector<double> alpha(b.size()), beta(b.size()), sigma(b.size()), pAp(b.size());

    // Whether or not we also need to compute the L2 norm
    const bool L2_required = param.residual_type & (QUDA_L2_RELATIVE_RESIDUAL | QUDA_L2_ABSOLUTE_RESIDUAL);

    // set L2breakdown to be immediately true if we aren't requesting an L2 norm, alternatively,
    // it only gets set to true after some heuristics suggest the L2 norm has "stalled out"
    bool L2breakdown = !L2_required;
    const double L2breakdown_eps = 100. * uhigh;

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;

    PrintStats("CG", k, r2, b2, hq_res);

    bool converged = convergence(r2, hq_res, stop, stop_hq);

    // Various parameters related to restarts

    // Trackers for the L2 norm:
    //  rNorm: current iterated |r|
    // r0Norm: computed |r| at the last reliable update
    auto rNorm = sqrt(r2[0]);
    auto r0Norm = rNorm;

    // If the computed |r| goes above r0Norm between reliable updates,
    // update this ceiling. This goes into "R" type reliable updates.
    double maxrx = L2breakdown ? hq_res[0] : rNorm;
    double maxrr = L2breakdown ? hq_res[0] : rNorm;

    // Triggers for explicitly counting residual updates and checking for L2breakdown.
    // * updateX broadly maps to if the iterated residual has dropped by a factor of delta
    //   relative to the previously re-computed residual.
    // * updateR broadly maps to if the iterated residual has dropped by a factor of delta
    //   relative to the max of the previously re-computed residual and all iterated residuals
    //   since the last reliable update.
    bool updateX = false;
    bool updateR = false;

    // Counter for the number of times in a row the computed residual has jumped above the
    // previously computed residual.
    int resIncrease = 0;

    // Counter for the total number of times the computed residual has increased above the previously
    // computed residual, independent of when it happened.
    int resIncreaseTotal = 0;

    // Trackers for the HQ residual
    // hq0Res: computed HQ residual at the last reliable updated
    auto hq0Res = hq_res;

    // Counter for the number of times in a row the computed heavy quark residual has
    // jumped above the previously computed heavy quark residual.
    int hqresIncrease = 0;

    // Counter for the total number of times a reliable updated based on the heavy quark residual
    // has been triggered.
    int hqresRestartTotal = 0;

    // Count the steps since a reliable update and the total number of reliable updates.
    // The steps since a reliable update is also used to make sure final convergence is
    // based on the computed residual and not the iterated residual.
    int rUpdate = 0;
    int steps_since_reliable = 1;

    while (!converged && k < param.maxiter) {
      matSloppy(Ap, p);

      r2_old = r2;

      pAp = blas::reDotProduct(p, Ap);

      for (auto i = 0u; i < alpha.size(); i++) alpha[i] = r2[i] / pAp[i];

      // here we are deploying the alternative beta computation
      auto cg_norm = blas::axpyCGNorm(-alpha, Ap, r_sloppy);
      for (auto i = 0u; i < cg_norm.size(); i++) {
        r2[i] = cg_norm[i].x;                                  // (r_new, r_new)
        sigma[i] = cg_norm[i].y >= 0.0 ? cg_norm[i].y : r2[i]; // use r2 if (r_k+1, r_k+1-r_k) breaks
      }
      rNorm = sqrt(r2[0]);

      // If the iterated norm has dropped by more than a factor of delta, trigger
      // an update. The baseline we check against differs depending on if
      // we're still checking the L2 norm, or if that has converged/broken down and we're
      // now looking at the HQ residual.

      if (!L2breakdown && (L2_required || convergenceL2(r2, stop))) {
        // L2 based reliable update

        // If the iterated residual norm has gone above the most recent "baseline" norm,
        // update the baseline norm.
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;

        // Has the iterated norm dropped by a factor of delta from the last computed norm?
        updateX = (rNorm < param.delta * r0Norm && r0Norm <= maxrx);

        // Has the iterated norm dropped by a factor of delta relative to the largest the
        // iterated norm has been since the last update?
        updateR = ((rNorm < param.delta * maxrr && r0Norm <= maxrr) || updateX);
      } else {
        // hqresidual based reliable update
        if (hq_res[0] > maxrx) maxrx = hq_res[0];
        if (hq_res[0] > maxrr) maxrr = hq_res[0];

        // I'm making the decision to use `param.delta` for the hq_res check because
        // in some regards it's an L2-esque norm...

        // Has the iterated heavy quark residual dropped by a factor of delta^2 from the last
        // computed norm?
        updateX = (hq_res[0] < param.delta * param.delta * hq0Res[0] && r0Norm <= maxrx);

        // Has the iterated heavy quark residual dropped by a factor of delta relative
        // to the largest the iterated norm has been since the last update?
        updateR = ((hq_res[0] < param.delta * param.delta * maxrr && hq0Res[0] <= maxrr) || updateX);
      }

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if (convergence(r2, hq_res, stop, stop_hq) && param.delta >= param.tol) updateX = true;

      // force a reliable update based on the HQ residual if L2 breakdown has already happened
      if (L2breakdown && (convergenceHQ(hq_res, stop_hq) || (r2[0] / b2[0]) < hq_res_stall_check)
          && param.delta >= param.tol)
        updateX = true;

      if (!(updateR || updateX)) {
        // No reliable update needed

        for (auto i = 0u; i < beta.size(); i++) beta[i] = sigma[i] / r2_old[i]; // use the alternative beta computation

        blas::axpyZpbx(alpha, p, x_sloppy, r_sloppy, beta);

        auto get_hq_res2 = [](cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                              cvector_ref<const ColorSpinorField> &r) {
          auto hq_nrm = blas::xpyHeavyQuarkResidualNorm(x, y, r);
          vector<double> hq_res(hq_nrm.size());
          for (auto i = 0u; i < hq_nrm.size(); i++) hq_res[i] = sqrt(hq_nrm[i].z);
          return hq_res;
        };

        if (k % param.heavy_quark_check == 0) {
          if (param.precision != param.precision_sloppy) {
            blas::copy(r, r_sloppy);
            hq_res = get_hq_res2(x_sloppy, y, r);
          } else {
            hq_res = get_hq_res2(x_sloppy, y, r_sloppy);
          }
        }

        steps_since_reliable++;

      } else {
        // We're performing a reliable update

        // Accumulate p into x, accumulate x into the total solution y, explicitly recompute the residual vector
        blas::axpy(alpha, p, x_sloppy);
        blas::copy(x, x_sloppy); // no op when these pointers alias
        blas::xpy(x, y);
        mat(r, y);

        // Recompute the exact residual and heavy quark residual
        r2 = blas::xmyNorm(b, r);
        rNorm = sqrt(r2[0]);
        hq_res = get_hq_res(y, r);

        // Copy and update fields
        blas::copy(r_sloppy, r); // no op when these pointers alias
        blas::zero(x_sloppy);

        // Check and see if we're "done" with the L2 norm. This could be because
        // we were already done with it, we never needed it, or the L2 norm has finally converged.
        if (!L2breakdown && convergenceL2(r2, stop)) L2breakdown = true;

        // Depending on if we're still grinding on the L2 norm or if we've moved along to just
        // the HQ norm, we reset the baselines for reliable updates that get used on the
        // *next* iteration. We still need the baselines that were used for this iteration
        // for the checks down below.
        if (!L2breakdown) {
          // If we're still grinding on the L2 norm, the new baseline is the freshly
          // recomputed |r|.
          maxrr = rNorm;
          maxrx = rNorm;
        } else {
          // If we've made it to the HQ norm, the new baseline is the freshly recomputed
          // heavy quark residual
          maxrr = hq_res[0];
          maxrx = hq_res[0];

          // Once we're dealing with the heavy quark residual, we perform a *hard* CG
          // restart at every reliable update via setting the search vector `p` to the current
          // exact residual vector.
          heavy_quark_restart = true;

          // And then we keep track of the fact we're doing a HQ residual reliable update...
          hqresRestartTotal++;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual (total #inc %i)",
                      hqresRestartTotal);

          if (hqresRestartTotal > param.max_hq_res_restart_total) {
            // ...and if we've restarted too many times, flunk out of the solve.
            warningQuda("CG: solver exiting due to too many heavy quark residual restarts (%i/%i)", hqresRestartTotal,
                        param.max_hq_res_restart_total);
            break;
          }
        }

        // Check and see if we've reached the limit of the precision. There isn't necessarily
        // a great way to do this, so as a proxy we check to see if the new computed residual is
        // larger than the computed residual from the last reliable update, and if this is the case
        // enough times we throw up our hands, say "we're good here", and switch over to the HQ
        // residual.
        if (rNorm > r0Norm && updateX && !L2breakdown) {
          // Count the number of times in a row this has happened
          resIncrease++;

          // And count the total number of times this has happened outright
          resIncreaseTotal++;

          // ...tell the world about it too.
          warningQuda(
            "new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2[0]), r0Norm, resIncreaseTotal);

          // If the norm is ridiculously small in magnitude, we've exceeded the maximums on various
          // ways we keep track of residual increases, or the L2 norm converged, we say "we're good here"
          // and move over to the HQ residual norm.
          if (rNorm < L2breakdown_eps || resIncrease > param.max_res_increase
              || resIncreaseTotal > param.max_res_increase_total || r2 < stop) {
            L2breakdown = true;
            warningQuda("L2 breakdown %e, %e", rNorm, L2breakdown_eps);

            // We also have to do a logic correction, switching the reliable update baselines we set above
            // from the L2 norm over to the HQ residual.
            maxrr = hq_res[0];
            maxrx = hq_res[0];
          }
        } else {
          // This variable counts the number of times in a row the computed residual has gone up,
          // so if it hasn't gone up this time around we reset this counter.
          resIncrease = 0;
        }

        // If we've done checking the L2 norm, we do a similar check of if the HQ residual has increased
        // for multiple reliable updates in a row.
        if (hq_res > hq0Res && updateX && L2breakdown) {
          // Count the number of consecutive increases
          hqresIncrease++;

          // Tell the world about it
          warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e",
                      hq_res[0], hq0Res[0]);

          // And if it's increased too many times in a row, flunk out of the solve.
          if (hqresIncrease > param.max_hq_res_increase) {
            warningQuda("CG: solver exiting due to too many heavy quark residual norm increases (%i/%i)", hqresIncrease,
                        param.max_hq_res_increase);
            break;
          }
        } else {
          // This variable counts the number of times in a row the computed heavy quark residual has increased,
          // so if it hasn't gone up this time around we reset the counter.
          hqresIncrease = 0;
        }

        // Depending on if we're in the L2 norm part of the solve or a HQ residual part of the solve
        // we "reset" the solve in a different way.
        if (heavy_quark_restart) {
          // If we're in the HQ residual part of the solve, we just do a hard CG restart.
          logQuda(QUDA_DEBUG_VERBOSE, "HQ restart == hard CG restart\n");
          blas::copy(p, r_sloppy);
          heavy_quark_restart = false;
        } else {
          // If we're still in the L2 norm part of the solve, we explicitly restore
          // the orthogonality of the gradient vector, recompute beta, update `p`, and carry on with our lives.
          logQuda(QUDA_DEBUG_VERBOSE, "Regular restart == explicit gradient vector re-orthogonalization\n");
          auto rp = blas::cDotProduct(r_sloppy, p);
          for (auto i = 0u; i < b.size(); i++) rp[i] / r2[i];
          blas::caxpy(-rp, r_sloppy, p);

          for (auto i = 0u; i < b.size(); i++) beta[i] = r2[i] / r2_old[i];
          blas::xpayz(r_sloppy, beta, p, p);
        }

        // Last, we increment the reliable update counter, reset the number of steps since the last reliable update,
        // and reset the cached value of |r| and the heavy quark residual from the time of this
        // reliable update.
        rUpdate++;
        steps_since_reliable = 0;
        r0Norm = sqrt(r2[0]);

        hq0Res = hq_res;
      }

      k++;

      PrintStats("CG", k, r2, b2, hq_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, hq_res, stop, stop_hq);

      // check for recent enough reliable updates of the HQ residual if we use it

      // L2 is converged or precision maxed out for L2
      bool L2done = L2breakdown || convergenceL2(r2, stop);
      // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
      bool HQdone = (steps_since_reliable == 0 && param.delta > 0) && convergenceHQ(hq_res, stop_hq);
      converged = L2done && HQdone;
    }

    blas::copy(x, x_sloppy);
    blas::xpy(y, x);

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    logQuda(QUDA_VERBOSE, "CG: Reliable updates = %d\n", rUpdate);

    if (param.compute_true_res) {
      // compute the true residuals
      mat(r, x);
      auto true_r2 = blas::xmyNorm(b, r);
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(true_r2[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
    }

    PrintSummary("CG", k, r2, b2, stop, stop_hq);

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

// use BlockCGrQ algortithm or BlockCG (with / without GS, see BLOCKCG_GS option)
#define BCGRQ 1
#if BCGRQ

#ifndef BLOCKSOLVER

  void CG::blocksolve(ColorSpinorField &, ColorSpinorField &) { errorQuda("QUDA_BLOCKSOLVER not built."); }

#else

  void CG::blocksolve(ColorSpinorField &x, ColorSpinorField &b)
  {
    getProfile().TPSTART(QUDA_PROFILE_INIT);

    using Eigen::MatrixXcd;

    // Check to see that we're not trying to invert on a zero-field source
    // MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_MULTI_SHIFT];
    double b2avg = 0;
    for (int i = 0; i < param.num_src; i++) {
      b2[i] = blas::norm2(b.Component(i));
      b2avg += b2[i];
      if (b2[i] == 0) {
        getProfile().TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source - undefined for block solver\n");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    b2avg = b2avg / param.num_src;

    create(x, b);

    // calculate residuals for all vectors
    // and initialize r2 matrix
    double r2avg = 0;
    MatrixXcd r2(param.num_src, param.num_src);
    for (int i = 0; i < param.num_src; i++) {
      mat(r.Component(i), x.Component(i));
      r2(i, i) = blas::xmyNorm(b.Component(i), r.Component(i));
      r2avg += r2(i, i).real();
      printfQuda("r2[%i] %e\n", i, r2(i, i).real());
    }
    for (int i = 0; i < param.num_src; i++) {
      for (int j = i + 1; j < param.num_src; j++) {
        r2(i, j) = blas::cDotProduct(r.Component(i), r.Component(j));
        r2(j, i) = std::conj(r2(i, j));
      }
    }

    blas::copy(rSloppy, r);
    blas::copy(p, rSloppy);
    blas::copy(rnew, rSloppy);

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    if (use_heavy_quark_res) errorQuda("ERROR: heavy quark residual not supported in block solver");

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop[QUDA_MAX_MULTI_SHIFT];

    for (int i = 0; i < param.num_src; i++) {
      stop[i] = stopping(param.tol, b2[i], param.residual_type); // stopping condition of solver
    }

    // Eigen Matrices instead of scalars
    MatrixXcd alpha = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd beta = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd C = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd S = MatrixXcd::Identity(param.num_src, param.num_src);
    MatrixXcd pAp = MatrixXcd::Identity(param.num_src, param.num_src);
    quda::Complex *AC = new quda::Complex[param.num_src * param.num_src];

#ifdef MWVERBOSE
    MatrixXcd pTp = MatrixXcd::Identity(param.num_src, param.num_src);
#endif

    // FIXME:reliable updates currently not implemented
    /*
    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }
    bool L2breakdown = false;
    int rUpdate = 0;
    nt steps_since_reliable = 1;
    */

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;

    PrintStats("CG", k, r2avg / param.num_src, b2avg);
    bool allconverged = true;
    bool converged[QUDA_MAX_MULTI_SHIFT];
    for (int i = 0; i < param.num_src; i++) {
      converged[i] = convergence(r2(i, i).real(), 0., stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }

    // CHolesky decomposition
    MatrixXcd L = r2.llt().matrixL(); //// retrieve factor L  in the decomposition
    C = L.adjoint();
    MatrixXcd Linv = C.inverse();

#ifdef MWVERBOSE
    std::cout << "r2\n " << r2 << std::endl;
    std::cout << "L\n " << L.adjoint() << std::endl;
#endif

    // set p to QR decompsition of r
    // temporary hack - use AC to pass matrix arguments to multiblas
    for (int i = 0; i < param.num_src; i++) {
      blas::zero(p.Component(i));
      for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = Linv(i, j); }
    }
    //blas::legacy::caxpy(AC, r, p);

    // set rsloppy to to QR decompoistion of r (p)
    for (int i = 0; i < param.num_src; i++) { blas::copy(rSloppy.Component(i), p.Component(i)); }

#ifdef MWVERBOSE
    for (int i = 0; i < param.num_src; i++) {
      for (int j = 0; j < param.num_src; j++) { pTp(i, j) = blas::cDotProduct(p.Component(i), p.Component(j)); }
    }
    std::cout << " pTp  " << std::endl << pTp << std::endl;
    std::cout << " L " << std::endl << L.adjoint() << std::endl;
    std::cout << " C " << std::endl << C << std::endl;
#endif

    while (!allconverged && k < param.maxiter) {
      // apply matrix
      for (int i = 0; i < param.num_src; i++) {
        matSloppy(Ap.Component(i), p.Component(i)); // tmp as tmp
      }

      // calculate pAp
      for (int i = 0; i < param.num_src; i++) {
        for (int j = i; j < param.num_src; j++) {
          pAp(i, j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
          if (i != j) pAp(j, i) = std::conj(pAp(i, j));
        }
      }

      // update Xsloppy
      alpha = pAp.inverse() * C;
      // temporary hack using AC
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = alpha(i, j); }
      }
      //blas::legacy::caxpy(AC, p, xSloppy);

      // update rSloppy
      beta = pAp.inverse();
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = -beta(i, j); }
      }
      //blas::legacy::caxpy(AC, Ap, rSloppy);

      // orthorgonalize R
      // copy rSloppy to rnew as temporary
      for (int i = 0; i < param.num_src; i++) { blas::copy(rnew.Component(i), rSloppy.Component(i)); }
      for (int i = 0; i < param.num_src; i++) {
        for (int j = i; j < param.num_src; j++) {
          r2(i, j) = blas::cDotProduct(r.Component(i), r.Component(j));
          if (i != j) r2(j, i) = std::conj(r2(i, j));
        }
      }
      // Cholesky decomposition
      L = r2.llt().matrixL(); // retrieve factor L  in the decomposition
      S = L.adjoint();
      Linv = S.inverse();
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        blas::zero(rSloppy.Component(i));
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = Linv(i, j); }
      }
      //blas::legacy::caxpy(AC, rnew, rSloppy);

#ifdef MWVERBOSE
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) {
          pTp(i, j) = blas::cDotProduct(rSloppy.Component(i), rSloppy.Component(j));
        }
      }
      std::cout << " rTr " << std::endl << pTp << std::endl;
      std::cout << "QR" << S << std::endl << "QP " << S.inverse() * S << std::endl;
      ;
#endif

      // update p
      // use rnew as temporary again for summing up
      for (int i = 0; i < param.num_src; i++) { blas::copy(rnew.Component(i), rSloppy.Component(i)); }
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = std::conj(S(j, i)); }
      }
      //blas::legacy::caxpy(AC, p, rnew);
      // set p = rnew
      for (int i = 0; i < param.num_src; i++) { blas::copy(p.Component(i), rnew.Component(i)); }

      // update C
      C = S * C;

#ifdef MWVERBOSE
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { pTp(i, j) = blas::cDotProduct(p.Component(i), p.Component(j)); }
      }
      std::cout << " pTp " << std::endl << pTp << std::endl;
      std::cout << "S " << S << std::endl << "C " << C << std::endl;
#endif

      // calculate the residuals for all shifts
      r2avg = 0;
      for (int j = 0; j < param.num_src; j++) {
        r2(j, j) = C(0, j) * conj(C(0, j));
        for (int i = 1; i < param.num_src; i++) r2(j, j) += C(i, j) * conj(C(i, j));
        r2avg += r2(j, j).real();
      }

      k++;
      PrintStats("CG", k, r2avg / param.num_src, b2avg);
      // check convergence
      allconverged = true;
      for (int i = 0; i < param.num_src; i++) {
        converged[i] = convergence(r2(i, i).real(), 0, stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
      }
    }

    for (int i = 0; i < param.num_src; i++) { blas::xpy(y.Component(i), xSloppy.Component(i)); }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // logQuda(QUDA_VERBOSE, "CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    for (int i = 0; i < param.num_src; i++) {
      mat(r.Component(i), x.Component(i));
      param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;

      PrintSummary("CG", k, r2(i, i).real(), b2[i], stop[i]);
    }

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
    getProfile().TPSTART(QUDA_PROFILE_FREE);

    delete[] AC;
    getProfile().TPSTOP(QUDA_PROFILE_FREE);

    return;
  }
#endif

#else

// use Gram Schmidt in Block CG ?
#define BLOCKCG_GS 1
void CG::solve(ColorSpinorField& x, ColorSpinorField& b) {
  #ifndef BLOCKSOLVER
  errorQuda("QUDA_BLOCKSOLVER not built.");
  #else
  #ifdef BLOCKCG_GS
  printfQuda("BCGdQ Solver\n");
  #else
  printfQuda("BCQ Solver\n");
  #endif
  const bool use_block = true;

  getProfile().TPSTART(QUDA_PROFILE_INIT);

  using Eigen::MatrixXcd;
  MatrixXcd mPAP(param.num_src,param.num_src);
  MatrixXcd mRR(param.num_src,param.num_src);


  // Check to see that we're not trying to invert on a zero-field source
  //MW: it might be useful to check what to do here.
  double b2[QUDA_MAX_MULTI_SHIFT];
  double b2avg=0;
  double r2avg=0;
  for(int i=0; i< param.num_src; i++){
    b2[i]=blas::norm2(b.Component(i));
    b2avg += b2[i];
    if(b2[i] == 0){
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      errorQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }
  }

  #ifdef MWVERBOSE
  MatrixXcd b2m(param.num_src,param.num_src);
  // just to check details of b
  for(int i=0; i<param.num_src; i++){
    for(int j=0; j<param.num_src; j++){
      b2m(i,j) = blas::cDotProduct(b.Component(i), b.Component(j));
    }
  }
  std::cout << "b2m\n" <<  b2m << std::endl;
  #endif

  create(x, b);

  //  const int i = 0;  // MW: hack to be able to write Component(i) instead and try with i=0 for now

  for (int i = 0; i < param.num_src; i++) { mat(r.Component(i), x.Component(i)); }

  // double r2[QUDA_MAX_MULTI_SHIFT];
  MatrixXcd r2(param.num_src,param.num_src);
  for(int i=0; i<param.num_src; i++){
    r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
    printfQuda("r2[%i] %e\n", i, r2(i,i).real());
  }
  if(use_block){
    // MW need to initalize the full r2 matrix here
    for(int i=0; i<param.num_src; i++){
      for(int j=i+1; j<param.num_src; j++){
        r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
        r2(j,i) = std::conj(r2(i,j));
      }
    }
  }

  blas::copy(rSloppy, r);
  blas::copy(p, rSloppy);
  blas::copy(pnew, rSloppy);

  if (&x != &xSloppy) {
    blas::copy(y, x);
    blas::zero(xSloppy);
  } else {
    blas::zero(y);
  }

  const bool use_heavy_quark_res =
  (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
  bool heavy_quark_restart = false;

  getProfile().TPSTOP(QUDA_PROFILE_INIT);
  getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

  MatrixXcd r2_old(param.num_src, param.num_src);
  double heavy_quark_res[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark res idual
  double heavy_quark_res_old[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark residual
  double stop[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
    if (use_heavy_quark_res) {
      heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      heavy_quark_res_old[i] = heavy_quark_res[i];   // heavy quark residual
    }
  }
  const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

  MatrixXcd alpha = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd beta = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd gamma = MatrixXcd::Identity(param.num_src,param.num_src);
  //  gamma = gamma * 2.0;

  MatrixXcd pAp(param.num_src, param.num_src);
  MatrixXcd pTp(param.num_src, param.num_src);
  int rUpdate = 0;

  double rNorm[QUDA_MAX_MULTI_SHIFT];
  double r0Norm[QUDA_MAX_MULTI_SHIFT];
  double maxrx[QUDA_MAX_MULTI_SHIFT];
  double maxrr[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    rNorm[i] = sqrt(r2(i,i).real());
    r0Norm[i] = rNorm[i];
    maxrx[i] = rNorm[i];
    maxrr[i] = rNorm[i];
  }

  double delta = param.delta;//MW: hack no reliable updates param.delta;

  // this parameter determines how many consective reliable update
  // reisudal increases we tolerate before terminating the solver,
  // i.e., how long do we want to keep trying to converge
  const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
  const int maxResIncreaseTotal = param.max_res_increase_total;
  // 0 means we have no tolerance
  // maybe we should expose this as a parameter
  const int hqmaxresIncrease = maxResIncrease + 1;

  int resIncrease = 0;
  int resIncreaseTotal = 0;
  int hqresIncrease = 0;

  // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
  // only used if we use the heavy_quark_res
  bool L2breakdown = false;

  getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

  int k = 0;

  for(int i=0; i<param.num_src; i++){
    r2avg+=r2(i,i).real();
  }
  PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);
  int steps_since_reliable = 1;
  bool allconverged = true;
  bool converged[QUDA_MAX_MULTI_SHIFT];
  for(int i=0; i<param.num_src; i++){
    converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
    allconverged = allconverged && converged[i];
  }
  MatrixXcd sigma(param.num_src,param.num_src);

  #ifdef BLOCKCG_GS
  // begin ignore Gram-Schmidt for now

  for(int i=0; i < param.num_src; i++){
    double n = blas::norm2(p.Component(i));
    blas::ax(1/sqrt(n),p.Component(i));
    for(int j=i+1; j < param.num_src; j++) {
      auto ri = blas::cDotProduct(p.Component(i), p.Component(j));
      blas::caxpy(-ri,p.Component(i),p.Component(j));
    }
  }

  gamma = MatrixXcd::Zero(param.num_src,param.num_src);
  for ( int i = 0; i < param.num_src; i++){
    for (int j=i; j < param.num_src; j++){
      gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
    }
  }
  #endif
  // end ignore Gram-Schmidt for now

  #ifdef MWVERBOSE
  for(int i=0; i<param.num_src; i++){
    for(int j=0; j<param.num_src; j++){
      pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
    }
  }

  std::cout << " pTp " << std::endl << pTp << std::endl;
  std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
  #endif
  while ( !allconverged && k < param.maxiter ) {
    for(int i=0; i<param.num_src; i++){
      matSloppy(Ap.Component(i), p.Component(i)); // tmp as tmp
    }


    bool breakdown = false;
    // FIXME: need to check breakdown
    // current implementation sets breakdown to true for pipelined CG if one rhs triggers breakdown
    // this is probably ok


    if (param.pipeline) {
      errorQuda("pipeline not implemented");
    } else {
      r2_old = r2;
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j < param.num_src; j++){
          if(use_block or i==j)
          pAp(i,j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
          else
          pAp(i,j) = 0.;
        }
      }

      alpha = pAp.inverse() * gamma.adjoint().inverse() * r2;
      #ifdef MWVERBOSE
      std::cout << "alpha\n" << alpha << std::endl;

      if(k==1){
        std::cout << "pAp " << std::endl <<pAp << std::endl;
        std::cout << "pAp^-1 " << std::endl <<pAp.inverse() << std::endl;
        std::cout << "r2 " << std::endl <<r2 << std::endl;
        std::cout << "alpha " << std::endl <<alpha << std::endl;
        std::cout << "pAp^-1r2" << std::endl << pAp.inverse()*r2 << std::endl;
      }
      #endif
      // here we are deploying the alternative beta computation
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j < param.num_src; j++){

          blas::caxpy(-alpha(j,i), Ap.Component(j), rSloppy.Component(i));
        }
      }
      // MW need to calculate the full r2 matrix here, after update. Not sure how to do alternative sigma yet ...
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j<param.num_src; j++){
          if(use_block or i==j)
          r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
          else
          r2(i,j) = 0.;
        }
      }
      sigma = r2;
    }


    bool updateX=false;
    bool updateR=false;
    //      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? true : false;
    //      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? true : false;
    //
    // printfQuda("Checking reliable update %i %i\n",updateX,updateR);
    // reliable update conditions
    for(int i=0; i<param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      if (rNorm[i] > maxrx[i]) maxrx[i] = rNorm[i];
      if (rNorm[i] > maxrr[i]) maxrr[i] = rNorm[i];
      updateX = (rNorm[i] < delta * r0Norm[i] && r0Norm[i] <= maxrx[i]) ? true : false;
      updateR = ((rNorm[i] < delta * maxrr[i] && r0Norm[i] <= maxrr[i]) || updateX) ? true : false;
    }
    if ( (updateR || updateX )) {
      // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
      updateX=false;
      updateR=false;
      // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
    }

    if ( !(updateR || updateX )) {

      beta = gamma * r2_old.inverse() * sigma;
      #ifdef MWVERBOSE
      std::cout << "beta\n" << beta << std::endl;
      #endif
      if (param.pipeline && !breakdown)
      errorQuda("pipeline not implemented");

      else{
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j<param.num_src; j++){
            blas::caxpy(alpha(j,i),p.Component(j),xSloppy.Component(i));
          }
        }

        // set to zero
        for(int i=0; i < param.num_src; i++){
          blas::ax(0,pnew.Component(i)); // do we need components here?
        }
        // add r
        for(int i=0; i<param.num_src; i++){
          // for(int j=0;j<param.num_src; j++){
          // order of updating p might be relevant here
          blas::axpy(1.0,r.Component(i),pnew.Component(i));
          // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
          // }
        }
        // beta = beta * gamma.inverse();
        for(int i=0; i<param.num_src; i++){
          for(int j=0;j<param.num_src; j++){
            double rcoeff= (j==0?1.0:0.0);
            // order of updating p might be relevant hereq
            blas::caxpy(beta(j,i),p.Component(j),pnew.Component(i));
            // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
          }
        }
        // now need to do something with the p's

        for(int i=0; i< param.num_src; i++){
          blas::copy(p.Component(i), pnew.Component(i));
        }


        #ifdef BLOCKCG_GS
        for(int i=0; i < param.num_src; i++){
          double n = blas::norm2(p.Component(i));
          blas::ax(1/sqrt(n),p.Component(i));
          for(int j=i+1; j < param.num_src; j++) {
            auto ri = blas::cDotProduct(p.Component(i), p.Component(j));
            blas::caxpy(-ri,p.Component(i),p.Component(j));

          }
        }


        gamma = MatrixXcd::Zero(param.num_src,param.num_src);
        for ( int i = 0; i < param.num_src; i++){
          for (int j=i; j < param.num_src; j++){
            gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
          }
        }
        #endif

        #ifdef MWVERBOSE
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j<param.num_src; j++){
            pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
          }
        }
        std::cout << " pTp " << std::endl << pTp << std::endl;
        std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
        #endif
      }


      if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
        if (&x != &xSloppy) {
          blas::copy(tmp, y);   //  FIXME: check whether copy works here
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy.Component(i), tmp.Component(i), rSloppy.Component(i)).z);
          }
        } else {
          blas::copy(r, rSloppy);  //  FIXME: check whether copy works here
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(x.Component(i), y.Component(i), r.Component(i)).z);
          }
        }
      }

      steps_since_reliable++;
    } else {
      printfQuda("reliable update\n");
      for(int i=0; i<param.num_src; i++){
        blas::axpy(alpha(i,i).real(), p.Component(i), xSloppy.Component(i));
      }
      blas::copy(x, xSloppy); // nop when these pointers alias

      for(int i=0; i<param.num_src; i++){
        blas::xpy(x.Component(i), y.Component(i)); // swap these around?
      }
      for(int i=0; i<param.num_src; i++){
        mat(r.Component(i), y.Component(i)); //  here we can use x as tmp
      }
      for(int i=0; i<param.num_src; i++){
        r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
      }

      for(int i=0; i<param.num_src; i++){
        blas::copy(rSloppy.Component(i), r.Component(i)); //nop when these pointers alias
        blas::zero(xSloppy.Component(i));
      }

      // calculate new reliable HQ resididual
      if (use_heavy_quark_res){
        for(int i=0; i<param.num_src; i++){
          heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(y.Component(i), r.Component(i)).z);
        }
      }

      // MW: FIXME as this probably goes terribly wrong right now
      for(int i = 0; i<param.num_src; i++){
        // break-out check if we have reached the limit of the precision
        if (sqrt(r2(i,i).real()) > r0Norm[i] && updateX) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2(i,i).real()), r0Norm[i], resIncreaseTotal);
          if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("CG: solver exiting due to too many true residual norm increases");
              break;
            }
          }
        } else {
          resIncrease = 0;
        }
      }
      // if L2 broke down already we turn off reliable updates and restart the CG
      for(int i = 0; i<param.num_src; i++){
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res[i] > heavy_quark_res_old[i]) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res[i], heavy_quark_res_old[i]);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }
      }

      for(int i=0; i<param.num_src; i++){
        rNorm[i] = sqrt(r2(i,i).real());
        maxrr[i] = rNorm[i];
        maxrx[i] = rNorm[i];
        r0Norm[i] = rNorm[i];
        heavy_quark_res_old[i] = heavy_quark_res[i];
      }
      rUpdate++;

      if (use_heavy_quark_res and heavy_quark_restart) {
        // perform a restart
        blas::copy(p, rSloppy);
        heavy_quark_restart = false;
      } else {
        // explicitly restore the orthogonality of the gradient vector
        for(int i=0; i<param.num_src; i++){
          double rp = blas::reDotProduct(rSloppy.Component(i), p.Component(i)) / (r2(i,i).real());
          blas::axpy(-rp, rSloppy.Component(i), p.Component(i));

          beta(i,i) = r2(i,i) / r2_old(i,i);
          blas::xpay(rSloppy.Component(i), beta(i,i).real(), p.Component(i));
        }
      }

      steps_since_reliable = 0;
    }

    breakdown = false;
    k++;

    allconverged = true;
    r2avg=0;
    for(int i=0; i<param.num_src; i++){
      r2avg+= r2(i,i).real();
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }
    PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);

    // check for recent enough reliable updates of the HQ residual if we use it
    if (use_heavy_quark_res) {
      for(int i=0; i<param.num_src; i++){
        // L2 is concverged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (ru.steps_since_reliable == 0 and param.delta > 0)
          and convergenceHQ(r2(i, i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        converged[i] = L2done and HQdone;
      }
    }

  }

  blas::copy(x, xSloppy);
  for(int i=0; i<param.num_src; i++){
    blas::xpy(y.Component(i), x.Component(i));
  }

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

  param.iter += k;

  if (k == param.maxiter)
  warningQuda("Exceeded maximum iterations %d", param.maxiter);

  if (getVerbosity() >= QUDA_VERBOSE)
  printfQuda("CG: Reliable updates = %d\n", rUpdate);

  // compute the true residuals
  for(int i=0; i<param.num_src; i++){
    mat(r.Component(i), x.Component(i));
    param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
    param.true_res_offset[i] = param.true_res;
    param.true_res_hq_offset[i] = param.true_res_hq;

    PrintSummary("CG", k, r2(i, i).real(), b2[i], stop[i]);
  }

  getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  getProfile().TPSTART(QUDA_PROFILE_FREE);

  getProfile().TPSTOP(QUDA_PROFILE_FREE);

  return;

  #endif

}
#endif


}  // namespace quda
