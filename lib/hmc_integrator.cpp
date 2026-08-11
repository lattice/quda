/**
 * @file hmc_integrator.cpp
 * @brief Implementations of quda::Integrator and its symplectic derived
 *        classes (leapfrog, Omelyan, force-gradient).
 *
 * The nested-FGI integrator lives in nested_fgi_integrator.cpp; it shares
 * the base class but its two-timescale structure doesn't map onto the base
 * kick/drift/fgStep primitives.
 */

#include <hmc_integrator.h>
#include <hmc_quda.h>
#include <eigen_tracking_state.h>
#include <inv_tracker.h>

#include <blas_quda.h>
#include <clover_field.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <gauge_field.h>
#include <gauge_path_quda.h>
#include <gauge_tools.h>
#include <gauge_update_quda.h>
#include <invert_quda.h>
#include <malloc_quda.h>
#include <momentum.h>
#include <quda.h>
#include <quda_internal.h>
#include <timer.h>
#include <util_quda.h>

namespace quda
{

  // --------------------------------------------------------------------
  // Externally-provided state.
  //
  // The gauge-hierarchy and momentum live as file-scope statics in
  // interface_quda.cpp; we reach them via the same extern pattern already
  // used by hmc.cpp and eigen_tracking_state.cpp. updateExtendedGaugeResident
  // and loadCloverQuda are library-internal entry points declared in
  // interface_quda.cpp (extern linkage by default).
  // --------------------------------------------------------------------

} // namespace quda

extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSloppy;
extern quda::GaugeField *gaugePrecondition;
extern quda::GaugeField *gaugeRefinement;
extern quda::GaugeField *gaugeEigensolver;
extern quda::GaugeField *gaugeExtended;
extern quda::GaugeField *extendedGaugeResident;
extern quda::GaugeField momResident;
extern quda::CloverField *cloverPrecise;
extern quda::CloverField *cloverSloppy;
extern quda::CloverField *cloverPrecondition;
extern quda::CloverField *cloverRefinement;
extern quda::CloverField *cloverEigensolver;

void updateExtendedGaugeResident(bool new_gauge, const quda::lat_dim_t &R, quda::TimeProfile &profile,
                                 bool redundant_comms = false, QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID);

namespace quda
{

  // --------------------------------------------------------------------
  // Wilson plaquette gauge-force path tables.
  //
  // Allocated lazily on first kick, torn down via releaseIntegratorState()
  // (called from interface_quda.cpp's destroyHMCQuda / endQuda). Kept here
  // because they're only read by Integrator::kick; co-locating them avoids
  // extern plumbing.
  // --------------------------------------------------------------------

  static int **g_gauge_paths[4] = {nullptr, nullptr, nullptr, nullptr};
  static int g_gauge_path_length[6];
  static double g_gauge_path_coeff[6];
  static bool g_gauge_paths_initialized = false;

  /**
   * @brief Set up Wilson plaquette gauge force paths.
   *
   * Creates 6 staple paths per direction (length 3 each) following
   * the MILC createGaugeForcePaths pattern.
   * Direction encoding: XUP=0, YUP=1, ZUP=2, TUP=3, TDOWN=4..XDOWN=7.
   */
  static void setupWilsonGaugePaths()
  {
    if (g_gauge_paths_initialized) return;

    for (int i = 0; i < 6; i++) {
      g_gauge_path_length[i] = 3;
      g_gauge_path_coeff[i] = 1.0;
    }

    for (int dir = 0; dir < 4; dir++) {
      g_gauge_paths[dir] = static_cast<int **>(safe_malloc(6 * sizeof(int *)));
      int idx = 0;
      for (int i = 0; i < 4; i++) {
        if (i == dir) continue;
        int opp_dir = 7 - dir;
        int opp_i = 7 - i;
        g_gauge_paths[dir][idx] = static_cast<int *>(safe_malloc(3 * sizeof(int)));
        g_gauge_paths[dir][idx][0] = i;
        g_gauge_paths[dir][idx][1] = opp_dir;
        g_gauge_paths[dir][idx][2] = opp_i;
        idx++;
        g_gauge_paths[dir][idx] = static_cast<int *>(safe_malloc(3 * sizeof(int)));
        g_gauge_paths[dir][idx][0] = opp_i;
        g_gauge_paths[dir][idx][1] = opp_dir;
        g_gauge_paths[dir][idx][2] = i;
        idx++;
      }
    }
    g_gauge_paths_initialized = true;
  }

  /**
   * @brief Release the Wilson plaquette path tables.
   *
   * Called from destroyHMCQuda()/endQuda() in interface_quda.cpp so the
   * host allocations come down before QUDA's pool is flushed.
   */
  void releaseIntegratorState()
  {
    if (!g_gauge_paths_initialized) return;
    for (int dir = 0; dir < 4; dir++) {
      if (g_gauge_paths[dir]) {
        for (int i = 0; i < 6; i++) host_free(g_gauge_paths[dir][i]);
        host_free(g_gauge_paths[dir]);
        g_gauge_paths[dir] = nullptr;
      }
    }
    g_gauge_paths_initialized = false;
  }

  // --------------------------------------------------------------------
  // Per-trajectory CG counters. Read/cleared by hmcTrajectoryQuda via
  // integratorResetCGStats / integratorGetCGStats accessors.
  // --------------------------------------------------------------------

  static int g_traj_cg_iters = 0;
  static int g_traj_cg_solves = 0;

  void integratorResetCGStats()
  {
    g_traj_cg_iters = 0;
    g_traj_cg_solves = 0;
  }

  int integratorCGIters() { return g_traj_cg_iters; }
  int integratorCGSolves() { return g_traj_cg_solves; }

  void integratorBumpCGStats(int iters)
  {
    g_traj_cg_iters += iters;
    g_traj_cg_solves++;
  }

  // --------------------------------------------------------------------
  // Cached Integrator instance shared across trajectories. Built lazily
  // by getOrCreateIntegrator; rebuilt only when hmc_param.integrator
  // changes so nested-FGI's CoarseDeflationManager state survives.
  // --------------------------------------------------------------------

  namespace
  {
    std::unique_ptr<Integrator> g_currentIntegrator;
    QudaIntegratorType g_currentIntegratorType = QUDA_INVALID_INTEGRATOR;
    // Cache identity beyond integrator type: separate hmcRunQuda invocations
    // may pass distinct param/preconditioner pointers (e.g. one test enables
    // MG, another doesn't). Rebuilding on those changes prevents the cached
    // integrator from operating with a stale mg_instance or stale settings.
    const void *g_currentHmcParamAddr = nullptr;
    const void *g_currentInvParamAddr = nullptr;
    const void *g_currentMGInstance = nullptr;
  } // namespace

  Integrator &getOrCreateIntegrator(QudaHMCParam &hmc_param, QudaGaugeParam &gauge_param, QudaInvertParam &inv_param,
                                    EigenTrackingState *tracking, void *mg_instance)
  {
    bool needs_rebuild = !g_currentIntegrator || g_currentIntegratorType != hmc_param.integrator
      || g_currentHmcParamAddr != static_cast<const void *>(&hmc_param)
      || g_currentInvParamAddr != static_cast<const void *>(&inv_param) || g_currentMGInstance != mg_instance;

    if (needs_rebuild) {
      g_currentIntegrator.reset(Integrator::create(hmc_param, gauge_param, inv_param, tracking, mg_instance));
      g_currentIntegratorType = hmc_param.integrator;
      g_currentHmcParamAddr = &hmc_param;
      g_currentInvParamAddr = &inv_param;
      g_currentMGInstance = mg_instance;
    }
    return *g_currentIntegrator;
  }

  Integrator *currentIntegrator() { return g_currentIntegrator.get(); }

  void releaseIntegrator()
  {
    g_currentIntegrator.reset();
    g_currentIntegratorType = QUDA_INVALID_INTEGRATOR;
    g_currentHmcParamAddr = nullptr;
    g_currentInvParamAddr = nullptr;
    g_currentMGInstance = nullptr;
  }

  // --------------------------------------------------------------------
  // Local helpers.
  // --------------------------------------------------------------------

  namespace
  {
    /**
     * @brief Canonical R shell for HMC extended-gauge operations.
     *
     * Matches the pattern used by computeEOFermionForce and the clover
     * force paths: X direction gets a 2-wide shell, others 1-wide.
     */
    lat_dim_t hmcExtendedGaugeShell()
    {
      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
      return R;
    }

    /**
     * @brief Recompute the resident clover (+ inverse metadata + TrLog)
     *        from the current extendedGaugeResident, IN PLACE.
     *
     * In place is mandatory: reallocating (freeCloverQuda + loadCloverQuda)
     * dangles the clover pointers an MG hierarchy captured at setup.
     *
     * The compute must run at double and demote into the resident field —
     * mirroring loadCloverQuda, which creates at clover_cpu_prec (double),
     * computes, then demotes. Direct single-precision computeClover +
     * cloverInvert is a path loadCloverQuda never exercises and yields a
     * corrupt clover (NaN / systematically wrong TrLog, "diagonal appears
     * unset" reads downstream).
     */
    void refreshSloppyCloverFamily();

    void recomputeResidentClover(QudaInvertParam &inv_param)
    {
      if (!cloverPrecise) errorQuda("No resident clover field to recompute");

      if (cloverPrecise->Precision() >= QUDA_DOUBLE_PRECISION) {
        // createCloverQuda computes at the field precision (double): safe.
        createCloverQuda(&inv_param);
        refreshSloppyCloverFamily();
        return;
      }

      if (!extendedGaugeResident) errorQuda("No extended gauge field for clover recompute");
      GaugeField *gauge = extendedGaugeResident;

      // Promote the extended gauge to double for the compute
      GaugeField *ex = gauge;
      if (gauge->Precision() < QUDA_DOUBLE_PRECISION) {
        GaugeFieldParam param(*gauge);
        param.setPrecision(QUDA_DOUBLE_PRECISION, true);
        param.create = QUDA_NULL_FIELD_CREATE;
        ex = GaugeField::Create(param);
        ex->copy(*gauge);
      }

      GaugeFieldParam tensorParam(gaugePrecise->X(), QUDA_DOUBLE_PRECISION, QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
      tensorParam.location = QUDA_CUDA_FIELD_LOCATION;
      tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      tensorParam.setPrecision(tensorParam.Precision(), true);
      tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      GaugeField Fmunu(tensorParam);
      computeFmunu(Fmunu, *ex);

      CloverFieldParam cParam(*cloverPrecise);
      cParam.create = QUDA_NULL_FIELD_CREATE;
      cParam.setPrecision(QUDA_DOUBLE_PRECISION, true);
      CloverField tmp(cParam);
      computeClover(tmp, Fmunu, inv_param.clover_coeff);
      cloverInvert(tmp, tmp.Reconstruct());

      // Demote into the resident field in place (copies TrLog + metadata)
      cloverPrecise->copy(tmp);
      refreshSloppyCloverFamily();

      if (ex != gauge) delete ex;
    }

    /**
     * @brief In-place refresh of the sloppy clover family after a
     *        cloverPrecise recompute.
     *
     * When the sloppy/precondition/refinement/eigensolver clovers are
     * distinct objects (mixed precision), they hold now-stale copies. They
     * must be refreshed IN PLACE — reallocation (freeSloppyCloverQuda +
     * loadSloppyCloverQuda) would dangle the pointers an MG hierarchy
     * captured at setup. CloverField::copy performs the precision-demoting
     * conversion and refreshes the metadata (Diagonal, max) the compressed
     * accessors require.
     */
    void refreshSloppyCloverFamily()
    {
      if (cloverSloppy && cloverSloppy != cloverPrecise) cloverSloppy->copy(*cloverPrecise);
      if (cloverPrecondition && cloverPrecondition != cloverPrecise && cloverPrecondition != cloverSloppy)
        cloverPrecondition->copy(*cloverPrecise);
      if (cloverRefinement && cloverRefinement != cloverPrecise && cloverRefinement != cloverSloppy
          && cloverRefinement != cloverPrecondition)
        cloverRefinement->copy(*cloverPrecise);
      if (cloverEigensolver && cloverEigensolver != cloverPrecise && cloverEigensolver != cloverSloppy
          && cloverEigensolver != cloverPrecondition && cloverEigensolver != cloverRefinement)
        cloverEigensolver->copy(*cloverPrecise);
    }

    /**
     * @brief 2nd-order leapfrog schedule: P(h/2) [ Q(h) P(h) ]^{n-1} Q(h) P(h/2).
     *
     * Pure schedule — no force machinery. Parameterised on the kick/drift
     * callables so the *same* schedule drives two semantically distinct
     * integrators:
     *
     *   • LeapfrogIntegrator::operator() supplies the base-class
     *     kick()/drift() — full fermion CG + Wilson gauge force, plus
     *     extended-gauge rebuild on each drift.
     *   • NestedFGIIntegrator::innerLeapfrog supplies computeInnerForce()
     *     and gaugeStep() — *only* the cheap low-mode-projected force, no
     *     fine-grid CG, and a stripped-down drift that skips the
     *     extended-gauge rebuild (the inner force operates through the MG
     *     transfer, which doesn't need the fine-grid halo).
     *
     * If a reviewer is auditing the inner timescale for "are we doing CG
     * here?": no — the inner kick callable is computeInnerForce, which
     * calls LowModeForce::computeForce. The outer timescale is where the
     * full CG (and Wilson gauge force) lives.
     */
    template <typename Kick, typename Drift>
    void runLeapfrogSchedule(double tau, int nSteps, Kick &&kick, Drift &&drift, ColorSpinorField &phi)
    {
      const double h = tau / nSteps;
      kick(h / 2.0, phi);
      for (int s = 0; s < nSteps; s++) {
        drift(h);
        kick((s < nSteps - 1) ? h : h / 2.0, phi);
      }
    }

    /**
     * @brief 2nd-order Omelyan schedule:
     *        P(λh) [ Q(h/2) P((1-2λ)h) Q(h/2) P(2λh) ]^{n-1}
     *        Q(h/2) P((1-2λ)h) Q(h/2) P(λh).
     *
     * See runLeapfrogSchedule above for the kick/drift parametrisation
     * convention; the same caveats apply here. The outer
     * OmelyanIntegrator and the nested-FGI innerOmelyan share this body
     * but with different kick/drift implementations.
     */
    template <typename Kick, typename Drift>
    void runOmelyanSchedule(double tau, int nSteps, double lambda, Kick &&kick, Drift &&drift, ColorSpinorField &phi)
    {
      const double h = tau / nSteps;
      kick(lambda * h, phi);
      for (int s = 0; s < nSteps; s++) {
        drift(h / 2.0);
        kick((1.0 - 2.0 * lambda) * h, phi);
        drift(h / 2.0);
        kick((s < nSteps - 1) ? 2.0 * lambda * h : lambda * h, phi);
      }
    }
  } // namespace

  void hmcRefreshResidentGaugeState(QudaInvertParam &inv_param)
  {
    // Full state-refresh discipline after ANY device-side write to
    // gaugePrecise. Each element guards against a distinct staleness bug
    // observed in this codebase (see the ghost-pad / clover findings):
    // 1. ghost pads: copy() fills only the local volume, the Dirac
    //    operators read the pads;
    gaugePrecise->exchangeGhost();
    // 2. sloppy/precondition/refinement/eigensolver gauge copies (aliased at
    //    uniform precision). Each distinct copy must ALSO re-exchange its own
    //    ghost zone: the dslash reads gauge ghost pads even on a single GPU
    //    (no_comms_fill), and copy() does not reliably refresh the
    //    destination's ghost region on the in-place CUDA->CUDA path. Stale
    //    sloppy ghosts leave the mixed-precision inner operator inconsistent
    //    with the bulk (degraded reliable updates) and feed the MG
    //    setup-refresh a boundary-corrupted operator whose null-vector
    //    polish collapses to noise (coarse-op verification failure at
    //    L2 ~ 0.4 on evolved gauge; root-caused 2026-08-11).
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) {
      gaugeSloppy->copy(*gaugePrecise);
      gaugeSloppy->exchangeGhost();
    }
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy) {
      gaugePrecondition->copy(*gaugePrecise);
      gaugePrecondition->exchangeGhost();
    }
    if (gaugeRefinement && gaugeRefinement != gaugePrecise && gaugeRefinement != gaugeSloppy
        && gaugeRefinement != gaugePrecondition) {
      gaugeRefinement->copy(*gaugePrecise);
      gaugeRefinement->exchangeGhost();
    }
    if (gaugeEigensolver && gaugeEigensolver != gaugePrecise && gaugeEigensolver != gaugeSloppy
        && gaugeEigensolver != gaugePrecondition && gaugeEigensolver != gaugeRefinement) {
      gaugeEigensolver->copy(*gaugePrecise);
      gaugeEigensolver->exchangeGhost();
    }
    // 3. the cached extended gauge (createCloverQuda and the gauge force
    //    read it; new_gauge = true forces the rebuild);
    lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
    updateExtendedGaugeResident(true, R, getProfile());
    // 4. clover-type actions: in-place recompute at double + demote, plus
    //    the in-place sloppy-clover family refresh (mixed precision).
    if (cloverPrecise
        && (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH || inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
      recomputeResidentClover(inv_param);
    }
  }

  // --------------------------------------------------------------------
  // Primitive operations.
  // --------------------------------------------------------------------

  void Integrator::kick(double dt, ColorSpinorField &phi)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    setupWilsonGaugePaths();

    // --- Gauge force (internal API) ---
    // gaugeForce accumulates: mom -= epsilon * gauge_path_force (negative sign for HMC)
    lat_dim_t R = hmcExtendedGaugeShell();
    updateExtendedGaugeResident(false, R, getProfile());
    double eb3 = dt * hmc_param.beta / 3.0;

    std::vector<int> path_length_v(6);
    std::vector<double> path_coeff_v(6);
    for (int i = 0; i < 6; i++) {
      path_length_v[i] = g_gauge_path_length[i];
      path_coeff_v[i] = g_gauge_path_coeff[i];
    }
    std::vector<int **> input_path_v(4);
    for (int d = 0; d < 4; d++) input_path_v[d] = g_gauge_paths[d];

    gaugeForce(momResident, *extendedGaugeResident, eb3, input_path_v, path_length_v, path_coeff_v, 6, 4);

    // --- Fermion force (internal API) ---
    // 1. CG solve: x = (D†D)^{-1} phi using internal solve()
    // With MG: keep NORMOP_PC_SOLVE so GCR+MG preconditions M†M in one solve.
    // DIRECT_PC_SOLVE would split into M† and M solves, where M† is poorly preconditioned.
    QudaInvertParam ip = inv_param;

    // Create Dirac operators for the solve
    bool pc_solve = (ip.solve_type == QUDA_DIRECT_PC_SOLVE) || (ip.solve_type == QUDA_NORMOP_PC_SOLVE);
    Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
    createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, pc_solve, false);

    // Prepare x (solution) and b (source) on device in native order
    ColorSpinorParam csParam(phi);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    std::vector<ColorSpinorField> x(1, csParam);
    std::vector<ColorSpinorField> b(1, ColorSpinorField(phi)); // copy of phi

    // Enable QUDA's native chronological forecasting when eigentracking is active.
    if (tracking && tracking->isActive()) {
      ip.chrono_make_resident = 1;
      ip.chrono_use_resident = 1;
      ip.chrono_max_dim = 3;
      ip.chrono_index = 0;
      ip.chrono_replace_last = 0;
    }

    // Solve x = (M†M)⁻¹ b
    // Twisted actions: γ₅-hermiticity is M̂(μ)† = γ₅M̂(-μ)γ₅, so pass 1 of the
    // two-pass below must solve M(-μ) (see computeFermionAction for the
    // derivation; μ-symmetric actions alias the flipped set to the direct one).
    const bool kick_twisted
      = (ip.dslash_type == QUDA_TWISTED_MASS_DSLASH || ip.dslash_type == QUDA_TWISTED_CLOVER_DSLASH);
    Dirac *dirac_m = nullptr, *diracSloppy_m = nullptr, *diracPre_m = nullptr, *diracEig_m = nullptr;
    if (kick_twisted && ip.preconditioner && ip.solve_type == QUDA_DIRECT_PC_SOLVE) {
      QudaInvertParam ip_minus = ip;
      ip_minus.mu = -ip.mu;
      createDiracWithEig(dirac_m, diracSloppy_m, diracPre_m, diracEig_m, ip_minus, pc_solve, false);
    }

    if (ip.preconditioner && ip.solve_type == QUDA_DIRECT_PC_SOLVE) {
      // MG two-pass γ₅ trick: M† = γ₅ M γ₅, so (M†M)⁻¹ = M⁻¹ (γ₅ M γ₅)⁻¹ = M⁻¹ γ₅ M⁻¹ γ₅
      // Step 1: z = γ₅ b
      // Step 2: Solve M w = z   (~12 iters with MG)  → w = M⁻¹ γ₅ b
      // Step 3: y = γ₅ w        → y = γ₅ M⁻¹ γ₅ b = (M†)⁻¹ b
      // Step 4: Solve M x = y   (~12 iters with MG)  → x = M⁻¹ (M†)⁻¹ b = (M†M)⁻¹ b
      DiracM m(*dirac), mSloppy(*diracSloppy), mPre(*diracPre), mEig(*diracEig);
      SolverParam solverParam(ip);

      // Step 1: z = γ₅ b
      std::vector<ColorSpinorField> z(1, csParam);
      gamma5(z, b);

      // GCR-Krylov-residual capture for the eigentracker (gcr_tracker.h).
      // The two M solves below go through GCR-with-MG. The tracker stores
      // up to N normalised intermediate residuals — these span the
      // slow-converging modes the preconditioner could not flatten and
      // complement the converged-solution stash below. Cap pulled from
      // EigenTrackingParam::residualCap (CLI flag
      // --eigentracking-residual-cap, default 0=off, opt-in for
      // light-mass regimes).
      const int gcrResCap = (tracking && tracking->isActive()) ? tracking->getParam().residualCap : 0;
      // Promote captured residuals to inv_param.cuda_prec so the
      // EigenTracker pool (double-precision) absorption does not fault
      // inside multiCdot when mixing precisions.
      GCRTracker gcrTracker(gcrResCap, ip.cuda_prec);
      std::vector<ColorSpinorField> krylovVecs;

      // Step 2: Solve M(-μ) w = z (equals M for μ-symmetric actions)
      DiracM mMinus(dirac_m ? *dirac_m : *dirac), mMinusSloppy(diracSloppy_m ? *diracSloppy_m : *diracSloppy),
        mMinusPre(diracPre_m ? *diracPre_m : *diracPre), mMinusEig(diracEig_m ? *diracEig_m : *diracEig);
      std::vector<ColorSpinorField> w(1, csParam);
      {
        TrackerScope<GCRTracker> scope(activeGCRTracker, gcrTracker.isActive() ? &gcrTracker : nullptr);
        Solver *s = Solver::create(solverParam, mMinus, mMinusSloppy, mMinusPre, mMinusEig);
        (*s)(w, z);
        delete s;
        g_traj_cg_iters += solverParam.iter;
        g_traj_cg_solves++;
        logQuda(QUDA_VERBOSE, "γ₅ two-pass: M solve (pass 1) = %d iters\n", solverParam.iter);
        solverParam.iter = 0;
      }
      for (auto &q : takeRitzVectors(gcrTracker)) krylovVecs.push_back(std::move(q));

      // Step 3: y = γ₅ w
      std::vector<ColorSpinorField> y(1, csParam);
      gamma5(y, w);

      // Step 4: Solve M x = y
      {
        TrackerScope<GCRTracker> scope(activeGCRTracker, gcrTracker.isActive() ? &gcrTracker : nullptr);
        Solver *s = Solver::create(solverParam, m, mSloppy, mPre, mEig);
        (*s)(x, y);
        delete s;
        g_traj_cg_iters += solverParam.iter;
        g_traj_cg_solves++;
        logQuda(QUDA_VERBOSE, "γ₅ two-pass: M solve (pass 2) = %d iters\n", solverParam.iter);
      }
      for (auto &q : takeRitzVectors(gcrTracker)) krylovVecs.push_back(std::move(q));

      // Stash converged-solution vectors plus GCR-residual vectors for
      // eigentracking absorption. w = M⁻¹ γ₅ b and x = (M†M)⁻¹ b are
      // rich in low-mode content (each Krylov solve concentrates
      // amplitude on small-eigenvalue modes); the GCR residuals from the
      // tail of the iteration carry the modes the preconditioner could
      // not kill in the available iterations.
      if (tracking && tracking->isActive()) {
        std::vector<ColorSpinorField> solVecs;
        ColorSpinorField xNorm(x[0]);
        double xnrm = sqrt(blas::norm2(xNorm));
        if (xnrm > 1e-30) {
          blas::ax(1.0 / xnrm, xNorm);
          solVecs.push_back(std::move(xNorm));
        }
        ColorSpinorField wNorm(w[0]);
        double wnrm = sqrt(blas::norm2(wNorm));
        if (wnrm > 1e-30) {
          blas::ax(1.0 / wnrm, wNorm);
          solVecs.push_back(std::move(wNorm));
        }
        for (auto &q : krylovVecs) solVecs.push_back(std::move(q));
        if (!solVecs.empty()) tracking->stashRitzVectors(std::move(solVecs));
      }

      ip.iter = g_traj_cg_iters; // for logging
    } else {
      // Standard solve path (CG on M†M or direct without MG). When the
      // user has opted in via residualCap > 0, install a CGTracker for
      // the duration of this solve so the inv_cg_quda.cpp loop captures
      // alpha/beta/residuals for zero-cost Lanczos-tridiag Ritz pair
      // extraction. The extracted vectors feed the same stash path as
      // the MG branch above.
      const int cgRitzCap = (tracking && tracking->isActive()) ? tracking->getParam().residualCap : 0;
      CGTracker cgTracker(cgRitzCap);

      {
        TrackerScope<CGTracker> scope(activeCGTracker, cgTracker.isActive() ? &cgTracker : nullptr);
        solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);
        g_traj_cg_iters += ip.iter;
        g_traj_cg_solves++;
      }

      if (tracking && tracking->isActive()) {
        auto vecs = takeRitzVectors(cgTracker);
        if (!vecs.empty()) tracking->stashRitzVectors(std::move(vecs));
      }
    }

    // 2. Accumulate EO fermion force into momentum
    computeEOFermionForce(momResident, x[0], ip, dt);

    delete dirac;
    delete diracSloppy;
    if (diracPre != diracSloppy) delete diracPre;
    if (diracEig != diracPre) delete diracEig;
    if (dirac_m) {
      delete dirac_m;
      if (diracSloppy_m && diracSloppy_m != dirac_m) delete diracSloppy_m;
      if (diracPre_m && diracPre_m != dirac_m && diracPre_m != diracSloppy_m) delete diracPre_m;
      if (diracEig_m && diracEig_m != dirac_m && diracEig_m != diracPre_m) delete diracEig_m;
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  void Integrator::drift(double dt)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    logQuda(QUDA_VERBOSE, "Integrator::drift: dt=%e\n", dt);

    // Gauge update: U_out = exp(dt * P) * U_in
    // Must use separate in/out fields — in-place update is a race condition.
    GaugeFieldParam gParam(*gaugePrecise);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u_out(gParam);
    updateGaugeField(u_out, dt, *gaugePrecise, momResident, false, true);

    gaugePrecise->copy(u_out);
    hmcRefreshResidentGaugeState(inv_param);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  void Integrator::fgStep(double h, ColorSpinorField &phi)
  {
    const double lam = hmc_param.fgi_lambda;
    const double one_m_2lam = 1.0 - 2.0 * lam;
    const double xi_h3 = hmc_param.fgi_xi * h * h * h;
    // Hessian-free force-gradient displacement (Schäfers et al.,
    // arXiv:2501.17632, Eq. 8): U' = exp(-2 c_j h^2 / b_j × F^j_Schäfers × T) U,
    // where Schäfers' F^j = +∂S/∂U (gradient of action), NOT the HMC force.
    // In our convention `kick(α, phi)` produces mom = α × F_HMC = -α × ∂S/∂U,
    // and updateGaugeField does U' = exp(+1.0 × mom) U. Matching the two:
    //     U' = exp(-α ∂S/∂U) U  ⇔  Schäfers exp(-2 c h²/b × ∂S/∂U) U
    //   ⇒ α = +2 c h² / b = +2 ξ h² / (1-2λ).
    // The previous +ξh²/(1-2λ) was missing the factor of 2 — that's why the
    // FG correction cancelled only ~94% of the dt² leading error and the
    // integrator stayed at 2nd order (visible as p≈2 in HMC.dHScaling).
    const double fgCoeff = 2.0 * xi_h3 / (one_m_2lam * h);

    // Save gauge + momentum
    GaugeField gaugeSaved(*gaugePrecise);
    GaugeField momSaved(momResident);

    // Zero momentum, compute force, kick with FG coefficient
    momResident.zero();
    kick(fgCoeff, phi);

    // Displace gauge: U' = exp(1.0 * δπ) * U
    GaugeFieldParam gfParam(*gaugePrecise);
    gfParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u_out(gfParam);
    updateGaugeField(u_out, 1.0, *gaugePrecise, momResident, false, true);
    gaugePrecise->copy(u_out);
    // Full refresh at the displaced gauge: without it the excursion's force
    // evaluation uses the pre-displacement halo/clover and the FG correction
    // silently degenerates to 2nd order (p≈2 instead of 4 in HMC.dHScaling).
    hmcRefreshResidentGaugeState(inv_param);

    // Restore momentum, compute full kick at displaced gauge
    momResident.copy(momSaved);
    kick(one_m_2lam * h, phi);

    // Restore gauge — and rebuild extended at the original gauge so the
    // outer schedule's subsequent kick sees a consistent halo.
    gaugePrecise->copy(gaugeSaved);
    hmcRefreshResidentGaugeState(inv_param);
  }

  // --------------------------------------------------------------------
  // Derived class schedules (step 3 of refactor).
  // --------------------------------------------------------------------

  void LeapfrogIntegrator::operator()(double tau, int nSteps, ColorSpinorField &phi)
  {
    logQuda(QUDA_SUMMARIZE, "LeapfrogIntegrator: tau=%e, n_steps=%d, h=%e\n", tau, nSteps, tau / nSteps);
    runLeapfrogSchedule(
      tau, nSteps, [this](double dt, ColorSpinorField &p) { kick(dt, p); }, [this](double dt) { drift(dt); }, phi);
  }

  void OmelyanIntegrator::operator()(double tau, int nSteps, ColorSpinorField &phi)
  {
    const double lam = hmc_param.omelyan_lambda;
    logQuda(QUDA_SUMMARIZE, "OmelyanIntegrator: tau=%e, n_steps=%d, h=%e, lambda=%e\n", tau, nSteps, tau / nSteps, lam);
    runOmelyanSchedule(
      tau, nSteps, lam, [this](double dt, ColorSpinorField &p) { kick(dt, p); }, [this](double dt) { drift(dt); }, phi);
  }

  void FGIntegrator::operator()(double tau, int nSteps, ColorSpinorField &phi)
  {
    const double h = tau / nSteps;
    const double lam = hmc_param.fgi_lambda;
    logQuda(QUDA_SUMMARIZE, "FGIntegrator: tau=%e, n_steps=%d, h=%e, lambda=%e, xi=%e\n", tau, nSteps, h, lam,
            hmc_param.fgi_xi);

    // Initial P(lambda*h)
    kick(lam * h, phi);

    for (int step = 0; step < nSteps; step++) {
      logQuda(QUDA_VERBOSE, "FGI step %d/%d\n", step + 1, nSteps);

      drift(h / 2.0);
      fgStep(h, phi); // force-gradient sub-step covers the (1-2λ)h centre kick
      drift(h / 2.0);

      double kickDt = (step < nSteps - 1) ? 2.0 * lam * h : lam * h;
      kick(kickDt, phi);
    }
  }

  // --------------------------------------------------------------------
  // NestedFGIIntegrator — two-timescale outer FGI + inner leapfrog/Omelyan
  // with coarse-grid deflation. Co-located here with the other derived
  // classes; depends on the gauge-hierarchy externs declared above.
  // --------------------------------------------------------------------

  // solutionResident is the file-scope vector in interface_quda.cpp used by
  // the clover host-side invertQuda + computeCloverForceQuda path.
  extern std::vector<ColorSpinorField> solutionResident;

  NestedFGIIntegrator::NestedFGIIntegrator(QudaHMCParam &hmcParam, MG &mg, const DiracMatrix &matFine, void *mgPrec,
                                           QudaGaugeParam &gaugeParam_, QudaInvertParam &invParam_,
                                           EigenTrackingState *tracking_, GaugeField *&gaugePrecise_,
                                           GaugeField &momResident_, CloverField *&cloverPrecise_) :
    Integrator(hmcParam, gaugeParam_, invParam_, tracking_),
    deflManager(*mg.getTransfer(), *mg.getMatCoarseResidual(), *mg.getDiracCoarseResidual(), hmcParam.n_defl,
                hmcParam.eig_tol, hmcParam.eig_n_kr > 0 ? hmcParam.eig_n_kr : 3 * hmcParam.n_defl,
                hmcParam.eig_max_restarts, hmcParam.defl_refresh_interval),
    lowModeForce(deflManager, matFine, hmcParam.n_mr_smooth, hmcParam.mr_omega),
    lambda(hmcParam.fgi_lambda),
    xi(hmcParam.fgi_xi),
    beta(hmcParam.beta),
    nInnerSteps(hmcParam.n_inner_steps),
    innerIntegrator(hmcParam.inner_integrator),
    innerOmelyanLambda(hmcParam.inner_omelyan_lambda),
    gaugePrecise(gaugePrecise_),
    momResident(momResident_),
    cloverPrecise(cloverPrecise_),
    mgPreconditioner(mgPrec)
  {
    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: lambda=%f, xi=%f, n_inner=%d, n_defl=%d, inner=%s\n", lambda, xi,
            nInnerSteps, hmcParam.n_defl, innerIntegrator == QUDA_OMELYAN_INTEGRATOR ? "Omelyan" : "Leapfrog");
  }

  NestedFGIIntegrator::~NestedFGIIntegrator() = default;

  void NestedFGIIntegrator::gaugeStep(double dt)
  {
    GaugeFieldParam gParam(*gaugePrecise);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u_out(gParam);
    updateGaugeField(u_out, dt, *gaugePrecise, momResident, false, true);
    gaugePrecise->copy(u_out);
    hmcRefreshResidentGaugeState(inv_param);
  }

  /**
   * @brief Expensive outer-timescale force = gauge + (full fermion − low-mode).
   *
   * This is where the fine-grid CG solve lives in the nested-FGI scheme.
   * The Schwinger_MG split keeps outer + inner = total: the inner
   * timescale contributes only F_low (no gauge force, no fine-grid CG),
   * so this routine subtracts F_low at the end to cancel the inner's
   * contribution, leaving the residual high-mode component plus the
   * Wilson plaquette gauge force.
   */
  void NestedFGIIntegrator::computeOuterForce(double coeff, const ColorSpinorField &phi)
  {
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: computeOuterForce coeff=%e\n", coeff);

    QudaGaugeParam gp = gauge_param;
    gp.use_resident_gauge = 1;
    gp.use_resident_mom = 1;
    gp.make_resident_gauge = 1;
    gp.make_resident_mom = 1;
    gp.return_result_mom = 0;
    gp.overwrite_mom = 0;

    setupWilsonGaugePaths();
    double eb3 = coeff * beta / 3.0;
    computeGaugeForceQuda(nullptr, nullptr, g_gauge_paths, g_gauge_path_length, g_gauge_path_coeff, 6, 4, eb3, &gp);

    QudaInvertParam ip = inv_param;

    // Action-generic fermion force: solve x = (M̂†M̂)⁻¹φ, then
    // computeEOFermionForce assembles the hopping (+ clover sigma/TrLog,
    // + twisted variants) force under the validated F = -2·∂S/∂u
    // convention. This replaced a per-action branch whose clover leg
    // called the tmLQCD-convention computeCloverForceQuda (which has no
    // twist term and different normalizations).
    ColorSpinorParam csParam(phi);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    std::vector<ColorSpinorField> x(1, csParam);
    std::vector<ColorSpinorField> b(1, ColorSpinorField(phi));

    bool pc_solve = (ip.solve_type == QUDA_DIRECT_PC_SOLVE) || (ip.solve_type == QUDA_NORMOP_PC_SOLVE);
    Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
    createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, pc_solve, false);

    // For twisted actions γ₅-hermiticity reads M̂(μ)† = γ₅M̂(-μ)γ₅, so the
    // two-pass below must solve M(-μ) in its FIRST pass:
    // x = M(+μ)⁻¹ γ₅ M(-μ)⁻¹ γ₅ φ = (M̂†M̂)⁻¹φ. The +μ MG hierarchy remains
    // a valid preconditioner for M(-μ) (small diagonal shift — iteration
    // counts, never the converged solution). μ-symmetric actions reduce to
    // the standard two-pass (the flipped set aliases the direct one).
    const bool twisted = (ip.dslash_type == QUDA_TWISTED_MASS_DSLASH || ip.dslash_type == QUDA_TWISTED_CLOVER_DSLASH);
    Dirac *dirac_m = nullptr, *diracSloppy_m = nullptr, *diracPre_m = nullptr, *diracEig_m = nullptr;
    if (twisted && ip.preconditioner && ip.solve_type == QUDA_DIRECT_PC_SOLVE) {
      QudaInvertParam ip_minus = ip;
      ip_minus.mu = -ip.mu;
      createDiracWithEig(dirac_m, diracSloppy_m, diracPre_m, diracEig_m, ip_minus, pc_solve, false);
    }

    if (ip.preconditioner && ip.solve_type == QUDA_DIRECT_PC_SOLVE) {
      DiracM m(*dirac), mSloppy(*diracSloppy), mPre(*diracPre), mEig(*diracEig);
      DiracM mMinus(dirac_m ? *dirac_m : *dirac), mMinusSloppy(diracSloppy_m ? *diracSloppy_m : *diracSloppy),
        mMinusPre(diracPre_m ? *diracPre_m : *diracPre), mMinusEig(diracEig_m ? *diracEig_m : *diracEig);
      SolverParam solverParam(ip);
      std::vector<ColorSpinorField> z(1, csParam);
      gamma5(z, b);
      std::vector<ColorSpinorField> w(1, csParam);
      {
        Solver *s = Solver::create(solverParam, mMinus, mMinusSloppy, mMinusPre, mMinusEig);
        (*s)(w, z);
        delete s;
        solverParam.iter = 0;
      }
      std::vector<ColorSpinorField> y(1, csParam);
      gamma5(y, w);
      {
        Solver *s = Solver::create(solverParam, m, mSloppy, mPre, mEig);
        (*s)(x, y);
        delete s;
      }
    } else {
      solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);
    }

    computeEOFermionForce(momResident, x[0], ip, coeff);

    delete dirac;
    delete diracSloppy;
    if (diracPre != diracSloppy) delete diracPre;
    if (diracEig != diracPre) delete diracEig;
    if (dirac_m) {
      delete dirac_m;
      if (diracSloppy_m && diracSloppy_m != dirac_m) delete diracSloppy_m;
      if (diracPre_m && diracPre_m != dirac_m && diracPre_m != diracSloppy_m) delete diracPre_m;
      if (diracEig_m && diracEig_m != dirac_m && diracEig_m != diracPre_m) delete diracEig_m;
    }

    // Subtract low-mode force so outer + inner = total
    lowModeForce.computeForce(momResident, phi, -coeff, *gaugePrecise, cloverPrecise, gauge_param, inv_param);
  }

  /**
   * @brief Cheap inner-timescale fermion force.
   *
   * Approximates (D†D)⁻¹φ by restricting φ to the coarse grid, projecting
   * onto the tracked coarse eigenvectors, and prolonging back. This is
   * the entire fermion-force component on the inner timescale; *no
   * fine-grid CG runs here*. The outer timescale (computeOuterForce)
   * carries the difference F_full − F_low so that outer + inner = total
   * fermion force, and the gauge force lives exclusively in
   * computeOuterForce — adding it here would double-count.
   */
  void NestedFGIIntegrator::computeInnerForce(double coeff, const ColorSpinorField &phi)
  {
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: computeInnerForce coeff=%e\n", coeff);

    lowModeForce.computeForce(momResident, phi, coeff, *gaugePrecise, cloverPrecise, gauge_param, inv_param);

    deflManager.step();
    deflManager.maybeRefresh();
  }

  // The inner sub-integrators run on the cheap force only:
  //   kick  → computeInnerForce  (LowModeForce::computeForce, NO fine-grid CG)
  //   drift → gaugeStep          (gauge update + clover rebuild, no extended-gauge)
  // They reuse the same PQ schedules as the outer integrators; only the
  // kick/drift callables differ. See the runLeapfrogSchedule /
  // runOmelyanSchedule docstrings for the full convention.
  void NestedFGIIntegrator::innerLeapfrog(double dt, ColorSpinorField &phi)
  {
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: innerLeapfrog dt=%e, dti=%e, n=%d\n", dt, dt / nInnerSteps, nInnerSteps);
    runLeapfrogSchedule(
      dt, nInnerSteps, [this](double t, ColorSpinorField &p) { computeInnerForce(t, p); },
      [this](double t) { gaugeStep(t); }, phi);
  }

  void NestedFGIIntegrator::innerOmelyan(double dt, ColorSpinorField &phi)
  {
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: innerOmelyan dt=%e, dti=%e, n=%d, lam=%f\n", dt, dt / nInnerSteps,
            nInnerSteps, innerOmelyanLambda);
    runOmelyanSchedule(
      dt, nInnerSteps, innerOmelyanLambda, [this](double t, ColorSpinorField &p) { computeInnerForce(t, p); },
      [this](double t) { gaugeStep(t); }, phi);
  }

  void NestedFGIIntegrator::forceGradientStep(double h, const ColorSpinorField &phi)
  {
    double one_m_2lam = 1.0 - 2.0 * lambda;
    double xi_h3 = xi * h * h * h;

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: forceGradientStep h=%e\n", h);

    GaugeField gaugeSaved(*gaugePrecise);
    GaugeField momSaved(momResident);

    momResident.zero();

    // Hessian-free Yin-Mawhinney displacement coefficient: 2 ξ h² / (1-2λ)
    // (Schäfers et al. arXiv:2501.17632 Eq. 8). The factor of 2 is essential
    // — without it the FG correction cancels only ~94% of the leading dt²
    // error and the integrator stays at 2nd order. See standalone fgStep
    // for the convention-mapping derivation.
    double fgCoeff = 2.0 * xi_h3 / (one_m_2lam * h);
    computeOuterForce(fgCoeff, phi);

    {
      GaugeFieldParam gfParam(*gaugePrecise);
      gfParam.create = QUDA_NULL_FIELD_CREATE;
      GaugeField u_out(gfParam);
      updateGaugeField(u_out, 1.0, *gaugePrecise, momResident, false, true);
      gaugePrecise->copy(u_out);
    }
    // Full refresh at the displaced gauge (ghost exchange + gauge hierarchy +
    // extended halo + clover recompute), as in Integrator::fgStep. The manual
    // copy/extended-rebuild sequence previously used here skipped the ghost
    // exchange, so the central kick's solves read pre-displacement pads and
    // the FG correction degenerated to 2nd order (m-independent h² term in
    // the nested (h, m) theory grid).
    hmcRefreshResidentGaugeState(inv_param);

    momResident.copy(momSaved);

    computeOuterForce(one_m_2lam * h, phi);

    gaugePrecise->copy(gaugeSaved);
    hmcRefreshResidentGaugeState(inv_param);
  }

  void NestedFGIIntegrator::operator()(double tau, int nSteps, ColorSpinorField &phi)
  {
    auto profile = pushProfile(getEigenTrackProfile());
    double h = tau / nSteps;

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: trajectory tau=%e, n_outer=%d, h=%e, inner=%s\n", tau, nSteps, h,
            innerIntegrator == QUDA_OMELYAN_INTEGRATOR ? "Omelyan" : "Leapfrog");

    deflManager.resetCounter();

    auto runInner = [&](double dt) {
      if (innerIntegrator == QUDA_OMELYAN_INTEGRATOR)
        innerOmelyan(dt, phi);
      else
        innerLeapfrog(dt, phi);
    };

    // PQPQP_FGI:
    //   P(λh) [ inner(h/2) FG inner(h/2) P(2λh) ]^{n-1} inner(h/2) FG inner(h/2) P(λh)
    computeOuterForce(lambda * h, phi);

    for (int o = 0; o < nSteps; o++) {
      logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: outer step %d/%d\n", o + 1, nSteps);

      runInner(h / 2.0);
      forceGradientStep(h, phi);
      runInner(h / 2.0);

      double kickCoeff = (o < nSteps - 1) ? 2.0 * lambda * h : lambda * h;
      computeOuterForce(kickCoeff, phi);
    }

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: trajectory complete\n");
  }

  void NestedFGIIntegrator::afterAccepted(bool full_mg_update)
  {
    if (full_mg_update) {
      // updateMultigridQuda(refresh) just destroyed and recreated the
      // fine and coarse Dirac/DiracMatrix instances behind our backs
      // (see interface_quda.cpp updateMultigridQuda non-thin branch and
      // MG::reset → createCoarseDirac). The pointers cached in
      // deflManager (matCoarse, diracCoarse) and lowModeForce (matFine)
      // are now dangling. Rebind from the live MG hierarchy before
      // touching them.
      auto *mg_solver = static_cast<multigrid_solver *>(mgPreconditioner);
      MG &mg = *mg_solver->mg;
      deflManager.rebindCoarseRefs(*mg.getTransfer(), *mg.getMatCoarseResidual(), *mg.getDiracCoarseResidual());
      lowModeForce.rebindFineMatrix(*mg_solver->m);
      deflManager.solve();
    } else {
      deflManager.rayleighRitzUpdate();
    }
  }

  // --------------------------------------------------------------------
  // Factory.
  // --------------------------------------------------------------------

  // NestedFGIIntegrator captures the gauge-hierarchy and momentum statics
  // by reference; the externs at the top of this file resolve to
  // interface_quda.cpp's file-scope globals.
  Integrator *Integrator::create(QudaHMCParam &hmc_param, QudaGaugeParam &gauge_param, QudaInvertParam &inv_param,
                                 EigenTrackingState *tracking, void *mg_instance)
  {
    switch (hmc_param.integrator) {
    case QUDA_LEAPFROG_INTEGRATOR: return new LeapfrogIntegrator(hmc_param, gauge_param, inv_param, tracking);
    case QUDA_OMELYAN_INTEGRATOR: return new OmelyanIntegrator(hmc_param, gauge_param, inv_param, tracking);
    case QUDA_FORCE_GRADIENT_INTEGRATOR: return new FGIntegrator(hmc_param, gauge_param, inv_param, tracking);
    case QUDA_NESTED_FGI_INTEGRATOR: {
      if (!mg_instance) errorQuda("QUDA_NESTED_FGI_INTEGRATOR requires a non-null mg_instance");
      auto *mg_solver = static_cast<multigrid_solver *>(mg_instance);
      MG *mg = mg_solver->mg;
      return new NestedFGIIntegrator(hmc_param, *mg, *mg_solver->m, mg_instance, gauge_param, inv_param, tracking,
                                     ::gaugePrecise, ::momResident, ::cloverPrecise);
    }
    default: errorQuda("Unknown integrator type %d", hmc_param.integrator);
    }
    return nullptr; // unreachable; errorQuda does not return
  }

} // namespace quda
