/**
 * @file hmc.cpp
 * @brief Even-odd preconditioned HMC using QUDA internal primitives.
 *
 * Implements the EO fermion action, force, and pseudofermion generation
 * for 2-flavour Wilson fermions. The algorithm follows the verified
 * Schwinger_MG reference (Schwinger_MG/src/hmc.cpp) translated to QUDA's
 * 4D Dirac operator classes and GPU kernels.
 *
 * No existing upstream functions are modified. Internal C++ functions
 * in namespace quda are called by thin C-API wrappers in interface_quda.cpp.
 */

#include <hmc_quda.h>
#include <hmc_integrator.h>
#include <eigen_tracking_state.h>
#include <inv_tracker.h>
#include <dirac_quda.h>
#include <invert_quda.h>
#include <blas_quda.h>
#include <gauge_tools.h>
#include <gauge_path_quda.h>
#include <gauge_update_quda.h>
#include <momentum.h>
#include <multigrid.h>
#include <clover_field.h>
#include <quda_internal.h>
#include <ctime>
#include <random>

// Globals from interface_quda.cpp (file scope, not in quda namespace)
extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSloppy;
extern quda::GaugeField *gaugePrecondition;
extern quda::CloverField *cloverPrecise;
extern quda::GaugeField momResident;

namespace quda
{

  // --------------------------------------------------------------------
  // Persistent HMC RNG family. See hmc_quda.h for usage notes.
  // --------------------------------------------------------------------

  namespace
  {
    std::unique_ptr<RNG> hmcSpinorRNG;
    std::unique_ptr<RNG> hmcGaugeRNG;
    std::unique_ptr<std::mt19937_64> hmcMetropolisRNG;
    size_t hmcSpinorRNGVolume = 0;
    size_t hmcGaugeRNGVolume = 0;
    unsigned long long hmcBaseSeed = 0;
  } // namespace

  void releaseHMCRNG()
  {
    hmcSpinorRNG.reset();
    hmcGaugeRNG.reset();
    hmcMetropolisRNG.reset();
    hmcSpinorRNGVolume = 0;
    hmcGaugeRNGVolume = 0;
    hmcBaseSeed = 0;
  }

  void setHMCBaseSeed(unsigned long long seed)
  {
    if (seed == hmcBaseSeed && hmcMetropolisRNG) return;
    hmcSpinorRNG.reset();
    hmcGaugeRNG.reset();
    hmcSpinorRNGVolume = 0;
    hmcGaugeRNGVolume = 0;
    hmcBaseSeed = seed;
    hmcMetropolisRNG = std::make_unique<std::mt19937_64>(seed + 3ull);
  }

  RNG &getHMCGaugeRNG(const GaugeField &meta)
  {
    if (!hmcGaugeRNG || hmcGaugeRNGVolume != meta.Volume()) {
      hmcGaugeRNG = std::make_unique<RNG>(meta, hmcBaseSeed + 2ull);
      hmcGaugeRNGVolume = meta.Volume();
    }
    return *hmcGaugeRNG;
  }

  RNG &getHMCSpinorRNG(QudaInvertParam &inv_param)
  {
    if (!hmcSpinorRNG) {
      // First call: allocate a throwaway meta spinor purely to size the
      // RNG state array; subsequent calls are O(1).
      ColorSpinorParam csParam(nullptr, inv_param, gaugePrecise->X(), true, QUDA_CUDA_FIELD_LOCATION);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(gaugePrecise->Precision());
      csParam.fieldOrder = QUDA_NATIVE_FIELD_ORDER;
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      ColorSpinorField meta(csParam);
      hmcSpinorRNG = std::make_unique<RNG>(meta, hmcBaseSeed + 1ull);
      hmcSpinorRNGVolume = meta.Volume();
    }
    return *hmcSpinorRNG;
  }

  double hmcMetropolisUniform()
  {
    if (!hmcMetropolisRNG)
      errorQuda("hmcMetropolisUniform called before setHMCBaseSeed");
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(*hmcMetropolisRNG);
  }

  // --------------------------------------------------------------------
  // EigenTrackingState singleton + stored pseudofermion. Accessors keep
  // the storage local to this TU; lifecycle is driven by the HMC entry
  // points in interface_quda.cpp.
  // --------------------------------------------------------------------

  namespace
  {
    EigenTrackingState *g_eigenTrackingInstance = nullptr;
    std::unique_ptr<ColorSpinorField> g_storedPhi;
  } // namespace

  EigenTrackingState *getEigenTrackingInstance() { return g_eigenTrackingInstance; }

  void setEigenTrackingInstance(EigenTrackingState *inst)
  {
    if (g_eigenTrackingInstance && g_eigenTrackingInstance != inst) delete g_eigenTrackingInstance;
    g_eigenTrackingInstance = inst;
  }

  void releaseEigenTrackingInstance()
  {
    delete g_eigenTrackingInstance;
    g_eigenTrackingInstance = nullptr;
  }

  ColorSpinorField *getStoredPhi() { return g_storedPhi.get(); }

  void setStoredPhi(ColorSpinorField phi) { g_storedPhi = std::make_unique<ColorSpinorField>(std::move(phi)); }

  void releaseStoredPhi() { g_storedPhi.reset(); }

  int getMGNullVectorCount(void *mg_instance)
  {
    if (!mg_instance) return 0;
    return static_cast<int>(static_cast<multigrid_solver *>(mg_instance)->B.size());
  }

  void seedEigenTrackingFromMG(void *mg_instance, QudaHMCParam *hmc_param, QudaInvertParam *inv_param)
  {
    if (!mg_instance || !hmc_param->eigentracking_enabled) return;

    // Fast no-op when the tracker is already seeded (which it is on every
    // trajectory after the first / after a periodic refresh has been
    // followed by a re-seed). Skipping early avoids extracting and copying
    // the MG null vectors unnecessarily.
    if (auto *et = getEigenTrackingInstance(); et && et->getTracker().isInitialized()) return;

    auto *mg_solver = static_cast<multigrid_solver *>(mg_instance);
    int nVec = static_cast<int>(mg_solver->B.size());
    if (nVec == 0) return;

    auto profile = pushProfile(getEigenTrackProfile());
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    logQuda(QUDA_SUMMARIZE, "seedEigenTrackingFromMG: extracting %d even-parity vectors from MG null space\n", nVec);

    // Extract even-parity components, promoting to outer solve precision
    // and converting gamma basis from DEGRAND_ROSSI (MG internal) to UKQCD
    // (Dirac operator).
    std::vector<ColorSpinorField> evenVecs;
    for (int i = 0; i < nVec; i++) {
      ColorSpinorField evenRef = mg_solver->B[i][QUDA_EVEN_PARITY];
      ColorSpinorParam param(evenRef);
      param.create = QUDA_COPY_FIELD_CREATE;
      param.field = &evenRef;
      param.setPrecision(inv_param->cuda_prec);
      param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      evenVecs.push_back(ColorSpinorField(param));
    }

    // Create-and-configure the eigentracking singleton on first call.
    if (!getEigenTrackingInstance()) {
      setEigenTrackingInstance(new EigenTrackingState());
      EigenTrackingParam etp;
      etp.enabled = true;
      etp.nEv = hmc_param->eigentracking_n_ev;
      etp.poolCapacity = hmc_param->eigentracking_pool_capacity;
      etp.nRitz = hmc_param->eigentracking_n_ritz;
      etp.forecastOrder = hmc_param->eigentracking_forecast_order;
      etp.freshTRLMInterval = hmc_param->eigentracking_fresh_trlm_interval;
      etp.solutionHistoryDepth = hmc_param->eigentracking_solution_history;
      etp.trlmTol = hmc_param->eigentracking_trlm_tol;
      etp.trlmMaxRestarts = hmc_param->eigentracking_trlm_max_restarts;
      etp.trlmCheckInterval = hmc_param->eigentracking_trlm_check_interval;
      etp.eigType = hmc_param->eigentracking_eig_type;
      etp.blockSize = hmc_param->eigentracking_blk_size;
      etp.usePolyAcc = hmc_param->eigentracking_use_poly_acc != 0;
      etp.polyDeg = hmc_param->eigentracking_poly_deg;
      etp.aMin = hmc_param->eigentracking_a_min;
      etp.aMax = hmc_param->eigentracking_a_max;
      getEigenTrackingInstance()->configure(etp);
    }

    QudaInvertParam ip = *inv_param;
    ip.solve_type = QUDA_DIRECT_PC_SOLVE;
    bool pc = true;
    Dirac *d = nullptr, *dS = nullptr, *dP = nullptr, *dE = nullptr;
    createDiracWithEig(d, dS, dP, dE, ip, pc, false);
    DiracM matHalf(*d);

    getEigenTrackingInstance()->seedFromMGNullVectors(evenVecs, matHalf, ip);

    delete d;
    if (dS && dS != d) delete dS;
    if (dP && dP != d && dP != dS) delete dP;
    if (dE && dE != d && dE != dP) delete dE;
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  /**
     @brief Create a parity spinor field matching an existing field's metadata.
     @param ref Reference field whose metadata is copied
     @param create Field creation type (default: null allocation)
  */
  static ColorSpinorField createParityField(const ColorSpinorField &ref,
                                            QudaFieldCreate create = QUDA_NULL_FIELD_CREATE)
  {
    ColorSpinorParam param(ref);
    param.create = create;
    return ColorSpinorField(param);
  }

  /**
     @brief Create a full (both-parity) spinor field from a parity field's metadata.
     @param parityRef Parity field whose metadata is used as template
  */
  static ColorSpinorField createFullField(const ColorSpinorField &parityRef)
  {
    ColorSpinorParam param(parityRef);
    param.siteSubset = QUDA_FULL_SITE_SUBSET;
    param.x[0] *= 2;
    param.create = QUDA_ZERO_FIELD_CREATE;
    return ColorSpinorField(param);
  }

  /**
     @brief Create a zero-initialized force field matching a momentum field's metadata.
     @param mom Momentum field whose metadata is used as template
  */
  static GaugeField createForceField(const GaugeField &mom)
  {
    GaugeFieldParam param(mom);
    param.link_type = QUDA_GENERAL_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    return GaugeField(param);
  }

  double computeGaugeActionHMC(double beta)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
    GaugeField *gaugeEx = createExtendedGauge(*gaugePrecise, R, getProfile());
    double3 plaq3 = plaquette(*gaugeEx);
    int V = 1;
    for (int d = 0; d < 4; d++) V *= gaugePrecise->X()[d];
    delete gaugeEx;
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return beta * (1.0 - plaq3.x) * V * 6;
  }

  namespace
  {
    /**
     * @brief Build the csParam used by both generateEOPseudofermion overloads.
     */
    ColorSpinorParam eoPseudofermionCsParam(QudaInvertParam &inv_param)
    {
      ColorSpinorParam csParam(nullptr, inv_param, gaugePrecise->X(), true, QUDA_CUDA_FIELD_LOCATION);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(gaugePrecise->Precision());
      csParam.fieldOrder = QUDA_NATIVE_FIELD_ORDER;
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      return csParam;
    }
  }

  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, unsigned long long seed)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    ColorSpinorParam csParam = eoPseudofermionCsParam(inv_param);
    ColorSpinorField eta(csParam);
    ColorSpinorField phi(csParam);

    if (seed == 0) seed = static_cast<unsigned long long>(std::time(nullptr));
    spinorNoise(eta, seed, QUDA_NOISE_GAUSS);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, true);
    Dirac *dirac = Dirac::create(diracParam);
    // Heatbath: phi = M^dag eta so that S_f = phi^dag (M^dag M)^{-1} phi
    // = |eta|^2 exactly (phi = M eta is wrong for non-normal M; verified
    // by HMC.HeatbathActionIdentity)
    dirac->Mdag(phi, eta);
    delete dirac;

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return phi;
  }

  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, RNG &rng)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    ColorSpinorParam csParam = eoPseudofermionCsParam(inv_param);
    ColorSpinorField eta(csParam);
    ColorSpinorField phi(csParam);

    spinorNoise(eta, rng, QUDA_NOISE_GAUSS);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, true);
    Dirac *dirac = Dirac::create(diracParam);
    // Heatbath: phi = M^dag eta so that S_f = phi^dag (M^dag M)^{-1} phi
    // = |eta|^2 exactly (phi = M eta is wrong for non-normal M; verified
    // by HMC.HeatbathActionIdentity)
    dirac->Mdag(phi, eta);
    delete dirac;

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return phi;
  }

  double computeEOFermionAction(ColorSpinorField &phi_even, QudaInvertParam &inv_param)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    QudaInvertParam ip = inv_param;
    ip.solve_type = QUDA_NORMOP_PC_SOLVE;
    ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

    Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
    createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, true, false);

    ColorSpinorField x_even = createParityField(phi_even, QUDA_ZERO_FIELD_CREATE);
    std::vector<ColorSpinorField> x(1, std::move(x_even));
    std::vector<ColorSpinorField> b(1, ColorSpinorField(phi_even));

    solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);

    Complex dot = blas::cDotProduct(phi_even, x[0]);

    delete dirac;
    delete diracSloppy;
    if (diracPre != diracSloppy) delete diracPre;
    if (diracEig != diracPre) delete diracEig;

    double sf = dot.real();

    // Clover EO with EVEN_EVEN(_ASYMMETRIC) matpc: the odd-site clover
    // determinant contributes S ⊃ -2 TrLog[C_oo]. Must accompany the TrLog
    // force in computeEOFermionForce so the force oracles compare against
    // the same action the force derives from.
    if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH && cloverPrecise) {
      sf -= 2.0 * cloverPrecise->TrLog()[1]; // odd parity
    } else if (inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH && cloverPrecise) {
      // Twisted TrLog stores log det(A² + μ̃²)_oo = log[det Ã(μ) det Ã(-μ)],
      // i.e. both flavor determinant factors at once — coefficient 1, not 2.
      sf -= cloverPrecise->TrLog()[1];
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    return sf;
  }

  // solutionResident is the file-scope vector in interface_quda.cpp used by
  // the host-side invertQuda + computeCloverForceQuda path.
  extern std::vector<ColorSpinorField> solutionResident;

  double computeFermionAction(ColorSpinorField &phi, QudaInvertParam &inv_param, EigenTrackingState *tracking)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    QudaInvertParam ip = inv_param;

    ColorSpinorParam csParam(phi);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField x_dev(csParam);

    if (ip.preconditioner) {
      // MG γ₅ two-pass: (M†M)⁻¹ = M⁻¹ γ₅ M⁻¹ γ₅. Both passes are fast
      // because they're MG-preconditioned solves of M (not M†M).
      ip.solve_type = QUDA_DIRECT_PC_SOLVE;
      bool pc_solve = true;
      Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
      createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, pc_solve, false);

      DiracM m(*dirac), mSloppy(*diracSloppy), mPre(*diracPre), mEig(*diracEig);
      SolverParam solverParam(ip);

      // GCR-Krylov-residual capture, mirrors NestedFGI::computeOuterForce.
      // Cap pulled from EigenTrackingParam; 0 disables capture entirely.
      // Target precision = inv_param.cuda_prec so the pool absorption
      // does not fault on mixed-precision multiCdot.
      const int gcrResCap = (tracking && tracking->isActive()) ? tracking->getParam().residualCap : 0;
      GCRTracker gcrTracker(gcrResCap, ip.cuda_prec);
      std::vector<ColorSpinorField> krylovVecs;

      // Step 1: z = γ₅ phi
      std::vector<ColorSpinorField> z(1, csParam);
      std::vector<ColorSpinorField> b_vec(1, ColorSpinorField(phi));
      gamma5(z, b_vec);

      // Step 2: solve M w = z
      std::vector<ColorSpinorField> w(1, csParam);
      {
        TrackerScope<GCRTracker> scope(activeGCRTracker, gcrTracker.isActive() ? &gcrTracker : nullptr);
        Solver *s = Solver::create(solverParam, m, mSloppy, mPre, mEig);
        (*s)(w, z);
        delete s;
        integratorBumpCGStats(solverParam.iter);
        solverParam.iter = 0;
      }
      for (auto &q : takeRitzVectors(gcrTracker)) krylovVecs.push_back(std::move(q));

      // Step 3: y = γ₅ w
      std::vector<ColorSpinorField> y(1, csParam);
      gamma5(y, w);

      // Step 4: solve M x = y
      std::vector<ColorSpinorField> x_vec(1, csParam);
      {
        TrackerScope<GCRTracker> scope(activeGCRTracker, gcrTracker.isActive() ? &gcrTracker : nullptr);
        Solver *s = Solver::create(solverParam, m, mSloppy, mPre, mEig);
        (*s)(x_vec, y);
        delete s;
        integratorBumpCGStats(solverParam.iter);
      }
      for (auto &q : takeRitzVectors(gcrTracker)) krylovVecs.push_back(std::move(q));
      x_dev = x_vec[0];

      // Stash normalized solution + GCR-Krylov residuals for eigentracking
      // absorption. The action solve runs only twice per trajectory (start
      // and end), so the residual cap of 4 contributes at most 8 extra
      // pool candidates per trajectory from this site.
      if (tracking && tracking->isActive()) {
        ColorSpinorField xNorm(x_vec[0]);
        double xnrm = sqrt(blas::norm2(xNorm));
        std::vector<ColorSpinorField> solVecs;
        if (xnrm > 1e-30) {
          blas::ax(1.0 / xnrm, xNorm);
          solVecs.push_back(std::move(xNorm));
        }
        for (auto &q : krylovVecs) solVecs.push_back(std::move(q));
        if (!solVecs.empty()) tracking->stashRitzVectors(std::move(solVecs));
      }

      delete dirac;
      if (diracSloppy && diracSloppy != dirac) delete diracSloppy;
      if (diracPre && diracPre != dirac && diracPre != diracSloppy) delete diracPre;
      if (diracEig && diracEig != dirac && diracEig != diracPre) delete diracEig;
    } else {
      // Standard host-side path via invertQuda (no MG preconditioner).
      // Without MG the inv_type is typically CG / CGNE / CGNR, which
      // *does* admit zero-cost Lanczos-tridiag Ritz extraction
      // (cg_ritz_extractor.cpp). Install activeCGTracker for the
      // duration of the invertQuda call when the user has opted in via
      // residualCap > 0; the tracker captures alpha/beta and
      // normalised residuals from inside the CG loop, then we pull out
      // up to residualCap Ritz pairs to feed the ET pool.
      const int cgRitzCap = (tracking && tracking->isActive()) ? tracking->getParam().residualCap : 0;
      CGTracker cgTracker(cgRitzCap);

      {
        TrackerScope<CGTracker> scope(activeCGTracker, cgTracker.isActive() ? &cgTracker : nullptr);
        ip.make_resident_solution = 1;
        ip.use_resident_solution = 1;
        ColorSpinorParam cpuParam(nullptr, ip, gaugePrecise->X(), true, QUDA_CPU_FIELD_LOCATION);
        cpuParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField h_x(cpuParam);
        ColorSpinorField h_b(cpuParam);
        h_b = phi;
        invertQuda(h_x.data(), h_b.data(), &ip);
        integratorBumpCGStats(ip.iter);
        x_dev = solutionResident[0];
        solutionResident.clear();
      }

      if (tracking && tracking->isActive()) {
        auto vecs = takeRitzVectors(cgTracker);
        if (!vecs.empty()) tracking->stashRitzVectors(std::move(vecs));
      }
    }

    // S_f = Re(phi† x)
    Complex dot = blas::cDotProduct(phi, x_dev);
    double sf = dot.real();

    // Clover EO with EVEN_EVEN_ASYMMETRIC: S_f = phi^dag x - 2 * TrLog[ODD].
    // The force uses ASYMMETRIC matpc; the action must match. CG works with
    // SYMMETRIC and gives the same solution, but the log-det differs.
    if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH && cloverPrecise) {
      sf -= 2.0 * cloverPrecise->TrLog()[1]; // odd parity only
    } else if (inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH && cloverPrecise) {
      sf -= cloverPrecise->TrLog()[1]; // stores log det(A² + μ̃²): both flavor factors
    }

    solutionResident.clear();
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return sf;
  }

  /**
     @brief Accumulate the EO fermion force into momentum (Wilson hopping).

     Schwinger-style reconstruction of the full-site bilinear from the
     even-parity CG solution, followed by the outer-product kernel.

     @param mom       Momentum field (accumulated into)
     @param x_even    CG solution: D_hat^dag D_hat x = phi (even parity)
     @param inv_param Inverter parameters (provides kappa, dslash type)
     @param dt        Integration step size
  */
  void computeEOFermionForce(GaugeField &mom, ColorSpinorField &x_even,
                             QudaInvertParam &inv_param, double dt)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    double kappa = inv_param.kappa;
    double kappa2 = kappa * kappa;

    QudaInvertParam force_ip = inv_param;
    DiracParam diracParam;
    setDiracParam(diracParam, &force_ip, true);
    Dirac *dirac = Dirac::create(diracParam);

    // Operators with a site-diagonal term A (clover C, twisted 1 + i2κμγ₅)
    // require ASYMMETRIC matpc here: with dagger, only the asymmetric PC
    // kernel path applies the daggered diagonal inverse on the OUTPUT (odd)
    // parity; the symmetric path fuses the twist into the even input, which
    // yields the wrong chi_odd below.
    if (inv_param.dslash_type != QUDA_WILSON_DSLASH && inv_param.matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC)
      errorQuda("EO fermion force requires QUDA_MATPC_EVEN_EVEN_ASYMMETRIC for dslash type %d (got matpc %d)",
                inv_param.dslash_type, inv_param.matpc_type);

    // --- Common fields: y = M̂x, psi_odd, chi_odd ---
    ColorSpinorField y_even = createParityField(x_even);
    dirac->M(y_even, x_even);

    // psi_odd = A(+μ)_oo⁻¹ D_oe x
    ColorSpinorField psi_odd = createParityField(x_even);
    dirac->Dslash(psi_odd, x_even, QUDA_ODD_PARITY);

    // chi_odd = A(-μ)_oo⁻¹ D_eo† y, built dagger-explicitly. This replaces
    // the earlier γ₅·Dslash(γ₅y) sandwich, which is only correct when
    // A† = A: the twisted term commutes with γ₅ but Ã(μ)† = Ã(-μ), so the
    // sandwich fails to flip the twist sign. The PC Dslash under
    // (dagger, asymmetric matpc) applies the daggered diagonal inverse on
    // the output parity for Wilson (trivially), clover (C† = C), twisted
    // mass and twisted clover (μ → -μ flipped inside the kernels).
    ColorSpinorField chi_odd = createParityField(x_even);
    dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(chi_odd, y_even, QUDA_ODD_PARITY);
    dirac->Dagger(QUDA_DAG_NO);

    // Hopping fields (Schwinger reconstruction, same as Wilson)
    ColorSpinorField psi_full = createFullField(x_even);
    ColorSpinorField chi_full = createFullField(x_even);
    psi_full[QUDA_EVEN_PARITY] = x_even;
    psi_full[QUDA_ODD_PARITY] = psi_odd;
    chi_full[QUDA_EVEN_PARITY] = y_even;
    chi_full[QUDA_ODD_PARITY] = chi_odd;

    // --- Hopping force: coeff = 2κ² ---
    // QUDA momentum convention: with T = momAction = ¼Σ_a p_a² (Gell-Mann
    // components p_a of the stored anti-Hermitian momentum) and the MD flow
    // U' = exp(dt·P)U, conservation of H = T + S requires the stored force
    // to be F_a = -2·∂S/∂u_a — twice the naive generator gradient. The
    // production gauge force obeys the same convention (dS/dε = -2T in
    // HMC.GaugeForceActionConsistency solves to p_a = -2 ∂S/∂u_a).
    // Empirically (full-lattice 8192-component per-link sweep) coeff κ²
    // yields stored F_a = -∂S/∂u_a exactly, hence 2κ² here. The dynamical
    // arbiter is HMC.dHScaling: with κ² dH plateaus (dt⁰ — non-symplectic);
    // with 2κ² it scales as dt².
    GaugeField force = createForceField(mom);
    computeCloverOprod(force, *gaugePrecise, {chi_full}, {psi_full}, {2.0 * kappa2});

    // --- Clover sigma force (T1 even diagonal + T3 odd clover-inverse) ---
    // With C = 1 + c Σ_{μ>ν} σ(ν,μ) F_μν (the operator DiracClover applies,
    // c = csw·κ = cloverPrecise->Coeff()) and δS = -2Re[y†δM̂x]:
    //   T1 (even): δS ⊃ -2 Re[y† δC x]
    //   T3 (odd):  δS ⊃ -2κ² Re[χ_odd† δC ψ_odd]   (+κ² from the two minus
    //              signs in δ(C⁻¹) = -C⁻¹ δC C⁻¹ against -κ² D C⁻¹ D)
    // The odd fields are exactly the hopping fields ψ_odd = C⁻¹D_oe x,
    // χ_odd = C⁻¹D_eo†y, so the same full-site fields serve both terms with
    // parity coefficient ratio ε_odd/ε_even = κ². computeCloverSigmaOprod
    // supplies the (Q - Q†) antisymmetrization; cloverDerivative inserts the
    // oprod into all eight leaf corners of dQ/dU with external coefficient
    // c/8 (the 1/8 from the F_μν = (Q - Q†)/8 storage convention). ε = {1, κ²}
    // yields F_a = -∂S_σ/∂u_a; the overall 2 (ε = {2, 2κ²}) matches the
    // hopping coefficient's F = -2·∂S convention above. Full factor
    // accounting is cross-checked against computeCloverForceQuda's
    // coefficient ratios.
    if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH || inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      if (!cloverPrecise) errorQuda("No resident clover field for clover sigma force");

      GaugeFieldParam oParam(force);
      oParam.geometry = QUDA_TENSOR_GEOMETRY;
      oParam.create = QUDA_ZERO_FIELD_CREATE;
      GaugeField oprod(oParam);

      // TrLog force first: the trace kernel ASSIGNS its parity plane (the
      // sigma oprod below accumulates). Action term S ⊃ -2 TrLog[C_oo]
      // (clover) or -TrLog[(C² + μ̃²)_oo] (twisted clover); in both cases
      // δS = -2 Tr[W δC] with W = C⁻¹ resp. C(C² + μ̃²)⁻¹. The coefficient
      // composes to 2.0 under F_a = -∂S/∂u_a — doubled to 4.0 for the
      // F_a = -2·∂S/∂u_a momentum convention (same ×2 as the hopping and
      // sigma coefficients).
      //
      // Kernel normalization (clover_trace.cuh, branch selected from the
      // clover field's TwistFlavor): the untwisted branch restores the ×2
      // QUDA storage factor and yields exactly W = C⁻¹. The twist branch
      // multiplies the UNRESTORED stored field (C/2) by an explicit 0.5,
      // so it yields ¼·W — a fixed normalization inherited by production
      // callers (computeTMCloverForceQuda compensates in its own
      // coefficients, so the kernel cannot be changed without breaking
      // upstream). Compensate here: 4.0 × 4 = 16.0 for twisted clover.
      const bool tc = (inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH);
      computeCloverSigmaTrace(oprod, *cloverPrecise, tc ? 16.0 : 4.0, QUDA_ODD_PARITY);

      computeCloverSigmaOprod(oprod, {psi_full}, {chi_full}, {{2.0, 2.0 * kappa2}});

      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
      GaugeField *gaugeEx = createExtendedGauge(*gaugePrecise, R, getProfile());
      cloverDerivative(force, *gaugeEx, oprod, cloverPrecise->Coeff() / 8.0);
      delete gaugeEx;
    }

    // Momentum update: p += dt × TA(force)
    updateMomentum(mom, dt, force, "eo_fermion");

    delete dirac;
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
