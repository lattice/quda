#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <random_quda.h>
#include <timer.h>
#include <quda.h>

namespace quda
{

  // Forward declaration; full definition lives in eigen_tracking_state.h.
  // Only pulled in by translation units that actually dereference it.
  class EigenTrackingState;

  /**
   * @brief Accessor for the shared HMC eigentracking TimeProfile.
   *
   * Backed by a function-local static in interface_quda.cpp. All
   * eigentracking translation units push this via pushProfile so the
   * per-function cost aggregates into one "eigenTracking" row of the
   * end-of-run summary printed at endQuda.
   */
  TimeProfile &getEigenTrackProfile();

  /**
   * @brief Accessor for the one-shot eigentracking setup profile.
   *
   * Distinct from getEigenTrackProfile so that the wall-clock cost of
   * the initial TRLM (or MG-null seeding) is separated from the
   * per-trajectory tracker work in the hmcTrajectoryQuda child-profile
   * breakdown. Pushed only by EigenTrackingState::maybeInit and
   * EigenTrackingState::seedFromMGNullVectors.
   */
  TimeProfile &getEigenTrackSetupProfile();

  /**
   * @brief RAII guard for QUDA_PROFILE_COMPUTE on the active profile.
   *
   * Pair with `auto p = pushProfile(...)` so the body of a function is
   * fully accounted under the active profile's COMPUTE phase. Destruction
   * order (reverse of construction) ensures TPSTOP(COMPUTE) fires before
   * the outer pushProfile pops the active profile.
   *
   * Sub-phase TPSTART/TPSTOP are reference-counted in QUDA, so nested
   * ScopedComputePhase instances (e.g. when this function calls another
   * function that also wraps COMPUTE) compose correctly.
   */
  struct ScopedComputePhase {
    ScopedComputePhase() { getProfile().TPSTART(QUDA_PROFILE_COMPUTE); }
    ~ScopedComputePhase() { getProfile().TPSTOP(QUDA_PROFILE_COMPUTE); }
    ScopedComputePhase(const ScopedComputePhase &) = delete;
    ScopedComputePhase &operator=(const ScopedComputePhase &) = delete;
  };

  /**
   * @brief Gauge action: S_g = beta * (1 - plaq) * V * 6.
   */
  double computeGaugeActionHMC(double beta);

  /**
   * @brief Generate EO pseudofermion: φ_e = D̂ η_e, η_e ~ N(0,1).
   *
   * Creates a single-parity (even) pseudofermion field on the GPU. The
   * seed-based overload allocates a one-shot RNG state array; prefer the
   * RNG& overload for hot-path use so state advances between calls.
   */
  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, unsigned long long seed);
  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, RNG &rng);

  /**
   * @brief HMC RNG family.
   *
   * Three persistent streams seeded deterministically from a single user
   * base seed (offsets +1 spinor, +2 gauge, +3 Metropolis). The spinor /
   * gauge RNGs follow QUDA's multigrid.cpp pattern (one cuRAND state per
   * site, state advances across calls). The Metropolis RNG is a host-side
   * MT19937_64 used for the scalar accept/reject draw so the decision is
   * reproducible under a fixed user seed regardless of process-wide
   * drand48() state.
   */
  void setHMCBaseSeed(unsigned long long seed);

  /**
   * @brief Get-or-build the HMC spinor RNG for the preconditioned spinor
   *        geometry implied by the resident gauge field and @p inv_param.
   */
  RNG &getHMCSpinorRNG(QudaInvertParam &inv_param);

  /**
   * @brief Get-or-build the HMC gauge RNG matching @p meta's volume.
   */
  RNG &getHMCGaugeRNG(const GaugeField &meta);

  /**
   * @brief Draw one uniform [0,1) double from the Metropolis stream.
   */
  double hmcMetropolisUniform();

  /**
   * @brief Release all three RNG streams.
   *
   * Called from destroyHMCQuda and endQuda before the device pool is
   * flushed (the spinor/gauge state arrays live in the QUDA pool).
   */
  void releaseHMCRNG();

  /**
   * @brief Persistent EigenTrackingState singleton.
   *
   * Lifecycle: created lazily by the HMC entry points (or
   * seedEigenTrackingFromMG), configured from QudaHMCParam, used by
   * hmcTrajectoryQuda / hmcRunQuda for low-mode tracking, destroyed by
   * destroyHMCQuda before the device pool is flushed.
   */
  EigenTrackingState *getEigenTrackingInstance();
  void setEigenTrackingInstance(EigenTrackingState *inst);
  void releaseEigenTrackingInstance();

  /**
   * @brief Pseudofermion stored across trajectories for reversibility tests
   *        (used when QudaHMCParam::reuse_pseudofermion is set).
   */
  ColorSpinorField *getStoredPhi();
  void setStoredPhi(ColorSpinorField phi);
  void releaseStoredPhi();

  /**
   * @brief EO fermion action: S_f = Re(φ_e† x_e) where D̂†D̂ x_e = φ_e.
   *
   * Solves the even-odd preconditioned normal equations via CG.
   * Plain primitive: no MG γ₅ two-pass, no clover TrLog.
   */
  double computeEOFermionAction(ColorSpinorField &phi_even, QudaInvertParam &inv_param);

  /**
   * @brief Full fermion action used by hmcTrajectoryQuda for H_init / H_final.
   *
   * Differs from computeEOFermionAction:
   *   - Uses MG γ₅ two-pass when @p inv_param.preconditioner is set.
   *   - Subtracts the odd-parity clover TrLog for QUDA_CLOVER_WILSON_DSLASH
   *     (matches the asymmetric matpc convention used by the force).
   *   - Optionally stashes the normalized solution into the eigentracker for
   *     low-mode absorption.
   *
   * @param phi        Pseudofermion (parity field, native order, on device)
   * @param inv_param  Inverter parameters (CG / GCR+MG / etc.)
   * @param tracking   Optional eigentracking state; if non-null and active,
   *                   the normalized solution is stashed for absorption.
   */
  double computeFermionAction(ColorSpinorField &phi, QudaInvertParam &inv_param, EigenTrackingState *tracking = nullptr);

  /**
   * @brief EO fermion force: accumulates dS_eo/dU into momentum.
   *
   * Implements the verified Schwinger EO force algorithm:
   *   Y_e = D̂ x_e, p_o = Dslash(x_e), q_o = Dslash^dag(Y_e)
   *   X_full = (x_e, p_o), chi_full = (Y_e, q_o)
   *   force = -outer_product(chi_full, X_full)
   *
   * @param mom      Momentum field (accumulated into)
   * @param x_even   CG solution: D̂†D̂ x_e = φ_e (even parity)
   * @param inv_param Inverter parameters (provides kappa, dslash type)
   * @param dt       Integration step size
   */
  void computeEOFermionForce(GaugeField &mom, ColorSpinorField &x_even, QudaInvertParam &inv_param, double dt);

  /**
   * @brief Seed the eigentracker from MG null-space vectors.
   *
   * Extracts even-parity components of the MG null vectors and feeds
   * them into the eigentracker pool, avoiding the expensive initial TRLM.
   * Must be called after MG setup and before the first HMC trajectory.
   *
   * @param mg_instance  MG preconditioner (multigrid_solver *)
   * @param hmc_param    HMC parameters (for eigentracking config)
   * @param inv_param    Inverter parameters (for Dirac operator creation)
   */
  void seedEigenTrackingFromMG(void *mg_instance, QudaHMCParam *hmc_param, QudaInvertParam *inv_param);

  /**
     @brief Return the number of null-space vectors held by an MG instance.
     Lets test code size eigentracking pools from MG nvec without including
     the internal multigrid.h (whose header chain requires target-specific
     include paths that test executables do not have).
     @param mg_instance Opaque multigrid_solver pointer (may be null)
     @return Null-vector count, or 0 if mg_instance is null
  */
  int getMGNullVectorCount(void *mg_instance);

  /**
     @brief Full state-refresh discipline after any device-side write to
     gaugePrecise: ghost pads, sloppy gauge copies, extended gauge, and (for
     clover-type actions) the in-place clover recompute at double precision
     plus the in-place sloppy-clover family refresh. Preserves all field
     object identities (MG hierarchies hold pointers). Must be called after
     every gauge update, restore, or displacement in the HMC path.
     @param inv_param Inverter parameters (dslash type, clover coefficient)
  */
  void hmcRefreshResidentGaugeState(QudaInvertParam &inv_param);

} // namespace quda
