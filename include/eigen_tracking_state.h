#pragma once

/**
 * @file eigen_tracking_state.h
 * @brief Orchestrator for eigenspace tracking during HMC.
 *
 * Manages the lifecycle of an EigenTracker and EigenForecast,
 * plus chronological solution history for initial-guess extrapolation.
 */

#include <eigen_tracker.h>
#include <eigen_forecast.h>
#include <cg_ritz_extractor.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <string>

namespace quda
{

  /**
   * @brief Parameters controlling eigenspace tracking during HMC.
   *
   * Defaults here are the fallback values used if the caller does not set
   * anything. Callers that thread values from QudaHMCParam (the normal
   * production path) should set every field explicitly; the struct defaults
   * are informational only. In tests/hmc_test.cpp, the test fixture
   * substitutes small fixture-friendly values (nEv=6, poolCapacity=8, etc.)
   * when CLI flags are left at their "0 = derive" sentinel.
   */
  struct EigenTrackingParam {
    bool enabled = false;         /**< Master enable/disable switch */
    int nEv = 8;                  /**< Number of tracked eigenpairs */
    int poolCapacity = 32;        /**< Maximum pool size */
    int nRitz = 4;                /**< Ritz pairs to extract per CG solve */
    int forecastOrder = 1;        /**< Generator forecast order (0/1/2) */
    int freshTRLMInterval = 10;   /**< Trajectories between fresh TRLM (0=disabled) */
    int solutionHistoryDepth = 3; /**< Number of previous solutions to store */
    /** When false, stashRitzVectors becomes a no-op so the pool stays as
     *  the (RR-evolved) original MG null vectors. Mirrors the Schwinger
     *  pattern where MG-null-vector store is kept separate from any
     *  CG-derived Ritz-vector cache; preserves smoother-aware structure
     *  needed for MG-prolongator refresh at extreme mass. */
    bool absorbRitz = true;
    /** Pool-driven MG null-vector refresh on accepted-trajectory re-setup.
     *  -1 = disabled (standard CG re-setup), 0 = pure pool replacement,
     *  N>0 = hybrid pool + N CG inverse-iter polish steps per vector.
     *  Mirrors QudaHMCParam::eigentracking_mg_refresh_iters. */
    int mgRefreshIters = -1;
    /** Per-solve cap on extra ET-pool candidates contributed by each
     *  fermion solve. Single knob covering both the GCR (residual stash,
     *  gcr_tracker.h) and CG (Lanczos-tridiag Ritz extraction,
     *  cg_ritz_extractor.cpp) install paths. Default 0 (off): in
     *  safe-mass regimes the converged-solution stash alone saturates
     *  the pool's useful low-mode content. Set N>0 (typically 4) at
     *  light mass / large volume where the preconditioner's stuck modes
     *  are the signal worth capturing. Mirrors
     *  QudaHMCParam::eigentracking_residual_cap. */
    int residualCap = 0;
    /** Initial TRLM convergence knobs (also used for fresh-TRLM refreshes). */
    double trlmTol = 1e-6;      /**< TRLM convergence tolerance */
    int trlmMaxRestarts = 100;  /**< TRLM maximum restarts */
    int trlmCheckInterval = 10; /**< TRLM iterations between convergence checks */
    /** Eigensolver to use for the initial / fresh-TRLM solves. Defaults to
     *  standard TRLM. BLKTRLM operates on multiple Lanczos vectors per
     *  step (larger arithmetic intensity per kernel; n_kr and n_ev must be
     *  multiples of blockSize — auto-rounded if not).
     *  Other QudaEigType values flow through unchanged for advanced users. */
    QudaEigType eigType = QUDA_EIG_TR_LANCZOS;
    int blockSize = 4; /**< Block size for block solvers (BLKTRLM/BLKIRAM) */
    /** @name Initial TRLM polynomial acceleration (Chebyshev filter)
     *  For clustered or near-zero M^dag M spectra (hot-start gauges,
     *  critical-mass runs without MG), Chebyshev acceleration is needed
     *  to resolve the low modes reliably. Set a_min ~1 order of magnitude
     *  above the smallest target eigenvalue; a_max = 0 triggers QUDA's
     *  power-iteration auto-estimate. See QUDA eigensolver wiki. */
    bool usePolyAcc = false; /**< Enable Chebyshev acceleration in initial TRLM */
    int polyDeg = 50;        /**< Chebyshev polynomial degree */
    double aMin = 0.0;       /**< Lower suppression boundary (must be set when usePolyAcc) */
    double aMax = 0.0;       /**< Upper boundary; 0 => auto-estimate */
  };

  /**
   * @brief Orchestrator for eigenspace tracking during HMC.
   *
   * Owns an EigenTracker pool, an EigenForecast for inter-trajectory
   * prediction, and a chronological solution history for CG initial
   * guess extrapolation. Provides clean interface for HMC integrator
   * hooks at the appropriate points in the trajectory.
   */
  class EigenTrackingState
  {
  private:
    EigenTracker tracker_;
    EigenForecast forecast_;

    /** Chronological initial guess (Brower et al.) */
    std::vector<ColorSpinorField> solutionHistory_;

    /** Stashed Ritz vectors from CG solves during trajectory (absorbed between trajectories) */
    std::vector<ColorSpinorField> ritzStash_;

    /** Configuration */
    EigenTrackingParam param_;

    /** Cached invert params for creating reference spinor fields */
    QudaInvertParam invParam_;
    bool invParamSet_;

    /** Statistics */
    int trajectoryCount_;
    int totalRitzAbsorbed_;
    int totalCGSolves_;

  public:
    EigenTrackingState();
    ~EigenTrackingState() = default;

    /**
     * @brief Configure the tracking state from parameters.
     * @param p  Tracking parameters
     */
    void configure(const EigenTrackingParam &p);

    /**
     * @brief Initialize tracker via fresh TRLM if not already initialized.
     * @param mat         Normal operator D_hat^dag D_hat (for TRLM)
     * @param matHalf     D_hat operator (for Dpool computation)
     * @param inv_param   Inverter params (for creating reference spinor field)
     */
    void maybeInit(const DiracMatrix &mat, const DiracMatrix &matHalf, QudaInvertParam &inv_param);

    /**
     * @brief Called after each CG solve in the trajectory.
     *
     * Absorbs the normalized CG solution into the tracker pool.
     * Ritz extraction is now handled externally via CGTracker (zero cost).
     *
     * @param sol      CG solution vector
     * @param matHalf  D_hat operator
     */
    void afterCGSolve(const ColorSpinorField &sol, const DiracMatrix &matHalf);

    /**
     * @brief Called after each gauge update step in the trajectory.
     *
     * Updates Dpool cache using the new D_hat operator.
     *
     * @param matHalf  New D_hat operator (after gauge update)
     */
    void afterGaugeUpdate(const DiracMatrix &matHalf);

    /**
     * @brief Called between trajectories (after accept/reject).
     *
     * 1. Run full RR evolution to get rotation matrix.
     * 2. Record rotation in forecast state.
     * 3. Recompute Dpool with current operator.
     * 4. If fresh TRLM is due, re-initialize.
     *
     * @param mat      Normal operator D_hat^dag D_hat
     * @param matHalf  D_hat operator
     */
    void betweenTrajectories(const DiracMatrix &mat, const DiracMatrix &matHalf);

    /**
     * @brief Called before trajectory starts.
     *
     * Apply forecast pre-rotation to pool and Dpool,
     * then recompute Dpool with current operator.
     *
     * @param matHalf  D_hat operator
     */
    void beforeTrajectory(const DiracMatrix &matHalf);

    /**
     * @brief Get extrapolated initial guess for next CG solve.
     *
     * Uses simple extrapolation from solution history:
     *   depth 1: x0 = x_{n-1}
     *   depth 2: x0 = 2*x_{n-1} - x_{n-2}
     *   depth 3: x0 = 3*x_{n-1} - 3*x_{n-2} + x_{n-3}
     *
     * @param x0 Output: extrapolated initial guess (populated in-place)
     * @return true if a valid extrapolation was computed
     */
    bool getExtrapolatedX0(ColorSpinorField &x0) const;

    /** @brief Get tracking statistics summary string */
    std::string getSummary() const;

    /** @brief Check if tracking is enabled and initialized */
    bool isActive() const { return param_.enabled && tracker_.isInitialized(); }

    /**
     * @brief True when the periodic refresh interval has elapsed and the
     *        tracker should be re-seeded (or re-TRLM'd).
     *
     * The refresh policy lives outside this class because the orchestrator
     * (hmcTrajectoryQuda) has access to the MG instance and can prefer
     * MG re-seeding over a fresh TRLM. Returns false when the interval is
     * disabled (freshTRLMInterval = 0) or the tracker isn't yet initialised.
     */
    bool shouldRefresh() const
    {
      return tracker_.isInitialized() && param_.freshTRLMInterval > 0 && trajectoryCount_ > 0
        && trajectoryCount_ % param_.freshTRLMInterval == 0;
    }

    /**
     * @brief Mark the tracker uninitialised so the next seed / TRLM call
     *        rebuilds the pool from scratch. Preserves configuration.
     *
     * Caller follows up with seedEigenTrackingFromMG (when MG is available)
     * or maybeInit (TRLM fallback) — the existing init paths handle both.
     */
    void resetForRefresh()
    {
      tracker_ = EigenTracker();
      forecast_.reset();
    }

    /**
     * @brief Stash Ritz vectors from a CG solve for batch absorption later.
     * Zero matvec cost — vectors are just stored, not absorbed yet.
     * @param vecs  Ritz eigenvectors to stash
     */
    void stashRitzVectors(std::vector<ColorSpinorField> &&vecs);

    /**
     * @brief Seed the tracker pool from MG null-space vectors (skip TRLM).
     *
     * Absorbs even-parity vectors into the pool and computes Dpool.
     * Marks the tracker as initialized so maybeInit() becomes a no-op.
     *
     * @param evenVecs  Even-parity null-space vectors (already precision-promoted)
     * @param matHalf   D_hat operator for computing Dpool
     * @param inv_param Inverter params (cached for potential re-init)
     */
    void seedFromMGNullVectors(std::vector<ColorSpinorField> &evenVecs, const DiracMatrix &matHalf,
                               QudaInvertParam &inv_param);

    /** @brief Access parameters */
    const EigenTrackingParam &getParam() const { return param_; }

    /** @brief Access the tracker (read-only, for tests) */
    const EigenTracker &getTracker() const { return tracker_; }

    /** @brief Access the tracker (mutable, for forceUpdate on rejection) */
    EigenTracker &getTrackerMutable() { return tracker_; }

    /** @brief Cumulative count of Ritz vectors absorbed into the pool
     *         across the lifetime of this tracker. Used by the regression
     *         tests that verify the per-solve Krylov capture
     *         (--eigentracking-residual-cap > 0) actually feeds the pool. */
    int getTotalRitzAbsorbed() const { return totalRitzAbsorbed_; }

    /** @brief Number of trajectories the tracker has been live across.
     *         Increments inside betweenTrajectories. */
    int getTrajectoryCount() const { return trajectoryCount_; }
  };

} // namespace quda
