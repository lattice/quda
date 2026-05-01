#pragma once

/**
 * @file coarse_deflation_manager.h
 * @brief Manages coarse-grid eigenvectors for deflation tracking.
 *
 * Standalone class for tracking eigenvectors of the coarse MG operator.
 * Used by both eigentracking (HMC pool → MG refresh → coarse RR) and
 * the nested FGI (force splitting via coarse deflation).
 */

#include <vector>
#include <color_spinor_field.h>
#include <eigensolve_quda.h>
#include <transfer.h>
#include <dirac_quda.h>
#include <quda.h>

namespace quda
{

  /**
   * @brief Manages coarse-grid eigenvectors for deflation-based tracking.
   *
   * Supports three refresh tiers:
   *   Tier 1 (rayleighRitzUpdate): Re-diagonalize in the current eigenvector subspace.
   *     Cost: n_defl coarse matvecs (~free).
   *   Tier 2 (maybeRefresh): Periodic Galerkin rebuild + RR evolution within a trajectory.
   *     Cost: k fine matvecs per refresh (k = number of MG null vectors).
   *   Tier 3 (solve): Full TRLM re-eigensolve on the coarse operator.
   */
  class CoarseDeflationManager {
  private:
    std::vector<ColorSpinorField> coarseEvecs;
    std::vector<Complex> coarseEvals;

    const DiracMatrix *matCoarse;    /**< M wrapper for RR updates */
    const Dirac *diracCoarse;        /**< Raw Dirac for creating MdagM wrapper */
    const Transfer *transfer;

    int nDefl;
    int refreshInterval;
    int stepCounter;

    QudaEigParam eigParam;

    /** Workspace for RR update */
    std::vector<ColorSpinorField> workVecs;

  public:
    /**
     * @brief Construct the deflation manager.
     * @param transfer   Transfer operator (restrict/prolong) from MG level
     * @param matCoarse  Coarse Dirac matrix wrapper from MG
     * @param nDefl      Number of deflation eigenvectors
     * @param eigTol     TRLM convergence tolerance
     * @param nKr        Krylov space size (default 3*nDefl)
     * @param maxRestarts TRLM max restarts
     * @param refreshInterval Inner steps between Tier 2 refresh (0 = frozen)
     */
    CoarseDeflationManager(const Transfer &transfer, const DiracMatrix &matCoarse, const Dirac &diracCoarse,
                           int nDefl, double eigTol, int nKr = 0, int maxRestarts = 100, int refreshInterval = 0);

    ~CoarseDeflationManager() = default;

    /** @brief Tier 3: Run full TRLM eigensolver on the coarse operator */
    void solve();

    /** @brief Tier 1: Rayleigh-Ritz re-diagonalization (cheap, between trajectories) */
    void rayleighRitzUpdate();

    /**
     * @brief Rebind transfer / coarse Dirac / coarse DiracMatrix references.
     *
     * Call after a refresh-style updateMultigridQuda which destroys and
     * recreates the MG coarse operators. The previously stored pointers
     * dangle the moment MG::reset(refresh=true) returns; using them
     * (e.g. via rayleighRitzUpdate or solve) faults. After rebind the
     * stored coarseEvecs are still valid as ColorSpinorField containers
     * (block size and nVec are configuration-stable across refresh) but
     * their numerical contents are stale, so call solve() to repopulate.
     */
    void rebindCoarseRefs(const Transfer &transfer_, const DiracMatrix &matCoarse_, const Dirac &diracCoarse_);

    /** @brief Tier 2: Check step counter and refresh if due */
    void maybeRefresh();

    /** @brief Increment the inner-step counter */
    void step() { stepCounter++; }

    /** @brief Reset the step counter (e.g., at start of trajectory) */
    void resetCounter() { stepCounter = 0; }

    const std::vector<ColorSpinorField> &getEvecs() const { return coarseEvecs; }
    const std::vector<Complex> &getEvals() const { return coarseEvals; }
    const Transfer &getTransfer() const { return *transfer; }
    int getNDefl() const { return nDefl; }
  };

} // namespace quda
