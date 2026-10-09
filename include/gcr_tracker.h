#pragma once

/**
 * @file gcr_tracker.h
 * @brief Krylov-basis recording for GCR solves.
 *
 * GCR is the outer solver QUDA uses for MG-preconditioned fermion solves.
 * Unlike CG, it does not produce a Lanczos tridiagonal whose eigenvalues
 * are Ritz values of the operator (CGTracker exploits that for zero-cost
 * Ritz extraction in inv_cg_quda.cpp). GCR with right preconditioning
 * builds A * P_m = Q_m * T_m where P_m are the (preconditioned) search
 * directions and Q_m are the orthonormalised applied directions; T_m is
 * upper triangular but is the QR factor of A * P, not the Hessenberg of
 * A in any basis we have. Recovering Ritz pairs of A from (P, Q, T) in
 * preconditioned GCR requires either extra matvecs (one per stored q_i)
 * or a non-trivial inversion of the preconditioner's effect on P, both
 * of which are too expensive on the inner-loop hot path.
 *
 * What this tracker does instead is simpler and zero-extra-matvec:
 * during the solve it normalises and stores up to N most-recent GCR
 * residuals. The residuals span the affine space r_0 + Krylov subspace
 * and are progressively enriched in the modes the preconditioner could
 * not kill (= the modes MG most needs refreshed). After convergence the
 * eigentracker absorbs them as additional pool vectors; the pool's
 * Rayleigh-Ritz machinery handles the projection / orthogonalisation
 * downstream, just as it already does for the converged-solution vector
 * stashed by NestedFGI's outer force.
 *
 * Lifecycle mirrors CGTracker: the inv_gcr_quda.cpp loop reads the
 * extern global activeGCRTracker; eigentracking call sites set it
 * before invoking GCR and read out the stored vectors afterwards.
 */

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <quda_internal.h>
#include <vector>

namespace quda
{

  class GCRTracker;

  /**
   * @brief Global pointer for the active GCR tracker.
   *
   * Set before calling a GCR solve, read by inv_gcr_quda.cpp at each
   * outer-iteration's residual-update site, cleared after. Single-threaded
   * by HMC convention, same as activeCGTracker.
   */
  extern GCRTracker *activeGCRTracker;

  /**
   * @brief Records normalised GCR residuals for downstream pool absorption.
   *
   * Holds up to maxVecs_ vectors. When more iterations occur than the
   * cap allows, residuals beyond the cap are silently dropped (we keep
   * the FIRST maxVecs_ residuals, not the last; a FIFO would require
   * vector::erase, which conflicts with QUDA's ColorSpinorField
   * move-assignment guard). The early residuals span the same Krylov
   * subspace as the later ones, so the pool's downstream Rayleigh-Ritz
   * step recovers the low-mode content either way. Restart boundaries
   * are not treated specially — residuals from before and after a
   * restart go into the same pool. Strict orthogonality is not required
   * because EigenTracker::absorb re-orthonormalises against the
   * existing pool.
   *
   * recordIteration handles two normalisations needed before the pool
   * can absorb a captured residual:
   *
   *   - Precision promotion. GCR runs at precision_sloppy (typically
   *     single); the pool reference vectors are at the target precision
   *     passed to the constructor (typically inv_param.cuda_prec =
   *     double). Mismatched-precision multiCdot kernels are not
   *     instantiated, so without promotion the absorption faults.
   *
   *   - Site-subset is preserved by design. The EigenTracker pool is
   *     seeded by hmc.cpp's seedEigenTrackingFromMG from the
   *     even-parity half-site components of the MG null vectors, so
   *     pool reference vectors are half-site fine. Inside a PC solve
   *     (solve_type=DIRECT_PC_SOLVE etc.) GCR's r_sloppy is also
   *     half-site, which matches by construction. Embedding half-site
   *     into full-site here would invert that match and fault in
   *     MultiReduce's length check.
   *
   * recordIteration also silently filters out residuals from non-fine
   * spinor fields. The hook in inv_gcr_quda.cpp fires for every GCR
   * instance, including the coarse-grid GCR that QUDA's MG
   * preconditioner runs at level 1+. Coarse residuals have a
   * different (Ns, Nc) than the pool's fine-grid Wilson reference
   * vectors, so they are dropped at recordIteration entry. Tracking
   * the coarse spectrum is a separate concern handled by the nested
   * FGI's CoarseDeflationManager.
   */
  class GCRTracker
  {
  private:
    int maxVecs_;                             /**< Cap on stored residuals */
    QudaPrecision targetPrecision_;           /**< Precision to promote stored residuals to */
    std::vector<ColorSpinorField> residuals_; /**< Normalised residuals at targetPrecision_ */
    int totalIterations_;                     /**< Bookkeeping; total record() calls */
    bool active_;                             /**< Whether tracking is enabled */

  public:
    /**
     * @brief Construct a GCRTracker.
     *
     * @param maxVecs         Cap on stored residuals. 0 disables capture.
     * @param targetPrecision Precision the stored residuals are promoted to.
     *                        Should match the EigenTracker pool's precision
     *                        (typically inv_param.cuda_prec = double),
     *                        otherwise downstream multiCdot kernels that
     *                        mix double pool vectors with single-precision
     *                        residuals fault with "Y precision N not
     *                        supported". QUDA_INVALID_PRECISION (default)
     *                        means "no promotion, store at the source
     *                        precision."
     */
    explicit GCRTracker(int maxVecs = 0, QudaPrecision targetPrecision = QUDA_INVALID_PRECISION);

    /**
     * @brief Record one GCR iteration's residual.
     *
     * Called from the GCR loop after the residual update. Promotes @p r
     * to targetPrecision_ (if set), normalises in-place, and stores. If
     * the cap has been reached the new residual is silently dropped
     * (we keep the first maxVecs_; see implementation for the rationale).
     */
    void recordIteration(const ColorSpinorField &r);

    /**
     * @brief Move the stored residuals out for pool absorption.
     *
     * Returns the FIFO of normalised residuals captured during the last
     * GCR solve and clears internal storage. Caller (typically the
     * eigentracking call site) hands them to EigenTracker::absorb.
     */
    std::vector<ColorSpinorField> takeResiduals();

    /** @brief Clear state; preserves maxVecs_ / active_ configuration */
    void reset();

    bool isActive() const { return active_ && maxVecs_ > 0; }
    int numStored() const { return static_cast<int>(residuals_.size()); }
    int numIterations() const { return totalIterations_; }
  };

  // Use TrackerScope<GCRTracker> from inv_tracker.h to install / restore
  // the activeGCRTracker slot around a solve. Same pattern as
  // TrackerScope<CGTracker>; both are templated against the tracker
  // class and avoid duplicating the save / restore boilerplate at every
  // call site.

} // namespace quda
