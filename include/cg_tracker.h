#pragma once

/**
 * @file cg_tracker.h
 * @brief Zero-cost CG-Lanczos Ritz pair extraction.
 *
 * Records CG iteration coefficients (alpha, beta) and normalized residuals
 * during the CG solve. After convergence, extracts Ritz pairs from the
 * implicit Lanczos tridiagonal at zero additional matvec cost.
 *
 * Based on the CG-Lanczos equivalence described in eigenspace_tracking.tex.
 */

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <quda_internal.h>
#include <vector>

namespace quda
{

  // Forward declaration for the global pointer
  class CGTracker;

  /**
   * @brief Global pointer for the active CG tracker.
   *
   * Set before calling solve(), read by CG during iteration, cleared after.
   * This avoids modifying the QudaInvertParam C-API struct. Thread-safe
   * in practice because HMC CG solves are sequential.
   */
  extern CGTracker *activeCGTracker;

  /**
   * @brief Records CG iteration data for zero-cost Ritz extraction.
   *
   * During CG, stores:
   *   - alpha_k, beta_k coefficients (scalar, negligible storage)
   *   - Up to N_lcz = 3*N_ritz normalized residuals q_k = r_k/||r_k||
   *
   * After convergence, builds the Lanczos tridiagonal T_m from alpha/beta,
   * eigendecomposes T_m, and reconstructs Ritz vectors v_p = Q_m u_p.
   */
  class CGTracker {
  private:
    int nRitz_;                                  /**< Number of Ritz pairs to extract */
    int maxLanczos_;                             /**< Max Lanczos vectors to store (3*nRitz) */
    std::vector<double> alphas_;                 /**< CG alpha coefficients */
    std::vector<double> betas_;                  /**< CG beta coefficients */
    std::vector<ColorSpinorField> lanczosVecs_;  /**< Normalized residuals q_k */
    bool active_;                                /**< Whether tracking is enabled */

  public:
    CGTracker(int nRitz = 0, int maxLanczos = 0);

    /**
     * @brief Record one CG iteration's data.
     *
     * Called from within the CG loop after alpha_k and beta_k are computed.
     * Stores the coefficients and (if under the cap) a normalized copy of
     * the residual vector.
     *
     * @param alpha  CG alpha coefficient for this iteration
     * @param beta   CG beta coefficient for this iteration
     * @param r      Current residual vector (will be normalized and copied)
     */
    void recordIteration(double alpha, double beta, const ColorSpinorField &r);

    /**
     * @brief Extract Ritz pairs from the accumulated Lanczos tridiagonal.
     *
     * Builds T_m from stored alpha/beta:
     *   delta_0 = 1/alpha_0
     *   delta_k = 1/alpha_k + beta_{k-1}/alpha_{k-1}  (k > 0)
     *   gamma_k = sqrt(beta_k) / alpha_k
     *
     * Eigendecomposes T_m, reconstructs Ritz vectors via v_p = Q_m u_p.
     * Cost: O(m^2) host-side, zero GPU matvecs.
     *
     * @param vecs  Output: Ritz eigenvectors (GPU)
     * @param vals  Output: Ritz eigenvalues
     */
    void extractRitzPairs(std::vector<ColorSpinorField> &vecs, std::vector<Complex> &vals);

    /** @brief Reset state for next CG solve */
    void reset();

    /** @brief Check if tracker is active */
    bool isActive() const { return active_ && nRitz_ > 0; }

    /** @brief Get number of stored Lanczos vectors */
    int numLanczosVecs() const { return static_cast<int>(lanczosVecs_.size()); }

    /** @brief Get number of CG iterations recorded */
    int numIterations() const { return static_cast<int>(alphas_.size()); }
  };

} // namespace quda
