#pragma once

/**
 * @file eigen_tracker.h
 * @brief Pool-based eigenvector tracker for HMC eigenspace maintenance.
 *
 * Ported from Schwinger_MG/src/eigen_forecast.cpp (EigenTracker struct).
 */

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <eigen_helper.h>
#include <blas_quda.h>
#include <quda_internal.h>
#include <vector>

namespace quda
{

  /**
   * @brief Pool-based eigenvector tracker for HMC eigenspace maintenance.
   *
   * Maintains a pool of orthonormal GPU vectors spanning an approximate
   * low-eigenvalue subspace of the even-odd preconditioned normal operator
   * M = D_hat^dag D_hat.  For each pool vector v_i, caches Dv_i = D_hat v_i
   * so that the projected matrix v_i^dag M v_j = Dv_i^dag Dv_j can be
   * computed without any M matvecs (zero-matvec compress).
   */
  class EigenTracker
  {
  private:
    std::vector<ColorSpinorField> pool_;  /**< Pool vectors (orthonormal, Ritz-ordered) */
    std::vector<ColorSpinorField> Dpool_; /**< D_hat applied to each pool vector */
    std::vector<Complex> eigvals_;        /**< nEv eigenvalue estimates */
    int nEv_;                             /**< Number of wanted eigenpairs */
    int poolCapacity_;                    /**< Maximum pool size */
    bool initialized_;                    /**< Whether tracker has been seeded */

  public:
    EigenTracker();
    ~EigenTracker() = default;

    EigenTracker(const EigenTracker &) = delete;
    EigenTracker &operator=(const EigenTracker &) = delete;
    EigenTracker(EigenTracker &&) = default;
    EigenTracker &operator=(EigenTracker &&) = default;

    /**
     * @brief Seed the tracker from a converged TRLM eigensolve.
     * @param kSpace  Converged eigenvectors from TRLM (GPU vectors)
     * @param evals   Converged eigenvalues from TRLM
     * @param mat     D_hat operator (DiracM wrapper for EO-preconditioned Dirac)
     * @param nEv     Number of wanted eigenpairs to track
     * @param capacity Maximum pool size (must be >= nEv)
     */
    void init(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals, const DiracMatrix &mat, int nEv,
              int capacity);

    /**
     * @brief Compress pool via RR projection using cached Dpool.
     *
     * Builds poolSize x poolSize matrix T_ij = Dpool[i]^dag Dpool[j],
     * diagonalizes on host via Eigen SelfAdjointEigenSolver, rotates
     * GPU vectors, keeps best poolCapacity vectors (smallest eigenvalues).
     * Zero D^dag D matvecs.
     */
    void compress();

    /**
     * @brief Absorb new vectors into pool.
     *
     * Orthogonalizes against existing pool via double MGS,
     * computes D_hat v_new for each, auto-compresses if over capacity.
     *
     * @param newVecs Vectors to absorb (GPU)
     * @param mat     D_hat operator for computing Dpool entries
     * @return Number of vectors actually absorbed
     */
    int absorb(std::vector<ColorSpinorField> &newVecs, const DiracMatrix &mat);

    /**
     * @brief Update Dpool cache after gauge change.
     *
     * Recomputes Dpool[i] = mat(pool[i]) with the new operator,
     * then calls compress() to re-diagonalize.
     * Cost: poolSize single-operator matvecs.
     *
     * @param mat  New D_hat operator (after gauge update)
     */
    void forceUpdate(const DiracMatrix &mat);

    /**
     * @brief Full Rayleigh-Ritz evolution with the normal operator.
     *
     * Applies mat to each pool vector, builds projected matrix,
     * diagonalizes, rotates. Returns the k x k rotation matrix
     * in column-major layout (for use by EigenForecast).
     *
     * @param mat  Normal operator D_hat^dag D_hat (DiracMdagM wrapper)
     * @return k x k rotation matrix in column-major std::vector<Complex>
     */
    std::vector<Complex> rayleighRitzEvolve(const DiracMatrix &mat);

    /**
     * @brief Compute max eigenvector residual ||M v_i - lambda_i v_i|| / ||M v_i||.
     *
     * Cost: nEv matvecs of the normal operator.
     *
     * @param mat  Normal operator D_hat^dag D_hat
     * @return Maximum relative residual across first nEv vectors
     */
    double maxResidual(const DiracMatrix &mat);

    /** @brief Query whether the tracker has been initialized */
    bool isInitialized() const { return initialized_; }

    /** @brief Get current pool size */
    int poolSize() const { return static_cast<int>(pool_.size()); }

    /** @brief Get number of tracked eigenpairs */
    int getNEv() const { return nEv_; }

    /** @brief Read-only access to eigenvalues */
    const std::vector<Complex> &getEvals() const { return eigvals_; }

    /** @brief Read-only access to pool vectors */
    const std::vector<ColorSpinorField> &getPool() const { return pool_; }

    /** @brief Read-only access to Dpool vectors */
    const std::vector<ColorSpinorField> &getDpool() const { return Dpool_; }

    /** @brief Mutable access to pool vectors (for rotation by EigenForecast) */
    std::vector<ColorSpinorField> &getPoolMutable() { return pool_; }

    /** @brief Mutable access to Dpool vectors (for rotation by EigenForecast) */
    std::vector<ColorSpinorField> &getDpoolMutable() { return Dpool_; }
  };

} // namespace quda
