#pragma once

/**
 * @file eigen_forecast.h
 * @brief Chronological generator forecasting for eigenspace rotation prediction.
 *
 * Ported from Schwinger_MG/src/eigen_forecast.cpp.
 */

#include <color_spinor_field.h>
#include <eigen_helper.h>
#include <blas_quda.h>
#include <quda_internal.h>
#include <vector>

namespace quda
{

  /**
   * @brief Chronological generator forecasting for eigenspace rotation prediction.
   *
   * Tracks the Hermitian generator H of the RR rotation U = exp(iH) across
   * trajectories. Extrapolates H to predict the next rotation, enabling
   * pre-rotation of eigenvectors before RR to improve tracking accuracy.
   */
  class EigenForecast {
  private:
    int k_;                                      /**< Eigenvector count (matrix dimension) */
    int forecastOrder_;                           /**< 0=constant, 1=linear, 2=quadratic */
    static constexpr int maxHistory_ = 3;        /**< Max generators stored (quadratic needs 3) */
    std::vector<MatrixXcd> generatorHistory_;    /**< Circular buffer of k x k Hermitian generators */

  public:
    /**
     * @brief Construct the forecast state.
     * @param k              Eigenvector count
     * @param forecastOrder  Extrapolation order: 0=constant, 1=linear, 2=quadratic
     */
    EigenForecast(int k = 0, int forecastOrder = 1);

    /**
     * @brief Record a rotation matrix from a Rayleigh-Ritz step.
     *
     * Extracts the Hermitian generator H = -i log(U) from the unitary
     * rotation matrix U, and stores it in the history buffer.
     *
     * @param rotation  k x k unitary rotation matrix (column-major, std::vector<Complex>)
     * @param k         Dimension of the rotation matrix
     */
    void recordRotation(const std::vector<Complex> &rotation, int k);

    /**
     * @brief Predict the next rotation matrix.
     *
     * Uses polynomial extrapolation of generator history:
     *   order 0: H_pred = H[last]
     *   order 1: H_pred = 2*H[n-1] - H[n-2]
     *   order 2: H_pred = 3*H[n-1] - 3*H[n-2] + H[n-3]
     * Then R_pred = exp(i * H_pred).
     *
     * @return k x k unitary matrix in column-major layout, or empty if insufficient history
     */
    std::vector<Complex> forecastRotation() const;

    /**
     * @brief Apply a k x k rotation to a set of GPU vectors.
     *
     * new_v_i = sum_j R_{ji} * old_v_j.
     * Uses workspace vectors and blas::caxpy.
     *
     * @param vecs     Vector of GPU ColorSpinorFields to rotate in-place
     * @param rotation k x k matrix in column-major layout
     * @param k        Dimension (number of vectors to rotate, may be <= vecs.size())
     */
    static void applyRotation(std::vector<ColorSpinorField> &vecs,
                               const std::vector<Complex> &rotation, int k);

    /** @brief Reset history */
    void reset();

    /** @brief Get history length */
    int historyLength() const { return static_cast<int>(generatorHistory_.size()); }

    /** @brief Set the eigenvector count */
    void setK(int newK)
    {
      k_ = newK;
      if (k_ <= 0) reset();
    }
  };

} // namespace quda
