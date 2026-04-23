/**
 * @file eigen_forecast.cpp
 * @brief Chronological generator forecasting for eigenspace rotation prediction.
 *
 * Ported from Schwinger_MG/src/eigen_forecast.cpp (extract_generator,
 * forecast_rotation, apply_rotation).
 */

#include <eigen_forecast.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <quda_internal.h>
#include <cmath>

namespace quda
{

  EigenForecast::EigenForecast(int k, int forecastOrder) : k_(k), forecastOrder_(forecastOrder) { }

  void EigenForecast::reset()
  {
    generatorHistory_.clear();
    logQuda(QUDA_VERBOSE, "EigenForecast: history reset\n");
  }

  void EigenForecast::recordRotation(const std::vector<Complex> &rotation, int k)
  {
    if (k <= 0 || static_cast<int>(rotation.size()) != k * k) {
      logQuda(QUDA_VERBOSE, "EigenForecast: skipping recordRotation (k=%d, rotation size=%d)\n", k,
              static_cast<int>(rotation.size()));
      return;
    }

    // If dimension changed, reset history (old generators are incompatible)
    if (k != k_) {
      generatorHistory_.clear();
      logQuda(QUDA_VERBOSE, "EigenForecast: k changed from %d to %d, resetting history\n", k_, k);
    }
    k_ = k;

    // Convert column-major rotation vector to Eigen matrix U
    MatrixXcd U = MatrixXcd::Zero(k, k);
    for (int col = 0; col < k; col++)
      for (int row = 0; row < k; row++) {
        auto &c = rotation[col * k + row];
        U(row, col) = std::complex<double>(c.real(), c.imag());
      }

    // Extract Hermitian generator H = -i log(U)
    // Method: diagonalize S = (U - U†)/(2i) to get sin(θ),
    // then compute exact angles via atan2.

    // Build Hermitian S = (U - U†) / (2i)
    MatrixXcd S = (U - U.adjoint()) / std::complex<double>(0.0, 2.0);
    // Enforce exact Hermiticity
    S = (S + S.adjoint()) / 2.0;

    // Diagonalize S
    SelfAdjointEigenSolver<MatrixXcd> eigS(S);
    MatrixXcd V = eigS.eigenvectors(); // columns = eigenvectors

    // For each eigenvector v_j of S, compute the eigenvalue of U:
    // u_j = v_j† U v_j = cos(θ_j) + i sin(θ_j)
    // Then θ_j = atan2(Im(u_j), Re(u_j))
    VectorXd theta(k);
    for (int j = 0; j < k; j++) {
      std::complex<double> u_j = (V.col(j).adjoint() * U * V.col(j))(0, 0);
      theta(j) = std::atan2(u_j.imag(), u_j.real());
    }

    // Reconstruct H = V diag(θ) V†
    MatrixXcd H = V * theta.asDiagonal() * V.adjoint();
    // Enforce Hermiticity
    H = (H + H.adjoint()) / 2.0;

    // Push into history buffer (circular, max maxHistory_)
    if (static_cast<int>(generatorHistory_.size()) >= maxHistory_) {
      generatorHistory_.erase(generatorHistory_.begin());
    }
    generatorHistory_.push_back(H);

    logQuda(QUDA_VERBOSE, "EigenForecast: recorded generator (history length = %d)\n",
            static_cast<int>(generatorHistory_.size()));
  }

  std::vector<Complex> EigenForecast::forecastRotation() const
  {
    int n = static_cast<int>(generatorHistory_.size());
    if (n < 1 || k_ <= 0) return {};

    // Verify all generators have consistent dimension
    for (const auto &H : generatorHistory_) {
      if (H.rows() != k_ || H.cols() != k_) return {};
    }

    // Determine effective order based on available history
    int order = std::min(forecastOrder_, n - 1);

    // Extrapolate H_pred
    MatrixXcd H_pred = MatrixXcd::Zero(k_, k_);
    if (order >= 2 && n >= 3) {
      // Quadratic: 3*H[n-1] - 3*H[n-2] + H[n-3]
      H_pred = 3.0 * generatorHistory_[n - 1] - 3.0 * generatorHistory_[n - 2] + generatorHistory_[n - 3];
    } else if (order >= 1 && n >= 2) {
      // Linear: 2*H[n-1] - H[n-2]
      H_pred = 2.0 * generatorHistory_[n - 1] - generatorHistory_[n - 2];
    } else {
      // Constant: just use the last generator
      H_pred = generatorHistory_[n - 1];
    }

    // Enforce Hermiticity
    H_pred = (H_pred + H_pred.adjoint()) / 2.0;

    // Compute R = exp(i H_pred) via eigendecomposition
    SelfAdjointEigenSolver<MatrixXcd> eigH(H_pred);
    // R = V diag(exp(i θ)) V†
    VectorXcd expTheta(k_);
    for (int j = 0; j < k_; j++) {
      expTheta(j) = std::exp(std::complex<double>(0.0, eigH.eigenvalues()(j)));
    }
    MatrixXcd R = eigH.eigenvectors() * expTheta.asDiagonal() * eigH.eigenvectors().adjoint();

    // Convert to column-major layout
    std::vector<Complex> result(k_ * k_);
    for (int col = 0; col < k_; col++)
      for (int row = 0; row < k_; row++) { result[col * k_ + row] = Complex(R(row, col).real(), R(row, col).imag()); }

    logQuda(QUDA_VERBOSE, "EigenForecast: forecast rotation computed (order=%d, history=%d)\n", order, n);
    return result;
  }

  void EigenForecast::applyRotation(std::vector<ColorSpinorField> &vecs,
                                     const std::vector<Complex> &rotation, int k)
  {
    if (k <= 0 || static_cast<int>(rotation.size()) != k * k) return;
    if (static_cast<int>(vecs.size()) < k) return;

    // Allocate workspace
    std::vector<ColorSpinorField> work(k);
    for (int i = 0; i < k; i++) {
      ColorSpinorParam param(vecs[0]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      work[i] = ColorSpinorField(param);
    }

    // Rotate: work[i] = sum_j R[i*k + j] * vecs[j]  (column-major: R[col*k + row])
    // We want new_v_i = sum_j U_{ji} old_v_j = sum_j rotation[i*k + j] old_v_j
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        blas::caxpy(rotation[i * k + j], vecs[j], work[i]);
      }
    }

    // Copy back
    for (int i = 0; i < k; i++) { blas::copy(vecs[i], work[i]); }

    logQuda(QUDA_VERBOSE, "EigenForecast: applied %d x %d rotation to vectors\n", k, k);
  }

} // namespace quda
