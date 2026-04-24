/**
 * @file eigen_tracker.cpp
 * @brief Pool-based eigenvector tracker for HMC eigenspace maintenance.
 *
 * Ported from Schwinger_MG/src/eigen_forecast.cpp (EigenTracker).
 * Follows the RR pattern from coarse_deflation_manager.cpp.
 */

#include <eigen_tracker.h>
#include <eigensolve_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <quda_internal.h>
#include <timer.h>

namespace quda
{

  static TimeProfile profileEigenTracker("EigenTracker");

  EigenTracker::EigenTracker() : nEv_(0), poolCapacity_(0), initialized_(false) { }

  void EigenTracker::init(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals,
                          const DiracMatrix &mat, int nEv, int capacity)
  {
    auto profile = pushProfile(profileEigenTracker);
    if (capacity < nEv) errorQuda("EigenTracker: capacity (%d) must be >= nEv (%d)", capacity, nEv);

    nEv_ = nEv;
    poolCapacity_ = capacity;

    int kHave = std::min(static_cast<int>(kSpace.size()), capacity);

    // Copy eigenvectors into pool and compute D_hat * v for each
    pool_.resize(kHave);
    Dpool_.resize(kHave);
    for (int i = 0; i < kHave; i++) {
      pool_[i] = kSpace[i]; // GPU copy
      ColorSpinorParam param(kSpace[i]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      Dpool_[i] = ColorSpinorField(param);
      mat(Dpool_[i], pool_[i]);
    }

    // Store eigenvalues
    eigvals_.resize(std::min(nEv, kHave));
    for (int i = 0; i < static_cast<int>(eigvals_.size()); i++) { eigvals_[i] = evals[i]; }

    initialized_ = true;

    logQuda(QUDA_VERBOSE, "EigenTracker: initialized with %d pool vectors, nEv=%d, capacity=%d\n", kHave, nEv, capacity);
  }

  void EigenTracker::compress()
  {
    auto profile = pushProfile(profileEigenTracker);
    if (!initialized_) return;
    int k = static_cast<int>(pool_.size());
    if (k <= 1) return;

    logQuda(QUDA_VERBOSE, "EigenTracker: compressing pool of %d vectors (capacity=%d)\n", k, poolCapacity_);

    // Build k x k projected matrix: T_ij = Dpool[i]^dag Dpool[j]
    // This gives eigenvalues of D_hat^dag D_hat within the subspace
    MatrixXcd T = MatrixXcd::Zero(k, k);
    for (int j = 0; j < k; j++) {
      std::vector<Complex> dots(k);
      blas::block::cDotProduct(dots, {pool_.begin(), pool_.begin() + k}, Dpool_[j]);
      // dots[i] = pool_[i]^dag Dpool_[j], but we need Dpool_[i]^dag Dpool_[j]
      // Re-do with Dpool as the left operand
      blas::block::cDotProduct(dots, {Dpool_.begin(), Dpool_.begin() + k}, Dpool_[j]);
      for (int i = 0; i < k; i++) { T(i, j) = std::complex<double>(dots[i].real(), dots[i].imag()); }
    }

    // Enforce Hermiticity (fix floating-point asymmetry)
    T = (T + T.adjoint()) / 2.0;

    // Diagonalize on host
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(T);

    // Keep at most poolCapacity vectors (smallest eigenvalues first)
    int keep = std::min(k, poolCapacity_);

    // Allocate workspace for rotated vectors
    std::vector<ColorSpinorField> newPool(keep);
    std::vector<ColorSpinorField> newDpool(keep);
    for (int i = 0; i < keep; i++) {
      ColorSpinorParam param(pool_[0]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      newPool[i] = ColorSpinorField(param);
      newDpool[i] = ColorSpinorField(param);
    }

    // Rotate: new_v_i = sum_j U_{ji} * old_v_j
    for (int i = 0; i < keep; i++) {
      for (int j = 0; j < k; j++) {
        Complex alpha(eigensolver.eigenvectors().col(i)[j].real(), eigensolver.eigenvectors().col(i)[j].imag());
        blas::caxpy(alpha, pool_[j], newPool[i]);
        blas::caxpy(alpha, Dpool_[j], newDpool[i]);
      }
      // Renormalize pool vector (Dpool follows the same rotation)
      double nv = sqrt(blas::norm2(newPool[i]));
      if (nv > 1e-14 && std::abs(nv - 1.0) > 1e-10) {
        blas::ax(1.0 / nv, newPool[i]);
        blas::ax(1.0 / nv, newDpool[i]);
      }
    }

    pool_ = std::move(newPool);
    Dpool_ = std::move(newDpool);

    // Update eigenvalue estimates
    eigvals_.resize(std::min(nEv_, keep));
    for (int i = 0; i < static_cast<int>(eigvals_.size()); i++) {
      eigvals_[i] = Complex(eigensolver.eigenvalues()[i], 0.0);
    }

    logQuda(QUDA_VERBOSE, "EigenTracker: compressed to %d vectors. Smallest eval = %e, largest = %e\n", keep,
            eigvals_[0].real(), eigvals_[std::min(nEv_, keep) - 1].real());
  }

  int EigenTracker::absorb(std::vector<ColorSpinorField> &newVecs, const DiracMatrix &mat)
  {
    auto profile = pushProfile(profileEigenTracker);
    if (!initialized_) return 0;
    int absorbed = 0;

    for (auto &vIn : newVecs) {
      if (static_cast<int>(pool_.size()) >= poolCapacity_ + static_cast<int>(newVecs.size())) break;

      // Copy the input vector so we don't modify it
      ColorSpinorField v(vIn);

      // Orthogonalize against existing pool (double MGS for stability)
      int poolSz = static_cast<int>(pool_.size());
      for (int pass = 0; pass < 2; pass++) {
        for (int j = 0; j < poolSz; j++) {
          Complex proj = blas::cDotProduct(pool_[j], v);
          blas::caxpy(-proj, pool_[j], v);
        }
      }

      double nv = sqrt(blas::norm2(v));
      if (nv < 0.1) continue; // too collinear — skip

      blas::ax(1.0 / nv, v);

      // Compute D_hat * v_new
      ColorSpinorParam param(v);
      param.create = QUDA_ZERO_FIELD_CREATE;
      ColorSpinorField Dv(param);
      mat(Dv, v);

      pool_.push_back(std::move(v));
      Dpool_.push_back(std::move(Dv));
      absorbed++;
    }

    // If pool exceeds capacity, compress
    if (static_cast<int>(pool_.size()) > poolCapacity_) compress();

    logQuda(QUDA_SUMMARIZE, "EigenTracker: absorbed %d vectors, pool size = %d\n", absorbed,
            static_cast<int>(pool_.size()));
    return absorbed;
  }

  void EigenTracker::forceUpdate(const DiracMatrix &mat)
  {
    auto profile = pushProfile(profileEigenTracker);
    if (!initialized_) return;
    int k = static_cast<int>(pool_.size());

    logQuda(QUDA_SUMMARIZE, "EigenTracker: forceUpdate — recomputing Dpool for %d vectors\n", k);

    // Recompute Dpool with the new operator
    for (int i = 0; i < k; i++) { mat(Dpool_[i], pool_[i]); }

    // Note: no compress() here. Compress is only triggered by absorb() when pool
    // exceeds capacity. The Dpool is refreshed for use by future compress() calls.
  }

  std::vector<Complex> EigenTracker::rayleighRitzEvolve(const DiracMatrix &mat)
  {
    auto profile = pushProfile(profileEigenTracker);
    if (!initialized_) return {};
    int k = static_cast<int>(pool_.size());

    logQuda(QUDA_VERBOSE, "EigenTracker: RR evolution with %d pool vectors\n", k);

    // Compute A * v_i for each pool vector (k matvecs of the normal operator)
    std::vector<ColorSpinorField> AV(k);
    for (int i = 0; i < k; i++) {
      ColorSpinorParam param(pool_[0]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      AV[i] = ColorSpinorField(param);
      mat(AV[i], pool_[i]);
    }

    // Build k x k projected matrix: H_ij = pool[i]^dag AV[j]
    MatrixXcd H = MatrixXcd::Zero(k, k);
    for (int j = 0; j < k; j++) {
      std::vector<Complex> dots(k);
      blas::block::cDotProduct(dots, {pool_.begin(), pool_.begin() + k}, AV[j]);
      for (int i = 0; i < k; i++) { H(i, j) = std::complex<double>(dots[i].real(), dots[i].imag()); }
    }

    // Enforce Hermiticity
    H = (H + H.adjoint()) / 2.0;

    // Diagonalize on host
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(H);

    // Build rotation matrix in column-major layout for return
    std::vector<Complex> rotation(k * k);
    for (int col = 0; col < k; col++) {
      for (int row = 0; row < k; row++) {
        auto c = eigensolver.eigenvectors().col(col)[row];
        rotation[col * k + row] = Complex(c.real(), c.imag());
      }
    }

    // Update eigenvalues
    eigvals_.resize(std::min(nEv_, k));
    for (int i = 0; i < static_cast<int>(eigvals_.size()); i++) {
      eigvals_[i] = Complex(eigensolver.eigenvalues()[i], 0.0);
    }

    // Rotate pool AND Dpool vectors: new_v_i = sum_j U_{ji} * old_v_j
    // Dpool is rotated with the same matrix. The result is approximate (Dpool
    // was computed with the old D), but the caller will call forceUpdate() to
    // replace Dpool with exact values from the current operator.
    std::vector<ColorSpinorField> newPool(k);
    std::vector<ColorSpinorField> newDpool(k);
    for (int i = 0; i < k; i++) {
      ColorSpinorParam param(pool_[0]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      newPool[i] = ColorSpinorField(param);
      newDpool[i] = ColorSpinorField(param);
      for (int j = 0; j < k; j++) {
        blas::caxpy(rotation[i * k + j], pool_[j], newPool[i]);
        blas::caxpy(rotation[i * k + j], Dpool_[j], newDpool[i]);
      }
    }
    pool_ = std::move(newPool);
    Dpool_ = std::move(newDpool);

    logQuda(QUDA_SUMMARIZE, "EigenTracker: RR evolution complete. Smallest eval = %e, largest = %e\n",
            eigvals_[0].real(), eigvals_[std::min(nEv_, k) - 1].real());

    return rotation;
  }

  double EigenTracker::maxResidual(const DiracMatrix &mat)
  {
    if (!initialized_) return -1.0;
    int k = std::min(nEv_, static_cast<int>(pool_.size()));

    double maxRes = 0.0;
    ColorSpinorParam param(pool_[0]);
    param.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField Av(param);
    ColorSpinorField resid(param);

    for (int i = 0; i < k; i++) {
      // Av = M * v_i
      mat(Av, pool_[i]);
      double avNorm = sqrt(blas::norm2(Av));

      // resid = Av - lambda_i * v_i
      blas::copy(resid, Av);
      blas::caxpy(-eigvals_[i], pool_[i], resid);

      double relRes = sqrt(blas::norm2(resid)) / std::max(avNorm, 1e-30);
      maxRes = std::max(maxRes, relRes);
    }

    logQuda(QUDA_VERBOSE, "EigenTracker: max residual (nEv=%d) = %e\n", k, maxRes);
    return maxRes;
  }

} // namespace quda
