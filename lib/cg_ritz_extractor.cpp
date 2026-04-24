/**
 * @file cg_ritz_extractor.cpp
 * @brief Zero-cost CG-Ritz extraction via the CG-Lanczos equivalence.
 *
 * CGTracker records alpha/beta/residuals during the CG solve.
 * extractRitzPairs builds the Lanczos tridiagonal and eigendecomposes it.
 * CGRitzExtractor::extract is the legacy wrapper (now unused).
 */

#include <cg_ritz_extractor.h>
#include <cg_tracker.h>
#include <eigensolve_quda.h>
#include <eigen_helper.h>
#include <blas_quda.h>
#include <quda_internal.h>
#include <timer.h>
#include <algorithm>
#include <cmath>

namespace quda
{

  static TimeProfile profileCGRitz("CGRitzExtractor");

  // Global active tracker pointer (set before solve, read by CG, cleared after)
  CGTracker *activeCGTracker = nullptr;

  // ===================================================================
  // CGTracker implementation
  // ===================================================================

  CGTracker::CGTracker(int nRitz, int maxLanczos) :
    nRitz_(nRitz), maxLanczos_(maxLanczos > 0 ? maxLanczos : 3 * nRitz), active_(nRitz > 0)
  {
  }

  void CGTracker::reset()
  {
    alphas_.clear();
    betas_.clear();
    lanczosVecs_.clear();
  }

  void CGTracker::recordIteration(double alpha, double beta, const ColorSpinorField &r)
  {
    if (!active_) return;

    alphas_.push_back(alpha);
    betas_.push_back(beta);

    // Store normalized residual as Lanczos vector (capped at maxLanczos_)
    if (static_cast<int>(lanczosVecs_.size()) < maxLanczos_) {
      double rnorm = sqrt(blas::norm2(r));
      if (rnorm > 1e-30) {
        ColorSpinorField q(r); // copy
        blas::ax(1.0 / rnorm, q);
        lanczosVecs_.push_back(std::move(q));
      }
    }
  }

  void CGTracker::extractRitzPairs(std::vector<ColorSpinorField> &vecs, std::vector<Complex> &vals)
  {
    auto profile = pushProfile(profileCGRitz);
    vecs.clear();
    vals.clear();

    if (!active_ || nRitz_ <= 0) return;

    int m = std::min(static_cast<int>(alphas_.size()), static_cast<int>(lanczosVecs_.size()));
    if (m < nRitz_) {
      logQuda(QUDA_VERBOSE, "CGTracker: insufficient data for Ritz extraction (m=%d, nRitz=%d)\n", m, nRitz_);
      return;
    }

    logQuda(QUDA_VERBOSE, "CGTracker: extracting %d Ritz pairs from %d Lanczos vectors\n", nRitz_, m);

    // Build the Lanczos tridiagonal T_m from CG coefficients:
    //   delta_0 = 1/alpha_0
    //   delta_k = 1/alpha_k + beta_{k-1}/alpha_{k-1}  (k > 0)
    //   gamma_k = sqrt(beta_k) / alpha_k
    MatrixXd T = MatrixXd::Zero(m, m);

    T(0, 0) = 1.0 / alphas_[0];
    for (int k = 1; k < m; k++) {
      T(k, k) = 1.0 / alphas_[k] + betas_[k - 1] / alphas_[k - 1];
    }
    for (int k = 0; k < m - 1; k++) {
      double gamma = sqrt(std::max(betas_[k], 0.0)) / alphas_[k];
      T(k + 1, k) = gamma;
      T(k, k + 1) = gamma;
    }

    // Eigendecompose T (real symmetric tridiagonal)
    SelfAdjointEigenSolver<MatrixXd> eigT(T);

    // Extract the smallest nRitz eigenvalues and reconstruct Ritz vectors:
    //   v_p = Q_m u_p = sum_k u_p^(k) * q_k
    int nh = std::min(nRitz_, m);
    vecs.resize(nh);
    vals.resize(nh);

    for (int p = 0; p < nh; p++) {
      vals[p] = Complex(eigT.eigenvalues()(p), 0.0);

      ColorSpinorParam param(lanczosVecs_[0]);
      param.create = QUDA_ZERO_FIELD_CREATE;
      vecs[p] = ColorSpinorField(param);

      for (int k = 0; k < m; k++) {
        Complex coeff(eigT.eigenvectors().col(p)(k), 0.0);
        blas::caxpy(coeff, lanczosVecs_[k], vecs[p]);
      }

      // Normalize
      double nv = sqrt(blas::norm2(vecs[p]));
      if (nv > 1e-14) blas::ax(1.0 / nv, vecs[p]);
    }

    logQuda(QUDA_VERBOSE, "CGTracker: extracted %d Ritz pairs. Smallest Ritz value = %e\n", nh, vals[0].real());
  }

  // ===================================================================
  // Legacy CGRitzExtractor (fallback, not used when CGTracker is active)
  // ===================================================================

  void CGRitzExtractor::extract(std::vector<ColorSpinorField> &ritzVecs, std::vector<Complex> &ritzVals,
                                const ColorSpinorField &sol, const DiracMatrix &mat, int nRitz, int nKr,
                                int maxRestarts, double tol, bool usePolyAcc, int polyDeg, double aMin,
                                double aMax)
  {
    auto profile = pushProfile(profileCGRitz);
    if (nRitz <= 0) return;
    if (nKr <= 0) nKr = std::max(3 * nRitz, nRitz + 14);

    logQuda(QUDA_VERBOSE, "CGRitzExtractor (legacy TRLM): extracting %d Ritz pairs\n", nRitz);

    QudaEigParam ep;
    memset(&ep, 0, sizeof(QudaEigParam));
    ep.eig_type = QUDA_EIG_TR_LANCZOS;
    ep.spectrum = QUDA_SPECTRUM_SR_EIG;
    ep.n_ev = nRitz;
    ep.n_kr = nKr;
    ep.n_conv = nRitz;
    ep.n_ev_deflate = nRitz;
    ep.tol = tol;
    ep.max_restarts = maxRestarts;
    ep.require_convergence = QUDA_BOOLEAN_FALSE;
    ep.use_norm_op = QUDA_BOOLEAN_FALSE;
    ep.use_dagger = QUDA_BOOLEAN_FALSE;
    ep.use_pc = QUDA_BOOLEAN_FALSE;
    ep.use_poly_acc = usePolyAcc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    if (usePolyAcc) {
      ep.poly_deg = polyDeg;
      ep.a_min = aMin;
      ep.a_max = aMax; // 0 => QUDA auto-estimates via power iteration
    }
    ep.compute_svd = QUDA_BOOLEAN_FALSE;
    ep.compute_gamma5 = QUDA_BOOLEAN_FALSE;
    ep.batched_rotate = 0;
    ep.preserve_deflation = QUDA_BOOLEAN_FALSE;
    ep.check_interval = 5;
    ep.max_ortho_attempts = 10;
    ep.ortho_block_size = 0;
    ep.compute_evals_batch_size = nRitz;

    auto *eigSolve = EigenSolver::create(&ep, mat);

    std::vector<ColorSpinorField> kSpace;
    ritzVals.resize(nRitz);

    kSpace.resize(nKr);
    ColorSpinorParam param(sol);
    param.create = QUDA_ZERO_FIELD_CREATE;
    for (int i = 0; i < nKr; i++) { kSpace[i] = ColorSpinorField(param); }

    blas::copy(kSpace[0], sol);
    double nrm = sqrt(blas::norm2(kSpace[0]));
    if (nrm > 1e-30) blas::ax(1.0 / nrm, kSpace[0]);

    (*eigSolve)(kSpace, ritzVals);
    delete eigSolve;

    int nConverged = std::min(nRitz, static_cast<int>(kSpace.size()));
    ritzVecs.resize(nConverged);
    for (int i = 0; i < nConverged; i++) { ritzVecs[i] = kSpace[i]; }
    ritzVals.resize(nConverged);
  }

} // namespace quda
