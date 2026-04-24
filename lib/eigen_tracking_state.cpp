/**
 * @file eigen_tracking_state.cpp
 * @brief Orchestrator for eigenspace tracking during HMC.
 *
 * Manages the lifecycle of EigenTracker, EigenForecast, and
 * chronological solution history.
 */

#include <eigen_tracking_state.h>
#include <eigensolve_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <quda_internal.h>
#include <cstdio>

// gaugePrecise is at file scope (not in quda namespace) in interface_quda.cpp
extern quda::GaugeField *gaugePrecise;

namespace quda
{

  EigenTrackingState::EigenTrackingState() : invParamSet_(false), trajectoryCount_(0), totalRitzAbsorbed_(0), totalCGSolves_(0) { }

  void EigenTrackingState::configure(const EigenTrackingParam &p)
  {
    param_ = p;
    forecast_ = EigenForecast(p.nEv, p.forecastOrder);
    logQuda(QUDA_VERBOSE,
            "EigenTrackingState: configured (nEv=%d, pool=%d, nRitz=%d, forecast=%d, freshInterval=%d, history=%d)\n",
            p.nEv, p.poolCapacity, p.nRitz, p.forecastOrder, p.freshTRLMInterval, p.solutionHistoryDepth);
  }

  void EigenTrackingState::maybeInit(const DiracMatrix &mat, const DiracMatrix &matHalf,
                                     QudaInvertParam &inv_param)
  {
    if (tracker_.isInitialized()) return;

    // Cache inv_param for future re-initialization (periodic TRLM refresh)
    invParam_ = inv_param;
    invParamSet_ = true;

    logQuda(QUDA_SUMMARIZE, "EigenTrackingState: running initial TRLM (nEv=%d)\n", param_.nEv);

    // Configure TRLM eigensolver
    int nKr = 3 * param_.nEv;
    QudaEigParam ep;
    memset(&ep, 0, sizeof(QudaEigParam));
    ep.struct_size = sizeof(QudaEigParam);
    ep.invert_param = &inv_param;
    ep.eig_type = QUDA_EIG_TR_LANCZOS;
    ep.spectrum = QUDA_SPECTRUM_SR_EIG;
    ep.n_ev = param_.nEv;
    ep.n_kr = nKr;
    ep.n_conv = param_.nEv;
    ep.n_ev_deflate = param_.nEv;
    ep.tol = 1e-6;
    ep.max_restarts = 100;
    ep.require_convergence = QUDA_BOOLEAN_TRUE;
    ep.use_norm_op = QUDA_BOOLEAN_FALSE;
    ep.use_dagger = QUDA_BOOLEAN_FALSE;
    ep.use_pc = QUDA_BOOLEAN_FALSE;
    // Optional Chebyshev acceleration for ill-conditioned M^dag M spectra
    // (hot-start gauges, critical-mass runs without MG preconditioning).
    ep.use_poly_acc = param_.usePolyAcc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    if (param_.usePolyAcc) {
      ep.poly_deg = param_.polyDeg;
      ep.a_min = param_.aMin;
      ep.a_max = param_.aMax; // 0 => QUDA power-iteration auto-estimate
    }
    ep.compute_svd = QUDA_BOOLEAN_FALSE;
    ep.compute_gamma5 = QUDA_BOOLEAN_FALSE;
    ep.batched_rotate = 0;
    ep.preserve_deflation = QUDA_BOOLEAN_FALSE;
    ep.check_interval = 10;
    ep.max_ortho_attempts = 10;
    ep.ortho_block_size = 0;
    ep.compute_evals_batch_size = param_.nEv;

    auto *eigSolve = EigenSolver::create(&ep, mat);

    // Pre-allocate kSpace — the TRLM expects at least one vector to derive field params.
    // Use the same pattern as generatePseudofermion to create parity spinors.
    ColorSpinorParam csParam(nullptr, inv_param, ::gaugePrecise->X(), true, QUDA_CUDA_FIELD_LOCATION);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(gaugePrecise->Precision());
    csParam.fieldOrder = QUDA_NATIVE_FIELD_ORDER;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    std::vector<ColorSpinorField> kSpace(nKr, csParam);
    std::vector<Complex> evals(param_.nEv);

    (*eigSolve)(kSpace, evals);
    delete eigSolve;

    // Initialize the tracker from the TRLM result
    tracker_.init(kSpace, evals, matHalf, param_.nEv, param_.poolCapacity);
    forecast_.setK(param_.nEv);

    logQuda(QUDA_SUMMARIZE, "EigenTrackingState: initialized. Smallest eval = %e\n", evals[0].real());
  }

  void EigenTrackingState::seedFromMGNullVectors(std::vector<ColorSpinorField> &evenVecs,
                                                   const DiracMatrix &matHalf, QudaInvertParam &inv_param)
  {
    if (tracker_.isInitialized()) {
      logQuda(QUDA_VERBOSE, "EigenTrackingState: tracker already initialized, skipping MG seed\n");
      return;
    }

    invParam_ = inv_param;
    invParamSet_ = true;

    int nVec = static_cast<int>(evenVecs.size());
    logQuda(QUDA_SUMMARIZE, "EigenTrackingState: seeding from %d MG null vectors\n", nVec);

    // Use the null vectors as initial eigenvector estimates.
    // Assign dummy eigenvalues — compress() will compute real Rayleigh quotients.
    std::vector<Complex> dummyEvals(nVec, Complex(1.0, 0.0));

    // Initialize tracker with these vectors
    tracker_.init(evenVecs, dummyEvals, matHalf, param_.nEv, param_.poolCapacity);
    forecast_.setK(std::min(param_.nEv, tracker_.poolSize()));

    // Compress to get proper eigenvalue ordering via Rayleigh-Ritz
    tracker_.compress();

    logQuda(QUDA_SUMMARIZE, "EigenTrackingState: seeded from MG. Pool=%d, smallest eval=%e\n", tracker_.poolSize(),
            tracker_.getEvals().size() > 0 ? tracker_.getEvals()[0].real() : 0.0);
  }

  void EigenTrackingState::stashRitzVectors(std::vector<ColorSpinorField> &&vecs)
  {
    for (auto &v : vecs) ritzStash_.push_back(std::move(v));
    logQuda(QUDA_VERBOSE, "EigenTrackingState: stashed %lu Ritz vectors (total stash = %lu)\n", vecs.size(),
            ritzStash_.size());
  }

  void EigenTrackingState::afterCGSolve(const ColorSpinorField &sol, const DiracMatrix &matHalf)
  {
    totalCGSolves_++;

    // Absorb the normalized CG solution into the tracker pool.
    // Ritz extraction is now done externally via CGTracker (zero cost).
    if (tracker_.isInitialized()) {
      ColorSpinorField solNorm(sol);
      double snrm = sqrt(blas::norm2(solNorm));
      if (snrm > 1e-30) {
        blas::ax(1.0 / snrm, solNorm);
        std::vector<ColorSpinorField> solVec;
        solVec.push_back(std::move(solNorm));
        tracker_.absorb(solVec, matHalf);
      }
    }
  }

  void EigenTrackingState::afterGaugeUpdate(const DiracMatrix &matHalf)
  {
    if (!tracker_.isInitialized()) return;
    tracker_.forceUpdate(matHalf);
  }

  void EigenTrackingState::betweenTrajectories(const DiracMatrix &mat, const DiracMatrix &matHalf)
  {
    if (!tracker_.isInitialized()) return;

    trajectoryCount_++;

    // 1. Refresh Dpool with current operator (once per trajectory, not per gauge step)
    tracker_.forceUpdate(matHalf);

    // 2. Batch-absorb stashed Ritz vectors from CG solves during the trajectory
    if (!ritzStash_.empty()) {
      logQuda(QUDA_SUMMARIZE, "EigenTrackingState: absorbing %lu stashed Ritz vectors\n", ritzStash_.size());
      int absorbed = tracker_.absorb(ritzStash_, matHalf);
      totalRitzAbsorbed_ += absorbed;
      ritzStash_.clear();
    }

    // 3. Full RR evolution to get rotation matrix
    auto rotation = tracker_.rayleighRitzEvolve(mat);

    // 4. Recompute Dpool after RR rotation (RR rotates pool but Dpool needs refresh)
    tracker_.forceUpdate(matHalf);

    // 5. Record rotation in forecast state
    int k = tracker_.poolSize();
    if (!rotation.empty()) { forecast_.recordRotation(rotation, k); }

    // 6. Periodic fresh TRLM
    if (param_.freshTRLMInterval > 0 && trajectoryCount_ % param_.freshTRLMInterval == 0) {
      logQuda(QUDA_SUMMARIZE, "EigenTrackingState: periodic fresh TRLM at trajectory %d\n", trajectoryCount_);
      tracker_ = EigenTracker();
      forecast_.reset();
      if (invParamSet_) maybeInit(mat, matHalf, invParam_);
    }

    logQuda(QUDA_VERBOSE, "EigenTrackingState: betweenTrajectories complete (trajectory %d)\n", trajectoryCount_);
  }

  void EigenTrackingState::beforeTrajectory(const DiracMatrix &matHalf)
  {
    if (!tracker_.isInitialized()) return;

    // Apply forecast pre-rotation if we have sufficient history
    auto R = forecast_.forecastRotation();
    if (!R.empty()) {
      int k = tracker_.poolSize();
      EigenForecast::applyRotation(tracker_.getPoolMutable(), R, k);
      EigenForecast::applyRotation(tracker_.getDpoolMutable(), R, k);

      // Recompute Dpool with current operator (rotation was approximate)
      tracker_.forceUpdate(matHalf);

      logQuda(QUDA_SUMMARIZE, "EigenTrackingState: applied forecast pre-rotation\n");
    }
  }

  bool EigenTrackingState::getExtrapolatedX0(ColorSpinorField &x0) const
  {
    int n = static_cast<int>(solutionHistory_.size());
    if (n == 0) return false;

    // Binomial extrapolation from solution history
    // depth 1: x0 = x_{n-1}
    // depth 2: x0 = 2*x_{n-1} - x_{n-2}
    // depth 3: x0 = 3*x_{n-1} - 3*x_{n-2} + x_{n-3}
    blas::zero(x0);
    if (n >= 3) {
      blas::axpy(3.0, solutionHistory_[0], x0);
      blas::axpy(-3.0, solutionHistory_[1], x0);
      blas::axpy(1.0, solutionHistory_[2], x0);
    } else if (n >= 2) {
      blas::axpy(2.0, solutionHistory_[0], x0);
      blas::axpy(-1.0, solutionHistory_[1], x0);
    } else {
      blas::copy(x0, solutionHistory_[0]);
    }

    return true;
  }

  std::string EigenTrackingState::getSummary() const
  {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "EigenTracking: trajectories=%d, CG solves=%d, Ritz absorbed=%d, "
             "pool=%d/%d, forecast history=%d",
             trajectoryCount_, totalCGSolves_, totalRitzAbsorbed_, tracker_.poolSize(),
             tracker_.isInitialized() ? tracker_.getNEv() : 0, forecast_.historyLength());
    return std::string(buf);
  }

} // namespace quda
