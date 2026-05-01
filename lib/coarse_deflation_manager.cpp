#include <coarse_deflation_manager.h>
#include <eigensolve_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <hmc_quda.h>

namespace quda
{

  CoarseDeflationManager::CoarseDeflationManager(const Transfer &transfer_, const DiracMatrix &matCoarse_,
                                                 const Dirac &diracCoarse_, int nDefl_, double eigTol, int nKr,
                                                 int maxRestarts, int refreshInterval_) :
    matCoarse(&matCoarse_),
    diracCoarse(&diracCoarse_),
    transfer(&transfer_),
    nDefl(nDefl_),
    refreshInterval(refreshInterval_),
    stepCounter(0)
  {
    if (nKr <= 0) nKr = 3 * nDefl;

    // Configure the TRLM eigensolver parameters
    memset(&eigParam, 0, sizeof(QudaEigParam));
    eigParam.eig_type = QUDA_EIG_TR_LANCZOS;
    eigParam.spectrum = QUDA_SPECTRUM_SR_EIG;
    eigParam.n_ev = nDefl;
    eigParam.n_kr = nKr;
    eigParam.n_conv = nDefl;
    eigParam.n_ev_deflate = nDefl;
    eigParam.tol = eigTol;
    eigParam.max_restarts = maxRestarts;
    eigParam.require_convergence = QUDA_BOOLEAN_TRUE;
    eigParam.use_norm_op = QUDA_BOOLEAN_TRUE; // coarse operator is non-Hermitian; solve M†M
    eigParam.use_dagger = QUDA_BOOLEAN_FALSE;
    eigParam.use_pc = QUDA_BOOLEAN_FALSE;
    eigParam.use_poly_acc = QUDA_BOOLEAN_FALSE;
    eigParam.compute_svd = QUDA_BOOLEAN_FALSE;
    eigParam.compute_gamma5 = QUDA_BOOLEAN_FALSE;
    eigParam.batched_rotate = 0;
    eigParam.preserve_deflation = QUDA_BOOLEAN_FALSE;
    eigParam.check_interval = 10;
    eigParam.max_ortho_attempts = 10;
    eigParam.compute_evals_batch_size = nDefl;

    // Run initial eigensolve
    solve();
  }

  void CoarseDeflationManager::solve()
  {
    auto profile = pushProfile(getEigenTrackProfile());
    ScopedComputePhase _scope_;
    logQuda(QUDA_SUMMARIZE, "CoarseDeflationManager: Running TRLM eigensolver for %d coarse eigenvectors\n", nDefl);

    // Make a local copy of eigParam so the solver can modify it
    QudaEigParam ep = eigParam;

    // The coarse operator is non-Hermitian. TRLM requires Hermitian input.
    // Wrap in DiracMdagM to solve for eigenvectors of M†M.
    DiracMdagM matNorm(*diracCoarse);
    auto *eigSolve = EigenSolver::create(&ep, matNorm);

    // Pre-allocate kSpace — TRLM expects at least one vector for field metadata.
    // Use the Transfer's null-space vectors to create coarse-grid fields via create_coarse().
    int nKr = ep.n_kr;
    if (coarseEvecs.empty()) {
      ColorSpinorField V(transfer->Vectors()); // non-const copy for create_coarse
      auto coarseRef = V.create_coarse(transfer->Geo_bs(), transfer->Spin_bs(), transfer->nvec(),
                                        V.Precision(), QUDA_CUDA_FIELD_LOCATION);
      ColorSpinorParam csParam(coarseRef);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      coarseEvecs.resize(nKr, csParam);
    }
    coarseEvals.resize(nDefl);

    (*eigSolve)(coarseEvecs, coarseEvals);

    delete eigSolve;

    stepCounter = 0;

    logQuda(QUDA_VERBOSE, "CoarseDeflationManager: TRLM converged. Smallest eval = %e, largest eval = %e\n",
            coarseEvals[0].real(), coarseEvals[nDefl - 1].real());
  }

  void CoarseDeflationManager::rayleighRitzUpdate()
  {
    auto profile = pushProfile(getEigenTrackProfile());
    ScopedComputePhase _scope_;
    logQuda(QUDA_VERBOSE, "CoarseDeflationManager: Performing Rayleigh-Ritz update\n");

    // Allocate workspace if needed
    if ((int)workVecs.size() != nDefl) { resize(workVecs, nDefl, QUDA_ZERO_FIELD_CREATE, coarseEvecs[0]); }

    // Apply coarse operator to each eigenvector: w_k = A_c v_k
    for (int k = 0; k < nDefl; k++) { (*matCoarse)(workVecs[k], coarseEvecs[k]); }

    // Build the projected Hamiltonian H_ij = <v_i | w_j> = <v_i | A_c | v_j>
    MatrixXcd H = MatrixXcd::Zero(nDefl, nDefl);
    for (int j = 0; j < nDefl; j++) {
      std::vector<Complex> dots(nDefl);
      blas::block::cDotProduct(dots, {coarseEvecs.begin(), coarseEvecs.begin() + nDefl}, workVecs[j]);
      for (int i = 0; i < nDefl; i++) { H(i, j) = std::complex<double>(dots[i].real(), dots[i].imag()); }
    }

    // Diagonalize on the host
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(H);

    // Update eigenvalues
    for (int k = 0; k < nDefl; k++) { coarseEvals[k] = Complex(eigensolver.eigenvalues()[k], 0.0); }

    // Rotate eigenvectors: v_k^new = sum_j U_{jk} v_j
    // Use workVecs as temporary storage for the rotated vectors
    for (int k = 0; k < nDefl; k++) { blas::zero(workVecs[k]); }

    for (int k = 0; k < nDefl; k++) {
      for (int j = 0; j < nDefl; j++) {
        Complex alpha(eigensolver.eigenvectors().col(k)[j].real(), eigensolver.eigenvectors().col(k)[j].imag());
        blas::caxpy(alpha, coarseEvecs[j], workVecs[k]);
      }
    }

    // Copy rotated vectors back
    for (int k = 0; k < nDefl; k++) { blas::copy(coarseEvecs[k], workVecs[k]); }

    logQuda(QUDA_VERBOSE, "CoarseDeflationManager: RR update complete. Smallest eval = %e, largest = %e\n",
            coarseEvals[0].real(), coarseEvals[nDefl - 1].real());
  }

  void CoarseDeflationManager::rebindCoarseRefs(const Transfer &transfer_, const DiracMatrix &matCoarse_,
                                                const Dirac &diracCoarse_)
  {
    transfer = &transfer_;
    matCoarse = &matCoarse_;
    diracCoarse = &diracCoarse_;
    stepCounter = 0;
  }

  void CoarseDeflationManager::maybeRefresh()
  {
    if (refreshInterval <= 0) return;
    if (stepCounter < refreshInterval) return;
    auto profile = pushProfile(getEigenTrackProfile());
    ScopedComputePhase _scope_;

    logQuda(QUDA_VERBOSE, "CoarseDeflationManager: Periodic refresh triggered at step %d\n", stepCounter);

    // Tier 2: The coarse operator has already been updated (the MG hierarchy
    // was rebuilt via createCoarseDirac). We just re-diagonalize.
    rayleighRitzUpdate();

    stepCounter = 0;
  }

} // namespace quda
