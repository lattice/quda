#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <qio_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>
#include <tune_quda.h>
#include <eigen_helper.h>

namespace quda
{
  // Thick Restarted Lanczos Method constructor
  TRLM::TRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    EigenSolver(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Tridiagonal/Arrow matrix
    alpha.resize(n_kr, 0.0);
    beta.resize(n_kr, 0.0);

    // Thick restart specific checks
    if (n_kr < n_ev + 6) errorQuda("n_kr=%d must be greater than n_ev+6=%d\n", n_kr, n_ev + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void TRLM::operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
  {
    // Override any user input for block size.
    block_size = 1;

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    queryPrec(kSpace[0].Precision());
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      logQuda(QUDA_VERBOSE, "Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess(kSpace);

    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);

    // Check for Chebyshev maximum estimation
    checkChebyOpMax(kSpace);

    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = setEpsilon(kSpace[0].Precision());

    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (int step = num_keep; step < n_kr; step++) lanczosStep(kSpace, step);
      iter += (n_kr - num_keep);

      // The eigenvalues are returned in the alpha array
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat();
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      // mat_norm is updated.
      for (int i = num_locked; i < n_kr; i++)
        if (fabs(alpha[i]) > mat_norm) mat_norm = fabs(alpha[i]);

      // Locking check
      iter_locked = 0;
      for (int i = 1; i < (n_kr - num_locked); i++) {
        if (residua[i + num_locked] < epsilon * mat_norm) {
          logQuda(QUDA_DEBUG_VERBOSE, "**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked],
                  epsilon * mat_norm);
          iter_locked = i;
        } else {
          // Unlikely to find new locked pairs
          break;
        }
      }

      // Convergence check
      iter_converged = iter_locked;
      for (int i = iter_locked + 1; i < n_kr - num_locked; i++) {
        if (residua[i + num_locked] < tol * mat_norm) {
          logQuda(QUDA_DEBUG_VERBOSE, "**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked],
                  tol * mat_norm);
          iter_converged = i;
        } else {
          // Unlikely to find new converged pairs
          break;
        }
      }

      iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - num_locked - 12);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      computeKeptRitz(kSpace);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      num_converged = num_locked + iter_converged;
      num_keep = num_locked + iter_keep;
      num_locked += iter_locked;

      logQuda(QUDA_VERBOSE, "%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);

      logQuda(QUDA_DEBUG_VERBOSE, "iter Conv = %d\n", iter_converged);
      logQuda(QUDA_DEBUG_VERBOSE, "iter Keep = %d\n", iter_keep);
      logQuda(QUDA_DEBUG_VERBOSE, "iter Lock = %d\n", iter_locked);
      logQuda(QUDA_DEBUG_VERBOSE, "num_converged = %d\n", num_converged);
      logQuda(QUDA_DEBUG_VERBOSE, "num_keep = %d\n", num_keep);
      logQuda(QUDA_DEBUG_VERBOSE, "num_locked = %d\n", num_locked);
      for (int i = 0; i < n_kr; i++) {
        logQuda(QUDA_DEBUG_VERBOSE, "Ritz[%d] = %.16e residual[%d] = %.16e\n", i, alpha[i], i, residua[i]);
      }

      // Check for convergence
      if (num_converged >= n_conv) {
        reorder(kSpace);
        converged = true;
      }

      restart_iter++;
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      if (eig_param->require_convergence) {
        errorQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                  "restart steps. Exiting.",
                  n_conv, n_ev, n_kr, max_restarts);
      } else {
        warningQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                    "restart steps. Continuing with current lanczos factorisation.",
                    n_conv, n_ev, n_kr, max_restarts);
      }
    } else {
      logQuda(QUDA_SUMMARIZE, "TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n",
              n_conv, restart_iter, iter);

      // Dump all Ritz values and residua if using Chebyshev
      for (int i = 0; i < n_conv && eig_param->use_poly_acc; i++) {
        logQuda(QUDA_SUMMARIZE, "RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
      }

      // Compute eigenvalues/singular values

      computeEvals(kSpace, evals);
      if (compute_svd) computeSVD(kSpace, evals);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Thick Restart Member functions
  //---------------------------------------------------------------------------
  void TRLM::lanczosStep(std::vector<ColorSpinorField> &v, int j)
  {
    // Compute r = A * v_j - b_{j-i} * v_{j-1}
    // r = A * v_j

    chebyOp(r[0], v[j]);

    // a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(v[j], r[0]);

    // r = r - a_j * v_j
    blas::axpy(-alpha[j], v[j], r[0]);

    int start = (j > num_keep) ? j - 1 : 0;

    if (j - start > 0) {
      std::vector<double> beta_ = {beta.begin() + start, beta.begin() + j};
      for (auto & bi : beta_) bi = -bi;

      // r = r - b_{j-1} * v_{j-1}
      blas::axpy(beta_, {v.begin() + start, v.begin() + j}, r[0]);
    }

    // Orthogonalise r against the Krylov space
    for (int k = 0; k < 1; k++) blockOrthogonalizeHMGS(v, r, ortho_block_size, j + 1);

    // b_j = ||r||
    beta[j] = sqrt(blas::norm2(r[0]));

    // Prepare next step.
    // v_{j+1} = r / b_j
    blas::axy(1.0 / beta[j], r[0], v[j + 1]);
  }

  void TRLM::reorder(std::vector<ColorSpinorField> &kSpace)
  {
    int i = 0;

    if (reverse) {
      while (i < n_kr) {
        if ((i == 0) || (alpha[i - 1] >= alpha[i]))
          i++;
        else {
          std::swap(alpha[i], alpha[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
    } else {
      while (i < n_kr) {
        if ((i == 0) || (alpha[i - 1] <= alpha[i]))
          i++;
        else {
          std::swap(alpha[i], alpha[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
    }
  }

  void TRLM::eigensolveFromArrowMat()
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    int dim = n_kr - num_locked;
    int arrow_pos = num_keep - num_locked;

    // Eigen objects
    MatrixXd A = MatrixXd::Zero(dim, dim);
    ritz_mat.resize(dim * dim, 0.0);

    // Invert the spectrum due to chebyshev
    if (reverse) {
      for (int i = num_locked; i < n_kr - 1; i++) {
        alpha[i] *= -1.0;
        beta[i] *= -1.0;
      }
      alpha[n_kr - 1] *= -1.0;
    }

    // Construct arrow mat A_{dim,dim}
    for (int i = 0; i < dim; i++) {

      // alpha populates the diagonal
      A(i, i) = alpha[i + num_locked];
    }

    for (int i = 0; i < arrow_pos; i++) {

      // beta populates the arrow
      A(i, arrow_pos) = beta[i + num_locked];
      A(arrow_pos, i) = beta[i + num_locked];
    }

    for (int i = arrow_pos; i < dim - 1; i++) {

      // beta populates the sub-diagonal
      A(i, i + 1) = beta[i + num_locked];
      A(i + 1, i) = beta[i + num_locked];
    }

    // Eigensolve the arrow matrix
    SelfAdjointEigenSolver<MatrixXd> eigensolver;
    eigensolver.compute(A);

    // repopulate ritz matrix
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];

    for (int i = 0; i < dim; i++) {
      residua[i + num_locked] = fabs(beta[n_kr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
      // Update the alpha array
      alpha[i + num_locked] = eigensolver.eigenvalues()[i];
    }

    // Put spectrum back in order
    if (reverse) {
      for (int i = num_locked; i < n_kr; i++) { alpha[i] *= -1.0; }
    }

    profile.TPSTOP(QUDA_PROFILE_EIGEN);
  }

  void TRLM::computeKeptRitz(std::vector<ColorSpinorField> &kSpace)
  {
    int offset = n_kr + 1;
    int dim = n_kr - num_locked;

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    std::vector<double> ritz_mat_keep(dim * iter_keep);
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < iter_keep; i++) { ritz_mat_keep[j * iter_keep + i] = ritz_mat[i * dim + j]; }
    }

    rotateVecs(kSpace, ritz_mat_keep, offset, dim, iter_keep, num_locked, profile);

    // Update residual vector
    std::swap(kSpace[num_locked + iter_keep], kSpace[n_kr]);

    // Update sub arrow matrix
    for (int i = 0; i < iter_keep; i++) beta[i + num_locked] = beta[n_kr - 1] * ritz_mat[dim * (i + 1) - 1];
  }
} // namespace quda
