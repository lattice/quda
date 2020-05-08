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

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

namespace quda
{

  using namespace Eigen;

  // Thick Restarted Lanczos Method constructor
  TRLM::TRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    EigenSolver(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Tridiagonal/Arrow matrix
    alpha = (double *)safe_malloc(nKr * sizeof(double));
    beta = (double *)safe_malloc(nKr * sizeof(double));
    for (int i = 0; i < nKr; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }

    // Thick restart specific checks
    if (nKr < nEv + 6) errorQuda("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void TRLM::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // In case we are deflating an operator, save the tunechache from the inverter
    saveTuneCache();

    // Override any user input for block size.
    block_size = 1;

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess(kSpace);

    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);

    // Check for Chebyshev maximum estimation
    checkChebyOpMax(mat, kSpace);

    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = setEpsilon(kSpace[0]->Precision());

    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (int step = num_keep; step < nKr; step++) lanczosStep(kSpace, step);
      iter += (nKr - num_keep);

      // The eigenvalues are returned in the alpha array
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat();
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      // mat_norm is updated.
      for (int i = num_locked; i < nKr; i++)
        if (fabs(alpha[i]) > mat_norm) mat_norm = fabs(alpha[i]);

      // Locking check
      iter_locked = 0;
      for (int i = 1; i < (nKr - num_locked); i++) {
        if (residua[i + num_locked] < epsilon * mat_norm) {
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
            printfQuda("**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked],
                       epsilon * mat_norm);
          iter_locked = i;
        } else {
          // Unlikely to find new locked pairs
          break;
        }
      }

      // Convergence check
      iter_converged = iter_locked;
      for (int i = iter_locked + 1; i < nKr - num_locked; i++) {
        if (residua[i + num_locked] < tol * mat_norm) {
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
            printfQuda("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol * mat_norm);
          iter_converged = i;
        } else {
          // Unlikely to find new converged pairs
          break;
        }
      }

      iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - num_locked - 12);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      computeKeptRitz(kSpace);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      num_converged = num_locked + iter_converged;
      num_keep = num_locked + iter_keep;
      num_locked += iter_locked;

      if (getVerbosity() >= QUDA_VERBOSE) {
        printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
      }

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
        printfQuda("iter Conv = %d\n", iter_converged);
        printfQuda("iter Keep = %d\n", iter_keep);
        printfQuda("iter Lock = %d\n", iter_locked);
        printfQuda("num_converged = %d\n", num_converged);
        printfQuda("num_keep = %d\n", num_keep);
        printfQuda("num_locked = %d\n", num_locked);
        for (int i = 0; i < nKr; i++) {
          printfQuda("Ritz[%d] = %.16e residual[%d] = %.16e\n", i, alpha[i], i, residua[i]);
        }
      }

      // Check for convergence
      if (num_converged >= nConv) {
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
                  nConv, nEv, nKr, max_restarts);
      } else {
        warningQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                    "restart steps. Continuing with current lanczos factorisation.",
                    nConv, nEv, nKr, max_restarts);
      }
    } else {
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", nConv,
                   restart_iter, iter);

        // Dump all Ritz values and residua
        for (int i = 0; i < nConv; i++) {
          printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
        }
      }

      // Compute eigenvalues
      computeEvals(mat, kSpace, evals);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Destructor
  TRLM::~TRLM()
  {
    ritz_mat.clear();
    ritz_mat.shrink_to_fit();
    host_free(alpha);
    host_free(beta);
  }

  // Thick Restart Member functions
  //---------------------------------------------------------------------------
  void TRLM::lanczosStep(std::vector<ColorSpinorField *> v, int j)
  {
    // Compute r = A * v_j - b_{j-i} * v_{j-1}
    // r = A * v_j

    chebyOp(mat, *r[0], *v[j]);

    // a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v[j], *r[0]);

    // r = r - a_j * v_j
    blas::axpy(-alpha[j], *v[j], *r[0]);

    int start = (j > num_keep) ? j - 1 : 0;

    if (j - start > 0) {
      std::vector<ColorSpinorField *> r_ {r[0]};
      std::vector<double> beta_;
      beta_.reserve(j - start);
      std::vector<ColorSpinorField *> v_;
      v_.reserve(j - start);
      for (int i = start; i < j; i++) {
        beta_.push_back(-beta[i]);
        v_.push_back(v[i]);
      }
      // r = r - b_{j-1} * v_{j-1}
      blas::axpy(beta_.data(), v_, r_);
    }

    // Orthogonalise r against the Krylov space
    for (int k = 0; k < 1; k++) blockOrthogonalize(v, r, j + 1);

    // b_j = ||r||
    beta[j] = sqrt(blas::norm2(*r[0]));

    // Prepare next step.
    // v_{j+1} = r / b_j
    blas::zero(*v[j + 1]);
    blas::axpy(1.0 / beta[j], *r[0], *v[j + 1]);

    // Save Lanczos step tuning
    saveTuneCache();
  }

  void TRLM::reorder(std::vector<ColorSpinorField *> &kSpace)
  {
    int i = 0;

    if (reverse) {
      while (i < nKr) {
        if ((i == 0) || (alpha[i - 1] >= alpha[i]))
          i++;
        else {
          std::swap(alpha[i], alpha[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
    } else {
      while (i < nKr) {
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
    int dim = nKr - num_locked;
    // int arrow_pos = std::max(num_keep - num_locked + 1, 2);
    int arrow_pos = num_keep - num_locked;

    // Eigen objects
    MatrixXd A = MatrixXd::Zero(dim, dim);
    ritz_mat.resize(dim * dim);
    for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;

    // Invert the spectrum due to chebyshev
    if (reverse) {
      for (int i = num_locked; i < nKr - 1; i++) {
        alpha[i] *= -1.0;
        beta[i] *= -1.0;
      }
      alpha[nKr - 1] *= -1.0;
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
      residua[i + num_locked] = fabs(beta[nKr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
      // Update the alpha array
      alpha[i + num_locked] = eigensolver.eigenvalues()[i];
    }

    // Put spectrum back in order
    if (reverse) {
      for (int i = num_locked; i < nKr; i++) { alpha[i] *= -1.0; }
    }

    profile.TPSTOP(QUDA_PROFILE_EIGEN);
  }

  void TRLM::computeKeptRitz(std::vector<ColorSpinorField *> &kSpace)
  {
    int offset = nKr + 1;
    int dim = nKr - num_locked;

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    double *ritz_mat_keep = (double *)safe_malloc((dim * iter_keep) * sizeof(double));

    // If we have memory availible, do the entire rotation
    if (batched_rotate <= 0 || batched_rotate >= iter_keep) {
      ColorSpinorParam csParamClone(*kSpace[0]);
      csParamClone.create = QUDA_ZERO_FIELD_CREATE;
      if ((int)kSpace.size() < offset + iter_keep) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + iter_keep);
        kSpace.reserve(offset + iter_keep);
        for (int i = kSpace.size(); i < offset + iter_keep; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }

      // Pointers to the relevant vectors
      std::vector<ColorSpinorField *> vecs_ptr;
      std::vector<ColorSpinorField *> kSpace_ptr;

      // Alias the extra space vectors, zero the workspace
      vecs_ptr.reserve(iter_keep);
      for (int i = 0; i < iter_keep; i++) {
        kSpace_ptr.push_back(kSpace[offset + i]);
        blas::zero(*kSpace_ptr[i]);
      }

      // Alias the vectors we wish to keep, populate the Ritz matrix and transpose.
      kSpace_ptr.reserve(dim);
      for (int j = 0; j < dim; j++) {
        vecs_ptr.push_back(kSpace[num_locked + j]);
        for (int i = 0; i < iter_keep; i++) { ritz_mat_keep[j * iter_keep + i] = ritz_mat[i * dim + j]; }
      }

      // multiBLAS axpy
      blas::axpy(ritz_mat_keep, vecs_ptr, kSpace_ptr);

      // Copy back to the Krylov space
      for (int i = 0; i < iter_keep; i++) std::swap(kSpace[i + num_locked], kSpace[offset + i]);
    } else {

      // Do batched rotation to save on memory
      int batch_size = batched_rotate;
      int full_batches = iter_keep / batch_size;
      int batch_size_r = iter_keep % batch_size;
      bool do_batch_remainder = (batch_size_r != 0 ? true : false);

      if ((int)kSpace.size() < offset + batch_size) {
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_ZERO_FIELD_CREATE;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + batch_size);
        kSpace.reserve(offset + batch_size);
        for (int i = kSpace.size(); i < offset + batch_size; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }

      profile.TPSTART(QUDA_PROFILE_EIGEN);
      MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
      for (int j = 0; j < iter_keep; j++)
        for (int i = 0; i < dim; i++) mat(i, j) = ritz_mat[j * dim + i];

      FullPivLU<MatrixXd> matLU(mat);

      // Extract the upper triangular matrix
      MatrixXd matUpper = MatrixXd::Zero(iter_keep, iter_keep);
      matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
      matUpper.conservativeResize(iter_keep, iter_keep);

      // Extract the lower triangular matrix
      MatrixXd matLower = MatrixXd::Identity(dim, dim);
      matLower.block(0, 0, dim, iter_keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();
      matLower.conservativeResize(dim, iter_keep);

      // Extract the desired permutation matrices
      MatrixXi matP = MatrixXi::Zero(dim, dim);
      MatrixXi matQ = MatrixXi::Zero(iter_keep, iter_keep);
      matP = matLU.permutationP().inverse();
      matQ = matLU.permutationQ().inverse();
      profile.TPSTOP(QUDA_PROFILE_EIGEN);

      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      // Compute V * A = V * PLUQ

      // Do P Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matP.data(), dim);

      // Do L Multiply
      //---------------------------------------------------------------------------
      // Loop over full batches
      for (int b = 0; b < full_batches; b++) {

        // batch triangle
        blockRotate(kSpace, matLower.data(), dim, {b * batch_size, (b + 1) * batch_size},
                    {b * batch_size, (b + 1) * batch_size}, LOWER_TRI);
        // batch pencil
        blockRotate(kSpace, matLower.data(), dim, {(b + 1) * batch_size, dim}, {b * batch_size, (b + 1) * batch_size},
                    PENCIL);
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }

      if (do_batch_remainder) {
        // remainder triangle
        blockRotate(kSpace, matLower.data(), dim, {full_batches * batch_size, iter_keep},
                    {full_batches * batch_size, iter_keep}, LOWER_TRI);
        // remainder pencil
        if (iter_keep < dim) {
          blockRotate(kSpace, matLower.data(), dim, {iter_keep, dim}, {full_batches * batch_size, iter_keep}, PENCIL);
        }
        blockReset(kSpace, full_batches * batch_size, iter_keep, offset);
      }

      // Do U Multiply
      //---------------------------------------------------------------------------
      if (do_batch_remainder) {
        // remainder triangle
        blockRotate(kSpace, matUpper.data(), iter_keep, {full_batches * batch_size, iter_keep},
                    {full_batches * batch_size, iter_keep}, UPPER_TRI);
        // remainder pencil
        blockRotate(kSpace, matUpper.data(), iter_keep, {0, full_batches * batch_size},
                    {full_batches * batch_size, iter_keep}, PENCIL);
        blockReset(kSpace, full_batches * batch_size, iter_keep, offset);
      }

      // Loop over full batches
      for (int b = full_batches - 1; b >= 0; b--) {
        // batch triangle
        blockRotate(kSpace, matUpper.data(), iter_keep, {b * batch_size, (b + 1) * batch_size},
                    {b * batch_size, (b + 1) * batch_size}, UPPER_TRI);
        if (b > 0) {
          // batch pencil
          blockRotate(kSpace, matUpper.data(), iter_keep, {0, b * batch_size}, {b * batch_size, (b + 1) * batch_size},
                      PENCIL);
        }
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }

      // Do Q Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matQ.data(), iter_keep);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    }

    // Update residual vector
    std::swap(kSpace[num_locked + iter_keep], kSpace[nKr]);

    // Update sub arrow matrix
    for (int i = 0; i < iter_keep; i++) beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];

    host_free(ritz_mat_keep);

    // Save Krylov rotation tuning
    saveTuneCache();
  }
} // namespace quda
