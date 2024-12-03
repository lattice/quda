#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <vector_io.h>
#include <qio_field.h>
#include <color_spinor_field.h>
#include <random_quda.h>
#include <blas_quda.h>
#include <blas_3d.h>
#include <util_quda.h>
#include <tune_quda.h>
#include <eigen_helper.h>
#include <split_grid.h>

namespace quda
{
  // Thick Restarted Lanczos Method constructor
  TRLM3D::TRLM3D(const DiracMatrix &mat, QudaEigParam *eig_param) : EigenSolver(mat, eig_param)
  {
    getProfile().TPSTART(QUDA_PROFILE_INIT);

    ortho_dim = eig_param->ortho_dim;
    ortho_dim_size = eig_param->ortho_dim_size_local;
    if (ortho_dim != 3)
      errorQuda("Only 3D spatial splitting (ortho_dim = 3) is supported, ortho_dim passed = %d", ortho_dim);
    verbose_rank = comm_coord(0) == 0 && comm_coord(1) == 0 && comm_coord(2) == 0;

    // Tridiagonal/Arrow matrices
    alpha_3D.resize(ortho_dim_size);
    beta_3D.resize(ortho_dim_size);

    residua_3D.resize(ortho_dim_size);
    ritz_mat_3D.resize(ortho_dim_size);
    converged_3D.resize(ortho_dim_size, false);
    active_3D.resize(ortho_dim_size, false);

    iter_locked_3D.resize(ortho_dim_size, 0);
    iter_keep_3D.resize(ortho_dim_size, 0);
    iter_converged_3D.resize(ortho_dim_size, 0);

    num_locked_3D.resize(ortho_dim_size, 0);
    num_keep_3D.resize(ortho_dim_size, 0);
    num_converged_3D.resize(ortho_dim_size, 0);

    for (int i = 0; i < ortho_dim_size; i++) {
      alpha_3D[i].resize(n_kr, 0.0);
      beta_3D[i].resize(n_kr, 0.0);
      residua_3D[i].resize(n_kr, 0.0);
    }

    // 3D thick restart specific checks
    if (n_kr < n_ev + 6) errorQuda("n_kr=%d must be greater than n_ev+6=%d\n", n_kr, n_ev + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  void TRLM3D::operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
  {
    // Create 3-d split communicators
    CommKey split_key = {1, 1, 1, comm_dim(3)};

    if (!split_key.is_valid()) {
      errorQuda("split_key = [%d,%d,%d,%d] is not valid", split_key[0], split_key[1], split_key[2], split_key[3]);
    }
    logQuda(QUDA_DEBUG_VERBOSE, "Spliting the grid into sub-partitions: (%2d,%2d,%2d,%2d) / (%2d,%2d,%2d,%2d)\n",
            comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3), split_key[0], split_key[1], split_key[2], split_key[3]);
    push_communicator(split_key);

    // Override any user input for block size.
    block_size = 1;

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    queryPrec(kSpace[0].Precision());
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess3D(kSpace, ortho_dim_size);

    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);

    // Check for Chebyshev maximum estimation
    checkChebyOpMax(kSpace);

    // Convergence and locking criteria
    std::vector<double> mat_norm_3D(ortho_dim_size, 0.0);
    double epsilon = setEpsilon(kSpace[0].Precision());

    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    int t_offset = ortho_dim_size * comm_coord(3);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      // Get min step
      int step_min = *std::min_element(num_locked_3D.begin(), num_locked_3D.end());
      comm_allreduce_min(step_min);

      for (int step = step_min; step < n_kr; step++) lanczosStep3D(kSpace, step);
      iter += (n_kr - step_min);

      // The eigenvalues are returned in the alpha array
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat3D();
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

      // mat_norm is updated and used for LR
      for (int t = 0; t < ortho_dim_size; t++)
        for (int i = num_locked_3D[t]; i < n_kr; i++)
          if (fabs(alpha_3D[t][i]) > mat_norm_3D[t]) mat_norm_3D[t] = fabs(alpha_3D[t][i]);

      // Lambda that returns mat_norm for LR and returns the relevant alpha
      // (the corresponding Ritz value) for SR
      auto check_norm = [&](double sr_norm, int t) -> double {
        if (eig_param->spectrum == QUDA_SPECTRUM_LR_EIG)
          return mat_norm_3D[t];
        else
          return sr_norm;
      };

      // Locking check
      for (int t = 0; t < ortho_dim_size; t++) {
        if (!converged_3D[t]) {
          iter_locked_3D[t] = 0;
          for (int i = 0; i < (n_kr - num_locked_3D[t]); i++) {
            if (residua_3D[t][i + num_locked_3D[t]] < epsilon * check_norm(alpha_3D[t][i + num_locked_3D[t]], t)) {
              logQuda(QUDA_DEBUG_VERBOSE, "**** Locking %d %d resid=%+.6e condition=%.6e ****\n", t, i,
                      residua_3D[t][i + num_locked_3D[t]], epsilon * check_norm(alpha_3D[t][i + num_locked_3D[t]], t));
              iter_locked_3D[t] = i + 1;
            } else {
              // Unlikely to find new locked pairs
              break;
            }
          }
        }
      }

      // Convergence check
      for (int t = 0; t < ortho_dim_size; t++) {
        if (!converged_3D[t]) {
          iter_converged_3D[t] = iter_locked_3D[t];
          for (int i = iter_locked_3D[t]; i < n_kr - num_locked_3D[t]; i++) {
            if (residua_3D[t][i + num_locked_3D[t]] < tol * check_norm(alpha_3D[t][i + num_locked_3D[t]], t)) {
              logQuda(QUDA_DEBUG_VERBOSE, "**** Converged %d %d resid=%+.6e condition=%.6e ****\n", t, i,
                      residua_3D[t][i + num_locked_3D[t]], tol * check_norm(alpha_3D[t][i + num_locked_3D[t]], t));
              iter_converged_3D[t] = i + 1;
            } else {
              // Unlikely to find new converged pairs
              break;
            }
          }
        }
      }

      for (int t = 0; t < ortho_dim_size; t++) {
        if (!converged_3D[t])
          iter_keep_3D[t]
            = std::min(iter_converged_3D[t] + (n_kr - num_converged_3D[t]) / 2, n_kr - num_locked_3D[t] - 12);
      }

      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      computeKeptRitz3D(kSpace);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

      int min_nconv = n_kr + 1;
      int min_nlock = n_kr + 1;
      int min_nkeep = n_kr + 1;

      for (int t = 0; t < ortho_dim_size; t++) {
        if (!converged_3D[t]) {
          num_converged_3D[t] = num_locked_3D[t] + iter_converged_3D[t];
          num_keep_3D[t] = num_locked_3D[t] + iter_keep_3D[t];
          num_locked_3D[t] += iter_locked_3D[t];

          if (num_converged_3D[t] < min_nconv) min_nconv = num_converged_3D[t];
          if (num_locked_3D[t] < min_nlock) min_nlock = num_locked_3D[t];
          if (num_keep_3D[t] < min_nkeep) min_nkeep = num_keep_3D[t];

          // Use printf to get data from t dim only
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE && verbose_rank) {
            printf("%04d converged eigenvalues for timeslice %d at restart iter %04d\n", num_converged_3D[t],
                   t_offset + t, restart_iter + 1);
            printf("iter Conv[%d] = %d\n", t_offset + t, iter_converged_3D[t]);
            printf("iter Keep[%d] = %d\n", t_offset + t, iter_keep_3D[t]);
            printf("iter Lock[%d] = %d\n", t_offset + t, iter_locked_3D[t]);
            printf("num_converged[%d] = %d\n", t_offset + t, num_converged_3D[t]);
            printf("num_keep[%d] = %d\n", t_offset + t, num_keep_3D[t]);
            printf("num_locked[%d] = %d\n", t_offset + t, num_locked_3D[t]);
            for (int i = 0; i < n_kr; i++) {
              printf("Ritz[%d][%d] = %.16e residual[%d] = %.16e\n", t_offset + t, i, alpha_3D[t][i], i, residua_3D[t][i]);
            }
          }
        }
      }

      // Check for convergence
      bool all_converged = true;
      for (int t = 0; t < ortho_dim_size; t++) {
        if (num_converged_3D[t] >= n_conv) {
          converged_3D[t] = true;
        } else {
          all_converged = false;
        }
      }

      if (getVerbosity() >= QUDA_VERBOSE && verbose_rank) {
        std::stringstream converged;
        std::copy(converged_3D.begin(), converged_3D.end(), std::ostream_iterator<int>(converged, ""));
        printf("iter = %d rank = %d converged = %s min nlock %3d nconv %3d nkeep %3d\n", restart_iter + 1,
               comm_rank_global(), converged.str().c_str(), min_nlock, min_nconv, min_nkeep);
      }

      if (all_converged) {
        reorder3D(kSpace);
        converged = true;
      }

      restart_iter++;
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      if (eig_param->require_convergence) {
        errorQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                  "restart steps. Exiting.",
                  n_conv, n_ev, n_kr, max_restarts);
      } else {
        warningQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                    "restart steps. Continuing with current Lanczos factorisation.",
                    n_conv, n_ev, n_kr, max_restarts);
        // Compute eigenvalues
        computeEvals(kSpace, evals);
      }
    } else {
      if (verbose_rank) {
        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printf("TRLM (rank = %d) computed the requested %d vectors in %d restart steps and %d OP*x operations\n",
                 comm_rank_global(), n_conv, restart_iter, iter);
        }

        if (getVerbosity() >= QUDA_VERBOSE) {
          // Dump all Ritz values and residua if using Chebyshev
          if (eig_param->use_poly_acc) {
            for (int t = 0; t < ortho_dim_size; t++) {
              for (int i = 0; i < n_conv; i++) {
                printf("RitzValue[%d][%04d]: (%+.16e, %+.16e) residual %.16e\n", t, i, alpha_3D[t][i], 0.0,
                       residua_3D[t][i]);
              }
            }
          }
        }
      }

      // Compute eigenvalues
      computeEvals(kSpace, evals);
    }

    push_communicator(default_comm_key);

    // ensure all processes have all e-values
    comm_allreduce_sum(evals);

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Thick Restart 3D Member functions
  //---------------------------------------------------------------------------
  void TRLM3D::lanczosStep3D(std::vector<ColorSpinorField> &v, int j)
  {
    // Compute r[t] = A[t] * v_j[t] - b_{j-i}[t] * v_{j-1}[t]
    // r[t] = A[t] * v_j[t]

    // Use this while we have only axpby in place of axpy;
    std::vector<double> unit(ortho_dim_size, 1.0);
    std::vector<double> alpha_j(ortho_dim_size, 0.0);
    std::vector<double> beta_j(ortho_dim_size, 0.0);

    // 3D vectors that hold data for individual t components
    ColorSpinorParam csParamClone(v[0]);
    csParamClone.change_dim(ortho_dim, 1);
    std::vector<ColorSpinorField> vecs_t(1, csParamClone);
    ColorSpinorField r_t(csParamClone);

    // Identify active 3D slices. The active_3D array
    // should be modified here only throughout the entire
    // algorithm.
    for (int t = 0; t < ortho_dim_size; t++) {
      // Every element of the active array must be assessed
      active_3D[t] = (num_keep_3D[t] <= j && !converged_3D[t]) ? true : false;
    }

    // This will be a blocked operator with no
    // connections in the ortho_dim (usually t)
    // hence the 3D sections of each vector
    // will be independent.
    chebyOp(r[0], v[j]);

    // a_j[t] = v_j^dag[t] * r[t]
    blas3d::reDotProduct(alpha_j, v[j], r[0]);
    for (int t = 0; t < ortho_dim_size; t++) {
      // Only active problem data is recorded
      if (active_3D[t]) alpha_3D[t][j] = alpha_j[t];
    }

    // r[t] = r[t] - a_j[t] * v_j[t]
    for (int t = 0; t < ortho_dim_size; t++) alpha_j[t] = active_3D[t] ? -alpha_j[t] : 0.0;
    blas3d::axpby(alpha_j, v[j], unit, r[0]);

    // r[t] = r[t] - b_{j-1}[t] * v_{j-1}[t]
    // Only orthogonalise active problems
    for (int t = 0; t < ortho_dim_size; t++) {
      if (active_3D[t]) {
        int start = (j > num_keep_3D[t]) ? j - 1 : 0;
        if (j - start > 0) {

          // Ensure we have enough 3D vectors
          if ((int)vecs_t.size() < j - start) resize(vecs_t, j - start, csParamClone);

          // Copy the 3D data into the 3D vectors, create beta array, and create
          // pointers to the 3D vectors
          std::vector<double> beta_t;
          beta_t.reserve(j - start);
          // Copy residual
          blas3d::copy(t, blas3d::copyType::COPY_TO_3D, r_t, r[0]);
          // Copy vectors
          for (int i = start; i < j; i++) {
            blas3d::copy(t, blas3d::copyType::COPY_TO_3D, vecs_t[i - start], v[i]);
            beta_t.push_back(-beta_3D[t][i]);
          }

          // r[t] = r[t] - beta[t]{j-1} * v[t]{j-1}
          blas::block::axpy(beta_t, {vecs_t.begin(), vecs_t.begin() + j - start}, r_t);

          // Copy residual back to 4D vector
          blas3d::copy(t, blas3d::copyType::COPY_FROM_3D, r_t, r[0]);
        }
      }
    }

    // Orthogonalise r against the Krylov space
    for (int k = 0; k < 1; k++) blockOrthogonalize3D(v, r, j + 1); //  future work: up to 4 times???

    // b_j[t] = ||r[t]||
    blas3d::reDotProduct(beta_j, r[0], r[0]);
    for (int t = 0; t < ortho_dim_size; t++) beta_j[t] = active_3D[t] ? sqrt(beta_j[t]) : 0.0;

    // Prepare next step.
    // v_{j+1}[t] = r[t] / b_{j}[t]
    for (int t = 0; t < ortho_dim_size; t++) {
      if (active_3D[t]) {
        beta_3D[t][j] = beta_j[t];
        beta_j[t] = 1.0 / beta_j[t];
      }
    }

    std::vector<double> c(ortho_dim_size);
    for (int t = 0; t < ortho_dim_size; t++) c[t] = beta_j[t] == 0.0 ? 1.0 : 0.0;
    blas3d::axpby(beta_j, r[0], c, v[j + 1]);
  }

  void TRLM3D::reorder3D(std::vector<ColorSpinorField> &kSpace)
  {
    for (int t = 0; t < ortho_dim_size; t++) {
      int i = 0;
      if (reverse) {
        while (i < n_kr) {
          if ((i == 0) || (alpha_3D[t][i - 1] >= alpha_3D[t][i]))
            i++;
          else {
            std::swap(alpha_3D[t][i], alpha_3D[t][i - 1]);
            blas3d::swap(t, kSpace[i], kSpace[i - 1]);
            i--;
          }
        }
      } else {
        while (i < n_kr) {
          if ((i == 0) || (alpha_3D[t][i - 1] <= alpha_3D[t][i]))
            i++;
          else {
            std::swap(alpha_3D[t][i], alpha_3D[t][i - 1]);
            blas3d::swap(t, kSpace[i], kSpace[i - 1]);
            i--;
          }
        }
      }
    }
  }

  void TRLM3D::eigensolveFromArrowMat3D()
  {
    getProfile().TPSTART(QUDA_PROFILE_EIGEN);

    // Loop over the 3D problems
#pragma omp parallel for
    for (int t = 0; t < ortho_dim_size; t++) {
      if (!converged_3D[t]) {

        int dim = n_kr - num_locked_3D[t];
        int arrow_pos = num_keep_3D[t] - num_locked_3D[t];

        // Eigen objects
        MatrixXd A = MatrixXd::Zero(dim, dim);
        ritz_mat_3D[t].resize(dim * dim);
        for (int i = 0; i < dim * dim; i++) ritz_mat_3D[t][i] = 0.0;

        // Invert the spectrum due to chebyshev
        if (reverse) {
          for (int i = num_locked_3D[t]; i < n_kr - 1; i++) {
            alpha_3D[t][i] *= -1.0;
            beta_3D[t][i] *= -1.0;
          }
          alpha_3D[t][n_kr - 1] *= -1.0;
        }

        // Construct arrow mat A_{dim,dim}
        for (int i = 0; i < dim; i++) {

          // alpha_3D populates the diagonal
          A(i, i) = alpha_3D[t][i + num_locked_3D[t]];
        }

        for (int i = 0; i < arrow_pos; i++) {

          // beta_3D populates the arrow
          A(i, arrow_pos) = beta_3D[t][i + num_locked_3D[t]];
          A(arrow_pos, i) = beta_3D[t][i + num_locked_3D[t]];
        }

        for (int i = arrow_pos; i < dim - 1; i++) {

          // beta_3D populates the sub-diagonal
          A(i, i + 1) = beta_3D[t][i + num_locked_3D[t]];
          A(i + 1, i) = beta_3D[t][i + num_locked_3D[t]];
        }

        // Eigensolve the arrow matrix
        SelfAdjointEigenSolver<MatrixXd> eigensolver;
        eigensolver.compute(A);

        // repopulate ritz matrix
        for (int i = 0; i < dim; i++)
          for (int j = 0; j < dim; j++) ritz_mat_3D[t][dim * i + j] = eigensolver.eigenvectors().col(i)[j];

        for (int i = 0; i < dim; i++) {
          residua_3D[t][i + num_locked_3D[t]] = fabs(beta_3D[t][n_kr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
          // Update the alpha_3D array
          alpha_3D[t][i + num_locked_3D[t]] = eigensolver.eigenvalues()[i];
        }

        // Put spectrum back in order
        if (reverse) {
          for (int i = num_locked_3D[t]; i < n_kr; i++) { alpha_3D[t][i] *= -1.0; }
        }
      }
    }

    getProfile().TPSTOP(QUDA_PROFILE_EIGEN);
  }

  void TRLM3D::computeKeptRitz3D(std::vector<ColorSpinorField> &kSpace)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    // Multi-BLAS friendly array to store part of Ritz matrix we want
    std::vector<std::vector<double>> ritz_mat_keep(ortho_dim_size);

    std::vector<ColorSpinorField> vecs_t;
    std::vector<ColorSpinorField> kSpace_t;

    ColorSpinorParam csParamClone(kSpace[0]);
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    csParamClone.change_dim(ortho_dim, 1);

    for (int t = 0; t < ortho_dim_size; t++) {
      if (!converged_3D[t]) {
        int dim = n_kr - num_locked_3D[t];
        int keep = iter_keep_3D[t];

        ritz_mat_keep[t].resize(dim * keep);
        for (int j = 0; j < dim; j++) {
          for (int i = 0; i < keep; i++) { ritz_mat_keep[t][j * keep + i] = ritz_mat_3D[t][i * dim + j]; }
        }

        // Alias the vectors we wish to keep.
        vector_ref<ColorSpinorField> vecs_locked(kSpace.begin() + num_locked_3D[t],
                                                 kSpace.begin() + num_locked_3D[t] + dim);

        // multiBLAS axpy. Create 3D vectors so that we may perform all t independent
        // vector rotations.
        vecs_t.reserve(dim);
        kSpace_t.reserve(keep);

        // Create 3D arrays
        for (int i = vecs_t.size(); i < dim; i++) vecs_t.push_back(csParamClone);
        for (int i = kSpace_t.size(); i < keep; i++) kSpace_t.push_back(csParamClone);

        blas::zero(kSpace_t);

        // Copy to data to 3D array, zero out workspace, make pointers
        for (int i = 0; i < dim; i++) blas3d::copy(t, blas3d::copyType::COPY_TO_3D, vecs_t[i], vecs_locked[i]);

        // Compute the axpy
        blas::block::axpy(ritz_mat_keep[t], {vecs_t.begin(), vecs_t.begin() + dim},
                          {kSpace_t.begin(), kSpace_t.begin() + keep});

        // Copy back to the 4D workspace array

        // Copy compressed Krylov
        for (int i = 0; i < keep; i++) {
          blas3d::copy(t, blas3d::copyType::COPY_FROM_3D, kSpace_t[i], kSpace[num_locked_3D[t] + i]);
        }

        // Update residual vector
        blas3d::copy(t, blas3d::copyType::COPY_TO_3D, vecs_t[0], kSpace[n_kr]);
        blas3d::copy(t, blas3d::copyType::COPY_FROM_3D, vecs_t[0], kSpace[num_locked_3D[t] + keep]);

        // Update sub arrow matrix
        for (int i = 0; i < keep; i++)
          beta_3D[t][i + num_locked_3D[t]] = beta_3D[t][n_kr - 1] * ritz_mat_3D[t][dim * (i + 1) - 1];
      }
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // Orthogonalise r[t][0:] against V_[t][0:j]
  void TRLM3D::blockOrthogonalize3D(std::vector<ColorSpinorField> &vecs, std::vector<ColorSpinorField> &rvecs, int j)
  {
    for (int i = 0; i < j; i++) {
      std::vector<Complex> s_t(ortho_dim_size, 0.0);
      std::vector<Complex> unit_new(ortho_dim_size, {1.0, 0.0});

      // Block dot products stored in s_t.
      blas3d::cDotProduct(s_t, vecs[i], rvecs[0]);
      for (int t = 0; t < ortho_dim_size; t++) { s_t[t] *= active_3D[t] ? -1.0 : 0.0; }

      // Block orthogonalise
      blas3d::caxpby(s_t, vecs[i], unit_new, rvecs[0]);
    }
  }

  void TRLM3D::prepareInitialGuess3D(std::vector<ColorSpinorField> &kSpace, int ortho_dim_size)
  {
    auto nrm = blas::norm2(kSpace[0]);
    if (!std::isfinite(nrm) || nrm == 0.0) {
      RNG rng(kSpace[0], 1234);
      spinorNoise(kSpace[0], rng, QUDA_NOISE_UNIFORM);
    }

    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<double> zeros(ortho_dim_size, 0.0);
    blas3d::reDotProduct(norms, kSpace[0], kSpace[0]);
    for (int t = 0; t < ortho_dim_size; t++) norms[t] = 1.0 / sqrt(norms[t]);
    blas3d::axpby(norms, kSpace[0], zeros, kSpace[0]);
  }

  double TRLM3D::estimateChebyOpMax(ColorSpinorField &out, ColorSpinorField &in)
  {
    RNG rng(in, 1234);
    spinorNoise(in, rng, QUDA_NOISE_UNIFORM);

    // Power iteration
    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<double> zeros(ortho_dim_size, 0.0);
    for (int i = 0; i < 100; i++) {
      if ((i + 1) % 10 == 0) {
        blas3d::reDotProduct(norms, in, in);
        for (int t = 0; t < ortho_dim_size; t++) norms[t] = 1.0 / sqrt(norms[t]);
        blas3d::axpby(norms, in, zeros, in);
      }
      mat(out, in);
      std::swap(out, in);
    }

    // Compute spectral radius estimate
    std::vector<double> inner_products(ortho_dim_size, 0.0);
    blas3d::reDotProduct(inner_products, out, in);

    auto result = *std::max_element(inner_products.begin(), inner_products.end());
    comm_allreduce_max(result);
    if (verbose_rank && getVerbosity() >= QUDA_VERBOSE)
      printf("Chebyshev max = %e (rank = %d)\n", result, comm_rank_global());

    // Increase final result by 10% for safety
    return result * 1.10;
  }

  void TRLM3D::computeEvals(std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals, int size)
  {
    if (size == 0) size = n_conv;
    if (size > (int)evecs.size())
      errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size())
      errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    ColorSpinorParam csParamClone(evecs[0]);
    ColorSpinorField tmp(csParamClone);

    std::vector<std::vector<Complex>> evals_t(size, std::vector<Complex>(ortho_dim_size, 0.0));
    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<Complex> unit(ortho_dim_size, 1.0);
    std::vector<Complex> n_unit(ortho_dim_size, {-1.0, 0.0});

    for (int i = 0; i < size; i++) {
      // r = A * v_i
      mat(tmp, evecs[i]);

      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      // Block dot products stored in s_t.
      blas3d::cDotProduct(evals_t[i], evecs[i], tmp);
      blas3d::reDotProduct(norms, evecs[i], evecs[i]);
      for (int t = 0; t < ortho_dim_size; t++) evals_t[i][t] /= sqrt(norms[t]);

      // Measure ||lambda_i*v_i - A*v_i||
      blas3d::caxpby(evals_t[i], evecs[i], n_unit, tmp);
      blas3d::reDotProduct(norms, tmp, tmp);
      for (int t = 0; t < ortho_dim_size; t++) residua_3D[t][i] = sqrt(norms[t]);
    }

    // If size = n_conv, this routine is called post sort
    if (size == n_conv) {
      evals.resize(ortho_dim_size * comm_dim_global(ortho_dim) * n_conv, 0.0);

      if (verbose_rank) {
        int t_offset = ortho_dim_size * comm_coord_global(3);
        for (int t = 0; t < ortho_dim_size; t++) {
          for (int i = 0; i < size; i++) {

            // Use printf to get data from t dim only
            if (getVerbosity() >= QUDA_VERBOSE) {
              printf("Eval[%02d][%04d] = (%+.16e,%+.16e) residual = %+.16e\n", t_offset + t, i, evals_t[i][t].real(),
                     evals_t[i][t].imag(), residua_3D[t][i]);
            }

            // Transfer evals to eval array
            evals[(t_offset + t) * size + i] = evals_t[i][t];
          }
        }
      }
    }
  }

} // namespace quda
