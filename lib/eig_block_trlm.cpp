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
  // Thick Restarted Block Lanczos Method constructor
  BLKTRLM::BLKTRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    TRLM(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Block Thick restart specific checks
    if (n_kr < n_ev + 6) errorQuda("n_kr=%d must be greater than n_ev+6=%d\n", n_kr, n_ev + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }

    if (n_kr % block_size != 0) {
      errorQuda("Block size %d is not a factor of the Krylov space size %d", block_size, n_kr);
    }

    if (n_ev % block_size != 0) {
      errorQuda("Block size %d is not a factor of the compressed space %d", block_size, n_ev);
    }

    if (block_size <= 0) errorQuda("Block size %d passed to block eigensolver must be positive", block_size);

    if (block_size > n_conv) errorQuda("block_size = %d cannot exceed n_conv = %d", block_size, n_conv);

    auto n_blocks = n_kr / block_size;
    block_data_length = block_size * block_size;
    auto arrow_mat_array_size = block_data_length * n_blocks;
    // Tridiagonal/Arrow matrix
    block_alpha.resize(arrow_mat_array_size, 0.0);
    block_beta.resize(arrow_mat_array_size, 0.0);

    // Temp storage used in blockLanczosStep
    jth_block.resize(block_data_length);

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void BLKTRLM::operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
  {
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
    // DMH: This is an important step. With block solvers, initial guesses
    //      of block sizes N can be subspaces rich in extremal eigenmodes,
    //      N times more rich than non-blocked solvers.
    //      Final paragraph, IV.B https://arxiv.org/pdf/1902.02064.pdf
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

    // Begin BLOCK TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (int step = num_keep; step < n_kr; step += block_size) blockLanczosStep(kSpace, step);
      iter += (n_kr - num_keep);

      // Solve current block tridiag
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromBlockArrowMat();
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

      // In order to maintain the block structure, we truncate the
      // algorithmic variables to be multiples of the block size
      iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - num_locked - 12);
      iter_keep = (iter_keep / block_size) * block_size;
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      computeBlockKeptRitz(kSpace);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      num_converged = num_locked + iter_converged;
      num_keep = num_locked + iter_keep;
      num_locked += iter_locked;

      // In order to maintain the block structure, we truncate the
      // algorithmic variables to be multiples of the block size
      num_converged = (num_converged / block_size) * block_size;
      num_keep = (num_keep / block_size) * block_size;
      num_locked = (num_locked / block_size) * block_size;

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
      if (num_converged >= n_conv) converged = true;
      restart_iter++;
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      if (eig_param->require_convergence) {
        errorQuda("BLOCK TRLM failed to compute the requested %d vectors with a %d search space, %d block size, "
                  "and %d Krylov space in %d restart steps. Exiting.",
                  n_conv, n_ev, n_kr, block_size, max_restarts);
      } else {
        warningQuda("BLOCK TRLM failed to compute the requested %d vectors with a %d search space, %d block size, "
                    "and %d Krylov space in %d restart steps. Continuing with current lanczos factorisation.",
                    n_conv, n_ev, n_kr, block_size, max_restarts);
      }
    } else {
      logQuda(QUDA_SUMMARIZE,
              "BLOCK TRLM computed the requested %d vectors in %d restart steps with %d block size and "
              "%d BLOCKED OP*x operations.\n",
              n_conv, restart_iter, block_size, iter / block_size);

      // Dump all Ritz values and residua
      for (int i = 0; i < n_conv; i++) {
        logQuda(QUDA_SUMMARIZE, "RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
      }

      // Compute eigenvalues
      computeEvals(kSpace, evals);
      if (compute_svd) computeSVD(kSpace, evals);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Block Thick Restart Member functions
  //---------------------------------------------------------------------------
  void BLKTRLM::blockLanczosStep(std::vector<ColorSpinorField> &v, int j)
  {
    // Compute r = A * v_j - b_{j-i} * v_{j-1}

    // Offset for alpha, beta matrices
    int arrow_offset = j * block_size;
    int idx = 0, idx_conj = 0;

    // r = A * v_j
    //for (int b = 0; b < block_size; b++) chebyOp(mat, r[b], v[j + b]);
    chebyOp({r.begin(), r.begin() + block_size}, {v.begin() + j, v.begin() + j + block_size});

    // r = r - b_{j-1} * v_{j-1}
    int start = (j > num_keep) ? j - block_size : 0;

    if (j - start > 0) {
      int blocks = (j - start) / block_size;
      std::vector<Complex> beta_(blocks * block_data_length);

      // Switch beta block order from COLUMN to ROW major
      // This switches the block from upper to lower triangular
      for (int i = 0; i < blocks; i++) {
        int block_offset = (i + start / block_size) * block_data_length;
        for (int b = 0; b < block_size; b++) {
          for (int c = 0; c < block_size; c++) {
            idx = c * block_size + b;
            beta_[(i * block_size + b) * block_size + c] = -block_beta[block_offset + idx];
          }
        }
      }

      if (blocks == 1)
        blas::caxpy_L(beta_, {v.begin() + start, v.begin() + j}, {r.begin(), r.begin() + block_size});
      else
        blas::caxpy(beta_, {v.begin() + start, v.begin() + j}, {r.begin(), r.begin() + block_size});
    }

    // a_j = v_j^dag * r
    // Block dot products stored in alpha_block.
    std::vector<Complex> block_alpha_(block_size);
    blas::cDotProduct(block_alpha_, {v.begin() + j, v.begin() + j + block_size}, {r.begin(), r.end()});
    for (auto i = 0u; i < block_alpha_.size(); i++) block_alpha[arrow_offset + i] = block_alpha_[i];

    // Use jth_block to negate alpha data and apply block BLAS.
    // Data is in square hermitian form, no need to switch to ROW major
    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < block_size; c++) {
        idx = b * block_size + c;
        jth_block[idx] = -1.0 * block_alpha[arrow_offset + idx];
      }
    }

    // r = r - a_j * v_j
    blas::caxpy(jth_block, {v.begin() + j, v.begin() + j + block_size}, {r.begin(), r.end()});

    // Orthogonalise R[0:block_size] against the Krylov space V[0:j + block_size]
    for (int k = 0; k < 1; k++) blockOrthogonalizeHMGS(v, r, ortho_block_size, j + block_size);

    // QR decomposition via modified Gram-Schmidt
    // NB, QR via modified Gram-Schmidt is numerically unstable.
    // We perform the QR iteratively to recover numerical stability.
    //
    // Q_0 * R_0(V)   -> Q_0 * R_0 = V
    // Q_1 * R_1(Q_0) -> Q_1 * R_1 = V * R_0^-1 -> Q_1 * R_1 * R_0 = V
    // ...
    // Q_k * R_k(Q_{k-1}) -> Q_k * R_k * R_{k-1} * ... * R_0 = V
    //
    // Where the Q_k are orthonormal to MP and (R_k * R_{k-1} * ... * R_0)^1
    // is the matrix that maps V -> Q_k.

    // Column major order
    bool orthed = false;
    int k = 0;
    while (!orthed && k < max_ortho_attempts) {
      // Compute R_{k}
      logQuda(QUDA_DEBUG_VERBOSE, "Orthonormalisation attempt k = %d\n", k);
      for (int b = 0; b < block_size; b++) {
        double norm = sqrt(blas::norm2(r[b]));
        blas::ax(1.0 / norm, r[b]);
        jth_block[b * (block_size + 1)] = norm;
        for (int c = b + 1; c < block_size; c++) {

          Complex cnorm = blas::cDotProduct(r[b], r[c]);
          blas::caxpy(-cnorm, r[b], r[c]);

          idx = c * block_size + b;
          idx_conj = b * block_size + c;

          jth_block[idx] = cnorm;
          jth_block[idx_conj] = 0.0;
        }
      }
      // Accumulate R_{k} products
      updateBlockBeta(k, arrow_offset);
      orthed = orthoCheck(r, block_size);
      k++;
    }

    if (!orthed) errorQuda("Block TRLM unable to orthonormalise the block residual after %d iterations", k + 1);

    // Prepare next step.
    // v_{j+1} = r
    for (int b = 0; b < block_size; b++) v[j + block_size + b] = r[b];
  }

  void BLKTRLM::updateBlockBeta(int k, int arrow_offset)
  {
    if (k == 0) {
      // Copy over the jth_block matrix to block beta, Beta = R_0
      int idx = 0;
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < b + 1; c++) {
          idx = b * block_size + c;
          block_beta[arrow_offset + idx] = jth_block[idx];
        }
      }
    } else {
      // Compute BetaNew_ac = (R_k)_ab * Beta_bc
      // Use Eigen, it's neater
      MatrixXcd betaN = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd beta = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd Rk = MatrixXcd::Zero(block_size, block_size);
      int idx = 0;

      // Populate matrices
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < b + 1; c++) {
          idx = b * block_size + c;
          beta(c, b) = block_beta[arrow_offset + idx];
          Rk(c, b) = jth_block[idx];
        }
      }

      // Multiply using Eigen
      betaN = Rk * beta;

      // Copy back to beta array
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < b + 1; c++) {
          idx = b * block_size + c;
          block_beta[arrow_offset + idx] = betaN(c, b);
        }
      }
    }
  }

  void BLKTRLM::eigensolveFromBlockArrowMat()
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    int dim = n_kr - num_locked;
    if (dim % block_size != 0) errorQuda("dim = %d modulo block_size = %d != 0", dim, block_size);
    int blocks = dim / block_size;

    int arrow_pos = num_keep - num_locked;
    if (arrow_pos % block_size != 0) errorQuda("arrow_pos = %d modulo block_size = %d != 0", arrow_pos, block_size);
    int block_arrow_pos = arrow_pos / block_size;
    int num_locked_offset = (num_locked / block_size) * block_data_length;

    // Eigen objects
    MatrixXcd T = MatrixXcd::Zero(dim, dim);
    block_ritz_mat.resize(dim * dim);
    int idx = 0;

    // Populate the r and eblocks
    for (int i = 0; i < block_arrow_pos; i++) {
      for (int b = 0; b < block_size; b++) {
        // E block
        idx = i * block_size + b;
        T(idx, idx) = alpha[idx + num_locked];

        for (int c = 0; c < block_size; c++) {
          // r blocks
          idx = num_locked_offset + b * block_size + c;
          T(arrow_pos + c, i * block_size + b) = block_beta[i * block_data_length + idx];
          T(i * block_size + b, arrow_pos + c) = conj(block_beta[i * block_data_length + idx]);
        }
      }
    }

    // Add the alpha blocks
    for (int i = block_arrow_pos; i < blocks; i++) {
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < block_size; c++) {
          idx = num_locked_offset + b * block_size + c;
          T(i * block_size + b, i * block_size + c) = block_alpha[i * block_data_length + idx];
        }
      }
    }

    // Add the beta blocks
    for (int i = block_arrow_pos; i < blocks - 1; i++) {
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < b + 1; c++) {
          idx = num_locked_offset + b * block_size + c;
          // Sub diag
          T((i + 1) * block_size + c, i * block_size + b) = block_beta[i * block_data_length + idx];
          // Super diag
          T(i * block_size + b, (i + 1) * block_size + c) = conj(block_beta[i * block_data_length + idx]);
        }
      }
    }

    // Invert the spectrum due to Chebyshev (except the arrow diagonal)
    if (reverse) {
      for (int b = 0; b < dim; b++) {
        for (int c = 0; c < dim; c++) {
          T(c, b) *= -1.0;
          if (restart_iter > 0)
            if (b == c && b < arrow_pos && c < arrow_pos) T(c, b) *= -1.0;
        }
      }
    }

    // Eigensolve the arrow matrix
    SelfAdjointEigenSolver<MatrixXcd> eigensolver;
    eigensolver.compute(T);

    // Populate the alpha array with eigenvalues
    for (int i = 0; i < dim; i++) alpha[i + num_locked] = eigensolver.eigenvalues()[i];

    // Repopulate ritz matrix: COLUMN major
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) block_ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];

    // Use Sum of all beta values in the final block for
    // the convergence condition
    double beta_sum = 0;
    for (int i = 0; i < block_data_length; i++) beta_sum += fabs(block_beta[n_kr * block_size - block_data_length + i]);

    for (int i = 0; i < blocks; i++) {
      for (int b = 0; b < block_size; b++) {
        idx = b * (block_size + 1);
        residua[i * block_size + b + num_locked] = fabs(beta_sum * block_ritz_mat[dim * (i * block_size + b + 1) - 1]);
      }
    }

    profile.TPSTOP(QUDA_PROFILE_EIGEN);
  }

  void BLKTRLM::computeBlockKeptRitz(std::vector<ColorSpinorField> &kSpace)
  {
    int offset = n_kr + block_size;
    int dim = n_kr - num_locked;

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    std::vector<Complex> ritz_mat_keep(dim * iter_keep);
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < iter_keep; i++) { ritz_mat_keep[j * iter_keep + i] = block_ritz_mat[i * dim + j]; }
    }

    rotateVecs(kSpace, ritz_mat_keep, offset, dim, iter_keep, num_locked, profile);

    // Update residual vectors
    for (int i = 0; i < block_size; i++) std::swap(kSpace[num_locked + iter_keep + i], kSpace[n_kr + i]);

    // Compute new r blocks
    // Use Eigen, it's neater
    MatrixXcd beta = MatrixXcd::Zero(block_size, block_size);
    MatrixXcd ri = MatrixXcd::Zero(block_size, block_size);
    MatrixXcd ritzi = MatrixXcd::Zero(block_size, block_size);
    int blocks = iter_keep / block_size;
    int idx = 0;
    int beta_offset = n_kr * block_size - block_data_length;
    int num_locked_offset = num_locked * block_size;

    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < b + 1; c++) {
        idx = b * block_size + c;
        beta(c, b) = block_beta[beta_offset + idx];
      }
    }
    for (int i = 0; i < blocks; i++) {
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < block_size; c++) {
          idx = i * block_size * dim + b * dim + (dim - block_size) + c;
          ritzi(c, b) = block_ritz_mat[idx];
        }
      }

      ri = beta * ritzi;
      for (int b = 0; b < block_size; b++) {
        for (int c = 0; c < block_size; c++) {
          idx = num_locked_offset + b * block_size + c;
          block_beta[i * block_data_length + idx] = ri(c, b);
        }
      }
    }
  }

} // namespace quda
