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
  // Implicitly Restarted Arnoldi Method constructor
  IRAM::IRAM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    EigenSolver(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Upper Hessenberg, Q and R matrices
    upperHess.resize(n_kr);
    Qmat.resize(n_kr);
    Rmat.resize(n_kr);
    for (int i = 0; i < n_kr; i++) {
      upperHess[i].resize(n_kr, 0.0);
      Qmat[i].resize(n_kr, 0.0);
      Rmat[i].resize(n_kr, 0.0);
    }

    if (eig_param->qr_tol == 0) { eig_param->qr_tol = eig_param->tol * 1e-2; }

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  // Arnoldi Member functions
  //---------------------------------------------------------------------------
  void IRAM::arnoldiStep(std::vector<ColorSpinorField> &v, std::vector<ColorSpinorField> &r, double &beta, int j)
  {
    beta = sqrt(blas::norm2(r[0]));
    if (j > 0) upperHess[j][j - 1] = beta;

    // v_{j} = r_{j-1}/beta
    blas::ax(1.0 / beta, r[0]);
    std::swap(v[j], r[0]);

    // r_{j} = M * v_{j};
    mat(r[0], v[j]);

    double beta_pre = sqrt(blas::norm2(r[0]));

    // Compute the j-th residual corresponding
    // to the j step factorization.
    // Use Classical Gram Schmidt and compute:
    // w_{j} <-  V_{j}^dag * M * v_{j}
    // r_{j} <-  M * v_{j} - V_{j} * w_{j}

    // H_{j,i}_j = v_i^dag * r
    std::vector<Complex> tmp(j + 1);
    blas::cDotProduct(tmp, {v.begin(), v.begin() + j + 1}, r);

    // Orthogonalise r_{j} against V_{j}.
    // r = r - H_{j,i} * v_j
    for (int i = 0; i < j + 1; i++) tmp[i] *= -1.0;
    blas::caxpy(tmp, {v.begin(), v.begin() + j + 1}, r);
    for (int i = 0; i < j + 1; i++) upperHess[i][j] = -1.0 * tmp[i];

    // Re-orthogonalization / Iterative refinement phase
    // Maximum 100 tries.

    // s      = V_{j}^T * B * r_{j}
    // r_{j}  = r_{j} - V_{j}*s
    // alphaj = alphaj + s_{j}

    // The stopping criteria used for iterative refinement is
    // discussed in Parlett's book SEP, page 107 and in Gragg &
    // Reichel ACM TOMS paper; Algorithm 686, Dec. 1990.
    // Determine if we need to correct the residual. The goal is
    // to enforce ||v(:,1:j)^T * r_{j}|| .le. eps * || r_{j} ||
    // The following test determines whether the sine of the
    // angle between  OP*x and the computed residual is less
    // than or equal to 0.717. In practice, more than one
    // step of iterative refinement is rare.

    int orth_iter = 0;
    int orth_iter_max = 100;
    beta = sqrt(blas::norm2(r[0]));
    while (beta < 0.717 * beta_pre && orth_iter < orth_iter_max) {

      logQuda(QUDA_DEBUG_VERBOSE, "beta = %e > 0.717*beta_pre = %e: Reorthogonalise at step %d, iter %d\n", beta,
              0.717 * beta_pre, j, orth_iter);

      beta_pre = beta;

      // Compute the correction to the residual:
      // r_{j} = r_{j} - V_{j} * r_{j}
      // and adjust for the correction in the
      // upper Hessenberg matrix.
      blas::cDotProduct(tmp, {v.begin(), v.begin() + j + 1}, r);
      for (int i = 0; i < j + 1; i++) tmp[i] *= -1.0;
      blas::caxpy(tmp, {v.begin(), v.begin() + j + 1}, r);
      for (int i = 0; i < j + 1; i++) upperHess[i][j] -= tmp[i];

      beta = sqrt(blas::norm2(r[0]));
      orth_iter++;
    }

    if (orth_iter == orth_iter_max) { errorQuda("Unable to orthonormalise r"); }
  }

  void IRAM::rotateBasis(std::vector<ColorSpinorField> &kSpace, int keep)
  {
    // Multi-BLAS friendly array to store the part of the rotation matrix
    std::vector<Complex> Qmat_keep(n_kr * keep, 0.0);
    for (int j = 0; j < n_kr; j++)
      for (int i = 0; i < keep; i++) { Qmat_keep[j * keep + i] = Qmat[j][i]; }

    rotateVecs(kSpace, Qmat_keep, n_kr, n_kr, keep, 0, profile);
  }

  void IRAM::reorder(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals,
                     const QudaEigSpectrumType spec_type)
  {
    int n = n_kr;
    std::vector<std::tuple<Complex, double, ColorSpinorField>> array(n);
    for (int i = 0; i < n; i++) array[i] = std::make_tuple(evals[i], residua[i], std::move(kSpace[i]));

    switch (spec_type) {
    case QUDA_SPECTRUM_LM_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (abs(std::get<0>(a)) > abs(std::get<0>(b)));
                });
      break;
    case QUDA_SPECTRUM_SM_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (abs(std::get<0>(a)) < abs(std::get<0>(b)));
                });
      break;
    case QUDA_SPECTRUM_LR_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (std::get<0>(a).real() > std::get<0>(b).real());
                });
      break;
    case QUDA_SPECTRUM_SR_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (std::get<0>(a).real() < std::get<0>(b).real());
                });
      break;
    case QUDA_SPECTRUM_LI_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (std::get<0>(a).imag() > std::get<0>(b).imag());
                });
      break;
    case QUDA_SPECTRUM_SI_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::tuple<Complex, double, ColorSpinorField> &a,
                   const std::tuple<Complex, double, ColorSpinorField> &b) {
                  return (std::get<0>(a).imag() < std::get<0>(b).imag());
                });
      break;
    default: errorQuda("Undefined spectrum type %d given", spec_type);
    }

    // Repopulate arrays with sorted elements
    for (int i = 0; i < n; i++) {
      std::swap(evals[i], std::get<0>(array[i]));
      std::swap(residua[i], std::get<1>(array[i]));
      std::swap(kSpace[i], std::get<2>(array[i]));
    }
  }

  void IRAM::qrShifts(const std::vector<Complex> evals, const int num_shifts)
  {
    // This isn't really Eigen, but it's morally equivalent
    profile.TPSTART(QUDA_PROFILE_HOST_COMPUTE);

    // Reset Q to the identity, copy upper Hessenberg
    MatrixXcd UHcopy = MatrixXcd::Zero(n_kr, n_kr);
    for (int i = 0; i < n_kr; i++) {
      for (int j = 0; j < n_kr; j++) {
        if (i == j)
          Qmat[i][j] = 1.0;
        else
          Qmat[i][j] = 0.0;
        UHcopy(i, j) = upperHess[i][j];
      }
    }

    for (int shift = 0; shift < num_shifts; shift++) {

      // Shift the eigenvalue
      for (int i = 0; i < n_kr; i++) upperHess[i][i] -= evals[shift];

      qrIteration(Qmat, upperHess);

      for (int i = 0; i < n_kr; i++) upperHess[i][i] += evals[shift];
    }

    profile.TPSTOP(QUDA_PROFILE_HOST_COMPUTE);
  }

  void IRAM::qrIteration(std::vector<std::vector<Complex>> &Q, std::vector<std::vector<Complex>> &R)
  {
    Complex T11, T12, T21, T22, U1, U2;
    double dV;

    double tol = eig_param->qr_tol;

    // Allocate the rotation matrices.
    std::vector<Complex> R11(n_kr - 1, 0.0);
    std::vector<Complex> R12(n_kr - 1, 0.0);
    std::vector<Complex> R21(n_kr - 1, 0.0);
    std::vector<Complex> R22(n_kr - 1, 0.0);

    for (int i = 0; i < n_kr - 1; i++) {

      // If the sub-diagonal element is numerically
      // small enough, floor it to 0;
      if (abs(R[i + 1][i]) < tol) {
        R[i + 1][i] = 0.0;
        continue;
      }

      U1 = R[i][i];
      dV = sqrt(norm(R[i][i]) + norm(R[i + 1][i]));
      dV = (U1.real() > 0) ? dV : -dV;
      U1 += dV;
      U2 = R[i + 1][i];

      T11 = conj(U1) / dV;
      R11[i] = conj(T11);

      T12 = conj(U2) / dV;
      R12[i] = conj(T12);

      T21 = conj(T12) * conj(U1) / U1;
      R21[i] = conj(T21);

      T22 = T12 * U2 / U1;
      R22[i] = conj(T22);

      // Do the H_kk and set the H_k+1k to zero
      R[i][i] -= (T11 * R[i][i] + T12 * R[i + 1][i]);
      R[i + 1][i] = 0;

      // Continue for the other columns
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 32)
#endif
      for (int j = i + 1; j < n_kr; j++) {
        Complex temp = R[i][j];
        R[i][j] -= (T11 * temp + T12 * R[i + 1][j]);
        R[i + 1][j] -= (T21 * temp + T22 * R[i + 1][j]);
      }
    }

    // Rotate R and V, i.e. H->RQ. V->VQ
    // Loop over columns of upper Hessenberg
    for (int j = 0; j < n_kr - 1; j++) {
      if (abs(R11[j]) > tol) {
        // Loop over the rows, up to the sub diagonal element i=j+1
#ifdef _OPENMP
#pragma omp parallel
        {
#pragma omp for schedule(static, 32) nowait
#endif
          for (int i = 0; i < j + 2; i++) {
            Complex temp = R[i][j];
            R[i][j] -= (R11[j] * temp + R12[j] * R[i][j + 1]);
            R[i][j + 1] -= (R21[j] * temp + R22[j] * R[i][j + 1]);
          }
#ifdef _OPENMP
#pragma omp for schedule(static, 32) nowait
#endif
          for (int i = 0; i < n_kr; i++) {
            Complex temp = Q[i][j];
            Q[i][j] -= (R11[j] * temp + R12[j] * Q[i][j + 1]);
            Q[i][j + 1] -= (R21[j] * temp + R22[j] * Q[i][j + 1]);
          }
#ifdef _OPENMP
        }
#endif
      }
    }
  }

  void IRAM::eigensolveFromUpperHess(std::vector<Complex> &evals, const double beta)
  {
    if (eig_param->use_eigen_qr) {
      profile.TPSTART(QUDA_PROFILE_EIGENQR);
      // Construct the upper Hessenberg matrix
      MatrixXcd Q = MatrixXcd::Identity(n_kr, n_kr);
      MatrixXcd R = MatrixXcd::Zero(n_kr, n_kr);
      for (int i = 0; i < n_kr; i++) {
        for (int j = 0; j < n_kr; j++) { R(i, j) = upperHess[i][j]; }
      }

      // QR the upper Hessenberg matrix
      Eigen::ComplexSchur<MatrixXcd> schurUH;
      schurUH.computeFromHessenberg(R, Q);
      profile.TPSTOP(QUDA_PROFILE_EIGENQR);

      profile.TPSTART(QUDA_PROFILE_EIGENEV);
      // Extract the upper triangular matrix, eigensolve, then
      // get the eigenvectors of the upper Hessenberg
      MatrixXcd matUpper = MatrixXcd::Zero(n_kr, n_kr);
      matUpper = schurUH.matrixT().triangularView<Eigen::Upper>();
      matUpper.conservativeResize(n_kr, n_kr);
      Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver(matUpper);
      Q = schurUH.matrixU() * eigenSolver.eigenvectors();

      // Update eigenvalues, residuia, and the Q matrix
      for (int i = 0; i < n_kr; i++) {
        evals[i] = eigenSolver.eigenvalues()[i];
        residua[i] = abs(beta * Q.col(i)[n_kr - 1]);
        for (int j = 0; j < n_kr; j++) Qmat[i][j] = Q(i, j);
      }
      profile.TPSTOP(QUDA_PROFILE_EIGENEV);
    } else {
      profile.TPSTART(QUDA_PROFILE_HOST_COMPUTE);
      // Copy the upper Hessenberg matrix into Rmat, and set Qmat to the identity
      for (int i = 0; i < n_kr; i++) {
        for (int j = 0; j < n_kr; j++) {
          Rmat[i][j] = upperHess[i][j];
          if (i == j)
            Qmat[i][j] = 1.0;
          else
            Qmat[i][j] = 0.0;
        }
      }

      // This is about as high as one cat get in double without causing
      // the Arnoldi to compute more restarts.
      double tol = eig_param->qr_tol;
      int max_iter = 100000;
      int iter = 0;

      Complex temp, discriminant, sol1, sol2, eval;
      for (int i = n_kr - 2; i >= 0; i--) {
        while (iter < max_iter) {
          if (abs(Rmat[i + 1][i]) < tol) {
            Rmat[i + 1][i] = 0.0;
            break;
          } else {

            // Compute the 2 eigenvalues via the quadratic formula
            //----------------------------------------------------
            // The discriminant
            temp = (Rmat[i][i] - Rmat[i + 1][i + 1]) * (Rmat[i][i] - Rmat[i + 1][i + 1]) / 4.0;
            discriminant = sqrt(Rmat[i + 1][i] * Rmat[i][i + 1] + temp);

            // Reuse temp
            temp = (Rmat[i][i] + Rmat[i + 1][i + 1]) / 2.0;

            sol1 = temp - Rmat[i + 1][i + 1] + discriminant;
            sol2 = temp - Rmat[i + 1][i + 1] - discriminant;
            //----------------------------------------------------

            // Deduce the better eval to shift
            eval = Rmat[i + 1][i + 1] + (norm(sol1) < norm(sol2) ? sol1 : sol2);

            // Shift the eigenvalue
            for (int j = 0; j < n_kr; j++) Rmat[j][j] -= eval;

            // Do the QR iteration
            qrIteration(Qmat, Rmat);

            // Shift back
            for (int j = 0; j < n_kr; j++) Rmat[j][j] += eval;
          }
          iter++;
        }
      }
      profile.TPSTOP(QUDA_PROFILE_HOST_COMPUTE);

      profile.TPSTART(QUDA_PROFILE_EIGENEV);
      // Compute the eigevectors of the origial upper Hessenberg
      // This is now very cheap because the input matrix to Eigen
      // is upper triangular.
      MatrixXcd Q = MatrixXcd::Zero(n_kr, n_kr);
      MatrixXcd R = MatrixXcd::Zero(n_kr, n_kr);
      for (int i = 0; i < n_kr; i++) {
        for (int j = 0; j < n_kr; j++) {
          Q(i, j) = Qmat[i][j];
          R(i, j) = Rmat[i][j];
        }
      }

      MatrixXcd matUpper = MatrixXcd::Zero(n_kr, n_kr);
      matUpper = R.triangularView<Eigen::Upper>();
      matUpper.conservativeResize(n_kr, n_kr);
      Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver(matUpper);
      Q *= eigenSolver.eigenvectors();

      // Update eigenvalues, residuia, and the Q matrix
      for (int i = 0; i < n_kr; i++) {
        evals[i] = eigenSolver.eigenvalues()[i];
        residua[i] = abs(beta * Q.col(i)[n_kr - 1]);
        for (int j = 0; j < n_kr; j++) Qmat[i][j] = Q(i, j);
      }

      logQuda(QUDA_VERBOSE, "QR iterations = %d\n", iter);
      profile.TPSTOP(QUDA_PROFILE_EIGENEV);
    }
  }

  void IRAM::operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
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

    // Apply a matrix op to the residual to place it in the
    // range of the operator
    mat(r[0], kSpace[0]);

    // Convergence criteria
    double epsilon = setEpsilon(kSpace[0].Precision());
    double epsilon23 = pow(epsilon, 2.0 / 3.0);
    double beta = 0.0;

    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin IRAM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    num_keep = 0;
    while (restart_iter < max_restarts && !converged) {
      for (int step = num_keep; step < n_kr; step++) arnoldiStep(kSpace, r, beta, step);
      iter += n_kr - num_keep;

      // Ritz values and their errors are updated.
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromUpperHess(evals, beta);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      num_keep = n_ev;
      int num_shifts = n_kr - num_keep;

      // Put unwanted Ritz values first. We will shift these out of the
      // Krylov space using QR.
      sortArrays(eig_param->spectrum, n_kr, evals, residua);

      // Put smallest errors on the unwanted Ritz first to aid in forward stability
      sortArrays(QUDA_SPECTRUM_LM_EIG, num_shifts, residua, evals);

      // Convergence test
      iter_converged = 0;
      for (int i = 0; i < n_ev; i++) {
        int idx = n_kr - 1 - i;
        double rtemp = std::max(epsilon23, abs(evals[idx]));
        if (residua[idx] < tol * rtemp) {
          iter_converged++;
          logQuda(QUDA_DEBUG_VERBOSE, "residuum[%d] = %e, condition = %e\n", i, residua[idx], tol * abs(evals[idx]));
        } else {
          // Unlikely to find new converged eigenvalues
          break;
        }
      }

      int num_keep0 = num_keep;
      iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - 12);

      num_converged = iter_converged;
      num_keep = iter_keep;
      num_shifts = n_kr - num_keep;

      logQuda(QUDA_VERBOSE, "%04d converged eigenvalues at iter %d\n", num_converged, restart_iter);

      if (num_converged >= n_conv) {

        profile.TPSTOP(QUDA_PROFILE_COMPUTE);
        eigensolveFromUpperHess(evals, beta);
        // Rotate the Krylov space
        rotateBasis(kSpace, n_kr);
        // Reorder the Krylov space and Ritz values
        reorder(kSpace, evals, eig_param->spectrum);

        // Compute the eigen/singular values.
        profile.TPSTART(QUDA_PROFILE_COMPUTE);
        computeEvals(kSpace, evals);
        if (compute_svd) computeSVD(kSpace, evals);
        converged = true;

      } else {

        // If num_keep changed, we resort the Ritz values and residua
        if (num_keep0 < num_keep) {
          sortArrays(eig_param->spectrum, n_kr, evals, residua);
          sortArrays(QUDA_SPECTRUM_LM_EIG, num_shifts, residua, evals);
        }

        profile.TPSTOP(QUDA_PROFILE_COMPUTE);
        // Apply the shifts of the unwated Ritz values via QR
        qrShifts(evals, num_shifts);

        // Compress the Krylov space using the accumulated Givens rotations in Qmat
        rotateBasis(kSpace, num_keep + 1);
        profile.TPSTART(QUDA_PROFILE_COMPUTE);

        // Update the residual vector
        blas::caxpby(upperHess[num_keep][num_keep - 1], kSpace[num_keep], Qmat[n_kr - 1][num_keep - 1], r[0]);

        if (sqrt(blas::norm2(r[0])) < epsilon) { errorQuda("IRAM has encountered an invariant subspace..."); }
      }
      restart_iter++;
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      if (eig_param->require_convergence) {
        errorQuda("IRAM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                  "restart steps. Exiting.",
                  n_conv, n_ev, n_kr, max_restarts);
      } else {
        warningQuda("IRAM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                    "restart steps. Continuing with current arnoldi factorisation.",
                    n_conv, n_ev, n_kr, max_restarts);
      }
    } else {
      logQuda(QUDA_SUMMARIZE,
              "IRAM computed the requested %d vectors with a %d search space and %d Krylov space in %d "
              "restart steps and %d OP*x operations.\n",
              n_conv, n_ev, n_kr, restart_iter, iter);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

} // namespace quda
