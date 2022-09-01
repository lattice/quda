#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cfloat>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <random_quda.h>
#include <qio_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>
#include <tune_quda.h>
#include <vector_io.h>
#include <eigen_helper.h>

namespace quda
{

  // Eigensolver class
  //-----------------------------------------------------------------------------
  EigenSolver::EigenSolver(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    mat(mat),
    eig_param(eig_param),
    profile(profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaEigParam(eig_param);

    // Problem parameters
    n_ev = eig_param->n_ev;
    n_kr = eig_param->n_kr;
    n_conv = eig_param->n_conv;
    n_ev_deflate = (eig_param->n_ev_deflate == -1 ? n_conv : eig_param->n_ev_deflate);
    tol = eig_param->tol;
    reverse = false;

    // Algorithm variables
    converged = false;
    restart_iter = 0;
    max_restarts = eig_param->max_restarts;
    check_interval = eig_param->check_interval;
    batched_rotate = eig_param->batched_rotate;
    block_size = eig_param->block_size;
    iter = 0;
    iter_converged = 0;
    iter_locked = 0;
    iter_keep = 0;
    num_converged = 0;
    num_locked = 0;
    num_keep = 0;

    save_prec = eig_param->save_prec;

    // Sanity checks
    if (n_kr <= n_ev) errorQuda("n_kr = %d is less than or equal to n_ev = %d", n_kr, n_ev);
    if (n_ev < n_conv) errorQuda("n_conv=%d is greater than n_ev=%d", n_conv, n_ev);
    if (n_ev == 0) errorQuda("n_ev=0 passed to Eigensolver");
    if (n_kr == 0) errorQuda("n_kr=0 passed to Eigensolver");
    if (n_conv == 0) errorQuda("n_conv=0 passed to Eigensolver");
    if (n_ev_deflate > n_conv) errorQuda("deflation vecs = %d is greater than n_conv = %d", n_ev_deflate, n_conv);

    residua.resize(n_kr, 0.0);

    // Part of the spectrum to be computed.
    switch (eig_param->spectrum) {
    case QUDA_SPECTRUM_LM_EIG: strcpy(spectrum, "LM"); break;
    case QUDA_SPECTRUM_SM_EIG: strcpy(spectrum, "SM"); break;
    case QUDA_SPECTRUM_LR_EIG: strcpy(spectrum, "LR"); break;
    case QUDA_SPECTRUM_SR_EIG: strcpy(spectrum, "SR"); break;
    case QUDA_SPECTRUM_LI_EIG: strcpy(spectrum, "LI"); break;
    case QUDA_SPECTRUM_SI_EIG: strcpy(spectrum, "SI"); break;
    default: errorQuda("Unexpected spectrum type %d", eig_param->spectrum);
    }

    // Deduce whether to reverse the sorting
    if (strncmp("L", spectrum, 1) == 0 && !eig_param->use_poly_acc) {
      reverse = true;
    } else if (strncmp("S", spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = true;
      spectrum[0] = 'L';
    } else if (strncmp("L", spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = true;
      spectrum[0] = 'S';
    }

    // For normal operators (MdagM, MMdag) the SVD of the
    // underlying operators (M, Mdag) is computed.
    compute_svd = eig_param->compute_svd;

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  // We bake the matrix operator 'mat' and the eigensolver parameters into the
  // eigensolver.
  EigenSolver *EigenSolver::create(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile)
  {
    EigenSolver *eig_solver = nullptr;

    switch (eig_param->eig_type) {
    case QUDA_EIG_IR_ARNOLDI:
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating IR Arnoldi eigensolver\n");
      eig_solver = new IRAM(mat, eig_param, profile);
      break;
    case QUDA_EIG_BLK_IR_ARNOLDI: errorQuda("Block IR Arnoldi not implemented"); break;
    case QUDA_EIG_TR_LANCZOS:
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating TR Lanczos eigensolver\n");
      eig_solver = new TRLM(mat, eig_param, profile);
      break;
    case QUDA_EIG_BLK_TR_LANCZOS:
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating Block TR Lanczos eigensolver\n");
      eig_solver = new BLKTRLM(mat, eig_param, profile);
      break;
    default: errorQuda("Invalid eig solver type");
    }

    // Sanity checks
    //--------------------------------------------------------------------------
    // Cannot solve non-hermitian systems with Lanczos
    if (!mat.hermitian() && eig_solver->hermitian())
      errorQuda("Cannot solve non-Hermitian system with strictly Hermitian eigensolver %d, %d", (int)!mat.hermitian(),
                (int)eig_solver->hermitian());

    // Support for Chebyshev only in strictly Hermitian solvers
    if (eig_param->use_poly_acc) {
      if (!mat.hermitian()) errorQuda("Cannot use polynomial acceleration with non-Hermitian operator");
      if (!eig_solver->hermitian()) errorQuda("Polynomial acceleration not supported with non-Hermitian solver");
    }

    // Cannot solve for imaginary spectrum of hermitian systems
    if (mat.hermitian() && (eig_param->spectrum == QUDA_SPECTRUM_SI_EIG || eig_param->spectrum == QUDA_SPECTRUM_LI_EIG))
      errorQuda("The imaginary spectrum of a Hermitian operator cannot be computed");

    // Cannot compute SVD of non-normal operators
    if (!eig_param->use_norm_op && eig_param->compute_svd)
      errorQuda("Computation of SVD supported for normal operators only");
    //--------------------------------------------------------------------------

    return eig_solver;
  }

  // Utilities and functions common to all Eigensolver instances
  //------------------------------------------------------------------------------
  void EigenSolver::prepareInitialGuess(std::vector<ColorSpinorField> &kSpace)
  {
    // Use 0th vector to extract meta data for the RNG.
    RNG rng(kSpace[0], 1234);
    for (int b = 0; b < block_size; b++) {
      // If the spinor contains initial data from the user
      // preserve it, else populate with rands.
      if (sqrt(blas::norm2(kSpace[b])) == 0.0) { spinorNoise(kSpace[b], rng, QUDA_NOISE_UNIFORM); }
    }

    bool orthed = false;
    int k = 0, kmax = 5;
    while (!orthed && k < kmax) {
      orthonormalizeMGS(kSpace, block_size);
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        if (block_size > 1)
          printfQuda("Orthonormalising initial guesses with Modified Gram Schmidt, iter k=%d/5\n", (k + 1));
        else
          printfQuda("Orthonormalising initial guess\n");
      }
      orthed = orthoCheck(kSpace, block_size);
      k++;
    }
    if (!orthed) errorQuda("Failed to orthonormalise initial guesses");
  }

  void EigenSolver::checkChebyOpMax(const DiracMatrix &mat, std::vector<ColorSpinorField> &kSpace)
  {
    if (eig_param->use_poly_acc && eig_param->a_max <= 0.0) {
      // Use part of the kSpace as temps
      eig_param->a_max = estimateChebyOpMax(mat, kSpace[block_size + 2], kSpace[block_size + 1]);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Chebyshev maximum estimate: %e.\n", eig_param->a_max);
    }
  }

  void EigenSolver::prepareKrylovSpace(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
  {
    resize(kSpace, n_kr + block_size, QUDA_ZERO_FIELD_CREATE); // increase Krylov space to n_kr + block_size
    resize(r, block_size, QUDA_ZERO_FIELD_CREATE, kSpace[0]);  // create residual
    evals.resize(n_kr, 0.0);                                   // increase evals space to n_ev
  }

  void EigenSolver::printEigensolverSetup()
  {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("********************************\n");
      printfQuda("**** START QUDA EIGENSOLVER ****\n");
      printfQuda("********************************\n");
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("spectrum %s\n", spectrum);
      printfQuda("tol %.4e\n", tol);
      printfQuda("n_conv %d\n", n_conv);
      printfQuda("n_ev %d\n", n_ev);
      printfQuda("n_kr %d\n", n_kr);
      if (block_size > 1) printfQuda("block size %d\n", block_size);
      if (batched_rotate > 0) printfQuda("batched rotation size %d\n", batched_rotate);
      if (eig_param->use_poly_acc) {
        printfQuda("polyDeg %d\n", eig_param->poly_deg);
        printfQuda("a-min %f\n", eig_param->a_min);
        printfQuda("a-max %f\n", eig_param->a_max);
      }
    }
  }

  double EigenSolver::setEpsilon(const QudaPrecision prec)
  {
    double eps = 0.0;
    switch (prec) {
    case QUDA_DOUBLE_PRECISION: eps = DBL_EPSILON; break;
    case QUDA_SINGLE_PRECISION: eps = FLT_EPSILON; break;
    case QUDA_HALF_PRECISION: eps = 2e-3; break;
    case QUDA_QUARTER_PRECISION: eps = 5e-2; break;
    default: errorQuda("Invalid precision %d", prec);
    }
    return eps;
  }

  void EigenSolver::queryPrec(const QudaPrecision prec)
  {
    switch (prec) {
    case QUDA_DOUBLE_PRECISION: printfQuda("Running eigensolver in double precision\n"); break;
    case QUDA_SINGLE_PRECISION: printfQuda("Running eigensolver in single precision\n"); break;
    case QUDA_HALF_PRECISION: printfQuda("Running eigensolver in half precision\n"); break;
    case QUDA_QUARTER_PRECISION: printfQuda("Running eigensolver in quarter precision\n"); break;
    default: errorQuda("Invalid precision %d", prec);
    }
  }

  void EigenSolver::cleanUpEigensolver(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals)
  {
    r.clear();

    // Resize Krylov Space
    int n_eig = n_conv;
    if (compute_svd) n_eig *= 2;
    kSpace.resize(n_eig);
    evals.resize(n_conv);

    // Only save if outfile is defined
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving eigenvectors\n");
      const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
      for (auto &k : kSpace) k.setSuggestedParity(mat_parity);

      // save the vectors
      VectorIO io(eig_param->vec_outfile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.save(kSpace, save_prec, n_eig);
    }

    mat.flops();

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("********************************\n");
      printfQuda("***** END QUDA EIGENSOLVER *****\n");
      printfQuda("********************************\n");
    }
  }

  void EigenSolver::chebyOp(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {
    // Just do a simple mat-vec if no poly acc is requested
    if (!eig_param->use_poly_acc) {
      mat(out, in);
      return;
    }

    if (eig_param->poly_deg == 0) errorQuda("Polynomial acceleration requested with zero polynomial degree");

    // Compute the polynomial accelerated operator.
    double a = eig_param->a_min;
    double b = eig_param->a_max;
    double delta = (b - a) / 2.0;
    double theta = (b + a) / 2.0;
    double sigma1 = -delta / theta;
    double sigma;
    double d1 = sigma1 / delta;
    double d2 = 1.0;
    double d3;

    // out = d2 * in + d1 * out
    // C_1(x) = x
    mat(out, in);
    blas::caxpby(d2, const_cast<ColorSpinorField &>(in), d1, out);
    if (eig_param->poly_deg == 1) return;

    // C_0 is the current 'in'  vector.
    // C_1 is the current 'out' vector.

    // Clone 'in' to two temporary vectors.
    auto tmp1 = in;
    auto tmp2 = out;

    // Using Chebyshev polynomial recursion relation,
    // C_{m+1}(x) = 2*x*C_{m} - C_{m-1}

    double sigma_old = sigma1;

    // construct C_{m+1}(x)
    for (int i = 2; i < eig_param->poly_deg; i++) {
      sigma = 1.0 / (2.0 / sigma1 - sigma_old);

      d1 = 2.0 * sigma / delta;
      d2 = -d1 * theta;
      d3 = -sigma * sigma_old;

      // FIXME - we could introduce a fused mat + blas kernel here, eliminating one temporary
      // mat*C_{m}(x)
      mat(out, tmp2);

      blas::axpbypczw(d3, tmp1, d2, tmp2, d1, out, tmp1);
      std::swap(tmp1, tmp2);

      sigma_old = sigma;
    }
    blas::copy(out, tmp2);
  }

  double EigenSolver::estimateChebyOpMax(const DiracMatrix &mat, ColorSpinorField &out, ColorSpinorField &in)
  {
    RNG rng(in, 1234);
    spinorNoise(in, rng, QUDA_NOISE_UNIFORM);

    // Power iteration
    double norm = 0.0;
    for (int i = 0; i < 100; i++) {
      if ((i + 1) % 10 == 0) {
        norm = sqrt(blas::norm2(in));
        blas::ax(1.0 / norm, in);
      }
      mat(out, in);
      std::swap(out, in);
    }

    // Compute spectral radius estimate
    double result = blas::reDotProduct(out, in);

    // Increase final result by 10% for safety
    return result * 1.10;
  }

  bool EigenSolver::orthoCheck(std::vector<ColorSpinorField> &vecs, int size)
  {
    bool orthed = true;
    const Complex Unit(1.0, 0.0);

    std::vector<Complex> H(size * size);
    blas::hDotProduct(H, {vecs.begin(), vecs.begin() + size}, {vecs.begin(), vecs.begin() + size});

    double epsilon = setEpsilon(vecs[0].Precision());

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        auto cnorm = H[i * size + j];
        if (j != i) {
          if (abs(cnorm) > 5.0 * epsilon) {
            if (getVerbosity() >= QUDA_SUMMARIZE)
              printfQuda("Norm <%d|%d>^2 = ||(%e,%e)|| = %e\n", i, j, cnorm.real(), cnorm.imag(), abs(cnorm));
            orthed = false;
          }
        } else {
          if (abs(Unit - cnorm) > 5.0 * epsilon) {
            if (getVerbosity() >= QUDA_SUMMARIZE)
              printfQuda("1 - Norm <%d|%d>^2 = 1 - ||(%e,%e)|| = %e\n", i, j, cnorm.real(), cnorm.imag(),
                         abs(Unit - cnorm));
            orthed = false;
          }
        }
      }
    }

    return orthed;
  }

  void EigenSolver::orthonormalizeMGS(std::vector<ColorSpinorField> &vecs, int size)
  {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < i; j++) {
        Complex cnorm = blas::cDotProduct(vecs[j], vecs[i]); // <j|i> with i normalised.
        blas::caxpy(-cnorm, vecs[j], vecs[i]);               // i = i - proj_{j}(i) = i - <j|i> * j
      }
      double norm = sqrt(blas::norm2(vecs[i]));
      blas::ax(1.0 / norm, vecs[i]); // i/<i|i>
    }
  }

  // Orthogonalise r[0:] against V_[0:j]
  void EigenSolver::blockOrthogonalize(std::vector<ColorSpinorField> &vecs, std::vector<ColorSpinorField> &rvecs, int j)
  {
    auto vecs_size = j;
    auto array_size = vecs_size * rvecs.size();
    std::vector<Complex> s(array_size);

    // Block dot products stored in s.
    blas::cDotProduct(s, {vecs.begin(), vecs.begin() + vecs_size}, {rvecs.begin(), rvecs.end()});

    // Block orthogonalise
    for (auto i = 0u; i < array_size; i++) s[i] *= -1.0;
    blas::caxpy(s, {vecs.begin(), vecs.begin() + vecs_size}, {rvecs.begin(), rvecs.end()});
  }

  void EigenSolver::permuteVecs(std::vector<ColorSpinorField> &kSpace, MatrixXi &mat, int size)
  {
    std::vector<int> pivots(size);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (mat.data()[j * size + i] == 1) { pivots[j] = i; }
      }
    }

    // Identify cycles in the permutation array.
    // We shall use the sign bit as a marker. If the
    // sign is negative, the vector has already been
    // swapped into the correct place. A positive
    // value indicates the start of a new cycle.

    for (int i = 0; i < size; i++) {
      // First cycle always starts at 0, hence OR statement
      if (pivots[i] > 0 || i == 0) {
        int k = i;
        // Identify vector to be placed at i
        int j = pivots[i];
        pivots[i] = -pivots[i];
        while (j > i) {
          std::swap(kSpace[k + num_locked], kSpace[j + num_locked]);
          pivots[j] = -pivots[j];
          k = j;
          j = -pivots[j];
        }
      }
    }
    // Sanity check
    for (int i = 0; i < size; i++) {
      if (pivots[i] > 0) errorQuda("Error at pivot %d", i);
    }
  }

  void EigenSolver::blockReset(std::vector<ColorSpinorField> &kSpace, int j_start, int j_end, int offset)
  {
    // copy back to correct position, zero out the workspace
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      std::swap(kSpace[j + num_locked], kSpace[k]);
      blas::zero(kSpace[k]);
    }
  }

  template <typename T, typename matrix_t>
  void EigenSolver::blockRotate(std::vector<ColorSpinorField> &kSpace, matrix_t &array, int rank,
                                const range &i_range, const range &j_range, blockType b_type, int offset)
  {
    auto block_i_rank = i_range.second - i_range.first;
    auto block_j_rank = j_range.second - j_range.first;

    // Quick return if no op.
    if (block_i_rank == 0 || block_j_rank == 0) return;

    std::vector<T> batch_array(block_i_rank * block_j_rank);
    // Populate batch array (COLUMN major -> ROW major)
    for (int j = j_range.first; j < j_range.second; j++) {
      for (int i = i_range.first; i < i_range.second; i++) {
        int j_arr = j - j_range.first;
        int i_arr = i - i_range.first;
        batch_array[i_arr * block_j_rank + j_arr] = array.data()[j * rank + i];
      }
    }

    // Range of vectors we wish to keep
    auto v = {kSpace.begin() + num_locked + i_range.first, kSpace.begin() + num_locked + i_range.second};

    // Range of the extra space vectors
    auto k = {kSpace.begin() + offset, kSpace.begin() + offset + j_range.second - j_range.first};

    switch (b_type) {
    case PENCIL: blas::axpy(batch_array, v, k); break;
    case LOWER_TRI: blas::axpy_L(batch_array, v, k); break;
    case UPPER_TRI: blas::axpy_U(batch_array, v, k); break;
    default: errorQuda("Undefined MultiBLAS type in blockRotate");
    }
  }

  void EigenSolver::computeSVD(const DiracMatrix &mat, std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals)
  {
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing SVD of M\n");

    int n_conv = eig_param->n_conv;
    if (evecs.size() < (unsigned int)(2 * n_conv))
      errorQuda("Incorrect deflation space sized %d passed to computeSVD, expected %d", (int)(evecs.size()), 2 * n_conv);

    std::vector<double> sigma_tmp(n_conv);

    for (int i = 0; i < n_conv; i++) {

      // This function assumes that you have computed the eigenvectors
      // of MdagM(MMdag), ie, the right(left) SVD of M. The ith eigen vector in the
      // array corresponds to the ith right(left) singular vector. We place the
      // computed left(right) singular vectors in the second half of the array. We
      // assume that right vectors are given and we compute the left.
      //
      // As a cross check, we recompute the singular values from mat vecs rather
      // than make the direct relation (sigma_i)^2 = |lambda_i|
      //--------------------------------------------------------------------------

      // Lambda already contains the square root of the eigenvalue of the norm op.
      Complex lambda = evals[i];

      // M*Rev_i = M*Rsv_i = sigma_i Lsv_i
      mat.Expose()->M(evecs[n_conv + i], evecs[i]);

      // sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
      sigma_tmp[i] = sqrt(blas::norm2(evecs[n_conv + i]));

      // Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
      blas::ax(1.0 / sigma_tmp[i], evecs[n_conv + i]);

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Sval[%04d] = %+.16e sigma - sqrt(|lambda|) = %+.16e\n", i, sigma_tmp[i],
                   sigma_tmp[i] - sqrt(abs(lambda.real())));

      evals[i] = sigma_tmp[i];
      //--------------------------------------------------------------------------
    }
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflateSVD(vector_ref<ColorSpinorField> &&sol, vector_ref<const ColorSpinorField> &&src,
                               vector_ref<const ColorSpinorField> &&evecs, const std::vector<Complex> &evals,
                               bool accumulate) const
  {
    // number of evecs
    if (n_ev_deflate == 0) {
      warningQuda("deflateSVD called with n_ev_deflate = 0");
      return;
    }

    int n_defl = n_ev_deflate;
    if (evecs.size() != (unsigned int)(2 * eig_param->n_conv))
      errorQuda("Incorrect deflation space sized %d passed to deflateSVD, expected %d", (int)(evecs.size()),
                2 * eig_param->n_conv);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Deflating %d left and right singular vectors\n", n_defl);

    // Perform Sum_i R_i * (\sigma_i)^{-1} * L_i^dag * vec = vec_defl
    // for all i computed eigenvectors and values.

    // 1. Take block inner product: L_i^dag * vec = A_i
    std::vector<Complex> s(n_defl * src.size());
    blas::cDotProduct(s, {evecs.begin() + eig_param->n_conv, evecs.begin() + eig_param->n_conv + n_defl},
                      {src.begin(), src.end()});

    // 2. Perform block caxpy
    //    A_i -> (\sigma_i)^{-1} * A_i
    //    vec_defl = Sum_i (R_i)^{-1} * A_i
    if (!accumulate) for (auto &x : sol) blas::zero(x);
    for (int i = 0; i < n_defl; i++) s[i] /= evals[i].real();

    blas::caxpy(s, {evecs.begin(), evecs.begin() + n_defl}, {sol.begin(), sol.end()});
  }

  void EigenSolver::computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField> &evecs,
                                 std::vector<Complex> &evals, int size)
  {
    if (size > (int)evecs.size())
      errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size())
      errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    ColorSpinorParam csParamClone(evecs[0]);
    csParamClone.create = QUDA_NULL_FIELD_CREATE;
    ColorSpinorField temp(csParamClone);

    for (int i = 0; i < size; i++) {
      // r = A * v_i
      mat(temp, evecs[i]);

      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = blas::cDotProduct(evecs[i], temp) / sqrt(blas::norm2(evecs[i]));
      // Measure ||lambda_i*v_i - A*v_i||
      Complex n_unit(-1.0, 0.0);
      blas::caxpby(evals[i], evecs[i], n_unit, temp);
      residua[i] = sqrt(blas::norm2(temp));
      // eig_param->invert_param->true_res_offset[i] = residua[i];

      // If size = n_conv, this routine is called post sort
      if (getVerbosity() >= QUDA_SUMMARIZE && size == n_conv)
        printfQuda("Eval[%04d] = (%+.16e,%+.16e) ||%+.16e|| Residual = %+.16e\n", i, evals[i].real(), evals[i].imag(), abs(evals[i]), residua[i]);
    }
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflate(vector_ref<ColorSpinorField> &&sol, vector_ref<const ColorSpinorField> &&src,
                            vector_ref<const ColorSpinorField> &&evecs, const std::vector<Complex> &evals,
                            bool accumulate) const
  {
    // number of evecs
    if (n_ev_deflate == 0) {
      warningQuda("deflate called with n_ev_deflate = 0");
      return;
    }

    int n_defl = n_ev_deflate;
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Deflating %d vectors\n", n_defl);

    // Perform Sum_i V_i * (L_i)^{-1} * (V_i)^dag * vec = vec_defl
    // for all i computed eigenvectors and values.

    // 1. Take block inner product: (V_i)^dag * vec = A_i
    std::vector<Complex> s(n_defl * src.size());
    blas::cDotProduct(s, {evecs.begin(), evecs.begin() + n_defl}, {src.begin(), src.end()});

    // 2. Perform block caxpy: V_i * (L_i)^{-1} * A_i
    for (int i = 0; i < n_defl; i++) { s[i] /= evals[i].real(); }

    // 3. Accumulate sum vec_defl = Sum_i V_i * (L_i)^{-1} * A_i
    if (!accumulate) for (auto &x : sol) blas::zero(x);

    blas::caxpy(s, {evecs.begin(), evecs.begin() + n_defl}, {sol.begin(), sol.end()});
  }

  void EigenSolver::loadFromFile(const DiracMatrix &mat, std::vector<ColorSpinorField> &kSpace,
                                 std::vector<Complex> &evals)
  {
    // Set suggested parity of fields
    const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
    for (int i = 0; i < n_conv; i++) { kSpace[i].setSuggestedParity(mat_parity); }

    {
      // load the vectors
      VectorIO io(eig_param->vec_infile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.load({kSpace.begin(), kSpace.begin() + n_conv});
    }

    // Create the device side residual vector by cloning
    // the kSpace passed to the function.
    resize(r, 1, QUDA_ZERO_FIELD_CREATE, kSpace[0]);

    // Error estimates (residua) given by ||A*vec - lambda*vec||
    computeEvals(mat, kSpace, evals);
  }

  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<Complex> &y)
  {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      switch (spec_type) {
      case QUDA_SPECTRUM_LM_EIG: printfQuda("'LM' -> sort into decreasing order of magnitude, largest first.\n"); break;
      case QUDA_SPECTRUM_SM_EIG: printfQuda("'SM' -> sort into increasing order of magnitude, smallest first.\n"); break;
      case QUDA_SPECTRUM_LR_EIG: printfQuda("'LR' -> sort with real(x) in decreasing algebraic order, largest first.\n"); break;
      case QUDA_SPECTRUM_SR_EIG: printfQuda("'SR' -> sort with real(x) in increasing algebraic order, smallest first.\n"); break;
      case QUDA_SPECTRUM_LI_EIG: printfQuda("'LI' -> sort with imag(x) in decreasing algebraic order, largest first.\n"); break;
      case QUDA_SPECTRUM_SI_EIG: printfQuda("'SI' -> sort with imag(x) in increasing algebraic order, smallest first\n"); break;
      default: errorQuda("Unkown spectrum type requested: %d", spec_type);
      }
    }

    std::vector<std::pair<Complex, Complex>> array(n);
    for (int i = 0; i < n; i++) array[i] = std::make_pair(x[i], y[i]);

    switch (spec_type) {
    case QUDA_SPECTRUM_LM_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (abs(a.first) < abs(b.first));
                });
      break;
    case QUDA_SPECTRUM_SM_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (abs(a.first) > abs(b.first));
                });
      break;
    case QUDA_SPECTRUM_LR_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (a.first).real() < (b.first).real();
                });
      break;
    case QUDA_SPECTRUM_SR_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (a.first).real() > (b.first).real();
                });
      break;
    case QUDA_SPECTRUM_LI_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (a.first).imag() < (b.first).imag();
                });
      break;
    case QUDA_SPECTRUM_SI_EIG:
      std::sort(array.begin(), array.begin() + n,
                [](const std::pair<Complex, Complex> &a, const std::pair<Complex, Complex> &b) {
                  return (a.first).imag() > (b.first).imag();
                });
      break;
    default: errorQuda("Undefined spectrum type %d given", spec_type);
    }

    // Repopulate x and y arrays with sorted elements
    for (int i = 0; i < n; i++) {
      x[i] = array[i].first;
      y[i] = array[i].second;
    }
  }

  // Overloaded version of sortArrays to deal with real y array.
  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<double> &y)
  {
    std::vector<Complex> y_tmp(n, 0.0);
    for (int i = 0; i < n; i++) y_tmp[i].real(y[i]);
    sortArrays(spec_type, n, x, y_tmp);
    for (int i = 0; i < n; i++) y[i] = y_tmp[i].real();
  }

  // Overloaded version of sortArrays to deal with real x array.
  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<double> &x, std::vector<Complex> &y)
  {
    std::vector<Complex> x_tmp(n, 0.0);
    for (int i = 0; i < n; i++) x_tmp[i].real(x[i]);
    sortArrays(spec_type, n, x_tmp, y);
    for (int i = 0; i < n; i++) x[i] = x_tmp[i].real();
  }

  // Overloaded version of sortArrays to deal with real x and y array.
  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<double> &x, std::vector<double> &y)
  {
    std::vector<Complex> x_tmp(n, 0.0);
    std::vector<Complex> y_tmp(n, 0.0);
    for (int i = 0; i < n; i++) {
      x_tmp[i].real(x[i]);
      y_tmp[i].real(y[i]);
    }
    sortArrays(spec_type, n, x_tmp, y_tmp);
    for (int i = 0; i < n; i++) {
      x[i] = x_tmp[i].real();
      y[i] = y_tmp[i].real();
    }
  }

  // Overloaded version of sortArrays to deal with complex x and integer y array.
  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<int> &y)
  {
    std::vector<Complex> y_tmp(n, 0.0);
    for (int i = 0; i < n; i++) y_tmp[i].real(y[i]);
    sortArrays(spec_type, n, x, y_tmp);
    for (int i = 0; i < n; i++) y[i] = (int)(y_tmp[i].real());
  }

  /**
     Template helper for selecting the Eigen matrix type based on the
     arithmetic (real or complex)
   */
  template <class T> struct eigen_matrix_map;
  template <> struct eigen_matrix_map<double> { using type = MatrixXd; };
  template <> struct eigen_matrix_map<Complex> { using type = MatrixXcd; };
  template <class T> using eigen_matrix_t = typename eigen_matrix_map<T>::type;

  template <typename T>
  void EigenSolver::rotateVecs(std::vector<ColorSpinorField> &kSpace, const std::vector<T> &rot_array,
                               int offset, int dim, int keep, int locked, TimeProfile &profile)
  {
    using matrix_t = eigen_matrix_t<T>;

    // If we have memory available, do the entire rotation
    if (batched_rotate <= 0 || batched_rotate >= keep) {
      if ((int)kSpace.size() < offset + keep) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", n_kr + keep);
        resize(kSpace, offset + keep, QUDA_ZERO_FIELD_CREATE, kSpace[0]);
      }

      // References to the relevant subsets
      auto vecs_ref = {kSpace.begin() + locked, kSpace.begin() + locked + dim};
      auto kSpace_ref = {kSpace.begin() + offset, kSpace.begin() + offset + keep};

      // zero the workspace
      for (auto &ki : kSpace_ref) blas::zero(*ki);

      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      blas::axpy(rot_array, vecs_ref, kSpace_ref);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

      // Copy compressed Krylov
      for (int i = 0; i < keep; i++) std::swap(kSpace[locked + i], kSpace[offset + i]);

    } else {

      // Do batched rotation to save on memory
      int batch_size = batched_rotate;
      int full_batches = keep / batch_size;
      int batch_size_r = keep % batch_size;
      bool do_batch_remainder = (batch_size_r != 0 ? true : false);

      if ((int)kSpace.size() < offset + batch_size) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + batch_size);
        resize(kSpace, offset + batch_size, QUDA_ZERO_FIELD_CREATE, kSpace[0]);
      }

      profile.TPSTART(QUDA_PROFILE_EIGENLU);
      matrix_t mat = matrix_t::Zero(dim, keep);
      for (int j = 0; j < keep; j++)
        for (int i = 0; i < dim; i++) mat(i, j) = rot_array[i * keep + j];

      FullPivLU<matrix_t> matLU(mat);

      // Extract the upper triangular matrix
      matrix_t matUpper = matLU.matrixLU().template triangularView<Eigen::Upper>();
      matUpper.conservativeResize(keep, keep);

      // Extract the lower triangular matrix
      matrix_t matLower = matrix_t::Identity(dim, dim);
      matLower.block(0, 0, dim, keep).template triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();
      matLower.conservativeResize(dim, keep);

      // Extract the desired permutation matrices
      MatrixXi matP = MatrixXi::Zero(dim, dim);
      MatrixXi matQ = MatrixXi::Zero(keep, keep);
      matP = matLU.permutationP().inverse();
      matQ = matLU.permutationQ().inverse();
      profile.TPSTOP(QUDA_PROFILE_EIGENLU);

      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      // Compute V * A = V * PLUQ

      // Do P Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matP, dim);

      // Do L Multiply
      //---------------------------------------------------------------------------
      // Loop over full batches
      for (int b = 0; b < full_batches; b++) {

        // batch triangle
        blockRotate<T>(kSpace, matLower, dim, {b * batch_size, (b + 1) * batch_size},
                       {b * batch_size, (b + 1) * batch_size}, LOWER_TRI, offset);
        // batch pencil
        blockRotate<T>(kSpace, matLower, dim, {(b + 1) * batch_size, dim},
                       {b * batch_size, (b + 1) * batch_size}, PENCIL, offset);
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }
      if (do_batch_remainder) {
        // remainder triangle
        blockRotate<T>(kSpace, matLower, dim, {full_batches * batch_size, keep},
                       {full_batches * batch_size, keep}, LOWER_TRI, offset);
        // remainder pencil
        if (keep < dim) {
          blockRotate<T>(kSpace, matLower, dim, {keep, dim}, {full_batches * batch_size, keep}, PENCIL, offset);
        }
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Do U Multiply
      //---------------------------------------------------------------------------
      if (do_batch_remainder) {
        // remainder triangle
        blockRotate<T>(kSpace, matUpper, keep, {full_batches * batch_size, keep},
                       {full_batches * batch_size, keep}, UPPER_TRI, offset);
        // remainder pencil
        blockRotate<T>(kSpace, matUpper, keep, {0, full_batches * batch_size},
                    {full_batches * batch_size, keep}, PENCIL, offset);
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Loop over full batches
      for (int b = full_batches - 1; b >= 0; b--) {
        // batch triangle
        blockRotate<T>(kSpace, matUpper, keep, {b * batch_size, (b + 1) * batch_size},
                       {b * batch_size, (b + 1) * batch_size}, UPPER_TRI, offset);
        if (b > 0) {
          // batch pencil
          blockRotate<T>(kSpace, matUpper, keep, {0, b * batch_size}, {b * batch_size, (b + 1) * batch_size},
                         PENCIL, offset);
        }
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }

      // Do Q Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matQ, keep);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    }
  }

  template void EigenSolver::rotateVecs<double>(std::vector<ColorSpinorField> &kSpace, const std::vector<double> &rot_array,
                                                int offset, int dim, int keep, int locked, TimeProfile &profile);

  template void EigenSolver::rotateVecs<Complex>(std::vector<ColorSpinorField> &kSpace, const std::vector<Complex> &rot_array,
                                                 int offset, int dim, int keep, int locked, TimeProfile &profile);

} // namespace quda
