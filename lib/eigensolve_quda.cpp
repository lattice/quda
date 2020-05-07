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
#include <vector_io.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

namespace quda
{

  using namespace Eigen;

  // Eigensolver class
  //-----------------------------------------------------------------------------
  EigenSolver::EigenSolver(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    mat(mat),
    eig_param(eig_param),
    profile(profile),
    tmp1(nullptr),
    tmp2(nullptr)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaEigParam(eig_param);

    // Problem parameters
    nEv = eig_param->nEv;
    nKr = eig_param->nKr;
    nConv = eig_param->nConv;
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

    // Sanity checks
    if (nKr <= nEv) errorQuda("nKr=%d is less than or equal to nEv=%d\n", nKr, nEv);
    if (nEv < nConv) errorQuda("nConv=%d is greater than nEv=%d\n", nConv, nEv);
    if (nEv == 0) errorQuda("nEv=0 passed to Eigensolver\n");
    if (nKr == 0) errorQuda("nKr=0 passed to Eigensolver\n");
    if (nConv == 0) errorQuda("nConv=0 passed to Eigensolver\n");

    residua = (double *)safe_malloc(nKr * sizeof(double));
    for (int i = 0; i < nKr; i++) { residua[i] = 0.0; }

    // Part of the spectrum to be computed.
    switch (eig_param->spectrum) {
    case QUDA_SPECTRUM_SR_EIG: strcpy(spectrum, "SR"); break;
    case QUDA_SPECTRUM_LR_EIG: strcpy(spectrum, "LR"); break;
    case QUDA_SPECTRUM_SM_EIG: strcpy(spectrum, "SM"); break;
    case QUDA_SPECTRUM_LM_EIG: strcpy(spectrum, "LM"); break;
    case QUDA_SPECTRUM_SI_EIG: strcpy(spectrum, "SI"); break;
    case QUDA_SPECTRUM_LI_EIG: strcpy(spectrum, "LI"); break;
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

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  // We bake the matrix operator 'mat' and the eigensolver parameters into the
  // eigensolver.
  EigenSolver *EigenSolver::create(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile)
  {
    EigenSolver *eig_solver = nullptr;

    switch (eig_param->eig_type) {
    case QUDA_EIG_IR_ARNOLDI: errorQuda("IR Arnoldi not implemented"); break;
    case QUDA_EIG_IR_LANCZOS: errorQuda("IR Lanczos not implemented"); break;
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

    if (!mat.hermitian() && eig_solver->hermitian()) errorQuda("Cannot solve non-Hermitian system with Hermitian eigensolver");
    return eig_solver;
  }

  // Utilities and functions common to all Eigensolver instances
  //------------------------------------------------------------------------------
  void EigenSolver::prepareInitialGuess(std::vector<ColorSpinorField *> &kSpace)
  {
    if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int b = 0; b < block_size; b++) {
        if (sqrt(blas::norm2(*kSpace[b])) == 0.0) { kSpace[b]->Source(QUDA_RANDOM_SOURCE); }
      }
    } else {
      RNG *rng = new RNG(*kSpace[0], 1234);
      rng->Init();
      for (int b = 0; b < block_size; b++) {
        if (sqrt(blas::norm2(*kSpace[b])) == 0.0) { spinorNoise(*kSpace[b], *rng, QUDA_NOISE_UNIFORM); }
      }
      rng->Release();
      delete rng;
    }
    bool orthed = false;
    int k = 0, kmax = 5;
    while (!orthed && k < kmax) {
      orthonormalizeMGS(kSpace, block_size);
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        if (block_size > 1)
          printfQuda("Orthonormalising initial guesses with Gram Schmidt, k=%d\n", k);
        else
          printfQuda("Orthonormalising initial guess\n");
      }
      orthed = orthoCheck(kSpace, block_size);
      k++;
    }
    if (!orthed) errorQuda("Failed to orthonormalise initial guesses");
  }

  void EigenSolver::checkChebyOpMax(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace)
  {
    if (eig_param->use_poly_acc && eig_param->a_max <= 0.0) {
      // Use part of the kSpace as temps
      eig_param->a_max = estimateChebyOpMax(mat, *kSpace[block_size + 2], *kSpace[block_size + 1]);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Chebyshev maximum estimate: %e.\n", eig_param->a_max);
    }
  }

  void EigenSolver::prepareKrylovSpace(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    ColorSpinorParam csParamClone(*kSpace[0]);
    // Increase Krylov space to nKr+block_size vectors, create residual
    kSpace.reserve(nKr + block_size);
    for (int i = nConv; i < nKr + block_size; i++) kSpace.push_back(ColorSpinorField::Create(csParamClone));
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    for (int b = 0; b < block_size; b++) { r.push_back(ColorSpinorField::Create(csParamClone)); }
    // Increase evals space to nEv
    evals.reserve(nEv);
    for (int i = nConv; i < nEv; i++) evals.push_back(0.0);
  }

  void EigenSolver::printEigensolverSetup()
  {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("**** START TRLM SOLUTION ****\n");
      printfQuda("*****************************\n");
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("spectrum %s\n", spectrum);
      printfQuda("tol %.4e\n", tol);
      printfQuda("nConv %d\n", nConv);
      printfQuda("nEv %d\n", nEv);
      printfQuda("nKr %d\n", nKr);
      if (block_size > 1) printfQuda("block size %d\n", block_size);
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
    case QUDA_DOUBLE_PRECISION:
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in double precision\n");
      eps = DBL_EPSILON;
      break;
    case QUDA_SINGLE_PRECISION:
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in single precision\n");
      eps = FLT_EPSILON;
      break;
    case QUDA_HALF_PRECISION:
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in half precision\n");
      eps = 2e-3;
      break;
    case QUDA_QUARTER_PRECISION:
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in quarter precision\n");
      eps = 5e-2;
      break;
    default: errorQuda("Invalid precision %d", prec);
    }
    return eps;
  }

  void EigenSolver::cleanUpEigensolver(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    for (int b = 0; b < block_size; b++) delete r[b];
    r.resize(0);

    // Resize Krylov Space
    for (unsigned int i = nConv; i < kSpace.size(); i++) { delete kSpace[i]; }
    kSpace.resize(nConv);
    evals.resize(nConv);

    // Only save if outfile is defined
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving eigenvectors\n");
      // Make an array of size nConv
      std::vector<ColorSpinorField *> vecs_ptr;
      vecs_ptr.reserve(nConv);
      const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
      for (int i = 0; i < nConv; i++) {
        kSpace[i]->setSuggestedParity(mat_parity);
        vecs_ptr.push_back(kSpace[i]);
      }
      // save the vectors
      VectorIO io(eig_param->vec_outfile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.save(vecs_ptr);
    }

    // Save TRLM tuning
    saveTuneCache();

    mat.flops();

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("***** END TRLM SOLUTION *****\n");
      printfQuda("*****************************\n");
    }
  }

  void EigenSolver::matVec(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {
    if (!tmp1 || !tmp2) {
      ColorSpinorParam param(in);
      if (!tmp1) tmp1 = ColorSpinorField::Create(param);
      if (!tmp2) tmp2 = ColorSpinorField::Create(param);
    }
    mat(out, in, *tmp1, *tmp2);

    // Save mattrix * vector tuning
    saveTuneCache();
  }

  void EigenSolver::chebyOp(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {
    // Just do a simple matVec if no poly acc is requested
    if (!eig_param->use_poly_acc) {
      matVec(mat, out, in);
      return;
    }

    if (eig_param->poly_deg == 0) { errorQuda("Polynomial acceleration requested with zero polynomial degree"); }

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
    matVec(mat, out, in);
    blas::caxpby(d2, const_cast<ColorSpinorField &>(in), d1, out);
    if (eig_param->poly_deg == 1) return;

    // C_0 is the current 'in'  vector.
    // C_1 is the current 'out' vector.

    // Clone 'in' to two temporary vectors.
    ColorSpinorField *tmp1 = ColorSpinorField::Create(in);
    ColorSpinorField *tmp2 = ColorSpinorField::Create(in);

    blas::copy(*tmp1, in);
    blas::copy(*tmp2, out);

    // Using Chebyshev polynomial recursion relation,
    // C_{m+1}(x) = 2*x*C_{m} - C_{m-1}

    double sigma_old = sigma1;

    // construct C_{m+1}(x)
    for (int i = 2; i < eig_param->poly_deg; i++) {
      sigma = 1.0 / (2.0 / sigma1 - sigma_old);

      d1 = 2.0 * sigma / delta;
      d2 = -d1 * theta;
      d3 = -sigma * sigma_old;

      // FIXME - we could introduce a fused matVec + blas kernel here, eliminating one temporary
      // mat*C_{m}(x)
      matVec(mat, out, *tmp2);

      Complex d1c(d1, 0.0);
      Complex d2c(d2, 0.0);
      Complex d3c(d3, 0.0);
      blas::caxpbypczw(d3c, *tmp1, d2c, *tmp2, d1c, out, *tmp1);
      std::swap(tmp1, tmp2);

      sigma_old = sigma;
    }
    blas::copy(out, *tmp2);

    delete tmp1;
    delete tmp2;

    // Save Chebyshev tuning
    saveTuneCache();
  }

  double EigenSolver::estimateChebyOpMax(const DiracMatrix &mat, ColorSpinorField &out, ColorSpinorField &in)
  {

    if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
      in.Source(QUDA_RANDOM_SOURCE);
    } else {
      RNG *rng = new RNG(in, 1234);
      rng->Init();
      spinorNoise(in, *rng, QUDA_NOISE_UNIFORM);
      rng->Release();
      delete rng;
    }

    ColorSpinorField *in_ptr = &in;
    ColorSpinorField *out_ptr = &out;

    // Power iteration
    double norm = 0.0;
    for (int i = 0; i < 100; i++) {
      if ((i + 1) % 10 == 0) {
        norm = sqrt(blas::norm2(*in_ptr));
        blas::ax(1.0 / norm, *in_ptr);
      }
      matVec(mat, *out_ptr, *in_ptr);
      std::swap(out_ptr, in_ptr);
    }

    // Compute spectral radius estimate
    double result = blas::reDotProduct(*out_ptr, *in_ptr);

    // Increase final result by 10% for safety
    return result * 1.10;

    // Save Chebyshev Max tuning
    saveTuneCache();
  }

  bool EigenSolver::orthoCheck(std::vector<ColorSpinorField *> vecs, int size)
  {
    bool orthed = true;
    const Complex Unit(1.0, 0.0);
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(size);
    for (int i = 0; i < size; i++) vecs_ptr.push_back(vecs[i]);

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        Complex cnorm = blas::cDotProduct(*vecs_ptr[j], *vecs_ptr[i]);
        if (j != i) {
          if (abs(cnorm) > 5e-16) {
            if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
              printfQuda("Norm <%d|%d>^2 = (%e,%e): %e\n", i, j, cnorm.real(), cnorm.imag(), abs(cnorm));
            orthed = false;
          }
        } else {
          if (abs(Unit - cnorm) > 5e-16) {
            if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
              printfQuda("Norm <%d|%d>^2 = (%e,%e): %e\n", i, j, cnorm.real(), cnorm.imag(), abs(Unit - cnorm));
            orthed = false;
          }
        }
      }
    }
    return orthed;
  }

  void EigenSolver::orthonormalizeMGS(std::vector<ColorSpinorField *> &vecs, int size)
  {
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(size);
    for (int i = 0; i < size; i++) vecs_ptr.push_back(vecs[i]);

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < i; j++) {
        Complex cnorm = blas::cDotProduct(*vecs_ptr[j], *vecs_ptr[i]); // <j|i> with i normalised.
        blas::caxpy(-cnorm, *vecs_ptr[j], *vecs_ptr[i]);               // i = i - proj_{j}(i) = i - <j|i> * i
      }
      double norm = sqrt(blas::norm2(*vecs_ptr[i]));
      blas::ax(1.0 / norm, *vecs_ptr[i]); // i/<i|i>
    }
  }

  // Orthogonalise r[0:] against V_[0:j]
  void EigenSolver::blockOrthogonalize(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> &rvecs, int j)
  {
    int vecs_size = j;
    int r_size = (int)rvecs.size();
    int array_size = vecs_size * r_size;
    std::vector<Complex> s(array_size);

    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(vecs_size);
    for (int i = 0; i < vecs_size; i++) { vecs_ptr.push_back(vecs[i]); }

    // Block dot products stored in s.
    blas::cDotProduct(s.data(), vecs_ptr, rvecs);

    // Block orthogonalise
    for (int i = 0; i < array_size; i++) s[i] *= -1.0;
    blas::caxpy(s.data(), vecs_ptr, rvecs);

    // Save orthonormalisation tuning
    saveTuneCache();
  }

  void EigenSolver::permuteVecs(std::vector<ColorSpinorField *> &kSpace, int *mat, int size)
  {
    std::vector<int> pivots(size);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (mat[j * size + i] == 1) { pivots[j] = i; }
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

  void EigenSolver::blockRotate(std::vector<ColorSpinorField *> &kSpace, double *array, int rank, const range &i_range,
                                const range &j_range, blockType b_type)
  {
    int block_i_rank = i_range.second - i_range.first;
    int block_j_rank = j_range.second - j_range.first;

    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;

    // Alias the vectors we wish to keep
    vecs_ptr.reserve(block_i_rank);
    for (int i = i_range.first; i < i_range.second; i++) { vecs_ptr.push_back(kSpace[num_locked + i]); }
    // Alias the extra space vectors
    kSpace_ptr.reserve(block_j_rank);
    for (int j = j_range.first; j < j_range.second; j++) {
      int k = nKr + 1 + j - j_range.first;
      kSpace_ptr.push_back(kSpace[k]);
    }

    double *batch_array = (double *)safe_malloc((block_i_rank * block_j_rank) * sizeof(double));
    // Populate batch array (COLUMN major -> ROW major)
    for (int j = j_range.first; j < j_range.second; j++) {
      for (int i = i_range.first; i < i_range.second; i++) {
        int j_arr = j - j_range.first;
        int i_arr = i - i_range.first;
        batch_array[i_arr * block_j_rank + j_arr] = array[j * rank + i];
      }
    }
    switch (b_type) {
    case PENCIL: blas::axpy(batch_array, vecs_ptr, kSpace_ptr); break;
    case LOWER_TRI: blas::axpy_L(batch_array, vecs_ptr, kSpace_ptr); break;
    case UPPER_TRI: blas::axpy_U(batch_array, vecs_ptr, kSpace_ptr); break;
    default: errorQuda("Undefined MultiBLAS type in blockRotate");
    }
    host_free(batch_array);

    // Save Krylov block rotation tuning
    saveTuneCache();
  }

  void EigenSolver::blockReset(std::vector<ColorSpinorField *> &kSpace, int j_start, int j_end, int offset)
  {
    // copy back to correct position, zero out the workspace
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      std::swap(kSpace[j + num_locked], kSpace[k]);
      blas::zero(*kSpace[k]);
    }
  }

  void EigenSolver::blockRotateComplex(std::vector<ColorSpinorField *> &kSpace, Complex *array, int rank,
                                       const range &i_range, const range &j_range, blockType b_type, int offset)
  {
    int block_i_rank = i_range.second - i_range.first;
    int block_j_rank = j_range.second - j_range.first;

    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;

    // Alias the vectors we wish to keep
    vecs_ptr.reserve(block_i_rank);
    for (int i = i_range.first; i < i_range.second; i++) { vecs_ptr.push_back(kSpace[num_locked + i]); }
    // Alias the extra space vectors
    kSpace_ptr.reserve(block_j_rank);
    for (int j = j_range.first; j < j_range.second; j++) {
      int k = offset + j - j_range.first;
      kSpace_ptr.push_back(kSpace[k]);
    }

    Complex *batch_array = (Complex *)safe_malloc((block_i_rank * block_j_rank) * sizeof(Complex));
    // Populate batch array (COLUM major -> ROW major)
    for (int j = j_range.first; j < j_range.second; j++) {
      for (int i = i_range.first; i < i_range.second; i++) {
        int j_arr = j - j_range.first;
        int i_arr = i - i_range.first;
        batch_array[i_arr * block_j_rank + j_arr] = array[j * rank + i];
      }
    }
    switch (b_type) {
    case PENCIL: blas::caxpy(batch_array, vecs_ptr, kSpace_ptr); break;
    case LOWER_TRI: blas::caxpy_L(batch_array, vecs_ptr, kSpace_ptr); break;
    case UPPER_TRI: blas::caxpy_U(batch_array, vecs_ptr, kSpace_ptr); break;
    default: errorQuda("Undefined MultiBLAS type in blockRotate");
    }
    host_free(batch_array);

    // Save Krylov block rotation tuning
    saveTuneCache();
  }

  void EigenSolver::computeSVD(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals)
  {
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing SVD of M\n");

    int nConv = eig_param->nConv;
    if (evecs.size() != (unsigned int)(2 * nConv))
      errorQuda("Incorrect deflation space sized %d passed to computeSVD, expected %d", (int)(evecs.size()), 2 * nConv);

    std::vector<double> sigma_tmp(nConv);

    for (int i = 0; i < nConv; i++) {

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
      mat.Expose()->M(*evecs[nConv + i], *evecs[i]);

      // sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
      sigma_tmp[i] = sqrt(blas::norm2(*evecs[nConv + i]));

      // Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
      blas::ax(1.0 / sigma_tmp[i], *evecs[nConv + i]);

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Sval[%04d] = %+.16e sigma - sqrt(|lambda|) = %+.16e\n", i, sigma_tmp[i],
                   sigma_tmp[i] - sqrt(abs(lambda.real())));

      evals[i] = sigma_tmp[i];
      //--------------------------------------------------------------------------
    }

    // Save SVD tuning
    saveTuneCache();
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflateSVD(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src,
                               const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
                               bool accumulate) const
  {
    // number of evecs
    int n_defl = eig_param->nConv;
    if (evecs.size() != (unsigned int)(2 * n_defl))
      errorQuda("Incorrect deflation space sized %d passed to computeSVD, expected %d", (int)(evecs.size()), 2 * n_defl);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Deflating %d left and right singular vectors\n", n_defl);

    // Perform Sum_i R_i * (\sigma_i)^{-1} * L_i^dag * vec = vec_defl
    // for all i computed eigenvectors and values.

    // 1. Take block inner product: L_i^dag * vec = A_i
    std::vector<ColorSpinorField *> left_vecs;
    left_vecs.reserve(n_defl);
    for (int i = n_defl; i < 2 * n_defl; i++) left_vecs.push_back(evecs[i]);

    std::vector<Complex> s(n_defl * src.size());
    std::vector<ColorSpinorField *> src_ = const_cast<decltype(src) &>(src);
    blas::cDotProduct(s.data(), left_vecs, src_);

    // 2. Perform block caxpy
    //    A_i -> (\sigma_i)^{-1} * A_i
    //    vec_defl = Sum_i (R_i)^{-1} * A_i
    if (!accumulate)
      for (auto &x : sol) blas::zero(*x);
    std::vector<ColorSpinorField *> right_vecs;
    right_vecs.reserve(n_defl);
    for (int i = 0; i < n_defl; i++) {
      right_vecs.push_back(evecs[i]);
      s[i] /= evals[i].real();
    }
    blas::caxpy(s.data(), right_vecs, sol);

    // Save SVD deflation tuning
    saveTuneCache();
  }

  void EigenSolver::computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs,
                                 std::vector<Complex> &evals, int size)
  {
    if (size > (int)evecs.size()) errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size()) errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    ColorSpinorParam csParamClone(*evecs[0]);
    std::vector<ColorSpinorField *> temp;
    temp.push_back(ColorSpinorField::Create(csParamClone));

    for (int i = 0; i < size; i++) {
      // r = A * v_i
      matVec(mat, *temp[0], *evecs[i]);
      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = blas::cDotProduct(*evecs[i], *temp[0]) / sqrt(blas::norm2(*evecs[i]));
      // Measure ||lambda_i*v_i - A*v_i||
      Complex n_unit(-1.0, 0.0);
      blas::caxpby(evals[i], *evecs[i], n_unit, *temp[0]);
      residua[i] = sqrt(blas::norm2(*temp[0]));

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Eval[%04d] = (%+.16e,%+.16e) residual = %+.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
    }
    delete temp[0];

    // Save Eval tuning
    saveTuneCache();
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflate(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src,
                            const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
                            bool accumulate) const
  {
    // number of evecs
    int n_defl = eig_param->nConv;

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Deflating %d vectors\n", n_defl);

    // Perform Sum_i V_i * (L_i)^{-1} * (V_i)^dag * vec = vec_defl
    // for all i computed eigenvectors and values.

    // Pointers to the required Krylov space vectors,
    // no extra memory is allocated.
    std::vector<ColorSpinorField *> eig_vecs;
    eig_vecs.reserve(n_defl);
    for (int i = 0; i < n_defl; i++) eig_vecs.push_back(evecs[i]);

    // 1. Take block inner product: (V_i)^dag * vec = A_i
    std::vector<Complex> s(n_defl * src.size());
    std::vector<ColorSpinorField *> src_ = const_cast<decltype(src) &>(src);
    blas::cDotProduct(s.data(), eig_vecs, src_);

    // 2. Perform block caxpy: V_i * (L_i)^{-1} * A_i
    for (int i = 0; i < n_defl; i++) { s[i] /= evals[i].real(); }

    // 3. Accumulate sum vec_defl = Sum_i V_i * (L_i)^{-1} * A_i
    if (!accumulate)
      for (auto &x : sol) blas::zero(*x);
    blas::caxpy(s.data(), eig_vecs, sol);

    // Save Deflation tuning
    saveTuneCache();
  }

  void EigenSolver::loadFromFile(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace,
                                 std::vector<Complex> &evals)
  {
    // Set suggested parity of fields
    const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
    for (int i = 0; i < nConv; i++) { kSpace[i]->setSuggestedParity(mat_parity); }

    // Make an array of size nConv
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(nConv);
    for (int i = 0; i < nConv; i++) { vecs_ptr.push_back(kSpace[i]); }

    {
      // load the vectors
      VectorIO io(eig_param->vec_infile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.load(vecs_ptr);
    }

    // Create the device side residual vector by cloning
    // the kSpace passed to the function.
    ColorSpinorParam csParam(*kSpace[0]);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    r.push_back(ColorSpinorField::Create(csParam));

    // Error estimates (residua) given by ||A*vec - lambda*vec||
    computeEvals(mat, kSpace, evals);
    delete r[0];
  }

  EigenSolver::~EigenSolver()
  {
    if (tmp1) delete tmp1;
    if (tmp2) delete tmp2;
    host_free(residua);
  }
} // namespace quda
