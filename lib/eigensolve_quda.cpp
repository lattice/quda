#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <invert_quda.h>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <qio_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#include <chrono>  // for high_resolution_clock

namespace quda
{

  using namespace Eigen;

  // Eigensolver class
  //-----------------------------------------------------------------------------
  EigenSolver::EigenSolver(QudaEigParam *eig_param, TimeProfile &profile) :
    eig_param(eig_param),
    profile(profile),
    tmp1(nullptr),
    tmp2(nullptr)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaEigParam(eig_param);

    // Problem parameters
    if (eig_param->eig_type != QUDA_EIG_DAV) {
      nEv = eig_param->nEv;
      nKr = eig_param->nKr;
    } else {
      mmin = eig_param->mmin;
      mmax = eig_param->mmax;
      corr_eq_tol = eig_param->corr_eq_tol;
      corr_eq_maxiter = eig_param->corr_eq_maxiter;
    }
    nConv = eig_param->nConv;
    tol = eig_param->tol;
    reverse = false;

    // Algorithm variables
    converged = false;
    restart_iter = 0;
    max_restarts = eig_param->max_restarts;
    check_interval = eig_param->check_interval;
    batched_rotate = eig_param->batched_rotate;
    iter = 0;
    iter_converged = 0;
    iter_locked = 0;
    iter_keep = 0;
    num_converged = 0;
    num_locked = 0;
    num_keep = 0;

    // Sanity checks
    if (eig_param->eig_type != QUDA_EIG_DAV) {
      if (nKr <= nEv) errorQuda("nKr=%d is less than or equal to nEv=%d\n", nKr, nEv);
      if (nEv < nConv) errorQuda("nConv=%d is greater than nEv=%d\n", nConv, nEv);
      if (nEv == 0) errorQuda("nEv=0 passed to Eigensolver\n");
      if (nKr == 0) errorQuda("nKr=0 passed to Eigensolver\n");
      if (nConv == 0) errorQuda("nConv=0 passed to Eigensolver\n");
    } else {
      if (mmax <= mmin) errorQuda("mmax=%d is less than or equal to mmin=%d\n", mmax, mmin);
      if (mmin == 0) errorQuda("mmin=0 passed to Eigensolver\n");
      if (mmax == 0) errorQuda("mmax=0 passed to Eigensolver\n");
      if (corr_eq_maxiter == 0) errorQuda("corr_eq_maxiter=0 passed to Eigensolver\n");
    }
    if (nConv == 0) errorQuda("nConv=0 passed to Eigensolver\n");

    if (eig_param->eig_type != QUDA_EIG_DAV) {
      residua = (double *)safe_malloc(nKr * sizeof(double));
      for (int i = 0; i < nKr; i++) { residua[i] = 0.0; }
    } else {
      residua = (double *)safe_malloc(mmax * sizeof(double));
      for (int i = 0; i < mmax; i++) { residua[i] = 0.0; }
    }

    if (eig_param->eig_type != QUDA_EIG_DAV) {
      // Quda MultiBLAS friendly array
      Qmat = (Complex *)safe_malloc(nEv * nKr * sizeof(Complex));
    }

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

    if(eig_param->eig_type==QUDA_EIG_DAV && strcmp(spectrum, "SR")){
      errorQuda("Jacobi-Davidson eigensolver only supports SR spectrum");
    }

    if (eig_param->eig_type != QUDA_EIG_DAV) {
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
      eig_solver = new TRLM(eig_param, mat, profile);
      break;
    case QUDA_EIG_DAV:
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating Jacobi-Davidson eigensolver\n");
      eig_solver = new JD(eig_param, mat, profile);
      break;
    default: errorQuda("Invalid eig solver type");
    }
    return eig_solver;
  }

  // Utilities and functions common to all Eigensolver instances
  //------------------------------------------------------------------------------

  void EigenSolver::matVec(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {
    if (!tmp1 || !tmp2) {
      ColorSpinorParam param(in);
      if (!tmp1) tmp1 = ColorSpinorField::Create(param);
      if (!tmp2) tmp2 = ColorSpinorField::Create(param);
    }
    mat(out, in, *tmp1, *tmp2);
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
  }

  // Orthogonalise r against V_[j]
  Complex EigenSolver::blockOrthogonalize(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> rvec,
                                          int j)
  {
    Complex *s = (Complex *)safe_malloc((j + 1) * sizeof(Complex));
    Complex sum(0.0, 0.0);
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(j + 1);
    for (int i = 0; i < j + 1; i++) { vecs_ptr.push_back(vecs[i]); }
    // Block dot products stored in s.
    blas::cDotProduct(s, vecs_ptr, rvec);

    // Block orthogonalise
    for (int i = 0; i < j + 1; i++) {
      sum += s[i];
      s[i] *= -1.0;
    }
    blas::caxpy(s, vecs_ptr, rvec);

    host_free(s);
    return sum;
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
    // Populate batch array (COLUM major -> ROW major)
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
  }

  void EigenSolver::blockReset(std::vector<ColorSpinorField *> &kSpace, int j_start, int j_end)
  {
    // copy back to correct position, zero out the workspace
    for (int j = j_start; j < j_end; j++) {
      int k = nKr + 1 + j - j_start;
      std::swap(kSpace[j + num_locked], kSpace[k]);
      blas::zero(*kSpace[k]);
    }
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
  }

  void EigenSolver::computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs,
                                 std::vector<Complex> &evals, int size)
  {
    if (size > (int)evecs.size()) errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size()) errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    ColorSpinorParam csParam(*evecs[0]);
    std::vector<ColorSpinorField *> temp;
    temp.push_back(ColorSpinorField::Create(csParam));

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
  }

  void EigenSolver::loadVectors(std::vector<ColorSpinorField *> &eig_vecs, std::string vec_infile)
  {

#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    auto spinor_parity = eig_vecs[0]->SuggestedParity();
    if (strcmp(vec_infile.c_str(), "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Start loading %04d vectors from %s\n", Nvec, vec_infile.c_str());

      std::vector<ColorSpinorField *> tmp;
      tmp.reserve(Nvec);
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        ColorSpinorParam csParam(*eig_vecs[0]);
        csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
                                                                                eig_vecs[0]->Precision());
        csParam.location = QUDA_CPU_FIELD_LOCATION;
        csParam.create = QUDA_NULL_FIELD_CREATE;
        // if we're loading nColor == 3 fields, they'll always be full-field. Copy into a single parity afterwards.
        if (csParam.nColor == 3 && csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) {
          csParam.x[0] *= 2;
          csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        }
        for (int i = 0; i < Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); }
      } else {
        ColorSpinorParam csParam(*eig_vecs[0]);
        if (csParam.nColor == 3 && csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) {
          csParam.x[0] *= 2;
          csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
          for (int i = 0; i < Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); }
        } else {
          for (int i = 0; i < Nvec; i++) { tmp.push_back(eig_vecs[i]); }
        }
      }

      void **V = static_cast<void **>(safe_malloc(Nvec * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        V[i] = tmp[i]->V();
        if (V[i] == NULL) {
          if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Could not allocate space for eigenVector[%d]\n", i);
        }
      }

      read_spinor_field(vec_infile.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), tmp[0]->SiteSubset(),
                        spinor_parity, tmp[0]->Ncolor(), tmp[0]->Nspin(), Nvec, 0, (char **)0);

      host_free(V);
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {

        ColorSpinorParam csParam(*eig_vecs[0]);
        if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || csParam.nColor != 3) { // we don't care for MG vectors
          for (int i = 0; i < Nvec; i++) {
            *eig_vecs[i] = *tmp[i];
            delete tmp[i];
          }
        } else { // nColor == 3 field with a single parity: need to copy it out of the full-field vector.
          // Create a temporary single-parity CPU field
          csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
                                                                                  eig_vecs[0]->Precision());
          csParam.location = QUDA_CPU_FIELD_LOCATION;
          csParam.create = QUDA_NULL_FIELD_CREATE;

          ColorSpinorField *tmp_intermediate = ColorSpinorField::Create(csParam);

          for (int i = 0; i < Nvec; i++) {
            if (spinor_parity == QUDA_EVEN_PARITY)
              blas::copy(*tmp_intermediate, tmp[i]->Even());
            else if (spinor_parity == QUDA_ODD_PARITY)
              blas::copy(*tmp_intermediate, tmp[i]->Odd());
            else
              errorQuda("When loading single parity vectors, the suggested parity must be set.");

            *eig_vecs[i] = *tmp_intermediate;
            delete tmp[i];
          }

          delete tmp_intermediate;
        }
      } else if (eig_vecs[0]->Location() == QUDA_CPU_FIELD_LOCATION
                 && eig_vecs[0]->SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
        for (int i = 0; i < Nvec; i++) {
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(*eig_vecs[i], tmp[i]->Even());
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(*eig_vecs[i], tmp[i]->Odd());
          else
            errorQuda("When loading single parity vectors, the suggested parity must be set.");

          delete tmp[i];
        }
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done loading vectors\n");
    } else {
      errorQuda("No eigenspace input file defined.");
    }
#else
    errorQuda("\nQIO library was not built.\n");
#endif
  }

  void EigenSolver::saveVectors(const std::vector<ColorSpinorField *> &eig_vecs, std::string vec_outfile)
  {

#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    std::vector<ColorSpinorField *> tmp;
    tmp.reserve(Nvec);
    auto spinor_parity = eig_vecs[0]->SuggestedParity();
    if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(*eig_vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
                                                                              eig_vecs[0]->Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;

      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || csParam.nColor != 3) { // we don't care for MG vectors
        // We're good, copy as is.
        csParam.create = QUDA_NULL_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));
          *tmp[i] = *eig_vecs[i];
        }
      } else { // QUDA_PARITY_SITE_SUBSET
        csParam.create = QUDA_NULL_FIELD_CREATE;

        // intermediate host single-parity field
        ColorSpinorField *tmp_intermediate = ColorSpinorField::Create(csParam);

        csParam.x[0] *= 2;                          // corrects for the factor of two in the X direction
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET; // create a full-parity field.
        csParam.create = QUDA_ZERO_FIELD_CREATE;    // to explicitly zero the odd sites.
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));

          // copy the single parity eigen/singular vector into an
          // intermediate device-side vector
          *tmp_intermediate = *eig_vecs[i];

          // copy the single parity only eigen/singular vector into the even components of the full parity vector
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i]->Even(), *tmp_intermediate);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i]->Odd(), *tmp_intermediate);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
        delete tmp_intermediate;
      }
    } else {
      ColorSpinorParam csParam(*eig_vecs[0]);
      if (csParam.nColor == 3 && csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i]->Even(), *eig_vecs[i]);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i]->Odd(), *eig_vecs[i]);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
      } else {
        for (int i = 0; i < Nvec; i++) { tmp.push_back(eig_vecs[i]); }
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start saving %d vectors to %s\n", Nvec, vec_outfile.c_str());

    void **V = static_cast<void **>(safe_malloc(Nvec * sizeof(void *)));
    for (int i = 0; i < Nvec; i++) {
      V[i] = tmp[i]->V();
      if (V[i] == NULL) {
        if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Could not allocate space for eigenVector[%04d]\n", i);
      }
    }

    write_spinor_field(vec_outfile.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), eig_vecs[0]->SiteSubset(),
                       spinor_parity, tmp[0]->Ncolor(), tmp[0]->Nspin(), Nvec, 0, (char **)0);

    host_free(V);
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
    if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION
        || (eig_vecs[0]->Location() == QUDA_CPU_FIELD_LOCATION && eig_vecs[0]->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) {
      for (int i = 0; i < Nvec; i++) delete tmp[i];
    }

#else
    errorQuda("\nQIO library was not built.\n");
#endif
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
    loadVectors(vecs_ptr, eig_param->vec_infile);

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
    if (eig_param->eig_type != QUDA_EIG_DAV) {
      host_free(Qmat);
    }
  }
  //-----------------------------------------------------------------------------
  //-----------------------------------------------------------------------------

  // Thick Restarted Lanczos Method constructor
  TRLM::TRLM(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile) :
    EigenSolver(eig_param, profile),
    mat(mat)
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
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Test for an initial guess
    double norm = sqrt(blas::norm2(*kSpace[0]));
    if (norm == 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Initial residual is zero. Populating with rands.\n");
      if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
        kSpace[0]->Source(QUDA_RANDOM_SOURCE);
      } else {
        RNG *rng = new RNG(*kSpace[0], 1234);
        rng->Init();
        spinorNoise(*kSpace[0], *rng, QUDA_NOISE_UNIFORM);
        rng->Release();
        delete rng;
      }
    }

    // Normalise initial guess
    norm = sqrt(blas::norm2(*kSpace[0]));
    blas::ax(1.0 / norm, *kSpace[0]);

    // Check for Chebyshev maximum estimation
    if (eig_param->use_poly_acc && eig_param->a_max <= 0.0) {
      eig_param->a_max = estimateChebyOpMax(mat, *kSpace[2], *kSpace[1]);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Chebyshev maximum estimate: %e.\n", eig_param->a_max);
    }

    // Create a device side residual vector by cloning
    // the kSpace passed to the function.
    ColorSpinorParam csParamClone(*kSpace[0]);
    csParam = csParamClone;
    // Increase Krylov space to nKr+1 one vector, create residual
    kSpace.reserve(nKr + 1);
    for (int i = nConv; i < nKr + 1; i++) kSpace.push_back(ColorSpinorField::Create(csParam));
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    r.push_back(ColorSpinorField::Create(csParam));
    // Increase evals space to nEv
    evals.reserve(nEv);
    for (int i = nConv; i < nEv; i++) evals.push_back(0.0);
    //---------------------------------------------------------------------------

    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = DBL_EPSILON;
    QudaPrecision prec = kSpace[0]->Precision();
    switch (prec) {
    case QUDA_DOUBLE_PRECISION:
      epsilon = DBL_EPSILON;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in double precision\n");
      break;
    case QUDA_SINGLE_PRECISION:
      epsilon = FLT_EPSILON;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in single precision\n");
      break;
    case QUDA_HALF_PRECISION:
      epsilon = 2e-3;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in half precision\n");
      break;
    case QUDA_QUARTER_PRECISION:
      epsilon = 5e-2;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in quarter precision\n");
      break;
    default: errorQuda("Invalid precision %d", prec);
    }

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("**** START TRLM SOLUTION ****\n");
      printfQuda("*****************************\n");
    }

    // Print Eigensolver params
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("spectrum %s\n", spectrum);
      printfQuda("tol %.4e\n", tol);
      printfQuda("nConv %d\n", nConv);
      printfQuda("nEv %d\n", nEv);
      printfQuda("nKr %d\n", nKr);
      if (eig_param->use_poly_acc) {
        printfQuda("polyDeg %d\n", eig_param->poly_deg);
        printfQuda("a-min %f\n", eig_param->a_min);
        printfQuda("a-max %f\n", eig_param->a_max);
      }
    }

    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (int step = num_keep; step < nKr; step++) lanczosStep(kSpace, step);
      iter += (nKr - num_keep);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Restart %d complete\n", restart_iter + 1);

      int arrow_pos = std::max(num_keep - num_locked + 1, 2);
      // The eigenvalues are returned in the alpha array
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat(num_locked, arrow_pos);
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

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
      printfQuda("kSpace size at convergence/max restarts = %d\n", (int)kSpace.size());
    // Prune the Krylov space back to size when passed to eigensolver
    for (unsigned int i = nConv; i < kSpace.size(); i++) { delete kSpace[i]; }
    kSpace.resize(nConv);
    evals.resize(nConv);

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
    delete r[0];

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
      saveVectors(vecs_ptr, eig_param->vec_outfile);
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("***** END TRLM SOLUTION *****\n");
      printfQuda("*****************************\n");
    }

    mat.flops();
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
    if (j > 0)
      for (int k = 0; k < 1; k++) blockOrthogonalize(v, r, j);

    // b_j = ||r||
    beta[j] = sqrt(blas::norm2(*r[0]));

    // Prepare next step.
    // v_{j+1} = r / b_j
    blas::zero(*v[j + 1]);
    blas::axpy(1.0 / beta[j], *r[0], *v[j + 1]);
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

  void TRLM::eigensolveFromArrowMat(int num_locked, int arrow_pos)
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    int dim = nKr - num_locked;

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

    for (int i = 0; i < arrow_pos - 1; i++) {

      // beta populates the arrow
      A(i, arrow_pos - 1) = beta[i + num_locked];
      A(arrow_pos - 1, i) = beta[i + num_locked];
    }

    for (int i = arrow_pos - 1; i < dim - 1; i++) {

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
      if ((int)kSpace.size() < offset + iter_keep) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + iter_keep);
        kSpace.reserve(offset + iter_keep);
        for (int i = kSpace.size(); i < offset + iter_keep; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParam));
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
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + batch_size);
        kSpace.reserve(offset + batch_size);
        for (int i = kSpace.size(); i < offset + batch_size; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParam));
        }
      }

      profile.TPSTART(QUDA_PROFILE_EIGEN);
      MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
      for (int j = 0; j < iter_keep; j++)
        for (int i = 0; i < dim; i++) mat(i, j) = ritz_mat[j * dim + i];

      FullPivLU<MatrixXd> matLU(mat);

      // Extract the upper triagnular matrix
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
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size);
      }

      if (do_batch_remainder) {
        // remainder triangle
        blockRotate(kSpace, matLower.data(), dim, {full_batches * batch_size, iter_keep},
                    {full_batches * batch_size, iter_keep}, LOWER_TRI);
        // remainder pencil
        if (iter_keep < dim) {
          blockRotate(kSpace, matLower.data(), dim, {iter_keep, dim}, {full_batches * batch_size, iter_keep}, PENCIL);
        }
        blockReset(kSpace, full_batches * batch_size, iter_keep);
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
        blockReset(kSpace, full_batches * batch_size, iter_keep);
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
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size);
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
  }

  // -------------------------------
  // Davidson-type eigensolver class

  // Jacobi-Davidson Method constructor
  JD::JD(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile) :
    EigenSolver(eig_param, profile),
    mat(mat)
  {

    if (eig_param->spectrum != QUDA_SPECTRUM_SR_EIG)
      errorQuda("Only smallest real spectrum type (SR) can be passed to the JD solver");

    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    corrEqTol = eig_param->corr_eq_tol;
    corrEqMaxiter = eig_param->corr_eq_maxiter;

    // Additional auxiliar profilers used in the solvers for the correction equation
    profileCorrEqInvs = new TimeProfile("profileCorrEqInvs");
    profileMatCorrEqInvs = new TimeProfile("profileMatCorrEqInvs");

    QudaInvertParam refineparam = *eig_param->invert_param;
    refineparam.cuda_prec_sloppy = eig_param->invert_param->cuda_prec_refinement_sloppy;
    solverParam = new SolverParam(refineparam);
    solverParam->tol = corrEqTol;
    solverParam->maxiter = corrEqMaxiter;
    solverParam->use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam->delta = eig_param->invert_param->reliable_delta_refinement;
    // disable preconditioning on solvers used in the correction equation
    solverParam->inv_type_precondition = QUDA_INVALID_INVERTER;

    solverParamMat = new SolverParam(*solverParam);
    solverParamPrec = new SolverParam(*solverParam);

    d = nullptr;
    // Create the dirac operator
    // IMPORTANT: cannot call createDirac(...) as in interface_quda.cpp
    {
      bool pc_solve = (eig_param->invert_param->solve_type == QUDA_DIRECT_PC_SOLVE) || \
                      (eig_param->invert_param->solve_type == QUDA_NORMOP_PC_SOLVE) || \
                      (eig_param->invert_param->solve_type == QUDA_NORMERR_PC_SOLVE);

      DiracParam diracParam;

      quda::setDiracParam(diracParam, eig_param->invert_param, pc_solve);
      d = Dirac::create(diracParam);
    }

    Dirac &dirac = *d;
    mmPP = new DiracPrecProjCorr(dirac);

    // Solvers used in the correction equation
    cgMat = new CG(const_cast<DiracMatrix&>(mat), const_cast<DiracMatrix&>(mat), const_cast<DiracMatrix&>(mat), \
                   *solverParamMat, *profileMatCorrEqInvs);
    cg = new CG(const_cast<DiracMatrix&>(mat), const_cast<DiracMatrix&>(mat), const_cast<DiracMatrix&>(mat), \
                *solverParam, *profileCorrEqInvs);
    gcrPrec = new GCR(*mmPP, *mmPP, *mmPP, *solverParamPrec, *profileCorrEqInvs);

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);

  }

  void JD::operator()(std::vector<ColorSpinorField *> &eigSpace, std::vector<Complex> &evals)
  {

    // TODO: general pendings:

    //	(DONE) 1. remove informal profiling via chronos (remove the chronos' header as well)
    //	       2. switch from X_tilde to eigSpace

    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, eigSpace, evals);
      return;
    }

    // Setting some parameters of the eigensolver
    int k=0, k_max=eig_param->nConv, m=0;
    int m_max=eig_param->mmax, m_min=eig_param->mmin;
    // 'tau' is the <target> for the eigensolver
    double theta=0.0;
    max_restarts = eig_param->max_restarts;

    // Test for an initial guess
    double norm = sqrt(blas::norm2(*eigSpace[0]));
    if (norm == 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Initial residual is zero. Populating with rands.\n");
      if (eigSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
        eigSpace[0]->Source(QUDA_RANDOM_SOURCE);
      } else {
        RNG *rng = new RNG(*eigSpace[0], 1234);
        rng->Init();
        spinorNoise(*eigSpace[0], *rng, QUDA_NOISE_UNIFORM);
        rng->Release();
        delete rng;
      }
    }

    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Clone eigSpace's CSF params
    ColorSpinorParam csParam(*eigSpace[0]);

    // TODO: move some of these allocations to the constructor, if possible
    // Init a zero residual
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    r.push_back(ColorSpinorField::Create(csParam));
    r_tilde.push_back(ColorSpinorField::Create(csParam));
    mmPP->y_hat.push_back(ColorSpinorField::Create(csParam));
    Qhat.reserve(k_max+1);
    for (int i=0; i<(k_max+1); i++) {Qhat.push_back(ColorSpinorField::Create(csParam));}
    csParam.create = QUDA_COPY_FIELD_CREATE;
    t.push_back(ColorSpinorField::Create(*eigSpace[0], csParam));

    // Create the vector subspaces used for the search of eigenpairs
    std::vector<ColorSpinorField *> u, u_A, V, V_A, X_tilde;
    // buffer spinors
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    u.push_back(ColorSpinorField::Create(csParam));
    u_A.push_back(ColorSpinorField::Create(csParam));
    for (int i=0; i<m_max; i++) {
      V.push_back(ColorSpinorField::Create(csParam));
      V_A.push_back(ColorSpinorField::Create(csParam));
    }
    std::vector<ColorSpinorField*> tmpV, tmpAV;
    for (int i=0; i<m_max; i++) {
      tmpV.push_back(ColorSpinorField::Create(csParam));
      tmpAV.push_back(ColorSpinorField::Create(csParam));
    }

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);

    // Convergence and locking criteria
    double epsilon = DBL_EPSILON;
    QudaPrecision prec = eigSpace[0]->Precision();
    switch (prec) {
    case QUDA_DOUBLE_PRECISION:
      epsilon = DBL_EPSILON;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in double precision\n");
      break;
    case QUDA_SINGLE_PRECISION:
      epsilon = FLT_EPSILON;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in single precision\n");
      break;
    case QUDA_HALF_PRECISION:
      epsilon = 2e-3;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in half precision\n");
      break;
    case QUDA_QUARTER_PRECISION:
      epsilon = 5e-2;
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Running Eigensolver in quarter precision\n");
      break;
    default: errorQuda("Invalid precision %d", prec);
    }

    // Begin JD Eigensolver computation
    //---------------------------------------------------------------------------
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("**** START JD SOLUTION ******\n");
      printfQuda("*****************************\n");
    }

    // Print Eigensolver params
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("spectrum %s\n", spectrum);
      printfQuda("tol %.4e\n", tol);
      printfQuda("nConv %d\n", nConv);
      printfQuda("mmin %d\n", m_min);
      printfQuda("mmax %d\n", m_max);
      printfQuda("corr-eq-maxiter %d\n", corrEqMaxiter);
      printfQuda("corr-eq-tol %f\n", corrEqTol);
    }

    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // TODO: remove this line, after the overall use of <evals> if fixed
    evals.resize(0);

    // Matrix with the compressed sub-space information to extract the eigenpairs
    MatrixXcd H;
    SelfAdjointEigenSolver<MatrixXcd> eigensolver;

    int loopr = 0;

    // Main loop
    while (restart_iter<max_restarts) {

      // Locking
      {
        int tmpSize = (int)X_tilde.size();

        // double reorthogonalization
        for (int i=0; i<tmpSize; i++) {
          Complex gamma = blas::cDotProduct(*X_tilde[i], *t[0]);
          blas::caxpy(-gamma, *X_tilde[i], *t[0]);
        }

        // normalize before the next orthogonalization
        norm = sqrt(blas::norm2(*t[0]));
        blas::ax(1.0 / norm, *t[0]);

        for (int i=0; i<tmpSize; i++) {
          Complex gamma = blas::cDotProduct(*X_tilde[i], *t[0]);
          blas::caxpy(-gamma, *X_tilde[i], *t[0]);
        }
      }

      // Project t orthogonal to V: t = t - ( v_i^* . t ) . v_i
      {
        // double reorthogonalization
        for (int i=0; i<m; i++) {
          Complex gamma = blas::cDotProduct(*V[i], *t[0]);
          blas::caxpy(-gamma, *V[i], *t[0]);
        }
        // normalize before the next orthogonalization
        norm = sqrt(blas::norm2(*t[0]));
        blas::ax(1.0 / norm, *t[0]);
        for (int i=0; i<m; i++) {
          Complex gamma = blas::cDotProduct(*V[i], *t[0]);
          blas::caxpy(-gamma, *V[i], *t[0]);
        }
      }

      m++;

      // Push: V[m-1] = t, and then normalize it
      blas::copy(*V[m-1], *t[0]);
      norm = sqrt(blas::norm2(*V[m-1]));
      blas::ax(1.0 / norm, *V[m-1]);
      // and then apply A to that newly added vector
      matVec(mat, *V_A[m-1], *V[m-1]);

      // Construction of H = Vdag . V_A
      H.conservativeResize(m,m);
      for (int i=0; i<(m-1); i++) {
        H(i,m-1) = blas::cDotProduct(*V[i], *V_A[m-1]);
        // TODO: is the next line necessary ?
        H(m-1,i) = conj(H(i,m-1));
      }
      // this next line ensures the case m=1
      H(m-1,m-1) = blas::cDotProduct(*V[m-1], *V_A[m-1]);

      // ith eigenvalue: eigensolver.eigenvalues()[i], ith eigenvector: eigensolver.eigenvectors().col(i)
      eigensolver.compute(H);

      // Moving the eigenpairs to a vector of std::pair to sort by eigenvalue
      // TODO: switch this whole treatment of eigenpairs to a (much) more efficient way ..
      std::vector< std::pair < double, std::vector<Complex>* > > eigenpairs;
      std::vector<Complex>* buffVec = 0;
      for (int i=0; i<m; i++) {
        // TODO: switch to safe_malloc
        buffVec = new std::vector<Complex>(eigensolver.eigenvectors().col(i).data(),
                                           eigensolver.eigenvectors().col(i).data() + eigensolver.eigenvectors().col(i).size());
        eigenpairs.push_back( std::make_pair( eigensolver.eigenvalues()[i], buffVec ) );
      }
      // Order the eigeninformation extracted from H in descending order of eigenvalues
      // TODO: switch to using a sort-function, due to general values of <tau>. Descending
      //       order is only applicable to the case tau=0 i.e. smallest eigenvalues
      std::sort(eigenpairs.begin(), eigenpairs.end());

      // Computing the residual
      // u = V * s_1 -- lifting the first Ritz vector through V
      blas::zero(*u[0]);
      for (int i=0; i<m; i++) {
        //blas::caxpy( (*(eigenpairs[0].second))[i], *V[i], *u[0] );
        // use this due to frequent-shrinking-avoiding
        blas::caxpy( (*(eigenpairs[loopr].second))[i], *V[i], *u[0] );
      }
      // and the same for u_A
      blas::zero(*u_A[0]);
      for (int i=0; i<m; i++) {
        //blas::caxpy( (*(eigenpairs[0].second))[i], *V_A[i], *u_A[0] );
        // use this due to frequent-shrinking-avoiding
        blas::caxpy( (*(eigenpairs[loopr].second))[i], *V_A[i], *u_A[0] );
      }
      // and then compute the residual
      theta = eigenpairs[loopr].first;
      blas::copy(*r[0], *u_A[0]);
      blas::caxpy(-theta, *u[0], *r[0]);

      norm = sqrt(blas::norm2(*r[0]));
      // Print the residual
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Iteration = %d, m = %d, residual = %f, converged eigenpairs = %d\n", iter, m, norm, k);
      }

      while(norm < eig_param->tol){
        evals.push_back(eigenpairs[loopr].first);
        csParam.create = QUDA_COPY_FIELD_CREATE;
        X_tilde.push_back(ColorSpinorField::Create(*u[0], csParam));
        k++;

        // Check for convergence
        if (k >= k_max) {
          converged = true;
          break;
        }

        // Shrinking the subspace
        /*
        {
          MatrixXcd tmpH = MatrixXcd::Zero(m-1,m-1);

          csParam.create = QUDA_ZERO_FIELD_CREATE;
          // the new subspace is of size m-1
          for(int i=0; i<(m-1); i++){
            blas::zero(*tmpV[i]);
            // project-up with a subspace still of size m
            for( int j=0; j<m; j++ ){
              blas::caxpy( (*(eigenpairs[i+1].second))[j], *V[j], *tmpV[i] );
              tmpH(i,i) = eigenpairs[i+1].first;
            }
            matVec(mat, *tmpAV[i], *tmpV[i]);
          }

          // pop first element of eigenpairs
          eigenpairs.erase(eigenpairs.begin());

          for(int i=0; i<(m-1); i++){
            for(int j=0; j<m; j++){
             //eigensolver.eigenvectors().col(i)[j] = 0;
             (*(eigenpairs[i].second))[j] = 0;
            }
            //eigensolver.eigenvectors().col(i)[i] = 1;
            (*(eigenpairs[i].second))[i] = 1;
          }

          // Assign new values of H
          // TODO: skip this resize() ?
          H.resize(m-1,m-1);
          H = tmpH;

          // Assign new values of V and W

          for (int i=0; i<(m-1); i++) {
            ColorSpinorField *tmpf = V[i];
            V[i] = tmpV[i];
            tmpV[i] = tmpf;
            tmpf = V_A[i];
            V_A[i] = tmpAV[i];
            tmpAV[i] = tmpf;
          }
        }
        */

        //m--;

        blas::zero(*u[0]);
        // project-up with a subspace still of size m
        for( int j=0; j<m; j++ ){
          blas::caxpy( (*(eigenpairs[loopr+1].second))[j], *V[j], *u[0] );
          //tmpH(i,i) = eigenpairs[i+1].first;
        }
        matVec(mat, *u_A[0], *u[0]);

        //*u[0] = *V[0];
        //*u_A[0] = *V_A[0];

        // and then compute the residual
        theta = eigenpairs[loopr+1].first;
        blas::copy(*r[0], *u_A[0]);
        blas::caxpy(-theta, *u[0], *r[0]);

        norm = sqrt(blas::norm2(*r[0]));

        loopr++;
      }

      // Check for convergence
      if (k >= k_max) {
        break;
      }

      // Restart: shrink the acceleration subspace
      // FIXME: add exceptions for case: (loopr+m_min) > m_max
      if(m == m_max){
        MatrixXcd tmpH = MatrixXcd::Zero(m_min,m_min);

        // 0-th element of the new (smaller) subspace
        csParam.create = QUDA_COPY_FIELD_CREATE;
        blas::copy(*tmpV[0], *u[0]);
        blas::copy(*tmpAV[0], *u_A[0]);
        tmpH(0,0) = eigenpairs[loopr].first;

        csParam.create = QUDA_ZERO_FIELD_CREATE;
        for(int i=1; i<m_min; i++){
          blas::zero(*tmpV[i]);
          for( int j=0; j<m; j++ ){
            blas::caxpy( (*(eigenpairs[loopr+i].second))[j], *V[j], *tmpV[i] );
            tmpH(i,i) = eigenpairs[loopr+i].first;
          }
          matVec(mat, *tmpAV[i], *tmpV[i]);
        }

        m = m_min;

        // Assign new values of H
        // TODO: skip this resize() ?
        H.resize(m_min,m_min);
        H = tmpH;

        // Assign new values of V and W
        for (int i=0; i<m; i++) {
          ColorSpinorField *tmpf = V[i];
          V[i] = tmpV[i];
          tmpV[i] = tmpf;
          tmpf = V_A[i];
          V_A[i] = tmpAV[i];
          tmpAV[i] = tmpf;
        }

        restart_iter++;

        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printfQuda("RESTART (#%d)\n", restart_iter);
        }

        loopr=0;
      }

      // Preparing for solving the correction equation
      theta = eigenpairs[0].first;
      X_tilde.push_back( u[0] );

      blas::copy(*t[0], *r[0]);
      //invertProjMat(theta, mat, *t[0], *r[0], QUDA_SILENT, k+1);

      {
        //invertProjMat(theta, mat, *t[0], *r[0], QUDA_SILENT, k+1, X_tilde);
        invertProjMat(theta, mat, *t[0], *r[0], QUDA_SILENT, 1, u);
      }

      //invertProjMat(theta, mat, *t[0], *r[0], QUDA_SILENT, ( (k+1)>=1?1:k+1 ));

      X_tilde.pop_back();

      for (auto p : eigenpairs) { delete p.second; }

      iter++;
    }

    //--------------------

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
      printfQuda("eigSpace size at convergence/max restarts = %d\n", (int)eigSpace.size());
    // Prune the Krylov space back to size when passed to eigensolver
    //for (unsigned int i = nConv; i < kSpace.size(); i++) { delete kSpace[i]; }

    // FIXME ?
    //kSpace.resize(nConv);
    //evals.resize(nConv);

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      if (eig_param->require_convergence) {
        //errorQuda("JD failed to compute the requested %d vectors with a search space of size between %d and %d in %d "
        //          "restart steps. Exiting.",
        //          nConv, m_min, m_min, max_restarts);
        printfQuda("JD failed to compute the requested %d vectors with a search space of size between %d and %d in %d "
                   "restart steps. Exiting.",
                   nConv, m_min, m_min, max_restarts);
      } else {
        warningQuda("JD failed to compute the requested %d vectors with a search space of size between %d and %d in %d "
                  "restart steps.",
                  nConv, m_min, m_min, max_restarts);
      }
    } else {
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("JD computed the requested %d vectors in %d restart steps and %d iterations.\n", nConv,
                   restart_iter, iter);

        // Dump all Ritz values and residua (?)
        for (int i = 0; i < nConv; i++) {
          //printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
        }
      }

      // Compute eigenvalues
      computeEvals(mat, X_tilde, evals);
    }

    // Local clean-up
    delete r[0];
    delete t[0];
    delete r_tilde[0];
    delete mmPP->y_hat[0];
    for (auto p : u) { delete p; }
    for (auto p : u_A) { delete p; }
    for (auto p : V) { delete p; }
    for (auto p : V_A) { delete p; }
    for (auto p : tmpV) { delete p; }
    for (auto p : tmpAV) { delete p; }
    for (auto p : X_tilde) { delete p; }
    for (auto p : Qhat) { delete p; }

    // Only save if outfile is defined
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving eigenvectors\n");
      // Make an array of size nConv
      std::vector<ColorSpinorField *> vecs_ptr;
      vecs_ptr.reserve(nConv);
      const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
      for (int i = 0; i < nConv; i++) {
        eigSpace[i]->setSuggestedParity(mat_parity);
        vecs_ptr.push_back(eigSpace[i]);
      }
      saveVectors(vecs_ptr, eig_param->vec_outfile);
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("*****************************\n");
      printfQuda("***** END JD SOLUTION *******\n");
      printfQuda("*****************************\n");
    }

    mat.flops();

  }

  // Jacobi-Davidson destructor
  JD::~JD()
  {
    delete profileCorrEqInvs;
    delete profileMatCorrEqInvs;

    delete solverParam;
    delete solverParamMat;
    delete solverParamPrec;
    delete cgMat;
    delete gcrPrec;
    delete cg;

    delete d;
    delete mmPP;
  }

  // JD Member functions
  //---------------------------------------------------------------------------

  void JD::invertProjMat(const double theta, const DiracMatrix &mat, ColorSpinorField &x, ColorSpinorField &b, QudaVerbosity verb, int k, std::vector<ColorSpinorField*> &eigSpace)
  {
    // TODO: correction equation pendings:

    //    (DONE) 1. move Qhat as a JD attribute, and fully pre-allocate it at the beginning of JD::operator()
    //	     (?) 2. stop alloc / dealloc of resultDotProduct on each call of this method
    //	     (?) 3. avoid always creating and destroying Eigen objects ( both within this method as in the three
    //		    forms of DiracPrecProjCorr::operator() )

    //std::vector<ColorSpinorField *> &qSpace = X_tilde;
    std::vector<ColorSpinorField *> &qSpace = eigSpace;

    // Clone x's CSF params
    ColorSpinorParam csParam(x);

    // Buffers for the shifts of the matrix operators
    double bareShift;

    // 1. solve for Qhat in K * Qhat = qSpace, with K a good preconditioner, mmSloppy for now

    int sizePS = k;

    // Casting away contractual constness
    DiracMatrix& matUnconst = const_cast<DiracMatrix&>(mat);

    csParam.create = QUDA_COPY_FIELD_CREATE;
    // Switching to the appropriate shift for JD
    bareShift = matUnconst.shift;
    matUnconst.shift = bareShift - theta;

    for(int i=0; i<sizePS; i++){
      blas::copy(*Qhat[i], *qSpace[i]);
      //cgWrapper(*cg, corrEqTol*10.0, corrEqMaxiter, verb, *solverParam, *Qhat[i], *qSpace[i]);
      cgWrapper(*cg, corrEqTol, corrEqMaxiter, verb, *solverParam, *Qhat[i], *qSpace[i]);
    }

    // and, switching back the shift parameters
    matUnconst.shift = bareShift;

    // 2. M = qSpacedag * Qhat

    std::vector<ColorSpinorField*> tmpQhat(Qhat.begin(), Qhat.begin()+sizePS);
    std::vector<ColorSpinorField*> tmpQSpace(qSpace.begin(), qSpace.begin()+sizePS);

    Complex* resultDotProd = (Complex*) safe_malloc( sizePS*sizePS * sizeof(Complex) );
    blas::cDotProduct(resultDotProd, const_cast<std::vector<ColorSpinorField*>&>(tmpQSpace),
                      const_cast<std::vector<ColorSpinorField*>&>(tmpQhat));

    // 3. LU decomposition of M -- TODO: move the application of .fullPivLu() to this sub-section ?

    // Init eigen object
    MatrixXcd M = MatrixXcd::Zero(sizePS,sizePS);
    for(int i=0; i<sizePS; i++){
      for(int j=0; j<sizePS; j++){
        M(i,j) = resultDotProd[i*sizePS + j];
      }
    }

    // 4. r_tilde = Ktilde^-1 * r

    // Switching to the appropriate shift for JD
    bareShift = matUnconst.shift;
    matUnconst.shift = bareShift - theta;

    // <r_tilde> is "r-hat" for a few of the upcoming lines

    blas::copy(*r_tilde[0], b);
    cgWrapper(*cg, corrEqTol, corrEqMaxiter, verb, *solverParam, *r_tilde[0], b);
    //cgWrapper(*cg, 1e-1, 2, verb, *solverParam, *r_tilde[0], b);

    // and, switching back the shift parameters
    matUnconst.shift = bareShift;

    MatrixXcd gamma(sizePS,1);
    for(int i=0; i<sizePS; i++){
      gamma(i,0) = blas::cDotProduct(*qSpace[i], *r_tilde[0]);
    }
    MatrixXcd alpha = M.fullPivLu().solve(gamma);
    for(int i=0; i<sizePS; i++){
      blas::caxpy(-alpha(i), *Qhat[i], *r_tilde[0]);
    }

    blas::ax(-1.0, *r_tilde[0]);

    // 5. solve: ( Ktilde^-1 * (I - QQdag)(A - \theta I)(I - QQdag) ) x = r_tilde

    mmPP->projSpace = qSpace;
    mmPP->theta = theta;
    mmPP->Mproj = M;
    mmPP->Qhat = Qhat;
    mmPP->solverParam_ = solverParamMat;
    mmPP->matUnconst_ = &matUnconst;
    mmPP->cg_ = cgMat;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    mmPP->eigSlvr = this;
    mmPP->k = k;

    // Test further for the best ratio for the following two parameters
    mmPP->tol = corrEqTol;
    mmPP->maxiter = corrEqMaxiter;

    blas::zero(x);
    //cgWrapper(*cgPrec, corrEqTol, corrEqMaxiter, verb, *solverParamPrec, x, *r_tilde[0]);
    //cgWrapper(*cgPrec, corrEqTol, corrEqMaxiter, getVerbosity(), *solverParamPrec, x, *r_tilde[0]);
    solverParamPrec->inv_type_precondition = QUDA_INVALID_INVERTER;
    QudaVerbosity verbTmp = getVerbosity();
    setVerbosity(QUDA_SILENT);
    //(*cgPrec)(x, *r_tilde[0]);
    (*gcrPrec)(x, *r_tilde[0]);
    setVerbosity(verbTmp);

    // Local clean-up
    host_free(resultDotProd);
  }

  void JD::cgWrapper(CG &cgw, double tol, int maxiter, QudaVerbosity verb, SolverParam &slvrPrm, ColorSpinorField &x, ColorSpinorField &b)
  {
    QudaVerbosity verbTmp = getVerbosity();
    setVerbosity(verb);

    double tol_buff = slvrPrm.tol;
    slvrPrm.tol = tol;
    int maxiter_buff = slvrPrm.maxiter;
    slvrPrm.maxiter = maxiter;

    cgw(x, b);

    slvrPrm.tol = tol_buff;
    slvrPrm.maxiter = maxiter_buff;

    setVerbosity(verbTmp);
  }

  // Projection matrix used in the solution of the correction equation
  //---------------------------------------------------------------------------

  void DiracPrecProjCorr::operator()(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double norm_bf = sqrt(blas::norm2(in));

    if( norm_bf==0 ){
      if (getVerbosity() >= QUDA_SUMMARIZE)
        warningQuda("Received a zero spinor in DiracPrecProjCorr::operator() within JD");
      blas::zero(out);
      return;
    }

    SolverParam &solverParam = *solverParam_;
    DiracMatrix &matUnconst = *matUnconst_;
    CG &cgx = *cg_;

    // 1. y = (A - \theta I)v

    matUnconst(out, in);
    blas::caxpy(-theta, const_cast<ColorSpinorField&>(in), out);

    // 2. y_hat = K^-1 * y

    blas::copy(*y_hat[0], out);

    //---------------------------------------------
    // Switching to the appropriate shift for JD
    double bareShift = matUnconst.shift;
    matUnconst.shift = bareShift - theta;

    eigSlvr->cgWrapper(cgx, tol, maxiter, QUDA_SILENT, solverParam, *y_hat[0], out);

    // Switching back the shift parameters
    matUnconst.shift = bareShift;
    //---------------------------------------------

    //int sizePS = projSpace.size();
    int sizePS = k;

    Eigen::MatrixXcd gamma(sizePS,1);
    for(int i=0; i<sizePS; i++){
      gamma(i,0) = blas::cDotProduct(*projSpace[i], *y_hat[0]);
    }
    Eigen::MatrixXcd alpha = Mproj.fullPivLu().solve(gamma);
    for(int i=0; i<sizePS; i++){
      blas::caxpy(-alpha(i), *Qhat[i], *y_hat[0]);
    }

    //out = *y_hat[0];
    blas::copy(out, *y_hat[0]);
  }

  void DiracPrecProjCorr::operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
  {
    double norm_bf = sqrt(blas::norm2(in));

    if( norm_bf==0 ){
      if (getVerbosity() >= QUDA_SUMMARIZE)
        warningQuda("Received a zero spinor in DiracPrecProjCorr::operator() within JD");
      blas::zero(out);
      return;
    }

    dirac->tmp1 = &tmp;

    SolverParam &solverParam = *solverParam_;
    DiracMatrix &matUnconst = *matUnconst_;
    CG &cgx = *cg_;

    // 1. y = (A - \theta I)v

    matUnconst(out, in);
    blas::caxpy(-theta, const_cast<ColorSpinorField&>(in), out);

    // 2. y_hat = K^-1 * y

    blas::copy(*y_hat[0], out);

    //---------------------------------------------
    // Switching to the appropriate shift for JD
    double bareShift = matUnconst.shift;
    matUnconst.shift = bareShift - theta;

    eigSlvr->cgWrapper(cgx, tol, maxiter, QUDA_SILENT, solverParam, *y_hat[0], out);

    // Switching back the shift parameters
    matUnconst.shift = bareShift;
    //---------------------------------------------

    //int sizePS = projSpace.size();
    int sizePS = k;

    Eigen::MatrixXcd gamma(sizePS,1);
    for(int i=0; i<sizePS; i++){
      gamma(i,0) = blas::cDotProduct(*projSpace[i], *y_hat[0]);
    }
    Eigen::MatrixXcd alpha = Mproj.fullPivLu().solve(gamma);
    for(int i=0; i<sizePS; i++){
      blas::caxpy(-alpha(i), *Qhat[i], *y_hat[0]);
    }

    //out = *y_hat[0];
    blas::copy(out, *y_hat[0]);

    dirac->tmp1 = NULL;
  }

  void DiracPrecProjCorr::operator()(ColorSpinorField &out, const ColorSpinorField &in, 
				     ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
  {
    double norm_bf = sqrt(blas::norm2(in));

    if( norm_bf==0 ){
      if (getVerbosity() >= QUDA_SUMMARIZE)
        warningQuda("Received a zero spinor in DiracPrecProjCorr::operator() within JD");
      blas::zero(out);
      return;
    }

    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;

    SolverParam &solverParam = *solverParam_;
    DiracMatrix &matUnconst = *matUnconst_;
    CG &cgx = *cg_;

    // 1. y = (A - \theta I)v

    matUnconst(out, in);
    blas::caxpy(-theta, const_cast<ColorSpinorField&>(in), out);

    // 2. y_hat = K^-1 * y

    blas::copy(*y_hat[0], out);

    //---------------------------------------------
    // Switching to the appropriate shift for JD
    double bareShift = matUnconst.shift;
    matUnconst.shift = bareShift - theta;

    eigSlvr->cgWrapper(cgx, tol, maxiter, QUDA_SILENT, solverParam, *y_hat[0], out);

    // Switching back the shift parameters
    matUnconst.shift = bareShift;
    //---------------------------------------------

    //int sizePS = projSpace.size();
    int sizePS = k;

    Eigen::MatrixXcd gamma(sizePS,1);
    for(int i=0; i<sizePS; i++){
      gamma(i,0) = blas::cDotProduct(*projSpace[i], *y_hat[0]);
    }
    Eigen::MatrixXcd alpha = Mproj.fullPivLu().solve(gamma);
    for(int i=0; i<sizePS; i++){
      blas::caxpy(-alpha(i), *Qhat[i], *y_hat[0]);
    }

    //out = *y_hat[0];
    blas::copy(out, *y_hat[0]);

    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }

} // namespace quda
