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
    profile(profile),
    tmp1(nullptr),
    tmp2(nullptr),
    transfer(nullptr)
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

    residua.reserve(n_kr);
    for (int i = 0; i < n_kr; i++) residua.push_back(0.0);

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

    // Parse compression parameters
    compressed_mode = false;
    if(eig_param->compress == QUDA_BOOLEAN_TRUE) {
      fine_n_ev = eig_param->comp_n_ev;
      fine_n_kr = eig_param->comp_n_kr;
      fine_n_conv = eig_param->comp_n_conv;
      fine_max_restarts = eig_param->comp_max_restarts;

      // Sanity checks
      if (fine_n_kr <= fine_n_ev) errorQuda("fine_n_kr = %d is less than or equal to fine_n_ev = %d", fine_n_kr, fine_n_ev);
      if (fine_n_ev < fine_n_conv) errorQuda("fine_n_conv=%d is greater than fine_n_ev=%d", fine_n_conv, fine_n_ev);
      if (fine_n_ev == 0) errorQuda("fine_n_ev=0 passed to Eigensolver");
      if (fine_n_kr == 0) errorQuda("fine_n_kr=0 passed to Eigensolver");
      if (fine_n_conv == 0) errorQuda("fine_n_conv=0 passed to Eigensolver");      

      if (batched_rotate < 1) errorQuda("compressed eigensolver requires batched rotation. batched_rotate = %d", batched_rotate);
      
      compress = true;
      spin_block_size = 2;
      n_block_ortho = eig_param->n_block_ortho;
      for(int i=0; i<4; i++) geo_block_size[i] = eig_param->geo_block_size[i];
    } else {
      compress = false;
    }
      
    
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
    case QUDA_EIG_BLK_IR_ARNOLDI:
      errorQuda("Block IR Arnoldi not implemented");
      break;
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
    if (!mat.hermitian() && eig_solver->hermitian())
      errorQuda("Cannot solve non-Hermitian system with strictly Hermitian eigensolver %d, %d", (int)!mat.hermitian(),
                (int)eig_solver->hermitian());

    // Support for Chebyshev only in strictly Hermitian solvers
    if (eig_param->use_poly_acc) {
      if (!mat.hermitian()) errorQuda("Cannot use polynomial acceleration with non-Hermitian operator");
      if (!eig_solver->hermitian()) errorQuda("Polynomial acceleration not supported with non-Hermitian solver");
    }

    if (mat.hermitian() && (eig_param->spectrum == QUDA_SPECTRUM_SI_EIG || eig_param->spectrum == QUDA_SPECTRUM_LI_EIG))
      errorQuda("The imaginary spectrum of a Hermitian operator cannot be computed");
    //--------------------------------------------------------------------------

    return eig_solver;
  }

  // Utilities and functions common to all Eigensolver instances
  //------------------------------------------------------------------------------
  void EigenSolver::prepareInitialGuess(std::vector<ColorSpinorField *> &kSpace)
  {
    double guess_norm = 0;
    if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int b = 0; b < block_size; b++) {
	guess_norm = sqrt(blas::norm2(*kSpace[b]));
	if (guess_norm == 0.0 || isnan(guess_norm)) { kSpace[b]->Source(QUDA_RANDOM_SOURCE); }
	guess_norm = sqrt(blas::norm2(*kSpace[b]));
	if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Initial guess %d norm = %e\n", b, guess_norm);
      }
    } else {
      RNG *rng = new RNG(*kSpace[0], 1234);
      for (int b = 0; b < block_size; b++) {
	guess_norm = sqrt(blas::norm2(*kSpace[b]));
	if (guess_norm == 0.0 || isnan(guess_norm)) { spinorNoise(*kSpace[b], *rng, QUDA_NOISE_UNIFORM); }
	guess_norm = sqrt(blas::norm2(*kSpace[b]));
	if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Initial guess %d norm = %e\n", b, guess_norm);
      }
      delete rng;
    }
    bool orthed = false;
    int k = 0, kmax = 5;
    while (!orthed && k < kmax) {
      orthonormalizeMGS(kSpace, block_size);
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        if (block_size > 1)
          printfQuda("Orthonormalising initial guesses with Modified Gram Schmidt, iter k=%d of %d\n", (k + 1), kmax);
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
    int k_size = kSpace.size();
    int r_size = r.size();
    int e_size = evals.size();
    ColorSpinorParam cs_param(*kSpace[0]);

    // Increase space to n_kr + block_size or n_conv vectors
    int max_size = (strcmp(eig_param->vec_infile, "") != 0 ? n_conv : n_kr + block_size);
    kSpace.reserve(max_size);
    for (int i = k_size; i < max_size; i++) kSpace.push_back(ColorSpinorField::Create(cs_param));
    cs_param.create = QUDA_ZERO_FIELD_CREATE;
    for (int b = r_size; b < block_size; b++) { r.push_back(ColorSpinorField::Create(cs_param)); }

    // Increase evals
    int eval_size = (strcmp(eig_param->coarse_vec_infile, "") != 0 ? n_conv : n_kr);
    evals.reserve(eval_size);
    for (int i = e_size; i < eval_size; i++) evals.push_back(0.0);
  }

  void EigenSolver::prepareCompressedKrylovSpace(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    int c_size = compressed_space.size();
    if(c_size != 0) errorQuda("A compressed space already exists");
    int f_size = fine_vector.size();
    int e_size = evals.size();

    if(!transfer) createTransferBasis(kSpace);
    QudaPrecision prec = kSpace[0]->Precision();
    QudaFieldLocation location = kSpace[0]->Location();
    ColorSpinorField *tmp_coarse = kSpace[0]->CreateCoarse(geo_block_size,
							   spin_block_size,
							   fine_n_conv, prec,
							   location);

    // Clone the coarse paramters
    ColorSpinorParam cs_param_coarse(*tmp_coarse);

    // Increase coarse space to n_kr + block_size or n_conv vectors
    int max_size = (strcmp(eig_param->coarse_vec_infile, "") != 0 ? n_conv : n_kr + block_size);
    compressed_space.reserve(max_size);
    for (int i = c_size; i < max_size; i++) compressed_space.push_back(ColorSpinorField::Create(cs_param_coarse));
    
    // Create fine vectors for workspace
    ColorSpinorParam cs_param_fine(*kSpace[0]);
    cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;
    for (int b = f_size; b < block_size; b++) { fine_vector.push_back(ColorSpinorField::Create(cs_param_fine)); }
    // Increase evals
    int eval_size = (strcmp(eig_param->coarse_vec_infile, "") != 0 ? n_conv : n_kr);
    evals.reserve(eval_size);
    for (int i = e_size; i < eval_size; i++) evals.push_back(0.0);
    
    // If we loaded the fine space from file with no coarse space we must start
    // the coarse solve from scratch
    if (strcmp(eig_param->vec_infile, "") != 0 && (strcmp(eig_param->coarse_vec_infile, "") == 0)) {
      // Restart the solver
      blas::zero(*fine_vector[0]);
      // Get some random noise
      prepareInitialGuess(fine_vector);
      // Add low mode components to the starting guess
      for(int i=0; i<fine_n_conv; i++) blas::caxpy(10.0, *kSpace[i], *fine_vector[0]);
      // The vector is non-zero, so this will just orthonormalise the guess
      prepareInitialGuess(fine_vector);
      // Compress the guess
      compressVectors(fine_vector, compressed_space, 0, 0, block_size);
    }
    if (strcmp(eig_param->vec_infile, "") == 0) {
      // If the fine space was just computed, we may continue the solve
      compressVectors(kSpace, compressed_space, 0, 0, num_locked + iter_keep + 1);
    }
    delete tmp_coarse;

    
    // Only save if outfile is defined. Make a function for this
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving fine eigenvectors\n");
      // Make an array of size n_conv
      std::vector<ColorSpinorField *> vecs_ptr;
      vecs_ptr.reserve(fine_n_conv);
      const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
      // We may wish to compute vectors in high prec, but use in a lower
      // prec. This allows the user to down copy the data for later use.
      QudaPrecision prec = kSpace[0]->Precision();
      if (save_prec < prec) {
        cs_param_fine.create = QUDA_REFERENCE_FIELD_CREATE;
        cs_param_fine.setPrecision(save_prec);
        for (unsigned int i = 0; i < kSpace.size(); i++) {
          kSpace[i]->setSuggestedParity(mat_parity);
          vecs_ptr.push_back(kSpace[i]->CreateAlias(cs_param_fine));
        }
        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printfQuda("kSpace successfully down copied from prec %d to prec %d\n", kSpace[0]->Precision(),
                     vecs_ptr[0]->Precision());
        }
      } else {
        for (int i = 0; i < fine_n_conv; i++) {
          kSpace[i]->setSuggestedParity(mat_parity);
          vecs_ptr.push_back(kSpace[i]);
        }
      }
      
      // save the vectors
      VectorIO io(eig_param->vec_outfile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.save(vecs_ptr);
      for (unsigned int i = 0; i < kSpace.size() && save_prec < prec; i++) delete vecs_ptr[i];
    }
    
    for(unsigned int i=block_size; i<kSpace.size(); i++) delete kSpace[i];
    kSpace.resize(block_size);
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

  void EigenSolver::cleanUpEigensolver(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    for (int b = 0; b < block_size; b++) delete r[b];
    r.resize(0);
    
    // Resize Krylov Space    
    for (unsigned int i = n_conv; i < kSpace.size(); i++) { delete kSpace[i]; }
    kSpace.resize(n_conv);
    evals.resize(n_conv);
    
    // Only save if outfile is defined
    if ((strcmp(eig_param->vec_outfile, "") != 0 && !compressed_mode) || 
        (strcmp(eig_param->coarse_vec_outfile, "") != 0 && compressed_mode)) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving eigenvectors\n");
      // Make an array of size n_conv
      std::vector<ColorSpinorField *> vecs_ptr;
      vecs_ptr.reserve(n_conv);
      const QudaParity mat_parity = impliedParityFromMatPC(mat.getMatPCType());
      // We may wish to compute vectors in high prec, but use in a lower
      // prec. This allows the user to down copy the data for later use.
      QudaPrecision prec = kSpace[0]->Precision();
      if (save_prec < prec) {
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_REFERENCE_FIELD_CREATE;
        csParamClone.setPrecision(save_prec);
        for (unsigned int i = 0; i < kSpace.size(); i++) {
          kSpace[i]->setSuggestedParity(mat_parity);
          vecs_ptr.push_back(kSpace[i]->CreateAlias(csParamClone));
        }
        if (getVerbosity() >= QUDA_SUMMARIZE) {
          printfQuda("kSpace successfully down copied from prec %d to prec %d\n", kSpace[0]->Precision(),
                     vecs_ptr[0]->Precision());
        }
      } else {
        for (int i = 0; i < n_conv; i++) {
	  kSpace[i]->setSuggestedParity(mat_parity);
          vecs_ptr.push_back(kSpace[i]);
        }
      }
      // save the vectors
      if(compressed_mode) {
	VectorIO io(eig_param->coarse_vec_outfile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
	io.save(vecs_ptr);
      } else {
	VectorIO io(eig_param->vec_outfile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
	io.save(vecs_ptr);
      }
      for (unsigned int i = 0; i < kSpace.size() && save_prec < prec; i++) delete vecs_ptr[i];
    }
    
    // Save tuning
    saveTuneCache();
    
    mat.flops();

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("********************************\n");
      printfQuda("***** END QUDA EIGENSOLVER *****\n");
      printfQuda("********************************\n");
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

    // Save matrix * vector tuning
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
      spinorNoise(in, *rng, QUDA_NOISE_UNIFORM);
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

    // Save Chebyshev Max tuning
    saveTuneCache();

    // Increase final result by 10% for safety
    return result * 1.10;
  }

  bool EigenSolver::orthoCheck(std::vector<ColorSpinorField *> vecs, int size)
  {
    bool orthed = true;
    const Complex Unit(1.0, 0.0);
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(size);
    for (int i = 0; i < size; i++) vecs_ptr.push_back(vecs[i]);

    std::vector<Complex> H(size * size);
    blas::hDotProduct(H.data(), vecs_ptr, vecs_ptr);

    double epsilon = setEpsilon(vecs[0]->Precision());

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
    int r_size = rvecs.size();
    int array_size = vecs_size * r_size;

    std::vector<ColorSpinorField *> vecs_ptr;

    if(!compressed_mode) {
      std::vector<Complex> s(array_size);

      vecs_ptr.reserve(vecs_size);
      for (int i = 0; i < vecs_size; i++) { vecs_ptr.push_back(vecs[i]); }

      // Block dot products stored in s.
      blas::cDotProduct(s.data(), vecs_ptr, rvecs);

      // Block orthogonalise
      for (int i = 0; i < array_size; i++) s[i] *= -1.0;
      blas::caxpy(s.data(), vecs_ptr, rvecs);
    } else {

      int batch_size = std::min((int)fine_vector.size(), j);
      int full_batches = (j)/batch_size;
      int remainder = (j)%batch_size;
      bool do_remainder = (j)%batch_size != 0 ? true : false;

      for (int b = 0; b < full_batches; b++) {
        int idx = b * batch_size;
        promoteVectors(fine_vector, vecs, 0, idx, batch_size);

        // Block dot products stored in s.
        std::vector<Complex> s(batch_size * r_size);
        blas::cDotProduct(s.data(), fine_vector, rvecs);
        // Block orthogonalise
        for (int i = 0; i < batch_size * r_size; i++) s[i] *= -1.0;
        blas::caxpy(s.data(), fine_vector, rvecs);
      }
      if(do_remainder) {
        int idx = full_batches * batch_size;
        promoteVectors(fine_vector, vecs, 0, idx, remainder);
        vecs_ptr.reserve(remainder);
        for (int i = 0; i < remainder; i++) { vecs_ptr.push_back(fine_vector[i]); }

        // Block dot products stored in s
        std::vector<Complex> s(remainder * r_size);
        blas::cDotProduct(s.data(), vecs_ptr, rvecs);

        // Block orthogonalise
        for (int i = 0; i < remainder * r_size; i++) s[i] *= -1.0;
        blas::caxpy(s.data(), vecs_ptr, rvecs);
      }
    }

    // Save orthonormalisation tuning
    saveTuneCache();
  }

#if 0
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
#endif

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
    bool local = false;
    int block_i_rank = i_range.second - i_range.first;
    int block_j_rank = j_range.second - j_range.first;

    // Quick return if no op.
    if (block_i_rank == 0 || block_j_rank == 0) return;

    //printfQuda("block_i_rank = %d\nblock_j_rank = %d\n", block_i_rank, block_j_rank);
    
    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;

    // If compressed vectors are passed, we create fine vectors
    // and prolongate the batches into them.
    if(compressed_mode && local) {
      // Create fine vectors
      int f_size = fine_vector.size();
      int num_vecs = block_i_rank + block_j_rank;
      fine_vector.reserve(block_i_rank + block_j_rank);
      
      ColorSpinorParam cs_param_fine(*r[0]);
      cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;
      for (int i = f_size; i < num_vecs; i++) fine_vector.push_back(ColorSpinorField::Create(cs_param_fine));      
    }
    int fine_pos = 0;
    // Alias the vectors we wish to keep
    vecs_ptr.reserve(block_i_rank);
    for (int i = i_range.first; i < i_range.second; i++) {
      if (compressed_mode && local) {
	promoteVectors(fine_vector, kSpace, fine_pos, num_locked + i, 1);
	vecs_ptr.push_back(fine_vector[fine_pos]);
	fine_pos++;
      }
      else vecs_ptr.push_back(kSpace[num_locked + i]);
    }
    
    // Alias the extra space vectors
    kSpace_ptr.reserve(block_j_rank);
    for (int j = j_range.first; j < j_range.second; j++) {
      int k = n_kr + 1 + j - j_range.first;
      if (compressed_mode && local) {
	promoteVectors(fine_vector, kSpace, fine_pos, k, 1);
	kSpace_ptr.push_back(fine_vector[fine_pos]);	
	fine_pos++;
      }
      else kSpace_ptr.push_back(kSpace[k]);
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

    // Compress the rotated fine vectors
    if (compressed_mode && local) {
      fine_pos = block_i_rank;
      for (int j = j_range.first; j < j_range.second; j++) {
	int k = n_kr + 1 + j - j_range.first;
	compressVectors(fine_vector, kSpace, fine_pos, k, 1);
	fine_pos++;
      }
    }
    
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

    // Quick return if no op.
    if (block_i_rank == 0 || block_j_rank == 0) return;

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

    int n_conv = eig_param->n_conv;
    if (evecs.size() != (unsigned int)(2 * n_conv))
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
      mat.Expose()->M(*evecs[n_conv + i], *evecs[i]);

      // sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
      sigma_tmp[i] = sqrt(blas::norm2(*evecs[n_conv + i]));

      // Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
      blas::ax(1.0 / sigma_tmp[i], *evecs[n_conv + i]);

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
    std::vector<ColorSpinorField *> left_vecs;
    left_vecs.reserve(n_defl);
    for (int i = eig_param->n_conv; i < eig_param->n_conv + n_defl; i++) left_vecs.push_back(evecs[i]);

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
    if (size > (int)evecs.size())
      errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size())
      errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    std::vector<ColorSpinorField *> temp;
    std::vector<ColorSpinorField *> evecs_ptr;
    evecs_ptr.reserve(size);
    ColorSpinorParam cs_param_fine(*r[0]);
    temp.push_back(ColorSpinorField::Create(cs_param_fine));
    
    for (int i = 0; i < size; i++) {      
      // r = A * v_i      
      if(compressed_mode) {
	profile.TPSTART(QUDA_PROFILE_COMPUTE);
	promoteVectors(fine_vector, evecs, 0, i, 1);
	profile.TPSTOP(QUDA_PROFILE_COMPUTE);
	evecs_ptr.push_back(fine_vector[0]);
      } else {
	evecs_ptr.push_back(evecs[i]);
      }
      
      matVec(mat, *temp[0], *evecs_ptr[i]);
      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = blas::cDotProduct(*evecs_ptr[i], *temp[0]) / sqrt(blas::norm2(*evecs_ptr[i]));
      // Measure ||lambda_i*v_i - A*v_i||
      Complex n_unit(-1.0, 0.0);
      blas::caxpby(evals[i], *evecs_ptr[i], n_unit, *temp[0]);
      residua[i] = sqrt(blas::norm2(*temp[0]));
      
      // If size = n_conv, this routine is called post sort
      if (getVerbosity() >= QUDA_SUMMARIZE && size == n_conv)
        printfQuda("Eval[%04d] = (%+.16e,%+.16e) residual = %+.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
    }
    delete temp[0];
    temp.resize(0);
    
    // Save Eval tuning
    saveTuneCache();
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflate(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src,
                            const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
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

    // Pointers to the required Krylov space vectors,
    // no extra memory is allocated.
    std::vector<ColorSpinorField *> eig_vecs;
    eig_vecs.reserve(n_defl);
    for (int i = 0; i < n_defl; i++) {
      eig_vecs.push_back(evecs[i]);
    }
    
    // 1. Take block inner product: (V_i)^dag * vec = A_i
    std::vector<Complex> s(n_defl * src.size());
    std::vector<ColorSpinorField *> src_ = const_cast<decltype(src) &>(src);
    blas::cDotProduct(s.data(), eig_vecs, src_);
    
    
    if(compressed_mode) {
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      promoteVectors(fine_vector, evecs, 0, i, 1);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      evecs_ptr.push_back(fine_vector[0]);
    } else {
      evecs_ptr.push_back(evecs[i]);
    }
    
    // 2. Perform block caxpy: V_i * (L_i)^{-1} * A_i
    for (int i = 0; i < n_defl; i++) {
      s[i] /= evals[i].real(); 
    }
    
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
    for (int i = 0; i < n_conv; i++) { kSpace[i]->setSuggestedParity(mat_parity); }

    // Make an array of size n_conv
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(n_conv);
    for (int i = 0; i < n_conv; i++) { vecs_ptr.push_back(kSpace[i]); }

    {
      // load the vectors
      if(compressed_mode) {
	VectorIO io(eig_param->coarse_vec_infile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
	io.load(vecs_ptr);
      } else {
	VectorIO io(eig_param->vec_infile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
	io.load(vecs_ptr);
      }
    }

    // Error estimates (residua) given by ||A*vec - lambda*vec||
    computeEvals(mat, kSpace, evals);
  }

  void EigenSolver::sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<Complex> &y)
  {

    //  'LM' -> sort into increasing order of magnitude.
    //  'SM' -> sort into decreasing order of magnitude.
    //  'LR' -> sort with real(x) in increasing algebraic order
    //  'SR' -> sort with real(x) in decreasing algebraic order
    //  'LI' -> sort with imag(x) in increasing algebraic order
    //  'SI' -> sort with imag(x) in decreasing algebraic order

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

  void EigenSolver::rotateVecsComplex(std::vector<ColorSpinorField *> &kSpace, const Complex *rot_array, const int offset,
                                      const int dim, const int keep, const int locked, TimeProfile &profile)
  {
    // If we have memory availible, do the entire rotation
    if (batched_rotate <= 0 || batched_rotate >= keep) {
      if ((int)kSpace.size() < offset + keep) {
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_ZERO_FIELD_CREATE;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", n_kr + keep);
        kSpace.reserve(offset + keep);
        for (int i = kSpace.size(); i < offset + keep; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }

      // Pointers to the relevant vectors
      std::vector<ColorSpinorField *> vecs_ptr;
      std::vector<ColorSpinorField *> kSpace_ptr;

      // Alias the extra space vectors, zero the workspace
      kSpace_ptr.reserve(keep);
      for (int i = 0; i < keep; i++) {
        kSpace_ptr.push_back(kSpace[offset + i]);
        blas::zero(*kSpace_ptr[i]);
      }

      // Alias the vectors we wish to compress.
      vecs_ptr.reserve(dim);
      for (int j = 0; j < dim; j++) vecs_ptr.push_back(kSpace[locked + j]);

      // multiBLAS caxpy
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      blas::caxpy(rot_array, vecs_ptr, kSpace_ptr);
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
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_ZERO_FIELD_CREATE;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + batch_size);
        kSpace.reserve(offset + batch_size);
        for (int i = kSpace.size(); i < offset + batch_size; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }

      profile.TPSTART(QUDA_PROFILE_EIGENLU);
      MatrixXcd mat = MatrixXcd::Zero(dim, keep);
      for (int j = 0; j < keep; j++)
        for (int i = 0; i < dim; i++) mat(i, j) = rot_array[i * keep + j];

      FullPivLU<MatrixXcd> matLU(mat);

      // Extract the upper triangular matrix
      MatrixXcd matUpper = MatrixXcd::Zero(keep, keep);
      matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
      matUpper.conservativeResize(keep, keep);

      // Extract the lower triangular matrix
      MatrixXcd matLower = MatrixXcd::Identity(dim, dim);
      matLower.block(0, 0, dim, keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();
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
      permuteVecs(kSpace, matP.data(), dim);

      // Do L Multiply
      //---------------------------------------------------------------------------
      // Loop over full batches
      for (int b = 0; b < full_batches; b++) {

        // batch triangle
        blockRotateComplex(kSpace, matLower.data(), dim, {b * batch_size, (b + 1) * batch_size},
                           {b * batch_size, (b + 1) * batch_size}, LOWER_TRI, offset);
        // batch pencil
        blockRotateComplex(kSpace, matLower.data(), dim, {(b + 1) * batch_size, dim},
                           {b * batch_size, (b + 1) * batch_size}, PENCIL, offset);
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }
      if (do_batch_remainder) {
        // remainder triangle
        blockRotateComplex(kSpace, matLower.data(), dim, {full_batches * batch_size, keep},
                           {full_batches * batch_size, keep}, LOWER_TRI, offset);
        // remainder pencil
        if (keep < dim) {
          blockRotateComplex(kSpace, matLower.data(), dim, {keep, dim}, {full_batches * batch_size, keep}, PENCIL,
                             offset);
        }
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Do U Multiply
      //---------------------------------------------------------------------------
      if (do_batch_remainder) {
        // remainder triangle
        blockRotateComplex(kSpace, matUpper.data(), keep, {full_batches * batch_size, keep},
                           {full_batches * batch_size, keep}, UPPER_TRI, offset);
        // remainder pencil
        blockRotateComplex(kSpace, matUpper.data(), keep, {0, full_batches * batch_size},
                           {full_batches * batch_size, keep}, PENCIL, offset);
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Loop over full batches
      for (int b = full_batches - 1; b >= 0; b--) {
        // batch triangle
        blockRotateComplex(kSpace, matUpper.data(), keep, {b * batch_size, (b + 1) * batch_size},
                           {b * batch_size, (b + 1) * batch_size}, UPPER_TRI, offset);
        if (b > 0) {
          // batch pencil
          blockRotateComplex(kSpace, matUpper.data(), keep, {0, b * batch_size}, {b * batch_size, (b + 1) * batch_size},
                             PENCIL, offset);
        }
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }

      // Do Q Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matQ.data(), keep);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    }
  }

  void EigenSolver::rotateVecs(std::vector<ColorSpinorField *> &kSpace, const double *rot_array, const int offset,
                               const int dim, const int keep, const int locked, TimeProfile &profile)
  {
    // If we have memory availible, do the entire rotation
    if ((batched_rotate <= 0 || batched_rotate >= keep) && !compressed_mode) {
      if ((int)kSpace.size() < offset + keep) {
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_ZERO_FIELD_CREATE;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace from %d to %d vectors\n", (int)kSpace.size(), offset + keep);
        kSpace.reserve(offset + keep);
        for (int i = kSpace.size(); i < offset + keep; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }

      // Pointers to the relevant vectors
      std::vector<ColorSpinorField *> vecs_ptr;
      std::vector<ColorSpinorField *> kSpace_ptr;

      // Alias the extra space vectors, zero the workspace
      kSpace_ptr.reserve(keep);
      for (int i = 0; i < keep; i++) {
        kSpace_ptr.push_back(kSpace[offset + i]);
        blas::zero(*kSpace_ptr[i]);
      }

      // Alias the vectors we wish to keep.
      vecs_ptr.reserve(dim);
      for (int j = 0; j < dim; j++) vecs_ptr.push_back(kSpace[locked + j]);

      // multiBLAS axpy
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      blas::axpy(rot_array, vecs_ptr, kSpace_ptr);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

      // Copy compressed Krylov
      for (int i = 0; i < keep; i++) std::swap(kSpace[locked + i], kSpace[offset + i]);

    } else {

      int batch_size = batched_rotate;
      int full_batches = keep / batch_size;
      int batch_size_r = keep % batch_size;
      bool do_batch_remainder = (batch_size_r != 0 ? true : false);

      // Ensure that the Krylov space is large enough
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
      MatrixXd mat = MatrixXd::Zero(dim, keep);
      for (int j = 0; j < keep; j++)
        for (int i = 0; i < dim; i++) mat(i, j) = rot_array[i * keep + j];

      FullPivLU<MatrixXd> matLU(mat);

      // Extract the upper triangular matrix
      MatrixXd matUpper = MatrixXd::Zero(keep, keep);
      matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
      matUpper.conservativeResize(keep, keep);

      // Extract the lower triangular matrix
      MatrixXd matLower = MatrixXd::Identity(dim, dim);
      matLower.block(0, 0, dim, keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();
      matLower.conservativeResize(dim, keep);

      // Extract the desired permutation matrices
      MatrixXi matP = MatrixXi::Zero(dim, dim);
      MatrixXi matQ = MatrixXi::Zero(keep, keep);
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
        blockRotate(kSpace, matLower.data(), dim, {full_batches * batch_size, keep}, {full_batches * batch_size, keep},
                    LOWER_TRI);
        // remainder pencil
        if (keep < dim) {
          blockRotate(kSpace, matLower.data(), dim, {keep, dim}, {full_batches * batch_size, keep}, PENCIL);
        }
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Do U Multiply
      //---------------------------------------------------------------------------
      if (do_batch_remainder) {
        // remainder triangle
        blockRotate(kSpace, matUpper.data(), keep, {full_batches * batch_size, keep}, {full_batches * batch_size, keep},
                    UPPER_TRI);
        // remainder pencil
        blockRotate(kSpace, matUpper.data(), keep, {0, full_batches * batch_size}, {full_batches * batch_size, keep},
                    PENCIL);
        blockReset(kSpace, full_batches * batch_size, keep, offset);
      }

      // Loop over full batches
      for (int b = full_batches - 1; b >= 0; b--) {
        // batch triangle
        blockRotate(kSpace, matUpper.data(), keep, {b * batch_size, (b + 1) * batch_size},
                    {b * batch_size, (b + 1) * batch_size}, UPPER_TRI);
        if (b > 0) {
          // batch pencil
          blockRotate(kSpace, matUpper.data(), keep, {0, b * batch_size}, {b * batch_size, (b + 1) * batch_size}, PENCIL);
        }
        blockReset(kSpace, b * batch_size, (b + 1) * batch_size, offset);
      }

      // Do Q Permute
      //---------------------------------------------------------------------------
      permuteVecs(kSpace, matQ.data(), keep);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    }
  }

  EigenSolver::~EigenSolver()
  {
    if (tmp1) delete tmp1;
    if (tmp2) delete tmp2;    
    if(transfer) delete transfer;
    
    for (auto &vec : compressed_space)
      if (vec) delete vec;    
    compressed_space.resize(0);
    
    for (auto &vec : r)
      if (vec) delete vec;    
    r.resize(0);    
    
    for (auto &vec : fine_vector)
      if (vec) delete vec;    
    fine_vector.resize(0);    
  }

  void EigenSolver::createTransferBasis(std::vector<ColorSpinorField *> &vec_space)
  {
    if(transfer) delete transfer;    
    // Create the transfer operator
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating Transfer basis of size %lu with order %d\n", vec_space.size(), vec_space[0]->Order());
    QudaPrecision prec = vec_space[0]->Precision();

    // Point to the converged fine vectors
    fine_space.reserve(fine_n_conv);
    for(int i=0; i<fine_n_conv; i++) fine_space.push_back(vec_space[i]);

    transfer = new Transfer(fine_space, fine_n_conv, n_block_ortho,
			    geo_block_size, spin_block_size, prec,
			    QUDA_TRANSFER_AGGREGATE, profile);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transfer basis created successfully\n");
    verifyCompression(fine_space);
  }
  
  void EigenSolver::verifyCompression(std::vector<ColorSpinorField *> &kSpace)
  {
    if(!transfer) createTransferBasis(kSpace);
    QudaPrecision prec = kSpace[0]->Precision();
    QudaFieldLocation location = kSpace[0]->Location();
    ColorSpinorField *tmp_coarse = kSpace[0]->CreateCoarse(geo_block_size,
							   spin_block_size,
							   fine_n_conv, prec,
							   location);

    // may want to revisit this---these were relaxed for cases where
    // ghost_precision < precision these were set while hacking in tests of quarter
    // precision ghosts
    double tol = (prec == QUDA_QUARTER_PRECISION || prec == QUDA_HALF_PRECISION) ? 5e-2 : prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-8;

    if (!tmp1 || !tmp2) {
      ColorSpinorParam param(*kSpace[0]);
      if (!tmp1) tmp1 = ColorSpinorField::Create(param);
      if (!tmp2) tmp2 = ColorSpinorField::Create(param);
    }
    
    bool error = false;
    for (unsigned int i = 0; i < kSpace.size(); i++) {
      *tmp1 = *kSpace[i];
      transfer->R(*tmp_coarse, *tmp1);
      transfer->P(*tmp2, *tmp_coarse);
      double deviation = sqrt(blas::xmyNorm(*tmp1, *tmp2) / blas::norm2(*tmp1));
      
      if (getVerbosity() >= QUDA_VERBOSE)
	printfQuda( "Vector %d: norms v_k = %e P^\\dagger v_k = %e (1 - P P^\\dagger) v_k = %e, L2 relative deviation = %e\n",
		    i, blas::norm2(*tmp1), blas::norm2(*tmp_coarse), blas::norm2(*tmp2), deviation);
      if (deviation > tol) {
	printfQuda("L2 relative deviation for k=%d failed, %e > %e\n", i, deviation, tol);
	if(!error) error = true;	
      }      
    }
    if(error) errorQuda("verifyCompression failed");
    delete tmp_coarse;
  }
  
  void EigenSolver::compressVectors(const std::vector<ColorSpinorField *> &fine,
				    std::vector<ColorSpinorField *> &coarse,
				    const unsigned int fine_vec_position,
				    const unsigned int coarse_vec_position,
				    const unsigned int num)
  {
    if(!transfer) errorQuda("Eigensolver transfer operator not yet constructed");
    unsigned int f_size = fine.size();
    unsigned int c_size = coarse.size();
    if(fine_vec_position >= f_size) errorQuda("fine_vec_position = %d is greater than or equal the fine space size = %d", fine_vec_position, f_size);
    if(coarse_vec_position >= c_size) errorQuda("coarse_vec_position = %d is greater than or equal the coarse space size = %d", coarse_vec_position, c_size);
    if(num > (f_size) - fine_vec_position) errorQuda("Attempting to compress %u vectors at position %u with a fine vector space of size %u", num, fine_vec_position, f_size);
    if(num > (c_size) - coarse_vec_position) errorQuda("Attempting to compress %u vectors to position %u with a coarse vector space of size %u", num, coarse_vec_position, c_size);
    
    for (unsigned int i = 0; i < num; i++) {
      bool compute_profile = profile.isRunning(QUDA_PROFILE_COMPUTE);
      if (compute_profile) profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      transfer->R(*coarse[coarse_vec_position+i], *fine[fine_vec_position + i]);
      if (compute_profile) profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  void EigenSolver::promoteVectors(const std::vector<ColorSpinorField *> &fine,
				   std::vector<ColorSpinorField *> &coarse,
				   const unsigned int fine_vec_position,
				   const unsigned int coarse_vec_position,
				   const unsigned int num)
  {
    if(!transfer) errorQuda("Eigensolver transfer operator not yet constructed");    
    unsigned int f_size = fine.size();
    unsigned int c_size = coarse.size();
    if(fine_vec_position >= f_size) errorQuda("fine_vec_position = %d is greater than or equal the fine space size = %d", fine_vec_position, f_size);
    if(coarse_vec_position >= c_size) errorQuda("coarse_vec_position = %d is greater than or equal the coarse space size = %d", coarse_vec_position, c_size);
    if(num > (f_size) - fine_vec_position) errorQuda("Attempting to promote %u vectors to position %u with a fine vector space of size %u", num, fine_vec_position, f_size);
    if(num > (c_size) - coarse_vec_position) errorQuda("Attempting to promote %u vectors at position %u with a coarse vector space of size %u", num, coarse_vec_position, c_size);
    
    for (unsigned int i = 0; i < num; i++) {
      bool compute_profile = profile.isRunning(QUDA_PROFILE_COMPUTE);
      if (compute_profile) profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      transfer->P(*fine[fine_vec_position + i], *coarse[coarse_vec_position + i]);
      if (compute_profile) profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }
} // namespace quda
