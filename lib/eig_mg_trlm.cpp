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
  MGTRLM::MGTRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    TRLM(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Tridiagonal/Arrow matrix
    int max_n_kr = std::max(n_kr, n_kr);
    alpha = (double *)safe_malloc(max_n_kr * sizeof(double));
    beta = (double *)safe_malloc(max_n_kr * sizeof(double));
    for (int i = 0; i < max_n_kr; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }

    transfer = nullptr;
    tmp_comp1 = nullptr;
    tmp_comp2 = nullptr;
    
    coarse_n_ev = eig_param->coarse_n_ev;
    coarse_n_kr = eig_param->coarse_n_kr;
    coarse_n_conv = eig_param->coarse_n_conv;
    coarse_max_restarts = eig_param->coarse_max_restarts;
    coarse_n_ev_deflate = eig_param->coarse_n_ev_deflate;
    
    // Sanity checks
    if (coarse_n_kr <= coarse_n_ev) errorQuda("coarse_n_kr = %d is less than or equal to coarse_n_ev = %d", coarse_n_kr, coarse_n_ev);
    if (coarse_n_ev < coarse_n_conv) errorQuda("coarse_n_conv = %d is greater than coarse_n_ev = %d", coarse_n_conv, coarse_n_ev);
    if (coarse_n_ev == 0) errorQuda("coarse_n_ev = 0 passed to Eigensolver");
    if (coarse_n_kr == 0) errorQuda("coarse_n_kr = 0 passed to Eigensolver");
    if (coarse_n_conv == 0) errorQuda("coarse_n_conv = 0 passed to Eigensolver");      
    if (coarse_n_ev_deflate > coarse_n_conv) errorQuda("deflation vecs = %d is greater than coarse_n_conv = %d", coarse_n_ev_deflate, coarse_n_conv);
    
    spin_block_size = 2;
    n_block_ortho = eig_param->n_block_ortho;
    for(int i=0; i<4; i++) geo_block_size[i] = eig_param->geo_block_size[i];
    
    // Thick restart specific checks
    if (coarse_n_kr < coarse_n_ev + 6) errorQuda("coarse_n_kr = %d must be greater than coarse_n_ev+6 = %d\n", coarse_n_kr, coarse_n_ev + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to a Lanczos type solver");
    }
    
    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void MGTRLM::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // In case we are deflating an operator, save the tunechache from the inverter
    saveTuneCache();

    // Override any user input for block size.
    block_size = 1;

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    if (getVerbosity() >= QUDA_VERBOSE) queryPrec(kSpace[0]->Precision());
    
    // Check for loading fine eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file %s\n", eig_param->vec_infile);

      prepareKrylovSpace(kSpace, evals);
      loadFromFile(mat, kSpace, evals);
      // Set converged to true as the fine problem is solved.
      converged = true;
      
      // Check for loading coarse eigenvectors
      if (strcmp(eig_param->coarse_vec_infile, "") != 0) {
	prepareCompressedKrylovSpace(kSpace, evals);
	compressed_mode = true;	  
	loadFromFile(mat, compressed_space, evals);
	// We are done
	return;
      }
    }

    // If no fine eigenspace was loaded, we must construct it.
    // Perform the fine computation, then switch to the
    // full problem when the fine problem is complete.
    if(!converged) {
      
      // Check for an initial guess. If none present, populate with rands, then
      // orthonormalise.
      prepareInitialGuess(kSpace);
      
      // Increase the size of kSpace passed to the function, will be trimmed to
      // original size before exit.
      prepareKrylovSpace(kSpace, evals);

      // Check for Chebyshev maximum estimation
      checkChebyOpMax(mat, kSpace);
      
      // Begin fine TRLM Eigensolver computation
      //---------------------------------------------------------------------------
      printEigensolverSetup();
      converged = mgTRLMSolve(kSpace);
    }
    
    // Switch to the coarse problem parameters
    if(converged) {
      if (getVerbosity() >= QUDA_SUMMARIZE) {
	printfQuda("FINE PROBLEM: TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", n_conv, restart_iter, iter);
      }
      
      // If we loaded a fine space to create a transfer basis, we start the
      // computation from scratch
      if (strcmp(eig_param->vec_infile, "") != 0) {            
	num_converged = 0;
	num_keep = 0;
	num_locked = 0;
	for (int i = 0; i < n_kr; i++) {
	  alpha[i] = 0.0;
	  beta[i] = 0.0;
	  residua[i] = 0.0;
	}
      }
      
      // Algorithm parameters (n_ev, n_kr, etc) are updated in this function
      prepareCompressedKrylovSpace(kSpace, evals);
      compressed_mode = true;
      
      // Print Eigensolver params
      printEigensolverSetup();
      converged = mgTRLMSolve(compressed_space);    
    }
    //---------------------------------------------------------------------------
    
    
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
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", n_conv,
                   restart_iter, iter);

        // Dump all Ritz values and residua if using Chebyshev
        for (int i = 0; i < n_conv && eig_param->use_poly_acc; i++) {
          printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
        }
      }
    }
    
    // Compute eigenvalues
    mgComputeEvals(mat, compressed_space, evals);
    
    // Local clean-up
    cleanUpEigensolver(compressed_space, evals);
  }
  
  // Destructor
  MGTRLM::~MGTRLM()
  {
    ritz_mat.clear();
    ritz_mat.shrink_to_fit();
    host_free(alpha);
    host_free(beta);

    if (tmp_comp1) delete tmp_comp1;
    if (tmp_comp2) delete tmp_comp2;    

    for (auto &vec : compressed_space)
      if (vec) delete vec;    
    compressed_space.resize(0);
    
    for (auto &vec : fine_vector_workspace)
      if (vec) delete vec;    
    fine_vector_workspace.resize(0);
    
    for (auto &vec : B)
      if (vec) delete vec;    
    B.resize(0);    
  }

  // Thick Restart Member functions
  //---------------------------------------------------------------------------
  bool MGTRLM::mgTRLMSolve(std::vector<ColorSpinorField *> &kSpace)
  {
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = setEpsilon(kSpace[0]->Precision());
    converged = false;
    
    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      // If true, promote the entire krylov space to the
      // fine representation and extend the space, then compress
      // back to coarse.
      bool fine_lanczos = true;
      if(fine_lanczos && compressed_mode) {
	int k_size = kSpace.size();
	int f_size = fine_vector_workspace.size();
	// Create fine vectors for test workspace
	ColorSpinorParam cs_param_fine(*fine_vector_workspace[0]);
	cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;
	
	for (int i = f_size; i < k_size; i++) fine_vector_workspace.push_back(ColorSpinorField::Create(cs_param_fine));
	promoteVectors(fine_vector_workspace, kSpace, 0, 0, k_size);
	compressed_mode = false;
	for (int step = num_keep; step < n_kr; step++) lanczosStep(fine_vector_workspace, step);
	
	int f_size_new = fine_vector_workspace.size();
	ColorSpinorParam cs_param_coarse(*kSpace[0]);
	for (int i = k_size; i < f_size_new; i++) kSpace.push_back(ColorSpinorField::Create(cs_param_coarse));
	compressVectors(fine_vector_workspace, kSpace, 0, 0, f_size_new);
      
	for (int i = f_size; i < f_size_new; i++) delete fine_vector_workspace[i];
	fine_vector_workspace.resize(f_size);      
	compressed_mode = true;
      } else {
	for (int step = num_keep; step < n_kr; step++) lanczosStep(kSpace, step);
      }
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
      for (int i = iter_locked + 1; i < n_kr - num_locked; i++) {
        if (residua[i + num_locked] < tol * mat_norm) {
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
            printfQuda("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol * mat_norm);
          iter_converged = i;
        } else {
          // Unlikely to find new converged pairs
          break;
        }
      }

      iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - num_locked - 12);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      mgComputeKeptRitz(kSpace);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      num_converged = num_locked + iter_converged;
      num_keep = num_locked + iter_keep;
      num_locked += iter_locked;

      if (getVerbosity() >= QUDA_VERBOSE) {
        printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
      }

      if (getVerbosity() >= QUDA_VERBOSE) {
        printfQuda("iter Conv = %d\n", iter_converged);
        printfQuda("iter Keep = %d\n", iter_keep);
	printfQuda("iter Lock = %d\n", iter_locked);
        printfQuda("num_converged = %d\n", num_converged);
        printfQuda("num_keep = %d\n", num_keep);
        printfQuda("num_locked = %d\n", num_locked);
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	  for (int i = 0; i < n_kr; i++) {
	    printfQuda("Ritz[%d] = %.16e residual[%d] = %.16e\n", i, alpha[i], i, residua[i]);
	  }
	}
      }
      
      // Check for convergence
      if (num_converged >= n_conv) {
        reorder(kSpace);
        converged = true;
      }

      restart_iter++;
    }
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    return converged;
  }
  
  void MGTRLM::mgLanczosStep(std::vector<ColorSpinorField *> v, int j)
  {
    // Compute r = A * v_j - b_{j-i} * v_{j-1}

    // If in compressed mode, we promote the current Lanczos vector
    // else just point to the current vector
    std::vector<ColorSpinorField *> v_j;
    if(compressed_mode) {
      promoteVectors(fine_vector_workspace, v, 0, j, 1);
      v_j.push_back(fine_vector_workspace[0]);
    } else {
      v_j.push_back(v[j]);
    }

    // r = A * v_j
    chebyOp(mat, *r[0], *v_j[0]);

    // a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v_j[0], *r[0]);

    // r = r - a_j * v_j
    blas::axpy(-alpha[j], *v_j[0], *r[0]);

    int start = (j > num_keep) ? j - 1 : 0;
    
    if (j - start > 0) {
      std::vector<ColorSpinorField *> r_ {r[0]};
      std::vector<double> beta_;
      std::vector<ColorSpinorField *> v_;
      
      if(compressed_mode) {	
	int batch_size = std::min((int)fine_vector_workspace.size(), j - start);
	int full_batches = (j - start)/batch_size;
	int remainder = (j - start)%batch_size;
	bool do_remainder = (j - start)%batch_size == 0 ? false : true;
	
	// We promote the v_idx + vectors.
	// No need to compress after as we read only.
	for (int b = 0; b < full_batches; b++) {

	  beta_.reserve(batch_size);
	  v_.reserve(batch_size);

	  int idx = start + b * batch_size;
	  promoteVectors(fine_vector_workspace, v, 0, idx, batch_size);
	  
	  for(int i = 0; i < batch_size; i++) {
	    beta_.push_back(-beta[idx + i]);
	    v_.push_back(fine_vector_workspace[i]);
	  }
	  // r = r - b_{j-1} * v_{j-1}
	  blas::axpy(beta_.data(), v_, r_);
	  
	  beta_.resize(0);
	  v_.resize(0);
	}
	if(do_remainder) {
	  beta_.reserve(remainder);
	  v_.reserve(remainder);
	  
	  int idx = start + full_batches * batch_size;
	  promoteVectors(fine_vector_workspace, v, 0, idx, remainder);
	  for(int i = 0; i < remainder; i++) {
	    beta_.push_back(-beta[idx + i]);
	    v_.push_back(fine_vector_workspace[i]);
	  }
	  // r = r - b_{j-1} * v_{j-1}
	  blas::axpy(beta_.data(), v_, r_);
	  beta_.resize(0);
	  v_.resize(0);
	}	
      } else {
	beta_.reserve(j - start);
	v_.reserve(j - start);	
	for (int i = start; i < j; i++) {
	  beta_.push_back(-beta[i]);
	  v_.push_back(v[i]);
	}
	// r = r - b_{j-1} * v_{j-1}
	blas::axpy(beta_.data(), v_, r_);
      }
    }

    // Orthogonalise r against the Krylov space
    for (int k = 0; k < 1; k++) mgBlockOrthogonalize(v, r, j + 1);

    // b_j = ||r||
    beta[j] = sqrt(blas::norm2(*r[0]));

    // Prepare next step.
    // v_{j+1} = r / b_j
    if(compressed_mode) {
      promoteVectors(fine_vector_workspace, v, 0, j+1, 1);
      blas::zero(*fine_vector_workspace[0]);
      blas::axpy(1.0 / beta[j], *r[0], *fine_vector_workspace[0]);
      compressVectors(fine_vector_workspace, v, 0, j+1, 1);      
    } else {
      blas::zero(*v[j + 1]);
      blas::axpy(1.0 / beta[j], *r[0], *v[j + 1]);
    }

    // Save Lanczos step tuning
    saveTuneCache();
  }
  
  void MGTRLM::mgComputeKeptRitz(std::vector<ColorSpinorField *> &kSpace)
  {
    int offset = n_kr + 1;
    int dim = n_kr - num_locked;

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    double *ritz_mat_keep = (double *)safe_malloc((dim * iter_keep) * sizeof(double));
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < iter_keep; i++) { ritz_mat_keep[j * iter_keep + i] = ritz_mat[i * dim + j]; }
    }

    // If true, promote the entire krylov space to the
    // fine representation and rotate the space, then compress
    // back to coarse.
    bool fine_rotate = true;
    if(fine_rotate && compressed_mode) {
      int k_size = kSpace.size();
      int f_size = fine_vector_workspace.size();

      // Create fine vectors for test workspace
      ColorSpinorParam cs_param_fine(*fine_vector_workspace[0]);
      cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;

      for (int i = f_size; i < k_size; i++) fine_vector_workspace.push_back(ColorSpinorField::Create(cs_param_fine));
      promoteVectors(fine_vector_workspace, kSpace, 0, 0, k_size);
      compressed_mode = false;
      rotateVecs(fine_vector_workspace, ritz_mat_keep, offset, dim, iter_keep, num_locked, profile);

      int f_size_new = fine_vector_workspace.size();
      ColorSpinorParam cs_param_coarse(*kSpace[0]);
      for (int i = k_size; i < f_size_new; i++) kSpace.push_back(ColorSpinorField::Create(cs_param_coarse));
      compressVectors(fine_vector_workspace, kSpace, 0, 0, f_size_new);
      
      for (int i = f_size; i < f_size_new; i++) delete fine_vector_workspace[i];
      fine_vector_workspace.resize(f_size);      
      compressed_mode = true;
    } else {
      rotateVecs(kSpace, ritz_mat_keep, offset, dim, iter_keep, num_locked, profile);
    }
    
    // Update residual vector
    std::swap(kSpace[num_locked + iter_keep], kSpace[n_kr]);

    // Update sub arrow matrix
    for (int i = 0; i < iter_keep; i++) beta[i + num_locked] = beta[n_kr - 1] * ritz_mat[dim * (i + 1) - 1];
    
    host_free(ritz_mat_keep);
  }

  void MGTRLM::createTransferBasis(std::vector<ColorSpinorField *> &vec_space)
  {
    if(transfer) delete transfer;    
    // Create the transfer operator
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating Transfer basis of size fine_n_conv = %d\n", n_conv);
    
    // Point to the converged fine vectors
    fine_space.reserve(n_conv);
    for(int i=0; i<n_conv; i++) fine_space.push_back(vec_space[i]);

#if 0
    bool pc_solution = false;
    if (pc_solution) X[0] *= 1;
#endif

    // Create a new vector space with native ordering for the transfer basis
    ColorSpinorParam csParam(*vec_space[0]);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    B.reserve(n_conv);
    for (int i = 0; i < n_conv; i++) {
      B.push_back(ColorSpinorField::Create(csParam));
      *B[i] = *vec_space[i];
    }
    
    transfer = new Transfer(B, n_conv, n_block_ortho, true, //?
			    geo_block_size, spin_block_size, vec_space[0]->Precision(),
			    QUDA_TRANSFER_AGGREGATE, profile);
    
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transfer basis created successfully\n");
    verifyCompression(B);
  }
  
  void MGTRLM::verifyCompression(std::vector<ColorSpinorField *> &kSpace)
  {
    printfQuda("Flag 5\n");
    if(!transfer) createTransferBasis(kSpace);
    printfQuda("Flag 5.1\n");
    QudaPrecision prec = kSpace[0]->Precision();
    printfQuda("Flag 5.2\n");
    QudaFieldLocation location = kSpace[0]->Location();    
    printfQuda("Flag 5.3 %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
    ColorSpinorField *tmp_coarse = kSpace[0]->CreateCoarse(geo_block_size,
							   spin_block_size,
							   n_conv, prec,
							   location);
    printfQuda("Flag 5.4\n");
    // may want to revisit this---these were relaxed for cases where
    // ghost_precision < precision these were set while hacking in tests of quarter
    // precision ghosts
    double tol = (prec == QUDA_QUARTER_PRECISION || prec == QUDA_HALF_PRECISION) ? 5e-2 : prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-8;
    ColorSpinorParam param(*kSpace[0]);
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    if(!tmp_comp1) tmp_comp1 = ColorSpinorField::Create(param);
    if(!tmp_comp2) tmp_comp2 = ColorSpinorField::Create(param);
    bool error = false;
    printfQuda("Flag 5.5\n");
    for (unsigned int i = 0; i < kSpace.size(); i++) {
      printfQuda("Flag 6.1 %d\n", i);
      *tmp_comp1 = *kSpace[i];
      printfQuda("Flag 6.2 %d\n", i);
      transfer->R(*tmp_coarse, *tmp_comp1);
      printfQuda("Flag 6.3 %d\n", i);
      transfer->P(*tmp_comp2, *tmp_coarse);
      printfQuda("Flag 6.4 %d\n", i);
      double deviation = 0;//sqrt(blas::xmyNorm(*tmp_comp1, *tmp_comp2) / blas::norm2(*tmp_comp1));
      printfQuda("Flag 6.5 %d\n", i);
      if (getVerbosity() >= QUDA_VERBOSE) {
	//printfQuda( "Vector %d: norms v_k = %e P^\\dagger v_k = %e (1 - P P^\\dagger) v_k = %e, L2 relative deviation = %e\n", i, blas::norm2(*tmp_comp1), blas::norm2(*tmp_coarse), blas::norm2(*tmp_comp2), deviation);
      }
      printfQuda("Flag 6.6 %d\n", i);
      if (deviation > tol) {
	printfQuda("L2 relative deviation for k=%d failed, %e > %e\n", i, deviation, tol);
	if(!error) error = true;	
      }
      printfQuda("Flag 6.7 %d\n", i);
    }
    if(error) errorQuda("verifyCompression failed");
    delete tmp_coarse;
  }
  
  void MGTRLM::compressVectors(const std::vector<ColorSpinorField *> &fine,
			       std::vector<ColorSpinorField *> &coarse,
			       const unsigned int fine_vec_position,
			       const unsigned int coarse_vec_position,
			       const unsigned int num) const
  {
    if(!transfer) errorQuda("Eigensolver transfer operator not yet constructed");
    unsigned int f_size = fine.size();
    unsigned int c_size = coarse.size();
    if(fine_vec_position >= f_size) errorQuda("fine_vec_position = %d is greater than or equal the fine space size = %d", fine_vec_position, f_size);
    if(coarse_vec_position >= c_size) errorQuda("coarse_vec_position = %d is greater than or equal the coarse space size = %d", coarse_vec_position, c_size);
    if(num > (f_size) - fine_vec_position) errorQuda("Attempting to compress %u vectors at position %u with a fine vector space of size %u", num, fine_vec_position, f_size);
    if(num > (c_size) - coarse_vec_position) errorQuda("Attempting to compress %u vectors to position %u with a coarse vector space of size %u", num, coarse_vec_position, c_size);

    printfQuda("f_size = %d, c_size = %d\n", f_size, c_size);
    
    for (unsigned int i = 0; i < num; i++) {
      bool compute_profile = profile.isRunning(QUDA_PROFILE_COMPUTE);
      if (compute_profile) profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      *tmp_comp1 = *fine[fine_vec_position + i];
      transfer->R(*coarse[coarse_vec_position+i], *tmp_comp1);
      if (compute_profile) profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  void MGTRLM::promoteVectors(std::vector<ColorSpinorField *> &fine,
			      const std::vector<ColorSpinorField *> &coarse,
			      const unsigned int fine_vec_position,
			      const unsigned int coarse_vec_position,
			      const unsigned int num) const
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
      transfer->P(*tmp_comp1, *coarse[coarse_vec_position + i]);
      *fine[fine_vec_position + i] = *tmp_comp1;
      if (compute_profile) profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  void MGTRLM::prepareCompressedKrylovSpace(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    int c_size = compressed_space.size();
    if(c_size != 0) errorQuda("A compressed space already exists");
    int f_size = fine_vector_workspace.size();
    int e_size = evals.size();

    if(!transfer) createTransferBasis(kSpace);
    
    QudaPrecision prec = kSpace[0]->Precision();
    QudaFieldLocation location = kSpace[0]->Location();
    ColorSpinorField *tmp_coarse = kSpace[0]->CreateCoarse(geo_block_size,
							   spin_block_size,
							   n_conv, prec,
							   location);
    
    // Clone the coarse paramters
    ColorSpinorParam cs_param_coarse(*tmp_coarse);
    delete tmp_coarse;

    // Update algorithm parameters
    int fine_n_conv = n_conv;    
    n_ev = eig_param->coarse_n_ev;
    n_kr = eig_param->coarse_n_kr;	  
    n_conv = eig_param->coarse_n_conv;	  
    max_restarts = eig_param->coarse_max_restarts;
    n_ev_deflate = eig_param->coarse_n_ev_deflate;	  
    
    // Increase coarse space
    int max_size = (strcmp(eig_param->coarse_vec_infile, "") != 0 ? n_conv : n_kr + 1 + batched_rotate);
    compressed_space.reserve(max_size);
    for (int i = c_size; i < max_size; i++) compressed_space.push_back(ColorSpinorField::Create(cs_param_coarse));
    
    // Create fine vectors for workspace
    ColorSpinorParam cs_param_fine(*kSpace[0]);
    cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;
    for (int b = f_size; b < batched_rotate; b++) { fine_vector_workspace.push_back(ColorSpinorField::Create(cs_param_fine)); }

    // Increase evals
    int eval_size = (strcmp(eig_param->coarse_vec_infile, "") != 0 ? n_conv : n_kr);
    evals.reserve(eval_size);
    for (int i = e_size; i < eval_size; i++) evals.push_back(0.0);
    
    // If we load the fine space we must restart the coarse solve from scratch.
    if (strcmp(eig_param->vec_infile, "") != 0) {
      // Restart the solver
      blas::zero(*fine_vector_workspace[0]);
      // Get some random noise
      prepareInitialGuess(fine_vector_workspace);
      // Compress the guess
      compressVectors(fine_vector_workspace, compressed_space, 0, 0, block_size);
    } else {
      printfQuda("prepareCompressedKrylovSpace compress vectors start\n");
      // We may simply compress the current factorisation and continue
      compressVectors(kSpace, compressed_space, 0, 0, kSpace.size());
      printfQuda("prepareCompressedKrylovSpace compress vectors end\n");
    }
    
    // Only save fine vectors if outfile is defined.    
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving fine eigenvectors\n");
      saveToFile(kSpace, fine_n_conv, impliedParityFromMatPC(mat.getMatPCType()));
    }
    
    for(unsigned int i=block_size; i<kSpace.size(); i++) delete kSpace[i];
    kSpace.resize(block_size);
  }

  // Orthogonalise r[0:] against V_[0:j]
  void MGTRLM::mgBlockOrthogonalize(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> &rvecs, int j)
  {
    int r_size = rvecs.size();
    std::vector<ColorSpinorField *> vecs_ptr;    
    int batch_size = std::min((int)fine_vector_workspace.size(), j);
    int full_batches = j/batch_size;
    int remainder = j%batch_size;
    bool do_remainder = j%batch_size != 0 ? true : false;
    
    for (int b = 0; b < full_batches; b++) {
      int idx = b * batch_size;
      promoteVectors(fine_vector_workspace, vecs, 0, idx, batch_size);
      
      // Block dot products stored in s.
      std::vector<Complex> s(batch_size * r_size);
      blas::cDotProduct(s.data(), fine_vector_workspace, rvecs);
      
      // Block orthogonalise
      for (int i = 0; i < batch_size * r_size; i++) s[i] *= -1.0;
      blas::caxpy(s.data(), fine_vector_workspace, rvecs);
    }
    if(do_remainder) {
      
      int idx = full_batches * batch_size;
      
      promoteVectors(fine_vector_workspace, vecs, 0, idx, remainder);
      vecs_ptr.reserve(remainder);
      for (int i = 0; i < remainder; i++) { vecs_ptr.push_back(fine_vector_workspace[i]); }
      
      // Block dot products stored in s
      std::vector<Complex> s(remainder * r_size);
      blas::cDotProduct(s.data(), vecs_ptr, rvecs);
      
      // Block orthogonalise
      for (int i = 0; i < remainder * r_size; i++) s[i] *= -1.0;
      blas::caxpy(s.data(), vecs_ptr, rvecs);
    }
    
    // Save orthonormalisation tuning
    saveTuneCache();
  }

  void MGTRLM::mgBlockRotate(std::vector<ColorSpinorField *> &kSpace, double *array, int rank, const range &i_range,
                                const range &j_range, blockType b_type)
  {
    int block_i_rank = i_range.second - i_range.first;
    int block_j_rank = j_range.second - j_range.first;
    
    // Quick return if no op.
    if (block_i_rank == 0 || block_j_rank == 0) return;
    
    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;

    // Populate batch array (COLUMN major -> ROW major)
    double *batch_array = (double *)safe_malloc((block_i_rank * block_j_rank) * sizeof(double));
    for (int j = j_range.first; j < j_range.second; j++) {
      for (int i = i_range.first; i < i_range.second; i++) {
	int j_arr = j - j_range.first;
	int i_arr = i - i_range.first;
	batch_array[i_arr * block_j_rank + j_arr] = array[j * rank + i];
      }
    }
    
    // If compressed vectors are passed, we create fine vectors
    // and prolongate the batches into them.
    int f_size = fine_vector_workspace.size();
    if(2*block_j_rank > f_size) {
	fine_vector_workspace.reserve(2*block_j_rank);      
	ColorSpinorParam cs_param_fine(*fine_vector_workspace[0]);
	cs_param_fine.create = QUDA_ZERO_FIELD_CREATE;
	for (int i = f_size; i < 2*block_j_rank; i++) fine_vector_workspace.push_back(ColorSpinorField::Create(cs_param_fine));
    }
    
    // Alias the extra space vectors
    kSpace_ptr.reserve(block_j_rank);
    promoteVectors(fine_vector_workspace, kSpace, 0, n_kr + 1, block_j_rank);
    for (int j = 0; j < block_j_rank; j++) kSpace_ptr.push_back(fine_vector_workspace[j]);
    
    // The `batch array` now contains either a PENCIL or TRI of data.
    // In compressed mode we must loop over SQUARES of the PENCIL,
    // each of side length `block_j_rank`.
    
    int full_batches = block_i_rank / block_j_rank;
    int batch_size = block_j_rank;
    int batch_size_r = block_i_rank % batch_size;
    bool do_batch_remainder = (batch_size_r != 0 ? true : false);
    
    int sq_size = batch_size * batch_size; // Elements in a square
    double *square_batch_array = (double *)safe_malloc((sq_size) * sizeof(double));
    
    for (int b = 0; b < full_batches; b++) {
      
      // Populate square batch array
      int i_offset = num_locked + i_range.first + b * batch_size;
      for (int i = b * batch_size; i < (b+1)*batch_size; i++) {
	int i_sq = i - b * batch_size;	  
	for (int j = 0; j < batch_size; j++) {
	  square_batch_array[i_sq * batch_size + j] = batch_array[i * batch_size + j];
	}
      } 
      
      // Alias the vectors we wish to keep
      vecs_ptr.reserve(batch_size);
      promoteVectors(fine_vector_workspace, kSpace, batch_size, i_offset, batch_size);
      for (int i = 0; i < batch_size; i++) vecs_ptr.push_back(fine_vector_workspace[batch_size + i]);
      
      switch (b_type) {
      case PENCIL: blas::axpy(square_batch_array, vecs_ptr, kSpace_ptr); break;
      case LOWER_TRI: blas::axpy_L(square_batch_array, vecs_ptr, kSpace_ptr); break;
      case UPPER_TRI: blas::axpy_U(square_batch_array, vecs_ptr, kSpace_ptr); break;
      default: errorQuda("Undefined MultiBLAS type in blockRotate");
      }
      
      compressVectors(fine_vector_workspace, kSpace, batch_size, i_offset, batch_size);
      vecs_ptr.resize(0);
    }
    host_free(square_batch_array);
    
    if (do_batch_remainder) {
      int r_size = batch_size * batch_size_r; // Elements in a remainder rectangle
      double *r_batch_array = (double *)safe_malloc((r_size) * sizeof(double));
      
      // Populate remainder batch array
      int b = full_batches;
      int i_offset = num_locked + i_range.first + b * batch_size;
      for (int i = b * batch_size; i < b * batch_size + batch_size_r; i++) {
	int i_r = i - b * batch_size;
	for (int j = 0; j < batch_size; j++) {
	  r_batch_array[i_r * batch_size + j] = batch_array[i * batch_size + j];
	}
      } 
      
      // Alias the vectors we wish to keep
      vecs_ptr.reserve(batch_size_r);
      promoteVectors(fine_vector_workspace, kSpace, batch_size, i_offset, batch_size_r);
      for (int i = 0; i < batch_size_r; i++) vecs_ptr.push_back(fine_vector_workspace[batch_size + i]);
      
      // There will never be a TRI type in the remainder of a compressed rotation.
      blas::axpy(r_batch_array, vecs_ptr, kSpace_ptr);
      compressVectors(fine_vector_workspace, kSpace, batch_size, i_offset, batch_size_r);	
      host_free(r_batch_array);
      vecs_ptr.resize(0);
    }
    
    // Compress the rotated fine vectors
    compressVectors(kSpace_ptr, kSpace, 0, n_kr + 1, batch_size);
    if(2*block_j_rank > f_size) {
      for (int i = f_size; i < 2*block_j_rank; i++) delete fine_vector_workspace[i];
      fine_vector_workspace.resize(f_size);
    }
  
    host_free(batch_array);
    // Save Krylov block rotation tuning
    saveTuneCache();    
  }
  
  void MGTRLM::mgRotateVecs(std::vector<ColorSpinorField *> &kSpace, const double *rot_array, const int offset,
			    const int dim, const int keep, TimeProfile &profile)
  {
    
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

  void MGTRLM::mgComputeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs,
			      std::vector<Complex> &evals, int size)
  {
    if (size > (int)evecs.size())
      errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size())
      errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());

    std::vector<ColorSpinorField *> temp;
    std::vector<ColorSpinorField *> evecs_ptr;
    evecs_ptr.reserve(size);
    ColorSpinorParam cs_param_fine(*fine_vector_workspace[0]);
    temp.push_back(ColorSpinorField::Create(cs_param_fine));
    
    for (int i = 0; i < size; i++) {      
      // r = A * v_i      
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      promoteVectors(fine_vector_workspace, evecs, 0, i, 1);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

#if 0
      if (i >= fine_space.size()) smoothEvec(mat, *fine_vector_workspace[0]);
#endif
      evecs_ptr.push_back(fine_vector_workspace[0]);
    
      matVec(mat, *temp[0], *evecs_ptr[i]);
      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = blas::cDotProduct(*evecs_ptr[i], *temp[0]) / sqrt(blas::norm2(*evecs_ptr[i]));
      // Measure || lambda_i * v_i - A * v_i ||
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
  void MGTRLM::mgDeflate(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src, const std::vector<Complex> &evals,
			 bool accumulate) const
  {
    // number of evecs
    if (n_ev_deflate == 0) {
      warningQuda("deflate called with n_ev_deflate = 0");
      return;
    }
    
    int n_defl = n_ev_deflate;
    
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Deflating %d vectors\n", n_defl);
    
    int n_batch = 1;
    std::vector<ColorSpinorField*> decompressed(n_batch);
    ColorSpinorParam cs_param(*fine_vector_workspace[0]);
    for (auto &v : decompressed) v = ColorSpinorField::Create(cs_param);
    
    for (int i = 0; i < n_defl; i += n_batch) {
      promoteVectors(decompressed, compressed_space, 0, i, n_batch);
      
      // 1. Take block inner product: (V_i)^dag * vec = A_i
      std::vector<Complex> s(n_batch * src.size());
      std::vector<ColorSpinorField *> src_ = const_cast<decltype(src) &>(src);
      blas::cDotProduct(s.data(), const_cast<std::vector<ColorSpinorField*> &>(decompressed), src_);
      
      // 2. Perform block caxpy: V_i * (L_i)^{-1} * A_i
      for (int j = 0; j < n_batch; j++) {
	for (int k = 0; k < (int)src.size(); k++) {
	  s[j * src.size() + k] /= evals[i + j].real(); // FIXME need to check if (j,j) are in correct order when src.size > 1
	}
      }
      
      // 3. Accumulate sum vec_defl = Sum_i V_i * (L_i)^{-1} * A_i
      if (!accumulate && i == 0)
	for (auto &x : sol) blas::zero(*x);
      blas::caxpy(s.data(), decompressed, sol);
    }
    
    for (auto &v : decompressed) delete v;
    
    // Save Deflation tuning
    saveTuneCache();
  }

} // namespace quda

