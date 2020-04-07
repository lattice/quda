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

//#define DEBUG_LOCAL

namespace quda
{

  using namespace Eigen;
  
  // Thick Restarted Block Lanczos Method constructor
  BLKTRLM::BLKTRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    TRLM(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Block Thick restart specific checks
    if (nKr < nEv + 6) errorQuda("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    
    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }
    
    if(nKr % block_size != 0) {
      errorQuda("Block size %d is not a factor of the Krylov space size %d", block_size, nKr);
    }

    if(block_size == 0) {
      errorQuda("Block size %d passed to block eigensolver", block_size);
    }
    
    n_blocks = nKr / block_size;
    block_data_length = block_size * block_size;
    int arrow_mat_size = block_data_length * n_blocks;
    // Tridiagonal/Arrow matrix
    block_alpha = (Complex *)safe_malloc(arrow_mat_size * sizeof(Complex));
    block_beta = (Complex *)safe_malloc(arrow_mat_size * sizeof(Complex));
    for (int i = 0; i < arrow_mat_size; i++) {
      block_alpha[i] = 0.0;
      block_beta[i] = 0.0;
    }    

    // Temp storage used in blockLanczosStep
    jth_block = (Complex *)safe_malloc(block_data_length * sizeof(Complex));
    beta_diag_inv = (double *)safe_malloc(block_size * sizeof(double));
    alpha_old = (double *)safe_malloc(nKr * sizeof(double));      
    
    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }
  
  void BLKTRLM::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // In case we are deflating an operator, save the tunechache from the inverter
    saveTuneCache();
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Create device side residual vectors by cloning
    // the kSpace passed to the function.
    ColorSpinorParam csParamClone(*kSpace[0]);
    // Increase Krylov space to nKr+block_size
    kSpace.reserve(nKr + block_size);
    for (int i = nConv; i < nKr + block_size; i++) kSpace.push_back(ColorSpinorField::Create(csParamClone));
    // create residual vectors
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    for(int b=0; b<block_size; b++) {
      r.push_back(ColorSpinorField::Create(csParamClone));
    }
    
    // Test for initial guess vectors
    // DMH: This is an important step. With block solvers, initial guesses
    //      of block sizes N can be subspaces rich in exremal eigenmodes,
    //      N times more rich than non-blocked solvers.
    //      Final paragraph, IV.B https://arxiv.org/pdf/1902.02064.pdf
    double norm = sqrt(blas::norm2(*kSpace[0]));
    if (norm == 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Initial residua are zero. Populating with rands.\n");
      if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
	for(int b=0; b<block_size; b++) {	  
	  kSpace[b]->Source(QUDA_RANDOM_SOURCE);
	}
      } else {
	RNG *rng = new RNG(*kSpace[0], 1234);
	rng->Init();	
	for(int b=0; b<block_size; b++) spinorNoise(*kSpace[b], *rng, QUDA_NOISE_UNIFORM);
	rng->Release();
	delete rng;
      }
    }

    printfQuda("Orthonormalising initial guesses with Gram-Schmidt.\n"); 
    orthonormalizeMGS(kSpace, block_size);
    //orthogonalizeGS(kSpace, block_size);
    
    printfQuda("Checking initial guesses.\n"); 
    orthoCheck(kSpace, block_size);
    
    printfQuda("Estimate Chebyshev max.\n"); 
    // Check for Chebyshev maximum estimation
    if (eig_param->use_poly_acc && eig_param->a_max <= 0.0) {
      // Use two vectors from kSpace as temps
      eig_param->a_max = estimateChebyOpMax(mat, *kSpace[block_size+1], *kSpace[block_size]);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Chebyshev maximum estimate: %e.\n", eig_param->a_max);
    }
    
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
      printfQuda("***********************************\n");
      printfQuda("**** START BLOCK TRLM SOLUTION ****\n");
      printfQuda("***********************************\n");
    }

    // Print Eigensolver params
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("spectrum %s\n", spectrum);
      printfQuda("tol %.4e\n", tol);
      printfQuda("nConv %d\n", nConv);
      printfQuda("nEv %d\n", nEv);
      printfQuda("nKr %d\n", nKr);
      printfQuda("block size %d\n", block_size);
      if (eig_param->use_poly_acc) {
        printfQuda("polyDeg %d\n", eig_param->poly_deg);
        printfQuda("a-min %f\n", eig_param->a_min);
        printfQuda("a-max %f\n", eig_param->a_max);
      }
    }

    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (int step = num_keep; step < nKr; step += block_size) {
	//printfQuda("Performing block step %d\n", step);
	blockLanczosStep(kSpace, step);
	//printfQuda("Block step %d complete\n", step);
      }
      iter += (nKr - num_keep);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      //printfQuda("Performing block Ritz\n");
      computeBlockKeptRitz(kSpace);
      for(int ev=0; ev<nEv; ev++) printfQuda("alpha %d = %.16e\n", ev, alpha[ev]);
      //printfQuda("Block Ritz complete\n");
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

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
      num_keep = nEv;
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
    for(int b=0; b<block_size; b++) delete r[b];
    r.resize(0);
    
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
      printfQuda("***********************************\n");
      printfQuda("***** END BLOCK TRLM SOLUTION *****\n");
      printfQuda("***********************************\n");
    }

    // Save TRLM tuning
    saveTuneCache();
    
    mat.flops();
  }
  
  // Destructor
  BLKTRLM::~BLKTRLM()
  {
    host_free(jth_block);
    host_free(beta_diag_inv);
    host_free(alpha_old);
    host_free(block_alpha);
    host_free(block_beta);
  }

  // Block Thick Restart Member functions
  //---------------------------------------------------------------------------
  void BLKTRLM::blockLanczosStep(std::vector<ColorSpinorField *> v, int j)
  {
    // Compute r = A * v_j - b_{j-i} * v_{j-1}

    // Offset for alpha, beta matrices
    int arrow_offset = j * block_size;
    int idx = 0, idx_conj = 0;
    
    // r = A * v_j
    for(int b=0; b<block_size; b++) chebyOp(mat, *r[b], *v[j + b]);

    // r = r - b_{j-1} * v_{j-1}
    /*
    int start = (j > num_keep) ? j - 1 : 0;
    if (j - start > 0) {
      std::vector<ColorSpinorField *> r_;
      r_.reserve(block_size);
      for(int i=0; i<block_size; i++) r_.push_back(r[i]);
      std::vector<Complex> beta_;
      beta_.reserve(block_data_length);
      for (int i = 0; i < block_data_length; i++) {
	beta_.push_back(-block_beta[arrow_offset - block_data_length + i]);
      }
      std::vector<ColorSpinorField *> v_;
      v_.reserve((j-start)*block_size);
      for (int i = start; i < j; i++) {
        v_.push_back(v[i]);
      }
      blas::caxpy(beta_.data(), v_, r_);
    }
    */
    
    // a_j = v_j^dag * r
    std::vector<ColorSpinorField *> vecs_ptr;
    vecs_ptr.reserve(block_size);
    for (int b = 0; b < block_size; b++) { vecs_ptr.push_back(v[j + b]); }
    // Block dot products stored in alpha_block.
    blas::cDotProduct(block_alpha + arrow_offset, vecs_ptr, r);

#ifdef DEBUG_LOCAL
    printfQuda("Current alpha\n");
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = c*block_size + b;
	printfQuda("(%e,%e) ", block_alpha[arrow_offset + idx].real(), block_alpha[arrow_offset + idx].imag());
      }
      printfQuda("\n");
    }
#endif
    
    // Solve current block tridiag
    eigensolveFromBlockArrowMat(j);
    
    // Use jth_block to negate alpha data and apply block BLAS.
    // Data is in square hermitian form, no need to switch to ROW major    
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	jth_block[idx] = -1.0 * block_alpha[arrow_offset + idx];
      }
    }
        
    // r = r - a_j * v_j
    blas::caxpy(jth_block, vecs_ptr, r);
    
    // Orthogonalise R[0:block_size] against the Krylov space V[0:block_offset] 
    if(j > 0) blockOrthogonalize(v, r, j);
    
    // QR decomposition via modified Gram Scmidt
    // NB, QR via modified Gram-Schmidt is numerically unstable.
    // May perform the QR iteratively to recover
    // numerical stability?
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
    while(!orthed && k<1) {
      // Compute R_{k}
      for(int b=0; b<block_size; b++) {      
	double norm = sqrt(blas::norm2(*r[b]));
	blas::ax(1.0 / norm, *r[b]);
	jth_block[b*(block_size + 1)] = norm;
	for(int c=0; c<b; c++) {
	  idx      = b*block_size + c;
	  idx_conj = c*block_size + b;
	  Complex cnorm = blas::cDotProduct(*r[c], *r[b]);
	  blas::caxpy(-cnorm, *r[c], *r[b]);	
	  jth_block[idx     ] = cnorm;
	  jth_block[idx_conj] = 0.0;	  
	}
      }
      // Accumulate R_{k}
      updateBlockBeta(k, arrow_offset);
      orthed = orthoCheck(r, block_size);
      k++;
    }
#ifdef DEBUG_LOCAL
    printfQuda("Orthed at k=%d\n", k);
#endif
    // Prepare next step.
    // v_{j+1} = r
    for(int b=0; b<block_size; b++) *v[j + block_size + b] = *r[b];

#ifdef DEBUG_LOCAL
    printfQuda("Current beta\n");
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = c*block_size + b;
	printfQuda("(%e,%e) ", block_beta[arrow_offset + idx].real(), block_beta[arrow_offset + idx].imag());
      }
      printfQuda("\n");
    }
#endif
    // Save Lanczos step tuning
    saveTuneCache();    
  }

  void BLKTRLM::updateBlockBeta(int k, int arrow_offset)
  {
    if(k == 0) {
      //Copy over the jth_block matrix to block beta, Beta = R_0
      int idx = 0, idx_conj = 0;
      for(int b=0; b<block_size; b++) {
	for(int c=0; c<b+1; c++) {
	  idx      = b*block_size + c;
	  block_beta[arrow_offset + idx] = jth_block[idx];
	}
      }
    } else {
      // Compute BetaNew_ac = (R_k)_ab * Beta_bc
      // Use Eigen, it's neater
      MatrixXcd betaN = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd beta  = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd Rk    = MatrixXcd::Zero(block_size, block_size);
      int idx = 0;
      for(int b=0; b<block_size; b++) {
	for(int c=0; c<b+1; c++) {
	  idx = b*block_size + c;
	  beta(c,b) = block_beta[arrow_offset + idx];
	  Rk(c,b)   = jth_block[idx];
	}
      }
      betaN = beta * Rk;
      for(int b=0; b<block_size; b++) {
	for(int c=0; c<b+1; c++) {
	  idx = b*block_size + c;
	  block_beta[arrow_offset + idx] = betaN(c,b);
	}
      }
    }
  }
  
  void BLKTRLM::eigensolveFromBlockArrowMat(int block_pos)
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    int dim = block_pos + block_size;
    int current_block = block_pos/block_size;
    if(block_pos%block_size != 0) errorQuda("Block pos %d != block size %d\n",
					    block_pos, block_size);
    
    // Eigen objects
    MatrixXcd T = MatrixXcd::Zero(dim, dim);
    block_ritz_mat.resize(dim * dim);
    for (int i = 0; i < dim * dim; i++) block_ritz_mat[i] = 0.0;
    int idx = 0, idx_conj = 0;

    // Populate the r and eblocks
    if(restart_iter > 0) {
      //printfQuda("R + E\n");
      for(int i=0; i<nEv/block_size; i++) {	
	for(int b=0; b<block_size; b++) {
	  // E block
	  idx = i*block_size + b;
	  T(idx, idx) = alpha[idx];	    
	  
	  for(int c=0; c<block_size; c++) {	    
	    // r blocks
	    idx = b*block_size + c;	    
	    T(nEv + c, i*block_size + b) =      block_beta[i*block_data_length + idx];
	    T(i*block_size + b, nEv + c) = conj(block_beta[i*block_data_length + idx]);
	  }
	}
      }
      // Inspect T
#ifdef DEBUG_LOCAL
      std::cout << T << std::endl << std::endl;
#endif
    }

    // Add the alpha blocks
    if(block_pos < nKr) {
      //printfQuda("Alpha\n");
      for(int i=num_keep/block_size; i<current_block + 1; i++) {
	for(int b=0; b<block_size; b++) {
	  for(int c=0; c<block_size; c++) {
	    idx = b*block_size + c;
	    T(i*block_size + c, i*block_size + b) = block_alpha[i*block_data_length + idx];
	  }
	}
      }
      // Inspect T
#ifdef DEBUG_LOCAL
      std::cout << T << std::endl << std::endl;
#endif
    }
    
    // Add the current beta blocks
    if(current_block > 0) {
      //printfQuda("Beta\n");
      for(int i=num_keep/block_size; i<current_block; i++) {
	for(int b=0; b<block_size; b++) {
	  for(int c=0; c<b+1; c++) {
	    idx = b*block_size + c;
	    T(i*block_size + b, (i+1)*block_size + c) = conj(block_beta[i*block_data_length + idx]);
	    T((i+1)*block_size + c, i*block_size + b) =      block_beta[i*block_data_length + idx];
	  }
	}
      }
      // Inspect T
#ifdef DEBUG_LOCAL
      std::cout << T << std::endl << std::endl;
#endif
    }
    
    // Invert the spectrum due to Chebyshev (except the current E block)
    if (reverse) {
      for(int b=0; b<dim; b++) {
	for(int c=0; c<dim; c++) {
	  if(b != c && b > num_keep && c > num_keep) T(c, b) *= -1.0;
	}
      }
    }
    
    // Eigensolve the arrow matrix
    SelfAdjointEigenSolver<MatrixXcd> eigensolver;
    eigensolver.compute(T);

    // Populate the alpha array with eigenvalues of negated operator
    for(int i=0; i<dim; i++) alpha[i] = eigensolver.eigenvalues()[i];
    
    // Repopulate ritz matrix: COLUMN major
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
	block_ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];
    
    profile.TPSTOP(QUDA_PROFILE_EIGEN);
    //printfQuda("eigensolveFromBlockArrowMat Complete\n");
  }
  
  void BLKTRLM::computeBlockKeptRitz(std::vector<ColorSpinorField *> &kSpace)
  {
    int offset = nKr + block_size;
    int dim = nKr - num_locked;
    
    // Multi-BLAS friendly array to store part of Ritz matrix we want
    Complex *ritz_mat_keep = (Complex *)safe_malloc((dim * nEv) * sizeof(Complex));
    
    // If we have memory availible, do the entire rotation
    if (batched_rotate <= 0 || batched_rotate >= nEv) {
      if ((int)kSpace.size() < offset + nEv) {
	ColorSpinorParam csParamClone(*kSpace[0]);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + nEv);
        kSpace.reserve(offset + nEv);
        for (int i = kSpace.size(); i < offset + nEv; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }
      
      // Pointers to the relevant vectors
      std::vector<ColorSpinorField *> vecs_ptr;
      std::vector<ColorSpinorField *> kSpace_ptr;

      // Alias the extra space vectors, zero the workspace
      vecs_ptr.reserve(nEv);
      for (int i = 0; i < nEv; i++) {
        kSpace_ptr.push_back(kSpace[offset + i]);
        blas::zero(*kSpace_ptr[i]);
      }

      // Alias the vectors we wish to keep, populate the Ritz matrix and transpose.
      kSpace_ptr.reserve(dim);
      for (int j = 0; j < dim; j++) {
        vecs_ptr.push_back(kSpace[j]);
        for (int i = 0; i < nEv; i++) { ritz_mat_keep[j * nEv + i] = block_ritz_mat[i * dim + j]; }
      }

      // multiBLAS caxpy
      blas::caxpy(ritz_mat_keep, vecs_ptr, kSpace_ptr);

      // Copy back to the Krylov space
      for (int i = 0; i < nEv; i++) std::swap(kSpace[i], kSpace[offset + i]);

      // Compute new r block
      // Use Eigen, it's neater
      MatrixXcd beta = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd ri = MatrixXcd::Zero(block_size, block_size);
      MatrixXcd ritzi = MatrixXcd::Zero(block_size, block_size);
      int blocks = nEv/block_size;
      int idx = 0;
      int beta_offset = ((nKr/block_size) - 1)*block_data_length;
      for(int b=0; b<block_size; b++) {
	for(int c=0; c<b+1; c++) {
	  idx = b*block_size + c;
	  beta(c,b) = block_beta[beta_offset + idx];
	}
      }
#ifdef DEBUG_LOCAL
      printfQuda("Current beta\n");
      for(int c=0; c<block_size; c++) {
	for(int b=0; b<block_size; b++) {
	  printfQuda("(%e,%e) ", beta(c,b).real(), beta(c,b).imag());
	}
	printfQuda("\n");
      }
#endif      
      for(int i=0; i<blocks; i++) {
	for(int b=0; b<block_size; b++) {
	  for(int c=0; c<block_size; c++) {
	    idx = b*dim + (dim-block_size) + c;
	    ritzi(c,b) = block_ritz_mat[idx];
	  }
	}
#ifdef DEBUG_LOCAL	
	printfQuda("Current RITZi\n");
	for(int c=0; c<block_size; c++) {
	  for(int b=0; b<block_size; b++) {	    
	    idx = c*block_size + b;
	    printfQuda("(%e,%e) ", ritzi(c,b).real(), ritzi(c,b).imag());
	  }
	  printfQuda("\n");
	}
#endif
	ri = beta * ritzi;
	for(int b=0; b<block_size; b++) {
	  for(int c=0; c<block_size; c++) {
	    idx = b*block_size + c;
	    block_beta[i*block_data_length + idx] = ri(c,b);
	  }
	}
      }
    } else {
      /*
      // Do batched rotation to save on memory
      int batch_size = batched_rotate;
      int full_batches = iter_keep / batch_size;
      int batch_size_r = iter_keep % batch_size;
      bool do_batch_remainder = (batch_size_r != 0 ? true : false);

      if ((int)kSpace.size() < offset + batch_size) {
      ColorSpinorParam csParamClone(*kSpace[0]);
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
      */
    }
      
    // Update residual vector
    std::swap(kSpace[num_locked + iter_keep], kSpace[nKr]);

    // Update sub arrow matrix
    //for (int i = 0; i < iter_keep; i++) beta[i + num_locked] = beta[nKr - 1] * block_ritz_mat[dim * (i + 1) - 1];

    host_free(ritz_mat_keep);

    // Save Krylov rotation tuning
    saveTuneCache();
  }
}
