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
#include <blas_quda_3d.h>
#include <util_quda.h>
#include <tune_quda.h>
#include <eigen_helper.h>

namespace quda
{
  // Thick Restarted Lanczos Method constructor
  TRLM3D::TRLM3D(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    EigenSolver(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    ortho_dim = eig_param->ortho_dim;
    ortho_dim_size = eig_param->ortho_dim_size;
    if(ortho_dim != 3) errorQuda("Only 3D spatial splitting (ortho_dim = 3) is supported, ortho_dim passed = %d", ortho_dim);
    
    // Tridiagonal/Arrow matrices
    alpha_3D = (double **)safe_malloc(ortho_dim_size * sizeof(double*));
    beta_3D = (double **)safe_malloc(ortho_dim_size * sizeof(double*));

    residua_3D.reserve(ortho_dim_size);
    ritz_mat_3D.resize(ortho_dim_size);
    converged_3D.resize(ortho_dim_size);
    active_3D.resize(ortho_dim_size);

    iter_locked_3D.resize(ortho_dim_size);
    iter_keep_3D.resize(ortho_dim_size);
    iter_converged_3D.resize(ortho_dim_size);
    
    num_locked_3D.resize(ortho_dim_size);
    num_keep_3D.resize(ortho_dim_size);
    num_converged_3D.resize(ortho_dim_size);
    
    for (int i = 0; i < ortho_dim_size; i++) {
      alpha_3D[i] = (double *)safe_malloc(n_kr * sizeof(double));
      beta_3D[i] = (double *)safe_malloc(n_kr * sizeof(double));
      residua_3D.push_back(std::vector<double>(n_kr, 0.0));
      converged_3D[i] = false;
      active_3D[i] = false;

      iter_locked_3D[i] = 0;
      iter_keep_3D[i] = 0;
      iter_converged_3D[i] = 0;
      
      num_locked_3D[i] = 0;
      num_keep_3D[i] = 0;
      num_converged_3D[i] = 0;
      
      for (int j = 0; j < n_kr; j++) {
	alpha_3D[i][j] = 0.0;
	beta_3D[i][j] = 0.0;
      }
    }
    
    // 3D thick restart specific checks
    if (n_kr < n_ev + 6) errorQuda("n_kr=%d must be greater than n_ev+6=%d\n", n_kr, n_ev + 6);

    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the TR Lanczos solver");
    }

    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void TRLM3D::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // In case we are deflating an operator, save the tunechache from the inverter
    saveTuneCache();
    
    // Override any user input for block size.
    block_size = 1;

    // For the 3D solver, we must ensure the eval array is the correct size
    evals.reserve(ortho_dim_size * comm_dim(3) * n_conv);
    for(int i=0; i<ortho_dim_size * comm_dim(3) * n_conv; i++)
      evals.push_back(0.0);
    
    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    if (getVerbosity() >= QUDA_VERBOSE) queryPrec(kSpace[0]->Precision());
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile3D(mat, kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess3D(kSpace, ortho_dim_size);
    
    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);
    
    // Check for Chebyshev maximum estimation
    checkChebyOpMax3D(mat, kSpace);

    // Convergence and locking criteria
    std::vector<double> mat_norm_3D(ortho_dim_size, 0.0);
    double epsilon = setEpsilon(kSpace[0]->Precision());
    
    // Print Eigensolver params
    ////printfQuda("Flag 1\n");
    printEigensolverSetup();
    ////printfQuda("Flag 2\n");
    //---------------------------------------------------------------------------

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      // Get min step

      int step_min = getArrayMinMax3D(num_locked_3D, n_kr, true);
      for (int step = step_min; step < n_kr; step++) {
	//printfQuda("Pre Step\n\n\n");
	lanczosStep3D(kSpace, step);
	//printfQuda("\n\nPost Step\n");
      }
      iter += (n_kr - step_min);
      
      // The eigenvalues are returned in the alpha array
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat3D();
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      // mat_norm is updated.
      for (int t = 0; t < ortho_dim_size; t++) {
	if(!converged_3D[t]) { 
	  for (int i = num_locked_3D[t]; i < n_kr; i++) {
	    if (fabs(alpha_3D[t][i]) > mat_norm_3D[t]) mat_norm_3D[t] = fabs(alpha_3D[t][i]);
	  }
	}
      }
      
      // Locking check
      for (int t = 0; t < ortho_dim_size; t++) {
	if(!converged_3D[t]) { 
	  iter_locked_3D[t] = 0;
	  for (int i = 1; i < (n_kr - num_locked_3D[t]); i++) {
	    if (residua_3D[t][i + num_locked_3D[t]] < epsilon * mat_norm_3D[t]) {
	      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
		printfQuda("**** Locking %d %d resid=%+.6e condition=%.6e ****\n", t, i, residua_3D[t][i + num_locked_3D[t]],
			   epsilon * mat_norm_3D[t]);
	      iter_locked_3D[t] = i;	    
	    } else {
	      // Unlikely to find new locked pairs	    
	      break;
	    }
	  }
	}
      }

      // Convergence check
      for (int t = 0; t < ortho_dim_size; t++) {
	if(!converged_3D[t]) { 
	  iter_converged_3D[t] = iter_locked_3D[t];
	  for (int i = iter_locked_3D[t] + 1; i < n_kr - num_locked_3D[t]; i++) {
	    if (residua_3D[t][i + num_locked_3D[t]] < tol * mat_norm_3D[t]) {
	      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
		printfQuda("**** Converged %d %d resid=%+.6e condition=%.6e ****\n", t, i, residua_3D[t][i + num_locked_3D[t]], tol * mat_norm_3D[t]);
	      iter_converged = i;
	    } else {
	      // Unlikely to find new converged pairs
	      break;
	    }
	  }
	}
      }

      for (int t = 0; t < ortho_dim_size; t++) {
	if(!converged_3D[t]) iter_keep_3D[t] = std::min(iter_converged_3D[t] + (n_kr - num_converged_3D[t]) / 2, n_kr - num_locked_3D[t] - 12);
      }
      
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      computeKeptRitz3D(kSpace);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      int t_offset = ortho_dim_size * comm_coord(3);      
      for (int t = 0; t < ortho_dim_size; t++) {
	if(!converged_3D[t]) { 
	  num_converged_3D[t] = num_locked_3D[t] + iter_converged_3D[t];
	  num_keep_3D[t] = num_locked_3D[t] + iter_keep_3D[t];
	  num_locked_3D[t] += iter_locked_3D[t];
	  
	  if (getVerbosity() >= QUDA_VERBOSE && comm_coord(0) == 0 && comm_coord(1) == 0 && comm_coord(2) == 0) {
	    printf("%04d converged eigenvalues for timeslice %d at restart iter %04d\n", num_converged_3D[t], t_offset + t, restart_iter + 1);
	    printf("iter Conv[%d] = %d\n", t_offset + t, iter_converged_3D[t]);
	    printf("iter Keep[%d] = %d\n", t_offset + t, iter_keep_3D[t]);
	    printf("iter Lock[%d] = %d\n", t_offset + t, iter_locked_3D[t]);
	    printf("num_converged[%d] = %d\n", t_offset + t, num_converged_3D[t]);
	    printf("num_keep[%d] = %d\n", t_offset + t, num_keep_3D[t]);
	    printf("num_locked[%d] = %d\n", t_offset + t, num_locked_3D[t]);
	    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	      for (int i = 0; i < n_kr; i++) {
		printf("Ritz[%d][%d] = %.16e residual[%d] = %.16e\n", t_offset + t, i, alpha_3D[t][i], i, residua_3D[t][i]);
	      }
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
      
      if(all_converged) {
	reorder3D(kSpace);
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
                  n_conv, n_ev, n_kr, max_restarts);
      } else {
        warningQuda("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                    "restart steps. Continuing with current lanczos factorisation.",
                    n_conv, n_ev, n_kr, max_restarts);
	// Compute eigenvalues
	computeEvals3D(mat, kSpace, evals);
      }
    } else {
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", n_conv,
                   restart_iter, iter);

        // Dump all Ritz values and residua if using Chebyshev
	if(eig_param->use_poly_acc) {
	  for (int t = 0; t < ortho_dim_size; t++) {
	    for (int i = 0; i < n_conv; i++) {
	      printfQuda("RitzValue[%d][%04d]: (%+.16e, %+.16e) residual %.16e\n", t, i, alpha_3D[t][i], 0.0, residua_3D[t][i]);
	    }
	  }
	}
      }
      
      // Compute eigenvalues
      computeEvals3D(mat, kSpace, evals);
    }
    
    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Destructor
  TRLM3D::~TRLM3D()
  {
    for(int i=0; i<ortho_dim_size; i++) {
      ritz_mat_3D[i].clear();
      ritz_mat_3D[i].shrink_to_fit();
      host_free(alpha_3D[i]);
      host_free(beta_3D[i]);
    }
    ritz_mat_3D.clear();
    ritz_mat_3D.shrink_to_fit();
    host_free(alpha_3D);
    host_free(beta_3D);   
  }

  // Thick Restart 3D Member functions
  //---------------------------------------------------------------------------
  void TRLM3D::lanczosStep3D(std::vector<ColorSpinorField *> v, int j)
  {
    // Compute r[t] = A[t] * v_j[t] - b_{j-i}[t] * v_{j-1}[t]
    // r[t] = A[t] * v_j[t]

    // Use this while we have only axpby in place of axpy;
    std::vector<double> unit(ortho_dim_size, 1.0);
    std::vector<double> alpha_j(ortho_dim_size, 0.0);
    std::vector<double> beta_j(ortho_dim_size, 0.0);

    // Clone a 4D workspace vector
    std::vector<ColorSpinorField *> workspace;
    ColorSpinorParam csParamClone(*v[0]);
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    workspace.push_back(ColorSpinorField::Create(csParamClone));    

    // 3D vectors that hold data for individual t components
    std::vector<ColorSpinorField *> vecs_t;
    std::vector<ColorSpinorField *> r_t;
    csParamClone.change_dim(ortho_dim, 1);
    vecs_t.push_back(ColorSpinorField::Create(csParamClone));
    r_t.push_back(ColorSpinorField::Create(csParamClone));    
    
    // Identify active 3D slices
    for(int t=0; t<ortho_dim_size; t++) {
      // Every element of the active array must be assessed
      active_3D[t] = (num_keep_3D[t] <= j && !converged_3D[t] ? true : false);
      // Copy the relevant slice to the workspace array if active
      if(active_3D[t]) {
	//printfQuda(" j = %d, slice %d is active\n", j, t);
	blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[0], *v[j]);	    
	blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[0], *workspace[0]);
      }
    }

    //printfQuda("Flag LS3D pre\n");

    // This will be a blocked operator with no
    // connections in the ortho_dim (usually t)
    // hence the 3D sections of each vector
    // will be independent.
    chebyOp(mat, *r[0], *workspace[0]);

    //printfQuda("Flag LS3D 0\n");
    
    // a_j[t] = v_j^dag[t] * r[t]    
    blas3d::reDotProduct(alpha_j, *workspace[0], *r[0]);
    for(int t=0; t<ortho_dim_size; t++) {
      // Only active problem data is recorded
      if(active_3D[t]) alpha_3D[t][j] = alpha_j[t];
    }

    //printfQuda("Flag LS3D 1\n");
    
    // r[t] = r[t] - a_j[t] * v_j[t]
    for(int t=0; t<ortho_dim_size; t++) {
      alpha_j[t] *= active_3D[t] ? -1.0 : 0.0;
    }
    blas3d::axpby(alpha_j, *workspace[0], unit, *r[0]);

    //printfQuda("Flag LS3D 2\n");
    
    // r[t] = r[t] - b_{j-1}[t] * v_{j-1}[t]
    // We do this problem by problem so that we can use the multiblas axpy
    // on 3D arrays. Only orthogonalise active problems
    
    for(int t=0; t<ortho_dim_size; t++) {
      if(active_3D[t]) {
	int start = (j > num_keep_3D[t]) ? j - 1 : 0;    
	if (j - start > 0) {
	  
	  // Ensure we have enough 3D vectors
	  if((int)vecs_t.size() < j - start) {
	    vecs_t.reserve(j - start);
	    for (int i = (int)vecs_t.size(); i < j - start; i++) {
	      vecs_t.push_back(ColorSpinorField::Create(csParamClone));
	    }
	  }
	  
	  // Copy the 3D data into the 3D vectors, create beta array, and create
	  // pointers to the 3D vectors
	  std::vector<ColorSpinorField *> vecs_t_ptr;
	  std::vector<double> beta_t;
	  vecs_t_ptr.reserve(j - start);
	  beta_t.reserve(j - start);
	  // Copy residual
	  blas3d::copy(t, blas3d::COPY_TO_3D, *r_t[0], *r[0]);
	  // Copy vectors
	  for (int i = start; i < j; i++) {
	    blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[i - start], *v[i]);
	    vecs_t_ptr.push_back(vecs_t[i - start]);
	    beta_t.push_back(-beta_3D[t][i]);
	  }
	  
	  // r[t] = r[t] - beta[t]{j-1} * v[t]{j-1}
	  blas::axpy(beta_t.data(), vecs_t_ptr, r_t);
	  
	  // Save Lanczos step tuning
	  saveTuneCache();
	  
	  // Copy residual back to 4D vector
	  blas3d::copy(t, blas3d::COPY_FROM_3D, *r_t[0], *r[0]);
	}
      }
    }
    
    //printfQuda("Flag LS3D 3\n");
    
    // Orthogonalise r against the Krylov space
    for (int k = 0; k < 1; k++) blockOrthogonalize3D(v, r, j + 1);

    //printfQuda("Flag LS3D 3.1\n");
    
    // b_j[t] = ||r[t]||
    blas3d::reDotProduct(beta_j, *r[0], *r[0]);
    for(int t=0; t<ortho_dim_size; t++) beta_j[t] = sqrt(beta_j[t]);

    //printfQuda("Flag LS3D 3.2\n");
    
    // Prepare next step.
    // v_{j+1}[t] = r[t] / b_{j}[t]
    blas::zero(*workspace[0]);    
    for(int t=0; t<ortho_dim_size; t++) {
      if(active_3D[t]) {
	beta_3D[t][j] = beta_j[t];
	beta_j[t] = 1.0/beta_j[t];
      }
    }

    //printfQuda("Flag LS3D 3.3\n");
    blas3d::axpby(beta_j, *r[0], unit, *workspace[0]);
    //printfQuda("Flag LS3D 3.4\n");
    
    // Copy data from workspace into the relevant 3D slice of the kSpace
    for(int t=0; t<ortho_dim_size; t++) {
      if(active_3D[t]) {
	blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[0], *workspace[0]);
	blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[0], *v[j+1]);
      }
    }

    //printfQuda("Flag LS3D 3.5\n");
    
    delete workspace[0];
    delete r_t[0];
    for(unsigned int i=0; i<vecs_t.size(); i++) delete vecs_t[i];
    //printfQuda("Flag LS3D 4\n");
    
    // Save Lanczos step tuning
    saveTuneCache();
  }
  
  void TRLM3D::reorder3D(std::vector<ColorSpinorField *> &kSpace)
  {
    std::vector<ColorSpinorField *> vecs_t;
    vecs_t.reserve(2);    
    ColorSpinorParam csParamClone(*kSpace[0]);
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    csParamClone.change_dim(ortho_dim, 1);
    
    for(int i=0; i<2; i++) vecs_t.push_back(ColorSpinorField::Create(csParamClone));    
    for(int t=0; t<ortho_dim_size; t++) {      
      int i = 0;
      if (reverse) {
	while (i < n_kr) {
	  if ((i == 0) || (alpha_3D[t][i - 1] >= alpha_3D[t][i]))
	    i++;
	  else {
	    std::swap(alpha_3D[t][i], alpha_3D[t][i - 1]);
	    blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[0], *kSpace[i]);	    
	    blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[1], *kSpace[i-1]);
	    blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[0], *kSpace[i-1]);	    
	    blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[1], *kSpace[i]);
	    i--;
	  }
	}
      } else {
	while (i < n_kr) {
	  if ((i == 0) || (alpha_3D[t][i - 1] <= alpha_3D[t][i]))
	    i++;
	  else {
	    std::swap(alpha_3D[t][i], alpha_3D[t][i - 1]);
	    blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[0], *kSpace[i]);	    
	    blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[1], *kSpace[i-1]);
	    blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[0], *kSpace[i-1]);	    
	    blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[1], *kSpace[i]); 
	    i--;
	  }
	}
      }
    }
    delete vecs_t[0];
    delete vecs_t[1];
  }
  
  void TRLM3D::eigensolveFromArrowMat3D()
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    
    // Loop over the 3D problems
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int t=0; t<ortho_dim_size; t++) {
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
    
    profile.TPSTOP(QUDA_PROFILE_EIGEN);    
  }
  
  void TRLM3D::computeKeptRitz3D(std::vector<ColorSpinorField *> &kSpace)
  {
    // Multi-BLAS friendly array to store part of Ritz matrix we want
    double **ritz_mat_keep = (double **)safe_malloc((ortho_dim_size) * sizeof(double));

    std::vector<ColorSpinorField *> vecs_t;
    std::vector<ColorSpinorField *> kSpace_t;

    for(int t=0; t<ortho_dim_size; t++) {
      if(active_3D[t]) {
	int dim = n_kr - num_locked_3D[t];
	int keep = iter_keep_3D[t];
	
	ritz_mat_keep[t] = (double *)safe_malloc((dim * keep) * sizeof(double));
	for (int j = 0; j < dim; j++) {
	  for (int i = 0; i < keep; i++) { ritz_mat_keep[t][j * keep + i] = ritz_mat_3D[t][i * dim + j]; }
	}
	
	// Pointers to the relevant vectors
	std::vector<ColorSpinorField *> vecs_ptr;
	
	// Alias the vectors we wish to keep.
	vecs_ptr.reserve(dim);
	for (int j = 0; j < dim; j++) vecs_ptr.push_back(kSpace[num_locked_3D[t] + j]);
	
	// multiBLAS axpy. Create 3D vectors so that we may perform all t independent
	// vector rotations.
	profile.TPSTART(QUDA_PROFILE_COMPUTE);
	
	vecs_t.reserve(dim);
	kSpace_t.reserve(keep);
	
	ColorSpinorParam csParamClone(*kSpace[0]);
	csParamClone.create = QUDA_ZERO_FIELD_CREATE;
	csParamClone.change_dim(ortho_dim, 1);
	
	// Create 3D arrays
	for(int i=vecs_t.size(); i<dim; i++) vecs_t.push_back(ColorSpinorField::Create(csParamClone));    
	for(int i=kSpace_t.size(); i<keep; i++) kSpace_t.push_back(ColorSpinorField::Create(csParamClone));
	
	// Copy to data to 3D array, zero out workspace, make pointers
	std::vector<ColorSpinorField *> vecs_t_ptr;
	std::vector<ColorSpinorField *> kSpace_t_ptr;
	for(int i=0; i<dim; i++) {
	  blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[i], *vecs_ptr[i]);
	  vecs_t_ptr.push_back(vecs_t[i]);
	}
	for(int i=0; i<keep; i++) {
	  blas::zero(*kSpace_t[i]);
	  kSpace_t_ptr.push_back(kSpace_t[i]);
	}
	
	// Compute the axpy      
	blas::axpy(ritz_mat_keep[t], vecs_t_ptr, kSpace_t_ptr);
	
	// Save rotation tuning
	saveTuneCache();      
	
	// Copy back to the 4D workspace array
	profile.TPSTOP(QUDA_PROFILE_COMPUTE);
	
	// Copy compressed Krylov
	for (int i = 0; i < keep; i++) {
	  blas3d::copy(t, blas3d::COPY_FROM_3D, *kSpace_t[i], *kSpace[num_locked_3D[t] + i]);
	}
	
	// Update residual vector
	blas3d::copy(t, blas3d::COPY_TO_3D, *vecs_t[0], *kSpace[n_kr]);
	blas3d::copy(t, blas3d::COPY_FROM_3D, *vecs_t[0], *kSpace[num_locked_3D[t] + keep]);
	
	// Update sub arrow matrix
	for (int i = 0; i < keep; i++) beta_3D[t][i + num_locked_3D[t]] = beta_3D[t][n_kr - 1] * ritz_mat_3D[t][dim * (i + 1) - 1];
	host_free(ritz_mat_keep[t]);
      }
    }
    
    // Delete all 3D vectors
    for(unsigned int i=0; i<vecs_t.size(); i++) delete vecs_t[i]; 
    for(unsigned int i=0; i<kSpace_t.size(); i++) delete kSpace_t[i];
    
    host_free(ritz_mat_keep);
  }

  // Orthogonalise r[t][0:] against V_[t][0:j]
  void TRLM3D::blockOrthogonalize3D(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> &rvecs, int j)
  {
    int vec_size = j;
    
    for (int i = 0; i < vec_size; i++) {

      Complex unit(1.0,0.0);
      std::vector<Complex> s_t(ortho_dim_size, 0.0);
      std::vector<double> r_t(ortho_dim_size, 0.0);
      std::vector<Complex> unit_new(ortho_dim_size, unit);
      
      std::vector<ColorSpinorField *> vec_ptr;
      vec_ptr.push_back(vecs[i]);

      // Block dot products stored in s_t.
      //printfQuda("*vec_ptr[%d] = %e *rvecs[%d] = %e\n", i, blas::norm2(*vec_ptr[0]), i, blas::norm2(*rvecs[0]));
      blas3d::cDotProduct(s_t, *vec_ptr[0], *rvecs[0]);
      blas3d::reDotProduct(r_t, *vec_ptr[0], *rvecs[0]);
      for(int t=0; t<ortho_dim_size; t++) {
	s_t[t] *= active_3D[t] ? -1.0 : 0.0;
	//printfQuda("Ortho active at %d = %s: S = (%e,%e)\n", t, active_3D[t] ? "T" : "F", s_t[t].real(), s_t[t].imag());
	//printfQuda("Ortho active at %d = %s: R =  %e\n", t, active_3D[t] ? "T" : "F", r_t[t]);
      }
      
      // Block orthogonalise
      blas3d::caxpby(s_t, *vec_ptr[0], unit_new, *rvecs[0]);
    }
    
    // Save orthonormalisation tuning
    saveTuneCache();      
  }

  void TRLM3D::prepareInitialGuess3D(std::vector<ColorSpinorField *> &kSpace, int ortho_dim_size)
  {
    if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
      if (sqrt(blas::norm2(*kSpace[0])) == 0.0) { kSpace[0]->Source(QUDA_RANDOM_SOURCE); }
    } else {
      RNG *rng = new RNG(*kSpace[0], 1234);
      if (sqrt(blas::norm2(*kSpace[0])) == 0.0) { spinorNoise(*kSpace[0], *rng, QUDA_NOISE_UNIFORM); }
      delete rng;
    }
    
    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<double> zeros(ortho_dim_size, 0.0);
    blas3d::reDotProduct(norms, *kSpace[0], *kSpace[0]);
    for(int t=0; t<ortho_dim_size; t++) {
      norms[t] = 1.0/sqrt(norms[t]);
    }
    
    blas3d::axpby(norms, *kSpace[0], zeros, *kSpace[0]);
  }
  
  double TRLM3D::estimateChebyOpMax3D(const DiracMatrix &mat, ColorSpinorField &out, ColorSpinorField &in)
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
    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<double> zeros(ortho_dim_size, 0.0);    
    for (int i = 0; i < 100; i++) {
      if ((i + 1) % 10 == 0) {
	blas3d::reDotProduct(norms, *in_ptr, *in_ptr);
	for(int t=0; t<ortho_dim_size; t++) norms[t] = 1.0/sqrt(norms[t]);
	blas3d::axpby(norms, *in_ptr, zeros, *in_ptr);
      }
      matVec(mat, *out_ptr, *in_ptr);
      std::swap(out_ptr, in_ptr);
    }

    // Compute spectral radius estimate
    std::vector<double> inner_products(ortho_dim_size, 0.0);
    blas3d::reDotProduct(inner_products, *out_ptr, *in_ptr);
    
    double result = 1.0;
    std::vector<double> all_results(comm_dim(ortho_dim) * ortho_dim_size, 0.0);
    for(int t=0; t<ortho_dim_size; t++) {
      all_results[comm_coord(ortho_dim) * ortho_dim_size + t] = inner_products[t];
    }
    
    comm_allreduce_array((double *)all_results.data(), comm_coord(ortho_dim) * ortho_dim_size);
    
    int spatial_comm_vol = 1;
    for(int i=0; i<4; i++)
      if(i != ortho_dim) spatial_comm_vol *= comm_dim(i);
    
    if (getVerbosity() >= QUDA_VERBOSE) {      
      for(int t=0; t<ortho_dim_size * comm_dim(ortho_dim); t++) {
	// scale out the redundant summations
	all_results[t] /= spatial_comm_vol;
	
	printf("Chebyshev max at slice %d = %e\n", t, all_results[t]);
	if(all_results[t] > result) result = all_results[t];
      }
    }
    
    // Save Chebyshev Max tuning
    saveTuneCache();
    
    // Increase final result by 10% for safety
    return result * 1.10;
  }

  void TRLM3D::checkChebyOpMax3D(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace)
  {
    if (eig_param->use_poly_acc && eig_param->a_max <= 0.0) {
      // Use part of the kSpace as temps
      eig_param->a_max = estimateChebyOpMax3D(mat, *kSpace[block_size + 2], *kSpace[block_size + 1]);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Chebyshev maximum estimate: %e\n", eig_param->a_max);
    }
  }

  void TRLM3D::computeEvals3D(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs,
			      std::vector<Complex> &evals, int size)
  {
    if (size > (int)evecs.size())
      errorQuda("Requesting %d eigenvectors with only storage allocated for %lu", size, evecs.size());
    if (size > (int)evals.size())
      errorQuda("Requesting %d eigenvalues with only storage allocated for %lu", size, evals.size());
    
    ColorSpinorParam csParamClone(*evecs[0]);
    std::vector<ColorSpinorField *> temp;
    temp.push_back(ColorSpinorField::Create(csParamClone));

    std::vector<std::vector<Complex>> evals_t(size, std::vector<Complex>(ortho_dim_size, 0.0));
    std::vector<double> norms(ortho_dim_size, 0.0);
    std::vector<Complex> unit(ortho_dim_size, 1.0);
    std::vector<Complex> n_unit(ortho_dim_size, {-1.0, 0.0});
    
    for (int i = 0; i < size; i++) {
      // r = A * v_i
      matVec(mat, *temp[0], *evecs[i]);

      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      // Block dot products stored in s_t.
      blas3d::cDotProduct(evals_t[i], *evecs[i], *temp[0]);
      blas3d::reDotProduct(norms, *evecs[i], *evecs[i]);
      for(int t=0; t<ortho_dim_size; t++) evals_t[i][t] /= sqrt(norms[t]);      
      
      // Measure ||lambda_i*v_i - A*v_i||
      blas3d::caxpby(evals_t[i], *evecs[i], n_unit, *temp[0]);
      blas3d::reDotProduct(norms, *temp[0], *temp[0]);
      for(int t=0; t<ortho_dim_size; t++) residua_3D[t][i] = sqrt(norms[t]);
    }

    // If size = n_conv, this routine is called post sort
    if (size == n_conv) {
      // We are computing T problems split across T nodes, so we must do an MPI gather
      // to display all the data to stdout.
      int t_size = ortho_dim_size * comm_dim(3);
      std::vector<Complex> evals_t_all(size * t_size, 0.0);      
      std::vector<double> resid_t_all(size * t_size, 0.0);
      for(int t=0; t<ortho_dim_size; t++)
	for (int i = 0; i < size; i++) {
	  evals_t_all[(comm_coord(3) * ortho_dim_size + t)*size + i] = evals_t[i][t];
	  resid_t_all[(comm_coord(3) * ortho_dim_size + t)*size + i] = residua_3D[t][i];
	}
      
      comm_allreduce_array((double *)&evals_t_all[0], 2 * t_size * size);
      comm_allreduce_array((double *)&resid_t_all[0], t_size * size);

      int spatial_comm_vol = 1;
      for(int i=0; i<4; i++)
	if(i != ortho_dim) spatial_comm_vol *= comm_dim(i); 
      
      for(int t=0; t<t_size; t++)
	for (int i = 0; i < size; i++) {
	  
	  // scale out the redundant summations
	  evals_t_all[t*size + i] /= spatial_comm_vol;
	  resid_t_all[t*size + i] /= spatial_comm_vol;
	  
	  if(getVerbosity() >= QUDA_SUMMARIZE) { 
	    printfQuda("Eval[%02d][%04d] = (%+.16e,%+.16e) residual = %+.16e\n", t, i,
		       evals_t_all[t*size + i].real(), evals_t_all[t*size + i].imag(),
		       resid_t_all[t*size + i]);
	  }
	  
	  // Transfer evals to eval array
	  evals.resize(size * evecs[0]->X()[3]);
	  evals[t*size + i] = evals_t_all[t*size + i];
	}
    }
  
    delete temp[0];
    
    // Save Eval tuning
    saveTuneCache();
  }

  void TRLM3D::loadFromFile3D(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace,
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
      VectorIO io(eig_param->vec_infile, eig_param->io_parity_inflate == QUDA_BOOLEAN_TRUE);
      io.load(vecs_ptr);
    }

    // Create the device side residual vector by cloning
    // the kSpace passed to the function.
    ColorSpinorParam csParam(*kSpace[0]);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    r.push_back(ColorSpinorField::Create(csParam));

    // Error estimates (residua) given by ||A*vec - lambda*vec||
    computeEvals3D(mat, kSpace, evals);
    delete r[0];
  }

  int TRLM3D::getArrayMinMax3D(const std::vector<int> &array, const int limit, const bool min)
  {
    int ret_val = limit;
    int spatial_comm_vol = 1;
    int t_size = comm_dim(ortho_dim) * ortho_dim_size;
    for(int i=0; i<4; i++)
      if(i != ortho_dim) spatial_comm_vol *= comm_dim(i); 
    
    std::vector<double> all_array(t_size, 0);
    for(int t=0; t<ortho_dim_size; t++) all_array[comm_coord(ortho_dim) * ortho_dim_size + t] = array[t];
    
    //printfQuda("Flag cara 1\n");
    comm_allreduce_array((double *)all_array.data(), t_size);
    //printfQuda("Flag cara 2\n");
    
    for(int t=0; t<t_size; t++) {
      // scale out the redundant summations
      all_array[t] /= spatial_comm_vol;
      if(all_array[t] < ret_val &&  min) ret_val = all_array[t];
      if(all_array[t] > ret_val && !min) ret_val = all_array[t];
    }
    //printfQuda("Flag cara 3\n");
    return ret_val;
  }
  
} // namespace quda
