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
    
    // Tridiagonal/Arrow matrices
    alpha_3D = (double **)safe_malloc(ortho_dim_size * sizeof(double*));
    beta_3D = (double **)safe_malloc(ortho_dim_size * sizeof(double*));
    residua_3D.reserve(ortho_dim_size);
    ritz_mat_3D.resize(ortho_dim_size);
    for (int i = 0; i < ortho_dim_size; i++) {
      alpha_3D[i] = (double *)safe_malloc(n_kr * sizeof(double));
      beta_3D[i] = (double *)safe_malloc(n_kr * sizeof(double));
      residua_3D.push_back(std::vector<double>(n_kr, 0.0));
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

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    if (getVerbosity() >= QUDA_VERBOSE) queryPrec(kSpace[0]->Precision());
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess3D(kSpace, ortho_dim, ortho_dim_size);
        
    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);

    // Check for Chebyshev maximum estimation
    checkChebyOpMax3D(mat, kSpace);

    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = setEpsilon(kSpace[0]->Precision());

    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin TRLM Eigensolver computation
    //---------------------------------------------------------------------------
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {
      
      for (int step = num_keep; step < n_kr; step++) lanczosStep3D(kSpace, step);
      iter += (n_kr - num_keep);

      // The eigenvalues are returned in the alpha array
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      eigensolveFromArrowMat3D();
      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      // mat_norm is updated. Find the smallest mat norm accross t, the largest
      // within each t
      double mat_norm_floor = 1e10;
      for (int t = 0; t < ortho_dim_size; t++) {
	for (int i = num_locked; i < n_kr; i++) {
	  if (fabs(alpha_3D[t][i]) > mat_norm) mat_norm = fabs(alpha_3D[t][i]);
	}
	if(mat_norm < mat_norm_floor) mat_norm_floor = mat_norm;
      }
      mat_norm = mat_norm_floor;
      
      // Locking check
      iter_locked = 0;
      for (int t = 0; t < ortho_dim_size; t++) {
	for (int i = 1; i < (n_kr - num_locked); i++) {
	  if (residua_3D[t][i + num_locked] < epsilon * mat_norm) {
	    if (getVerbosity() >= QUDA_VERBOSE)
	      printfQuda("**** Locking %d %d resid=%+.6e condition=%.6e ****\n", t, i, residua_3D[t][i + num_locked],
			 epsilon * mat_norm);
	    iter_locked = i;
	  } else {
	    // Unlikely to find new locked pairs
	    break;
	  }
	}
      }

      // Convergence check
      iter_converged = iter_locked;
      for (int t = 0; t < ortho_dim_size; t++) {
	for (int i = iter_locked + 1; i < n_kr - num_locked; i++) {
	  if (residua_3D[t][i + num_locked] < tol * mat_norm) {
	    if (getVerbosity() >= QUDA_VERBOSE)
	      printfQuda("**** Converged %d %d resid=%+.6e condition=%.6e ****\n", t, i, residua_3D[t][i + num_locked], tol * mat_norm);
	    iter_converged = i;
	  } else {
	    // Unlikely to find new converged pairs
	    break;
	  }
	}
      }

      iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - num_locked - 12);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      computeKeptRitz3D(kSpace);
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
	for (int t = 0; t < ortho_dim_size; t++) {
	  for (int i = 0; i < n_kr; i++) {
	    //printfQuda("Ritz[%d][%d] = %.16e residual[%d] = %.16e\n", t, i, alpha_3D[t][i], i, residua_3D[t][i]);
	  }
	}
      }

      // Check for convergence
      if (num_converged >= n_conv) {
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
      }
    } else {
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", n_conv,
                   restart_iter, iter);

        // Dump all Ritz values and residua if using Chebyshev
	for (int t = 0; t < ortho_dim_size; t++) {
	  for (int i = 0; i < n_conv && eig_param->use_poly_acc; i++) {
	    printfQuda("RitzValue[%d][%04d]: (%+.16e, %+.16e) residual %.16e\n", t, i, alpha_3D[t][i], 0.0, residua_3D[t][i]);
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

  // Thick Restart Member functions
  //---------------------------------------------------------------------------
  void TRLM3D::lanczosStep3D(std::vector<ColorSpinorField *> v, int j)
  {
    // Compute r[t] = A[t] * v_j[t] - b_{j-i}[t] * v_{j-1}[t]
    // r[t] = A[t] * v_j[t]

    // Use this while we have only axpby in place of axpy;
    std::vector<double> unit(ortho_dim_size, 1.0);
    std::vector<double> alpha_j(ortho_dim_size, 0.0);
    std::vector<double> beta_j(ortho_dim_size, 0.0);
    
    // DMH: OK for 3D
    chebyOp(mat, *r[0], *v[j]);

    // a_j[t] = v_j^dag[t] * r[t]    
    blas3d::reDotProduct(ortho_dim, alpha_j, *v[j], *r[0]);
    for(int t=0; t<ortho_dim_size; t++) {
      alpha_3D[t][j] = alpha_j[t];
      //printfQuda("alpha_%d[%d] = %e\n", j, t, alpha_j[t]);
    }
    
    // r[t] = r[t] - a_j[t] * v_j[t]
    for(int t=0; t<ortho_dim_size; t++) alpha_j[t] *= -1.0;    
    blas3d::axpby(ortho_dim, alpha_j, *v[j], unit, *r[0]);
        
    int start = (j > num_keep) ? j - 1 : 0;    
    if (j - start > 0) {
      std::vector<ColorSpinorField *> r_ {r[0]};
      for (int i = start; i < j; i++) {
	std::vector<ColorSpinorField *> v_;
	v_.push_back(v[i]);
        for(int t=0; t<ortho_dim_size; t++) beta_j[t] = beta_3D[t][i];
	
	// Perform each jth vector at a time
	// r[t] = r[t] - b_{j-1}[t] * v_{j-1}[t]
	// DMH this beta should be all the beta t values
	// for this i=start...j-1 vector
	blas3d::axpby(3, beta_j, *v_[0], unit, *r[0]);
      }
    }
    
    // Orthogonalise r against the Krylov space
    // DMH MG routine for this?
    for (int k = 0; k < 1; k++) blockOrthogonalize3D(v, r, j + 1);

    // b_j[t] = ||r[t]||
    // beta[j] = sqrt(blas::norm2(*r[0]));
    blas3d::reDotProduct(ortho_dim, beta_j, *r[0], *r[0]);
    for(int t=0; t<ortho_dim_size; t++) beta_j[t] = sqrt(beta_j[t]);
    
    // Prepare next step.
    // v_{j+1}[t] = r[t] / b_{j}[t]
    // // DMH: OK for 3D
    blas::zero(*v[j + 1]);    
    for(int t=0; t<ortho_dim_size; t++) {
      beta_3D[t][j] = beta_j[t];
      beta_j[t] = 1.0/beta_j[t];
    }
    blas3d::axpby(ortho_dim, beta_j, *r[0], unit, *v[j + 1]);
    
    // Save Lanczos step tuning
    saveTuneCache();
  }
  
  void TRLM3D::reorder3D(std::vector<ColorSpinorField *> &kSpace)
  {
  }

  void TRLM3D::eigensolveFromArrowMat3D()
  {
    profile.TPSTART(QUDA_PROFILE_EIGEN);
    // The same for all 3D problems
    int dim = n_kr - num_locked;
    int arrow_pos = num_keep - num_locked;

    // Loop over the 3D problems
    for(int t=0; t<ortho_dim_size; t++) {
      // Eigen objects
      MatrixXd A = MatrixXd::Zero(dim, dim);
      ritz_mat_3D[t].resize(dim * dim);
      for (int i = 0; i < dim * dim; i++) ritz_mat_3D[t][i] = 0.0;
      
      // Invert the spectrum due to chebyshev
      if (reverse) {
	for (int i = num_locked; i < n_kr - 1; i++) {
	  alpha_3D[t][i] *= -1.0;
	  beta_3D[t][i] *= -1.0;
	}
	alpha_3D[t][n_kr - 1] *= -1.0;
      }
      
      // Construct arrow mat A_{dim,dim}
      for (int i = 0; i < dim; i++) {
	
	// alpha_3D populates the diagonal
	A(i, i) = alpha_3D[t][i + num_locked];
      }
      
      for (int i = 0; i < arrow_pos; i++) {
	
	// beta_3D populates the arrow
	A(i, arrow_pos) = beta_3D[t][i + num_locked];
	A(arrow_pos, i) = beta_3D[t][i + num_locked];
      }
      
      for (int i = arrow_pos; i < dim - 1; i++) {
	
	// beta_3D populates the sub-diagonal
	A(i, i + 1) = beta_3D[t][i + num_locked];
	A(i + 1, i) = beta_3D[t][i + num_locked];
      }

      // Eigensolve the arrow matrix
      SelfAdjointEigenSolver<MatrixXd> eigensolver;
      eigensolver.compute(A);
      
      // repopulate ritz matrix
      for (int i = 0; i < dim; i++)
	for (int j = 0; j < dim; j++) ritz_mat_3D[t][dim * i + j] = eigensolver.eigenvectors().col(i)[j];
      
      for (int i = 0; i < dim; i++) {
	residua_3D[t][i + num_locked] = fabs(beta_3D[t][n_kr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
	//printfQuda("res[%d][%d] = %e\n", t, i, residua_3D[t][i + num_locked]);
	// Update the alpha_3D array
	alpha_3D[t][i + num_locked] = eigensolver.eigenvalues()[i];
      }
      
      // Put spectrum back in order
      if (reverse) {
	for (int i = num_locked; i < n_kr; i++) { alpha_3D[t][i] *= -1.0; }
      }     
    }
    profile.TPSTOP(QUDA_PROFILE_EIGEN);    
  }
  
  void TRLM3D::computeKeptRitz3D(std::vector<ColorSpinorField *> &kSpace)
  {
    int offset = n_kr + 1;
    int dim = n_kr - num_locked;

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    double **ritz_mat_keep = (double **)safe_malloc((ortho_dim_size) * sizeof(double));
    for(int t=0; t<ortho_dim_size; t++) {      
      ritz_mat_keep[t] = (double *)safe_malloc((dim * iter_keep) * sizeof(double));
      for (int j = 0; j < dim; j++) {
	for (int i = 0; i < iter_keep; i++) { ritz_mat_keep[t][j * iter_keep + i] = ritz_mat_3D[t][i * dim + j]; }
      }
    }

    if ((int)kSpace.size() < offset + iter_keep) {
      ColorSpinorParam csParamClone(*kSpace[0]);
      csParamClone.create = QUDA_ZERO_FIELD_CREATE;
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", offset + iter_keep);
      kSpace.reserve(offset + iter_keep);
      for (int i = kSpace.size(); i < offset + iter_keep; i++) {
	kSpace.push_back(ColorSpinorField::Create(csParamClone));
      }
    }

    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;

    // Alias the extra space vectors, zero the workspace
    kSpace_ptr.reserve(iter_keep);
    for (int i = 0; i < iter_keep; i++) {
      kSpace_ptr.push_back(kSpace[offset + i]);
      blas::zero(*kSpace_ptr[i]);
    }
    
    // Alias the vectors we wish to keep.
    vecs_ptr.reserve(dim);
    for (int j = 0; j < dim; j++) vecs_ptr.push_back(kSpace[num_locked + j]);
    
    // multiBLAS axpy
    //profile.TPSTART(QUDA_PROFILE_COMPUTE);
    std::vector<ColorSpinorField *> vecs_ptr_t(vecs_ptr.size());
    std::vector<ColorSpinorField *> kSpace_ptr_t(kSpace_ptr.size());
    ColorSpinorParam csParamClone(*kSpace[0]);
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    //csParamClone.change_dim(ortho_dim, 1);
    //for(unsigned int i=0; i<vecs_ptr.size(); i++) vecs_ptr_t.push_back(ColorSpinorField::Create(csParamClone));
    //for(unsigned int i=0; i<kSpace_ptr.size(); i++) kSpace_ptr_t.push_back(ColorSpinorField::Create(csParamClone));
    
    for(int t=0; t<ortho_dim_size; t++) {
      //for(unsigned int i=0; i<vecs_ptr.size(); i++) blas3d::copy(ortho_dim, t, *vecs_ptr_t[i], *vecs_ptr[i]);
      //for(unsigned int i=0; i<kSpace_ptr.size(); i++) blas3d::copy(ortho_dim, t, *kSpace_ptr_t[i], *kSpace_ptr[i]);
      //blas::axpy(ritz_mat_keep[t], vecs_ptr_t, kSpace_ptr_t);
      //blas::axpy(ritz_mat_keep[t], vecs_ptr, kSpace_ptr);
    }
    blas::axpy(ritz_mat_keep[0], vecs_ptr, kSpace_ptr);
    //profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    
    // Copy compressed Krylov
    for (int i = 0; i < iter_keep; i++) std::swap(kSpace[num_locked + i], kSpace[offset + i]);
    
    // Update residual vector
    std::swap(kSpace[num_locked + iter_keep], kSpace[n_kr]);    
    // Update sub arrow matrix
    for(int t=0; t<ortho_dim_size; t++) {      
      for (int i = 0; i < iter_keep; i++) beta_3D[t][i + num_locked] = beta_3D[t][n_kr - 1] * ritz_mat_3D[t][dim * (i + 1) - 1];
    }
    
    for(int t=0; t<ortho_dim_size; t++) host_free(ritz_mat_keep[t]);
    host_free(ritz_mat_keep);
  }

  // Orthogonalise r[t][0:] against V_[t][0:j]
  void TRLM3D::blockOrthogonalize3D(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> &rvecs, int j)
  {
    int vec_size = j;
    std::vector<Complex> s_t(ortho_dim_size, 0.0);
    std::vector<Complex> unit(ortho_dim_size, 1.0);
    for (int i = 0; i < vec_size; i++) {
      std::vector<ColorSpinorField *> vec_ptr;
      vec_ptr.push_back(vecs[i]);
      
      // Block dot products stored in s_t.
      blas3d::cDotProduct(ortho_dim, s_t, *vec_ptr[0], *rvecs[0]);
      for(int t=0; t<ortho_dim_size; t++) s_t[t] *= -1.0;
      
      // Block orthogonalise
      blas3d::caxpby(ortho_dim, s_t, *vec_ptr[0], unit, *rvecs[0]);
      
      // Save orthonormalisation tuning
      saveTuneCache();
    }
  }

  void TRLM3D::prepareInitialGuess3D(std::vector<ColorSpinorField *> &kSpace, int ortho_dim, int ortho_dim_size)
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
    blas3d::reDotProduct(ortho_dim, norms, *kSpace[0], *kSpace[0]);
    for(int t=0; t<ortho_dim_size; t++) norms[t] = 1.0/sqrt(norms[t]);
    blas3d::axpby(ortho_dim, norms, *kSpace[0], zeros, *kSpace[0]);
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
	blas3d::reDotProduct(ortho_dim, norms, *in_ptr, *in_ptr);
	for(int t=0; t<ortho_dim_size; t++) norms[t] = 1.0/sqrt(norms[t]);
	blas3d::axpby(ortho_dim, norms, *in_ptr, zeros, *in_ptr);
      }
      matVec(mat, *out_ptr, *in_ptr);
      std::swap(out_ptr, in_ptr);
    }

    // Compute spectral radius estimate
    std::vector<double> inner_products(ortho_dim_size, 0.0);
    blas3d::reDotProduct(3, inner_products, *out_ptr, *in_ptr);
    double result = 1.0;
    std::vector<double> all_results(comm_dim(ortho_dim) * ortho_dim_size);
    for(int t=0; t<ortho_dim_size; t++) {
      all_results[comm_coord(ortho_dim) * ortho_dim_size + t] = inner_products[t];
      if(inner_products[t] > result) result = inner_products[t];
    }
    
    comm_allreduce_array((double *)all_results.data(), comm_dim(ortho_dim) * ortho_dim_size);
    
    if (getVerbosity() >= QUDA_VERBOSE) {
      for(int t=0; t<ortho_dim_size * comm_dim(ortho_dim); t++) {
	printfQuda("Chebyshev max at slice %d = %e\n", t, all_results[t]);
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
    std::vector<Complex> n_unit(ortho_dim_size, -1.0);
    
    for (int i = 0; i < size; i++) {
      // r = A * v_i
      matVec(mat, *temp[0], *evecs[i]);

      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      // Block dot products stored in s_t.
      blas3d::cDotProduct(ortho_dim, evals_t[i], *evecs[i], *temp[0]);
      blas3d::reDotProduct(ortho_dim, norms, *evecs[i], *evecs[i]);
      for(int t=0; t<ortho_dim_size; t++) evals_t[i][t] /= sqrt(norms[t]);      
      
      // Measure ||lambda_i*v_i - A*v_i||
      blas3d::caxpby(ortho_dim, evals_t[i], *evecs[i], n_unit, *temp[0]);
      blas3d::reDotProduct(ortho_dim, norms, *temp[0], *temp[0]);
      for(int t=0; t<ortho_dim_size; t++) residua_3D[t][i] = sqrt(norms[t]);
    }

    // If size = n_conv, this routine is called post sort
    if (getVerbosity() >= QUDA_SUMMARIZE && size == n_conv)
      for(int t=0; t<ortho_dim_size; t++)
	for (int i = 0; i < size; i++) 
	  printfQuda("Eval[%02d][%04d] = (%+.16e,%+.16e) residual = %+.16e\n", t, i, evals_t[i][t].real(), evals_t[i][t].imag(), residua_3D[t][i]);
    
    
    delete temp[0];
    
    // Save Eval tuning
    saveTuneCache();
  }  
} // namespace quda
