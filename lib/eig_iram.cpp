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

namespace quda
{

  using namespace Eigen;

  // Implicitly Restarted Arnoldi Method constructor
  IRAM::IRAM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile) :
    EigenSolver(mat, eig_param, profile)
  {
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    // Upper Hessenberg matrix    
    upperHess = (Complex **)safe_malloc((n_kr+1) * sizeof(Complex*));
    for (int i = 0; i < n_kr+1; i++) {
      upperHess[i] = (Complex *)safe_malloc((n_kr+1) * sizeof(Complex));
      for (int j = 0; j < n_kr+1; j++) {
	upperHess[i][j] = 0.0;
      }
    }
    
    if (!(eig_param->spectrum == QUDA_SPECTRUM_LR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only real spectrum type (LR or SR) can be passed to the IR arnoldi solver");
    }
    
    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void IRAM::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // In case we are deflating an operator, save the tunechache from the inverter
    saveTuneCache();

    // Override any user input for block size.
    block_size = 1;

    // Pre-launch checks and preparation
    //---------------------------------------------------------------------------
    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Check for an initial guess. If none present, populate with rands, then
    // orthonormalise
    prepareInitialGuess(kSpace);

    // Increase the size of kSpace passed to the function, will be trimmed to
    // original size before exit.
    prepareKrylovSpace(kSpace, evals);

    // Check for Chebyshev maximum estimation
    checkChebyOpMax(mat, kSpace);

    // Convergence and locking criteria
    double mat_norm = 0.0;
    double epsilon = setEpsilon(kSpace[0]->Precision());

    Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;
    
    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin IRAM Eigensolver computation
    //---------------------------------------------------------------------------
    // Loop over restart iterations.    
    while (restart_iter < max_restarts && !converged) {
      
      for (int step = restart_iter; step < restart_iter + 1; step++) arnoldiStep(kSpace, step);
      iter += 1;
      restart_iter += 1;

      if(iter % check_interval == 0) {
	//Construct the Upper Hessenberg matrix H_k
	MatrixXcd upperHessEigen = MatrixXcd::Zero(iter, iter);
	for(int k=0; k<iter; k++) {
	  for(int i=0; i<iter; i++) {
	    upperHessEigen(k,i) = upperHess[k][i];
	  }
	}
	
	//std::cout << upperHessEigen << std::endl;
	
	// Eigensolve H_k
	eigenSolverUH.compute(upperHessEigen);
	
	// mat_norm and residua are updated.
	mat_norm = upperHessEigen.norm();
	printfQuda("mat_norm = %e\n", mat_norm);
	for (int i = 0; i < iter; i++) {
	  residua[i] = abs(upperHess[iter][iter-1] * eigenSolverUH.eigenvectors().col(i)[iter-1]);
	  printfQuda("Residuum[%d] = %e\n", i, residua[i]);
	}

	//Halting check
	if (n_ev <= n_kr) {
	  num_converged = 0;
	  for(int i=0; i<iter; i++) {
	    if(residua[i] < tol * abs(eigenSolverUH.eigenvalues()[i])) num_converged++;
	  }
	  printf("%04d converged eigenvalues at iter %d\n", num_converged, iter);	
	  if (num_converged >= n_ev) {
	    converged = true;
	    // Transfer ritz matrix
	    ritz_mat_complex.resize(iter*iter);
	    for (int i = 0; i < iter; i++) {
	      for (int j = 0; j < iter; j++) {	      
		ritz_mat_complex[j * iter + i] = eigenSolverUH.eigenvectors().col(i)[j];
	      }
	    }
	  }
	}
      }
    }
    
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
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("IRAM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", n_conv,
                   restart_iter, iter);

        // Dump all Ritz values and residua
        for (int i = 0; i < n_conv; i++) {
          //printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
        }
      }

      // Compute eigenvalues
      rotateVecsComplex(kSpace, iter);
      computeEvals(mat, kSpace, evals);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  // Destructor
  IRAM::~IRAM()
  {
    for(int i=0; i<n_kr+1; i++) host_free(upperHess[i]);
    host_free(upperHess);
  }
  
  // Thick Restart Member functions
  //---------------------------------------------------------------------------
  void IRAM::arnoldiStep(std::vector<ColorSpinorField *> &v, int j)
  {
    chebyOp(mat, *r[0], *v[j]);
    
    for (int i = 0; i < j+1; i++) {
      //H_{i,j} = v_i^dag * r
      upperHess[i][j] = blas::cDotProduct(*v[i], *r[0]);
      //printfQuda("iter:%d upperHess[%d][%d] = (%e,%e)\n", j, i, j, upperHess[i][j].real(), upperHess[i][j].imag());
      //r = r - v_i * H_{i,j}
      blas::caxpy(-1.0*upperHess[i][j], *v[i], *r[0]);
    }

    double norm_r = sqrt(blas::norm2(*r[0]));
    printfQuda("Residual norm = %e\n", norm_r);
    if(j < (int)v.size() - 1) {
      upperHess[j+1][j].real(norm_r);
    }

    if(j>0) {
      for(int i=0; i<2; i++) blockOrthogonalize(v, r, j);
    }
    
    blas::zero(*v[j + 1]);
    blas::axpy(1.0 / norm_r, *r[0], *v[j + 1]);
  }

  void IRAM::rotateVecsComplex(std::vector<ColorSpinorField *> &kSpace, int j)
  {
    ColorSpinorParam csParamClone(*kSpace[0]);
    csParamClone.create = QUDA_ZERO_FIELD_CREATE;
    
    // Pointers to the relevant vectors
    std::vector<ColorSpinorField *> vecs_ptr;
    std::vector<ColorSpinorField *> kSpace_ptr;
    
    // Alias the extra space vectors, zero the workspace
    kSpace_ptr.reserve(j);
    for (int i = 0; i < j; i++) {
      kSpace_ptr.push_back(ColorSpinorField::Create(csParamClone));
    }

    // Alias the vectors we wish to keep, populate the Ritz matrix and transpose.
    vecs_ptr.reserve(j);
    for (int i = 0; i < j; i++) {
      vecs_ptr.push_back(kSpace[i]);
    }
    
    // multiBLAS caxpy
    blas::caxpy(ritz_mat_complex.data(), vecs_ptr, kSpace_ptr);
    
    // Copy back to the Krylov space
    for (int i = 0; i < j; i++) {
      *kSpace[i] = *kSpace_ptr[i];
      delete kSpace_ptr[i];
    }
  }
}
