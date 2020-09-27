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

    // Upper Hessenberg and Q matrices
    upperHess = (Complex **)safe_malloc((n_kr) * sizeof(Complex*));
    Qmat = (Complex **)safe_malloc((n_kr) * sizeof(Complex*));
    for (int i = 0; i < n_kr; i++) {
      upperHess[i] = (Complex *)safe_malloc((n_kr) * sizeof(Complex));
      Qmat[i] = (Complex *)safe_malloc((n_kr) * sizeof(Complex));
      for (int j = 0; j < n_kr; j++) {
	upperHess[i][j] = 0.0;
	Qmat[i][j] = 0.0;
      }
    }
    
    if (!profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  // Arnoldi Member functions
  //---------------------------------------------------------------------------
  void IRAM::arnoldiStep(std::vector<ColorSpinorField *> &v, std::vector<ColorSpinorField *> &r, double &beta, int j)
  {
    //%---------------------------------------------------%
    //| STEP 1: Check if the B norm of j-th residual      |
    //| vector is zero. Equivalent to determine whether   |
    //| an exact j-step Arnoldi factorization is present. |
    //%---------------------------------------------------%
    beta = sqrt(blas::norm2(*r[0]));
    if(j > 0) upperHess[j][j-1] = beta;
    
    //%--------------------------------%
    //| STEP 2:  v_{j} = r_{j-1}/rnorm |
    //%--------------------------------%
    //blas::zero(*v[j]);
    //blas::axpy(1.0/beta, *r[0], *v[j]);
    blas::ax(1.0/beta, *r[0]);
    std::swap(v[j], r[0]);
    
    //%----------------------------%
    //| STEP 3:  r_{j} = OP*v_{j}; |
    //%----------------------------%  
    matVec(mat, *r[0], *v[j]);
    
    //%-------------------------------------%
    //| The following is needed for STEP 5. |
    //| Compute the B-norm of OP*v_{j}.     |
    //%-------------------------------------%
    
    double wnorm = sqrt(blas::norm2(*r[0]));
    
    //%-----------------------------------------%
    //| Compute the j-th residual corresponding |
    //| to the j step factorization.            |
    //| Use Classical Gram Schmidt and compute: |
    //| w_{j} <-  V_{j}^T * B * OP * v_{j}      |
    //| r_{j} <-  OP*v_{j} - V_{j} * w_{j}      |
    //%-----------------------------------------%

    //%------------------------------------------%
    //| Compute the j Fourier coefficients w_{j} |
    //| WORKD(IPJ:IPJ+N-1) contains B*OP*v_{j}.  |
    //%------------------------------------------%
    //H_{j,i}_j = v_i^dag * r
    std::vector<Complex> tmp(j+1);
    std::vector<ColorSpinorField *> v_;
    v_.reserve(j+1);
    for (int i = 0; i < j+1; i++) { v_.push_back(v[i]); }    
    blas::cDotProduct(tmp.data(), v_, r);

    //%--------------------------------------%
    //| Orthogonalize r_{j} against V_{j}.   |
    //| RESID contains OP*v_{j}. See STEP 3. | 
    //%--------------------------------------%
    //r = r - H_{j,i} * v_j 
    for (int i = 0; i < j+1; i++) tmp[i] *= -1.0;    
    blas::caxpy(tmp.data(), v_, r);
    for (int i = 0; i < j+1; i++) upperHess[i][j] = -1.0*tmp[i];
    
    //%-----------------------------------------------------------%
    //| STEP 5: Re-orthogonalization / Iterative refinement phase |
    //| Maximum NITER_ITREF tries.                                |
    //|                                                           |
    //|          s      = V_{j}^T * B * r_{j}                     |
    //|          r_{j}  = r_{j} - V_{j}*s                         |
    //|          alphaj = alphaj + s_{j}                          |
    //|                                                           |
    //| The stopping criteria used for iterative refinement is    |
    //| discussed in Parlett's book SEP, page 107 and in Gragg &  |
    //| Reichel ACM TOMS paper; Algorithm 686, Dec. 1990.         |
    //| Determine if we need to correct the residual. The goal is |
    //| to enforce ||v(:,1:j)^T * r_{j}|| .le. eps * || r_{j} ||  |
    //| The following test determines whether the sine of the     |
    //| angle between  OP*x and the computed residual is less     |
    //| than or equal to 0.717.                                   |
    //%-----------------------------------------------------------%

    int orth_iter = 0;
    int orth_iter_max = 100;
    beta = sqrt(blas::norm2(*r[0]));
    while(beta < 0.717*wnorm && orth_iter < orth_iter_max) {
    
      //%---------------------------------------------------%
      //| Enter the Iterative refinement phase. If further  |
      //| refinement is necessary, loop back here. The loop |
      //| variable is ITER. Perform a step of Classical     |
      //| Gram-Schmidt using all the Arnoldi vectors V_{j}  |
      //%---------------------------------------------------%

      //%---------------------------------------------%
      //| Compute the correction to the residual:     |
      //| r_{j} = r_{j} - V_{j} * WORKD(IRJ:IRJ+J-1). |
      //| The correction to H is v(:,1:J)*H(1:J,1:J)  |
      //| + v(:,1:J)*WORKD(IRJ:IRJ+J-1)*e'_j.         |
      //%---------------------------------------------%

      wnorm = beta;
    
      // reorthogonalise r against the Krylov space
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("beta[%d] = %e > 0.717*beta[%d] = %e: Reorthogonalise at step %d, iter %d\n", j+1, beta, j, 0.717*wnorm, j, orth_iter);
      blas::cDotProduct(tmp.data(), v_, r);
      for (int i = 0; i < j+1; i++) tmp[i] *= -1.0;    
      blas::caxpy(tmp.data(), v_, r);    
      for (int i = 0; i < j+1; i++) upperHess[i][j] -= tmp[i];
      
      beta = sqrt(blas::norm2(*r[0]));
      orth_iter++;
    }

    if(orth_iter == orth_iter_max) {
      //%---------------------------------------%
      //| RESID is numerically in the span of V |
      //%---------------------------------------%
      errorQuda("Unable to orthonormalise r");
    }
  }

  void IRAM::rotateVecsComplex(std::vector<ColorSpinorField *> &kSpace, int keep)
  {

    // Multi-BLAS friendly array to store part of Ritz matrix we want
    Complex *Qmat_keep = (Complex *)safe_malloc((n_kr * keep) * sizeof(Complex));

    // If we have memory availible, do the entire rotation
    if (batched_rotate <= 0 || batched_rotate >= keep || 1) {
      if ((int)kSpace.size() < n_kr + keep) {
        ColorSpinorParam csParamClone(*kSpace[0]);
        csParamClone.create = QUDA_ZERO_FIELD_CREATE;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Resizing kSpace to %d vectors\n", n_kr + keep);
        kSpace.reserve(n_kr + keep);
        for (int i = kSpace.size(); i < n_kr + keep; i++) {
          kSpace.push_back(ColorSpinorField::Create(csParamClone));
        }
      }
      
      // Pointers to the relevant vectors
      std::vector<ColorSpinorField *> vecs_ptr;
      std::vector<ColorSpinorField *> kSpace_ptr;

      // Alias the extra space vectors, zero the workspace
      kSpace_ptr.reserve(keep);
      for (int i = 0; i < keep; i++) {
        kSpace_ptr.push_back(kSpace[n_kr + i]);
        blas::zero(*kSpace_ptr[i]);
      }

      // Alias the vectors we wish to keep, populate the Ritz matrix and transpose.
      vecs_ptr.reserve(n_kr);
      for (int j = 0; j < n_kr; j++) {
        vecs_ptr.push_back(kSpace[j]);
        for (int i = 0; i < keep; i++) { Qmat_keep[j * keep + i] = Qmat[j][i]; }
      }
      
      // multiBLAS caxpy
      blas::caxpy(Qmat_keep, vecs_ptr, kSpace_ptr);

      // Copy compressed Krylov
      for (int i = 0; i < keep; i++) std::swap(kSpace[i], kSpace[n_kr + i]);
    }
    
    host_free(Qmat_keep);
    
    // Save Krylov rotation tuning
    saveTuneCache();
  }
  
  void IRAM::qrShifts(const std::vector<Complex> evals, const int num_shifts, const double epsilon)
  {
    Complex T11, T12, T21, T22, temp, temp2, temp3, U1, U2;
    double dV;
    
    // Allocate the rotation matrices.
    std::vector<Complex> R11(n_kr-1, 0.0);
    std::vector<Complex> R12(n_kr-1, 0.0);
    std::vector<Complex> R21(n_kr-1, 0.0);
    std::vector<Complex> R22(n_kr-1, 0.0);

    // Reset Q to the identity
    for(int i = 0; i < n_kr; i++) {
      for(int j = 0; j < n_kr; j++) {
	if(i == j) Qmat[i][j] = 1.0;
	else Qmat[i][j] = 0.0;
      }
    }
    
    for(int shift=0; shift<num_shifts; shift++) {

      // First pass, determine the matrices and do H -> R.
      for(int i=0; i<n_kr; i++) upperHess[i][i] -= evals[shift];
      
      for(int i=0; i<n_kr-1; i++) {
	if (abs(upperHess[i+1][i]) < epsilon) {
	  upperHess[i+1][i] = 0.0;
	  continue;
	}
	
	dV = sqrt(norm(upperHess[i][i]) + norm(upperHess[i+1][i]));
	U1 = upperHess[i][i];
	dV = (U1.real() > 0) ? dV : -dV;
	U1 += dV;
	U2 = upperHess[i+1][i];
	
	T11 = conj(U1);
	T11 /= dV;
	R11[i] = conj(T11);
	
	T12 = conj(U2);
	T12 /= dV;
	R12[i] = conj(T12);
	
	T21 = conj(T12);
	temp = conj(U1);
	temp /= U1;
	T21 *= temp;
	R21[i] = conj(T21);
	
	temp = U2 / U1;
	T22 = T12 * temp;
	R22[i] = conj(T22);
	
	// Do the H_kk and set the H_k+1k to zero
	temp = upperHess[i][i];
	temp2 = T11 * temp;
	temp3 = T12 * upperHess[i+1][i];
	temp2 += temp3;
	upperHess[i][i] -= temp2;
	upperHess[i+1][i] = 0;
	// Continue for the other columns
	for(int j=i+1; j < n_kr; j++) {
	  temp = upperHess[i][j];
	  temp2 = T11 * temp;
	  temp2 += T12 * upperHess[i+1][j];
	  upperHess[i][j] -= temp2;
	  
	  temp2 = T21 * temp;
	  temp2 += T22 * upperHess[i+1][j];
	  upperHess[i+1][j] -= temp2;
	}
      }
      
      // Rotate R and V, i.e. H->RQ. V->VQ 
      for(int j = 0; j < n_kr - 1; j++) {
	if(abs(R11[j]) > epsilon) {
	  for(int i = 0; i < j+2; i++) {
	    temp = upperHess[i][j];
	    temp2 = R11[j] * temp;
	    temp2 += R12[j] * upperHess[i][j+1];
	    upperHess[i][j] -= temp2;
	    
	    temp2 = R21[j] * temp;
	    temp2 += R22[j] * upperHess[i][j+1];
	    upperHess[i][j+1] -= temp2;
	  }
	  
	  for(int i = 0; i < n_kr; i++) {
	    temp = Qmat[i][j];
	    temp2 = R11[j] * temp;
	    temp2 += R12[j] * Qmat[i][j+1];
	    Qmat[i][j] -= temp2;
	    
	    temp2 = R21[j] * temp;
	    temp2 += R22[j] * Qmat[i][j+1];
	    Qmat[i][j+1] -= temp2;
	  }
	}
      }
      for(int i=0; i<n_kr; i++) upperHess[i][i] += evals[shift];
    }
  }
  
  void IRAM::eigensolveFromUpperHess(std::vector<Complex> &evals, const double beta)
  {
    //Construct the upper Hessenberg matrix       
    MatrixXcd upperHessEigen = MatrixXcd::Zero(n_kr, n_kr);
    for(int i=0; i<n_kr; i++) {
      for(int j=0; j<n_kr; j++) {
	upperHessEigen(i,j) = upperHess[i][j];
      }
    }
    
    // Eigensolve the upper Hessenberg matrix
    Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH(upperHessEigen);
    for(int i=0; i<n_kr; i++) {
      evals[i] = eigenSolverUH.eigenvalues()[i];
      residua[i] = abs(beta * eigenSolverUH.eigenvectors().col(i)[n_kr - 1]);
    }
    // Update the Q matrix
    for(int i = 0; i < n_kr; i++) {
      for(int j = 0; j < n_kr; j++) {
	Qmat[j][i] = eigenSolverUH.eigenvectors().col(i)[j];
      }
    }
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
    matVec(mat, *r[0], *kSpace[0]);
    
    // Check for Chebyshev maximum estimation
    checkChebyOpMax(mat, kSpace);

    // Convergence and locking criteria
    double epsilon = setEpsilon(kSpace[0]->Precision());
    double epsilon23 = pow(epsilon, 2.0/3.0);
    double beta = 0.0;

    // Eigen object for computing Ritz values from the upper Hessenberg matrix
    Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;
    
    // Print Eigensolver params
    printEigensolverSetup();
    //---------------------------------------------------------------------------

    // Begin IRAM Eigensolver computation
    //---------------------------------------------------------------------------

    // Do the first n_ev steps
    for (int step = 0; step < n_ev; step++) arnoldiStep(kSpace, r, beta, step);
    num_keep = n_ev;
    iter += n_ev;
    
    // Loop over restart iterations.    
    while (restart_iter < max_restarts && !converged) {
      for (int step = num_keep; step < n_kr; step++) arnoldiStep(kSpace, r, beta, step);
      iter += n_kr - num_keep;
      
      // Ritz values and their residua are updated.
      eigensolveFromUpperHess(evals, beta);

      num_keep = n_ev;
      int num_shifts = n_kr - num_keep;

      // Put unwanted Ritz(evals) first. We will shift these out of the
      // Krylov space using QR.
      sortArrays(eig_param->spectrum, n_kr, evals, residua);
      
      // Put smallest errors on the unwanted Ritz first to aid in forward stability
      sortArrays(QUDA_SPECTRUM_LM_EIG, num_shifts, residua, evals);

      // Convergence test
      iter_converged = 0;
      for(int i=0; i<n_ev; i++) {
	int idx = n_kr - 1 - i;
	double rtemp = std::max(epsilon23, abs(evals[idx]));
	if(residua[idx] < tol * rtemp) {
	  iter_converged++;
	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printf("residuum[%d] = %e, condition = %e\n", i, residua[idx], tol * abs(evals[idx]));
	} else {
	  // Unlikely to find new converged eigenvalues
	  break;
	}
      }
      
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("%04d converged eigenvalues at iter %d\n", num_converged, restart_iter);

      if (num_converged >= n_conv) {	
	converged = true;
	eigensolveFromUpperHess(evals, beta);
	rotateVecsComplex(kSpace, n_kr);
	reorder(kSpace, evals, eig_param->spectrum);
      } else {
	
	int num_keep0 = num_keep;
	iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - 12);
	
	num_converged = iter_converged;
	num_keep = iter_keep;      
	num_shifts = n_kr - num_keep;

	// If num_keep changed, we resort the Ritz values and residua
	if(num_keep0 < num_keep) {
	  sortArrays(eig_param->spectrum, n_kr, evals, residua);
	  sortArrays(QUDA_SPECTRUM_LM_EIG, num_shifts, residua, evals);    
	}

	// Apply the shifts of the unwated Ritz values via QR
	qrShifts(evals, num_shifts, epsilon);
	  
	// Compress the Krylov space using the accumulated Givens rotations in Qmat
	rotateVecsComplex(kSpace, num_keep+1);
	
	// Update the residual vector
	blas::caxpby(upperHess[num_keep][num_keep-1], *kSpace[num_keep], Qmat[n_kr-1][num_keep-1], *r[0]);

	if(sqrt(blas::norm2(*r[0])) < epsilon) {
	  errorQuda("IRAM has encountered an invariant subspace...");	 
	}
      }      
      restart_iter++;
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
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
	for(int i=0; i<n_conv; i++)
	  printfQuda("Eval[%04d] = (%+.16e,%+.16e) residual = %+.16e\n",
		     i, evals[i].real(), evals[i].imag(), residua[i]);
      }

      // Check order
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) computeEvals(mat, kSpace, evals, n_kr);
    }

    // Local clean-up
    cleanUpEigensolver(kSpace, evals);
  }

  void IRAM::reorder(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals, const QudaEigSpectrumType spec_type)
  {
    int i=0;
    switch(spec_type) {
    case QUDA_SPECTRUM_LM_EIG:
      while (i < n_kr) {
        if ((i == 0) || abs(evals[i - 1]) >= abs(evals[i]))
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    case QUDA_SPECTRUM_SM_EIG:
      while (i < n_kr) {
        if ((i == 0) || abs(evals[i - 1]) <= abs(evals[i]))
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    case QUDA_SPECTRUM_LR_EIG:      
      while (i < n_kr) {
        if ((i == 0) || evals[i - 1].real() >= evals[i].real())
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    case QUDA_SPECTRUM_SR_EIG:
      while (i < n_kr) {
        if ((i == 0) || evals[i - 1].real() <= evals[i].real())
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    case QUDA_SPECTRUM_LI_EIG:
      while (i < n_kr) {
        if ((i == 0) || evals[i - 1].imag() >= evals[i].imag())
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    case QUDA_SPECTRUM_SI_EIG:
      while (i < n_kr) {
        if ((i == 0) || evals[i - 1].imag() <= evals[i].imag())
	  i++;
        else {
          std::swap(evals[i], evals[i - 1]);
	  std::swap(residua[i], residua[i - 1]);
          std::swap(kSpace[i], kSpace[i - 1]);
          i--;
        }
      }
      break;
    default: errorQuda("Undefined sort %d given", spec_type);
    }
  }
  
  // Destructor
  IRAM::~IRAM()
  {
    for(int i=0; i<n_kr; i++) {
      host_free(upperHess[i]);
      host_free(Qmat[i]);
    }
    host_free(upperHess);
    host_free(Qmat);
  }  
}
