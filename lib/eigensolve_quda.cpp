#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <Eigen/Dense>

namespace quda {

  //Perform Rayleigh-Ritz in-place matrix multiplication.
  //           y_i = V_k s_i
  // where T_k s_i = theta_i s_i  
  template<typename Float>
  void computeEigVecs(std::complex<Float> **vecs, Float **mat,
		      int nEv, int nKr, int length) {

    //nEv is the number of Ritz vectors we use in the rotation
    //nKr is the number of Krylov space vectors to rotate.
    //length is the number of complex elements of the vector
    
    //loop over rows of kSpace
    for(int j=0; j<length; j++) {      
      
      //put jth row of kSpace in temp
      std::complex<Float> tmp[nKr];      
      for(int i=0; i<nKr; i++) {
	tmp[i] = vecs[i][j];
      }
      
      std::complex<Float> sum = 0.0;
      //loop over columns of Q 
      for(int k=0; k<nEv; k++) {
	//take product of jth row of Q and kth column of Y
	for(int l=0; l<nKr; l++) {
	  sum += tmp[l]*mat[k][l];
	}
	vecs[k][j] = sum;
	sum *= 0.0;
      }
    }
  }
  
  
  //Orthogonalise r against V_[j]
  template<typename Float>
  void orthogonalise(std::vector<ColorSpinorField*> v,
		     cudaColorSpinorField &r, int j) {
    
    std::complex<Float> s(0.0,0.0);
    for(int i=0; i<j; i++) {
      s = blas::cDotProduct(*v[i], r);
      blas::caxpy(-s, *v[i], r);
    }
  }

  template<typename Float>
  void lanczosStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   cudaColorSpinorField &r,
		   QudaEigParam *eig_param,
		   Float *alpha, Float *beta, int j) {

    //printfQuda("flag 1\n");
    
    //Compute r = A * v_j - b_{j-i} * v_{j-1}      
    //r = A * v_j
    //DMH: maybe enforce MdagM mat vec?
    if(eig_param->use_norm_op && eig_param->use_dagger) {
      mat.MMdag(r, *(v[j]) );
    }
    else if(eig_param->use_norm_op && !eig_param->use_dagger) {
      mat.MdagM(r, *(v[j]) );
    }
    else if (!eig_param->use_norm_op && eig_param->use_dagger) {
      mat.Mdag(r, *(v[j]) );
    }
    else {
      mat.M(r, *(v[j]) );
    }
    
    //r = r - b_{j-1} * v_{j-1}
    if(j>0) blas::axpy(-beta[j-1], *v[j-1], r);      
    
    //a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v[j], r);    

    //r = r - a_j * v_j
    blas::axpy(-alpha[j], *v[j], r);

    //b_j = ||r|| 
    beta[j] = sqrt(blas::norm2(r));

    //Orthogonalise
    if(beta[j] < (1.0)*sqrt(alpha[j]*alpha[j] + beta[j-1]*beta[j-1])) {

      //The residual vector r has been deemed insufficiently
      //orthogonal to the existing Krylov space. We must
      //orthogonalise it.
      //printfQuda("orthogonalising Beta %d = %e\n", j, beta[j]);
      orthogonalise<Float>(v, r, j);
      //b_j = ||r|| 
      beta[j] = sqrt(blas::norm2(r));    
    }

    //Prepare next step.
    //v_{j+1} = r / b_j
    blas::zero(*v[j+1]);
    blas::axpy(1.0/beta[j], r, *v[j+1]);

  }
  
  
  template<typename Float>
  void lanczos_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		     QudaEigParam *eig_param,
		     ColorSpinorParam *cpuParam){

    //profile.TPSTART(QUDA_PROFILE_INIT);
    
    // Preliminaries: Memory allocation, local variables.
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    Float tol = eig_param->tol;
    
    printfQuda("Nev requested = %d\n", nEv);
    printfQuda("NKrylov space = %d\n", nKr);
    
    Float *mod_evals_sorted = (Float*)malloc(nKr*sizeof(Float));
    Float *residual = (Float*)malloc(nKr*sizeof(Float));
    int *evals_sorted_idx = (int*)malloc(nKr*sizeof(int));
    
    //Tridiagonal matrix
    Float alpha[nKr];
    Float  beta[nKr];
    for(int i=0; i<nKr; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }
    
    //Alias pointers
    std::complex<Float> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<Float>*) (Float*)(h_evecs);    
    std::complex<Float> *h_evals_ = nullptr;
    h_evals_ = (std::complex<Float>*) (Float*)(h_evals);
    
    //Create the device side Krylov Space and residual vector
    cudaColorSpinorField *r = nullptr;
    ColorSpinorParam cudaParam(*cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    sizeof(Float) == sizeof(double) ?
      cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER :
      cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    r = new cudaColorSpinorField(cudaParam);

    //Create kryloc space on the device
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr+1; i++) {
      kSpace.push_back(cudaColorSpinorField::Create(cudaParam));
    }

    //Auxiliiary device vector
    ColorSpinorField *eVec = nullptr;
    eVec = new cudaColorSpinorField(cudaParam);
  
    //At the moment, basis rotation is done on the host. We must transfer the
    //krylov space each time we rotate, se we create (for now) host side arrays.
    //--------------------------------------------------------------------------
    int length = 12;
    for(int i=0; i<4; i++) {
      length *= cpuParam->x[i];
      printfQuda("dim[%d] = %d, length = %d\n", i, cpuParam->x[i], length);
    }
    std::vector<ColorSpinorField*> h_vecs;
    std::complex<Float> **vecs =
      (std::complex<Float>**)malloc((nKr+1)*sizeof(std::complex<Float*>));
    
    //Allocate space for the Ritz vectors.
    for(int i=0; i<nKr+1; i++) {
      vecs[i] = (std::complex<Float>*)malloc(length*sizeof(std::complex<Float>));
      cpuParam->v = &(vecs[i][0]);
      h_vecs.push_back(cpuColorSpinorField::Create(*cpuParam));
    }
    //--------------------------------------------------------------------------

    //Host side dense matrix of Kspace eigenvectors
    Float **Emat = (Float**)malloc((nKr)*sizeof(Float*));
    for(int i=0; i<nKr; i++) {
      Emat[i] = (Float*)malloc((nKr)*sizeof(Float));
    }
    
    //Populate source with randoms.
    printfQuda("Using random guess\n");
    kSpace[0] -> Source(QUDA_RANDOM_SOURCE);

    //Ensure we are not trying to compute on a zero-field source    
    const Float norm = sqrt(blas::norm2(*kSpace[0]));
    if(norm == 0){
      errorQuda("Initial residual is zero.\n");
      return;
    }
    //Normalise initial source
    blas::ax(1.0/norm, *kSpace[0]);
    //profile.TPSTOP(QUDA_PROFILE_INIT);
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------



    
    // START LANCZOS
    // Lanczos Method for Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------

    //profile.TPSTART(QUDA_PROFILE_COMPUTE);
    Float t1 = clock();
    printfQuda("START LANCZOS SOLUTION\n");
    bool converged = false;
    int numConverged = 0;
    int k=0;
    int check_interval = 100;
    while(k < nKr && !converged) {

      lanczosStep<Float>(mat, kSpace, *r, eig_param, alpha, beta, k);

      if((k+1)%check_interval == 0) {

	int check = k+1;
	printfQuda("%04d converged eigenvalues at iter %04d\n", numConverged, check);
	
	//Copy Krylov space to host for manipulation
	for(int i=0; i<check; i++) *h_vecs[i] = *kSpace[i];
	
	//Compute the Tridiagonal matrix T_{k,k} 
	using Eigen::MatrixXd;
	MatrixXd triDiag = MatrixXd::Zero(check, check);
	
	for(int i=0; i<check; i++) {
	  triDiag(i,i) = alpha[i];      
	  if(i<check-1) {	
	    triDiag(i+1,i) = beta[i];	
	    triDiag(i,i+1) = beta[i];
	  }
	}
	
	//Eigensolve the T_k matrix
	Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
	eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());
	//Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD(triDiag);
	
	//Ritz values are in ascending order if matrix is real.      
	//std::cout << eigenSolverTD.eigenvalues() << std::endl;
	
	//Place data in accessible array
	for(int i=0; i<check; i++) {
	  for(int j=0; j<check; j++) {
	    Emat[i][j] = eigenSolverTD.eigenvectors().col(i)[j];
	    //printfQuda("(%d,%d) %f\n", i, j, Emat[i][j]);
	  }
	}

	computeEigVecs<Float>(vecs, Emat, check, check, length);
	
	//Check for convergence
	int n_conv = 0;
	//Error estimates given by ||M*q_tilde - lambda q_tilde||
	for(int i=0; i<nEv; i++) {

	  *eVec = *h_vecs[i];

	  //r = A * v_i
	  if(eig_param->use_norm_op && eig_param->use_dagger) {
	    mat.MMdag(*r, *eVec);
	  }
	  else if(eig_param->use_norm_op && !eig_param->use_dagger) {
	    mat.MdagM(*r, *eVec);
	  }
	  else if (!eig_param->use_norm_op && eig_param->use_dagger) {
	    mat.Mdag(*r, *eVec);
	  }
	  else {  
	    mat.M(*r, *eVec);
	  }
	  
	  
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  h_evals_[i] = blas::cDotProduct(*eVec, *r)/sqrt(blas::norm2(*eVec));

	  //Convergence check ||A * v_i - lambda_i * v_i||
	  blas::caxpby(h_evals_[i], *eVec, -1.0, *r);
	  residual[i] = sqrt(blas::norm2(*r));
	  //printfQuda("Residual %d = %.6e\n", i, residual[i]);
	  if(i < nEv && residual[i] < tol) {
	    n_conv++;
	  }
	  if(n_conv == nEv) {
	    converged = true;
	  }
	}
	numConverged = n_conv;
      }
      k++;
    }
    //profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    

    //Post computation information
    //----------------------------
    if(!converged) {
      printf("Lanczos failed to compute the requested %d vectors in %d steps. Please either increase nKr or decrease nEv\n", nEv, nKr);
    } else {
      printf("Lanczos computed the requested %d vectors in %d steps\n", nEv, k);
      for(int i=0; i<nEv; i++) {
	printf("EigValue[%04d]: (%+.8e, %+.8e) residual %+.8e\n",
	       i, h_evals_[i].real(), h_evals_[i].imag(), residual[i]);
      }
    }
    
    Float t2 = clock() - t1;
    printfQuda("END LANCZOS SOLUTION\n");
    printfQuda("Time to solve problem using Lanczos = %e\n", t2/CLOCKS_PER_SEC);
    //-----------------------------

    //Local clean-up
    //--------------
    //profile.TPSTART(QUDA_PROFILE_FREE);
    delete r;
    delete eVec;
    for(int i=0; i<nKr+1; i++) {
      delete kSpace[i];
      delete h_vecs[i];
      free(vecs[i]);
    }
    //profile.TPSTOP(QUDA_PROFILE_FREE);
    //--------------
  }
  
  void lanczosSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam){
    
    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      lanczos_solve<double>(h_evecs, h_evals, mat, eig_param, cpuParam);
    } else {
      lanczos_solve<float>(h_evecs, h_evals, mat, eig_param, cpuParam);
    }    
  }  
}

#if 0

namespace quda {

  EigSolver *EigSolver::create(QudaEigParam &param, TimeProfile &profile) {
    
    EigSolver *eig_solver = nullptr;
    
    switch (param.eig_type) {
    case QUDA_LANCZOS:
      eig_solver = new Lanczos(ritz_mat, param, profile);
      break;
    case QUDA_IMP_RST_LANCZOS:
      eig_solver = new ImpRstLanczos(ritz_mat, param, profile);
      break;
    default:
      errorQuda("Invalid eig solver type");
    }
    return eig_solver;
  }
} // namespace quda

#endif //0
