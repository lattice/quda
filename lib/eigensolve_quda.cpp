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

  template<typename Float>
  void matVec(const Dirac &mat,
	      cudaColorSpinorField &out,
	      const cudaColorSpinorField &in,
	      QudaEigParam *eig_param){
    
    if(eig_param->use_norm_op && eig_param->use_dagger) {
      mat.MMdag(out,in);
      return;
    }
    else if(eig_param->use_norm_op && !eig_param->use_dagger) {
      mat.MdagM(out,in);
      return;
    }
    else if(!eig_param->use_norm_op && eig_param->use_dagger) {
      mat.Mdag(out,in);
      return;
    }
    else {  
      mat.M(out,in);
      return;
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

  
  //Orthogonalise r against V_[j]
  template<typename Float>
  void blockOrthogonalise(std::vector<ColorSpinorField*> v,
			  cudaColorSpinorField &r, int j) {

    cudaColorSpinorField *r_p = nullptr;
    r_p = new cudaColorSpinorField(r);    
    
    std::vector<ColorSpinorField*> r_v;
    r_v.push_back(cudaColorSpinorField::Create(*r_p));

    Complex *s = new Complex[j];    
    //Block dot products stored in s.
    blas::cDotProduct(s, v, r_v);
    
    printfQuda("Flag 1\n");
    
    //Block orthonormalise
    for(int i=0; i<j; i++) s[i] *= -1.0;
    blas::caxpy(s, v, r_v);
    delete r_v[0];
    delete r_p;
  }
  
  
  template<typename Float>
  void matVecOp(const Dirac &mat,
		cudaColorSpinorField &out,
		const cudaColorSpinorField &in,
		QudaEigParam *eig_param){
    
    //Just do a simple matVec if no poly acc is requested
    if(!eig_param->use_poly_acc) {    
      matVec<Float>(mat, out, in, eig_param);
      return;
    }

    if(eig_param->poly_deg == 0) {
      errorQuda("Polynomial accleration requested with zero polynomial degree");
    }
    
    //Compute the polynomial accelerated operator.
    Float delta,theta;
    Float sigma,sigma1,sigma_old;
    Float d1,d2,d3;

    double a = eig_param->a_min;
    double b = eig_param->a_max;

    delta = (b-a)/2.0;
    theta = (b+a)/2.0;

    sigma1 = -delta/theta;

    d1 = sigma1/delta;
    d2 = 1.0;

    //out = d2 * in + d1 * out
    //C_1(x) = x
    matVec<Float>(mat, out, in, eig_param);
    blas::caxpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);
    if(eig_param->poly_deg == 1) return;

    
    // C_0 is the current 'in'  vector.
    // C_1 is the current 'out' vector.
       
    //Clone 'in' to two temporary vectors.
    cudaColorSpinorField *tmp1 = new cudaColorSpinorField(in);
    cudaColorSpinorField *tmp2 = new cudaColorSpinorField(in);

    blas::copy(*tmp1,in);
    blas::copy(*tmp2,out);

    //Using Chebyshev polynomial recursion relation,
    //C_{m+1}(x) = 2*x*C_{m} - C_{m-1}

    sigma_old = sigma1;
    
    //construct C_{m+1}(x)
    for(int i=2; i < eig_param->poly_deg; i++){

      sigma = 1.0/(2.0/sigma1-sigma_old);
      
      d1 = 2.0*sigma/delta;
      d2 = -d1*theta;
      d3 = -sigma*sigma_old;

      //mat*C_{m}(x)
      matVec<Float>(mat, out, *tmp2, eig_param);
      
      blas::ax(d3,*tmp1);
      std::complex<double> d1c(d1,0.0);
      std::complex<double> d2c(d2,0.0);
      blas::cxpaypbz(*tmp1,d2c,*tmp2,d1c,out);
      
      blas::copy(*tmp1,*tmp2);
      blas::copy(*tmp2,out);
      sigma_old = sigma;
      
    }
    
    delete tmp1;
    delete tmp2;
  }
  
  template<typename Float>
  void lanczosStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   cudaColorSpinorField &r,
		   QudaEigParam *eig_param,
		   Float *alpha, Float *beta, int j) {

    //Compute r = A * v_j - b_{j-i} * v_{j-1}      
    //r = A * v_j
    //DMH: maybe enforce MdagM mat vec? Defo do a polyOp.
    matVecOp<Float>(mat, r, *v[j], eig_param);
    
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
      printfQuda("orthogonalising Beta %d = %e\n", j, beta[j]);
      orthogonalise<Float>(v, r, j);
      //blockOrthogonalise<Float>(v, r, j);
      //b_j = ||r|| 
      beta[j] = sqrt(blas::norm2(r));    
    }

    //Prepare next step.
    //v_{j+1} = r / b_j
    if(j < v.size() - 1) {
      blas::zero(*v[j+1]);
      blas::axpy(1.0/beta[j], r, *v[j+1]);
    }
  }
  
  
  template<typename Float>
  void lanczos_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		     QudaEigParam *eig_param,
		     ColorSpinorParam *cpuParam){

    // Preliminaries: Memory allocation, local variables.
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    Float tol = eig_param->tol;    

    printfQuda("\n\nSTART LANCZOS SOLUTION\n");
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

    //Create Krylov space on the device
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr; i++) {
      kSpace.push_back(cudaColorSpinorField::Create(cudaParam));
    }

    //Host side dense matrix of Q eigenvectors
    Float **Qmat_h = (Float**)malloc((nKr)*sizeof(Float*));
    for(int i=0; i<nKr; i++) {
      Qmat_h[i] = (Float*)malloc((nKr)*sizeof(Float));
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
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------



    
    // START LANCZOS
    // Lanczos Method for Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------

    //profile.TPSTART(QUDA_PROFILE_COMPUTE);
    Float t1 = clock();
    bool converged = false;
    int num_converged = 0;
    int k=0;
    int check_interval = eig_param->check_interval;
    //int check_interval = nEv;
    double time;
    double time_e = 0.0;
    double time_c = 0.0;
    double time_ls = 0.0;
    double time_dt = 0.0;
    double time_mb = 0.0;
    
    while(k < nKr && !converged) {

      time = -clock();      
      lanczosStep<Float>(mat, kSpace, *r, eig_param, alpha, beta, k);
      time += clock();
      time_ls += time;
      
      if((k+1)%check_interval == 0 && k > nEv) {
	
	int check = k+1;
	printfQuda("%04d converged eigenvalues at iter %04d\n", num_converged, check);
	
	time = -clock();
	//Device side linalg for Lanczos
	std::vector<ColorSpinorField*> d_vecs;
	for(int i=0; i<check; i++) {
	  d_vecs.push_back(cudaColorSpinorField::Create(cudaParam));
	}
	std::vector<ColorSpinorField*> d_vecs_out;
	for(int i=0; i<check; i++) {
	  d_vecs_out.push_back(cudaColorSpinorField::Create(cudaParam));
	}
		
	//Copy Krylov space to device for manipulation
	for(int i=0; i<check; i++) *d_vecs[i] = *kSpace[i];
	time += clock();
	time_dt += time;

	
	time = -clock();	
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

	for(int a=0; a<check; a++) printfQuda("TD eval %d = %e\n", a, eigenSolverTD.eigenvalues()[a]);
	
	Complex *Qmat_d = new Complex[check*check];
	
	//Place data in accessible array
	for(int i=0; i<check; i++) {
	  for(int j=0; j<check; j++) {
	    //DMH Careful: Order
	    Qmat_d[j*check + i].real(eigenSolverTD.eigenvectors().col(i)[j]);
	    Qmat_d[j*check + i].imag(0.0);
	    //printfQuda("(%d,%d) %f\n", i, j, Qmat_h[i][j]);
	  }
	}

	time += clock();
	time_e += time;
	//computeEigVecs<Float>(vecs, Qmat_h, check, check, length);
	
	//Use block basis rotation
	time = -clock();
	blas::caxpy(Qmat_d, d_vecs, d_vecs_out);
	time += clock();
	time_mb += time;
	//blas::caxpy((Complex*)eigenSolverTD.eigenvectors(),
	//d_vecs, d_vecs_out);

	for(int a=0; a<check; a++) printfQuda("Post Rotation norm %d = %.6e\n", a, sqrt(blas::norm2(*d_vecs_out[a])));
	
	//Check for convergence
	int n_conv = 0;
	//Error estimates given by ||M*q_tilde - lambda q_tilde||
	time = -clock();
	for(int i=0; i<nEv; i++) {
	  
	  //r = A * v_i
	  matVec<Float>(mat, *r, *d_vecs_out[i], eig_param);
	  
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  h_evals_[i] = blas::cDotProduct(*d_vecs_out[i], *r)/sqrt(blas::norm2(*d_vecs_out[i]));

	  //Convergence check ||A * v_i - lambda_i * v_i||
	  blas::caxpby(h_evals_[i], *d_vecs_out[i], -1.0, *r);
	  residual[i] = sqrt(blas::norm2(*r));
	  //printfQuda("Residual %d = %.6e\n", i, residual[i]);
	  if(i < nEv && residual[i] < tol) {
	    n_conv++;
	  }
	  if(n_conv == nEv) {
	    converged = true;
	  }
	}
	time += clock();
	time_c += time;
	num_converged = n_conv;
	
	for(int i=0; i<check; i++) {
	  delete d_vecs[i];
	  delete d_vecs_out[i];
	}
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
    Float total = (time_e + time_c + time_dt + time_ls + time_mb)/CLOCKS_PER_SEC;
    printfQuda("END LANCZOS SOLUTION\n");
    printfQuda("Time to solve problem using Lanczos = %e\n", t2/CLOCKS_PER_SEC);
    printfQuda("Time spent in all the regions       = %e\n", (time_e + time_c + time_dt + time_ls + time_mb)/CLOCKS_PER_SEC);    
    printfQuda("Time spent using EIGEN              = %e  %.1f%%\n", time_e/CLOCKS_PER_SEC, 100*(time_e/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in data transfer         = %e  %.1f%%\n", time_dt/CLOCKS_PER_SEC, 100*(time_dt/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in lanczos step          = %e  %.1f%%\n", time_ls/CLOCKS_PER_SEC, 100*(time_ls/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in multi blas            = %e  %.1f%%\n", time_mb/CLOCKS_PER_SEC, 100*(time_mb/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in convergence check     = %e  %.1f%%\n", time_c/CLOCKS_PER_SEC, 100*(time_c/CLOCKS_PER_SEC)/total);
    //-----------------------------

    //Local clean-up
    //--------------
    delete r;
    for(int i=0; i<nKr; i++) {
      delete kSpace[i];
    }
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
  

  template<typename Float>
  void irlm_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		  QudaEigParam *eig_param,
		  ColorSpinorParam *cpuParam){
    
    // Preliminaries: Memory allocation, local variables.
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    Float tol = eig_param->tol;    

    printfQuda("\n\nSTART IRLM SOLUTION\n");
    printfQuda("Nev requested = %d\n", nEv);
    printfQuda("NKrylov space = %d\n", nKr);
    
    Float *mod_evals_sorted = (Float*)malloc(nKr*sizeof(Float));
    Float *residual = (Float*)malloc(nKr*sizeof(Float));
    Float *residual_old = (Float*)malloc(nKr*sizeof(Float));
    int *evals_sorted_idx = (int*)malloc(nKr*sizeof(int));
    
    //Tridiagonal matrix
    Float alpha[nKr];
    Float  beta[nKr];
    for(int i=0; i<nKr; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }

    //Tracks if an eigenpair is locked
    bool *locked = (bool*)malloc((nKr)*sizeof(bool));
    for(int i=0; i<nKr; i++) locked[i] = false;

    //Tracks if an eigenvalue changes
    bool *changed = (bool*)malloc((nKr)*sizeof(bool));
    for(int i=0; i<nKr; i++) changed[i] = true;
    
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

    //Create Krylov space on the device
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr; i++) {
      kSpace.push_back(cudaColorSpinorField::Create(cudaParam));
    }

    //Host side dense matrix of Q eigenvectors
    Float **Qmat_h = (Float**)malloc((nKr)*sizeof(Float*));
    for(int i=0; i<nKr; i++) {
      Qmat_h[i] = (Float*)malloc((nKr)*sizeof(Float));
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
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------



    
    // START IRLM
    // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------

    Float t1 = clock();
    bool converged = false;
    int num_converged = 0;
    int restart_iter = 0;
    int max_restarts = eig_param->max_restarts;
    
    int k=0;
    int check_interval = eig_param->check_interval;

    double time;
    double time_e = 0.0;
    double time_c = 0.0;
    double time_ls = 0.0;
    double time_dt = 0.0;
    double time_mb = 0.0;

    //Device side linalg for Lanczos
    std::vector<ColorSpinorField*> d_vecs_nkr;
    for(int i=0; i<nKr; i++) {
      d_vecs_nkr.push_back(cudaColorSpinorField::Create(cudaParam));
    }
    std::vector<ColorSpinorField*> d_vecs_nkr_out;
    for(int i=0; i<nKr; i++) {
      d_vecs_nkr_out.push_back(cudaColorSpinorField::Create(cudaParam));
    }
    std::vector<ColorSpinorField*> d_vecs_nev;
    for(int i=0; i<nEv; i++) {
      d_vecs_nev.push_back(cudaColorSpinorField::Create(cudaParam));
    }
    std::vector<ColorSpinorField*> d_vecs_nev_out;
    for(int i=0; i<nEv; i++) {
      d_vecs_nev_out.push_back(cudaColorSpinorField::Create(cudaParam));
    }
    
    //Initial k step factorisation
    time = -clock();      
    for(k=0; k<nEv; k++) lanczosStep<Float>(mat, kSpace, *r, eig_param, alpha, beta, k);
    time += clock();
    time_ls += time;
        
    //Loop over restart iterations.
    while(restart_iter < max_restarts) {
      
      printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter);
      
      time = -clock();      
      for(k=nEv; k<nKr; k++) lanczosStep<Float>(mat, kSpace, *r, eig_param, alpha, beta, k);
      time += clock();
      time_ls += time;
      
      time = -clock();	
      //Compute the Tridiagonal matrix T_{k,k} 
      using Eigen::MatrixXd;
      MatrixXd triDiag = MatrixXd::Zero(nKr, nKr);
      
      for(int i=0; i<nKr; i++) {
	triDiag(i,i) = alpha[i];      
	if(i<nKr-1) {	
	  triDiag(i+1,i) = beta[i];	
	  triDiag(i,i+1) = beta[i];
	}
      }

      //Eigensolve the T_k matrix
      //Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
      //Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
      //eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());
      Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD(triDiag);

      for(int a=0; a<nKr; a++) printfQuda("TD eval %d = %e\n", a, eigenSolverTD.eigenvalues()[a]);
					  
      // (5) Initialise Q
      MatrixXd Q  = MatrixXd::Identity(nKr, nKr);
      MatrixXd Qj = MatrixXd::Zero(nKr, nKr);
      MatrixXd TmMuI;

      // (6) QR rotate the tridiag
      for(int j=nEv; j<nKr; j++) {
	
	//Apply the shift \mu_j
	TmMuI = triDiag;
	MatrixXd Id = MatrixXd::Identity(nKr, nKr);;
	TmMuI -= eigenSolverTD.eigenvalues()[j]*Id;

	// (7) QR decomposition of Tm - \mu_i I
	Eigen::HouseholderQR<MatrixXd> QR(TmMuI);

	// (8) Retain Qj matrices as a product.
	Qj = QR.householderQ();
	Q = Q * Qj;

	// (8) Update the Tridiag
	triDiag = Qj.adjoint() * triDiag;
	triDiag = triDiag * Qj;
      }
      time += clock();
      time_e += time;       
	
      Complex *Qmat_d = new Complex[nEv*nKr];	
      //Place data in accessible array
      for(int i=0; i<nEv; i++) {
	for(int j=0; j<nKr; j++) {
	  //DMH Careful: Order
	  Qmat_d[j*nEv + i].real(Q.col(i)[j]);
	  Qmat_d[j*nEv + i].imag(0.0);
	  //printfQuda("(%d,%d) %f\n", i, j, Q.col(i)[j]);
	}
      }

      //Use block basis rotation
      time = -clock();
      for(int i=0; i<nEv; i++) blas::zero(*d_vecs_nev_out[i]);
      blas::caxpy(Qmat_d, kSpace, d_vecs_nev_out);
      //Copy back to Krylov space array
      for(int i=0; i<nEv; i++) *kSpace[i] = *d_vecs_nev_out[i];      
      time += clock();
      time_mb += time;
      
      for(int a=0; a<nKr; a++)
	printfQuda("Post Rotation norm %d = %.6e\n", a, sqrt(blas::norm2(*kSpace[a])));
      
      //Update the residual
      //      r_{nev} = r_{nkv} * \sigma_{nev}  |  \sigma_{nev} = Q(nkv,nev)
      blas::ax(Q(nKr-1,nEv-1), *r);
      //      r_{nev} += v_{nev+1} * beta_k  |  beta_k = Tm(nev+1,nev)
      blas::axpy(triDiag(nEv,nEv-1), *kSpace[nEv], *r);

      //Construct the new starting vector
      double beta_k = sqrt(blas::norm2(*r));
      blas::zero(*kSpace[nEv]);
      blas::axpy(1.0/beta_k, *r, *kSpace[nEv]);
      beta[nEv-1] = beta_k;

      //Update the tridiag matrix
      triDiag(nEv-1,nEv) = beta_k;
      triDiag(nEv,nEv-1) = beta_k;
	
      if((restart_iter+1)%check_interval == 0) {

	printf("Convergence Check at restart iter %04d\n", restart_iter+1);

	//Construct the new Tridiagonal matrix for convergence check
	MatrixXd triDiagNew = MatrixXd::Identity(nEv, nEv);
	for(int i=0; i<nEv; i++) {
	  triDiagNew(i,i) = triDiag(i,i);
	  if(i<nEv-1) {
	    triDiagNew(i+1,i) = triDiag(i+1,i);
	    triDiagNew(i,i+1) = triDiag(i,i+1);
	  }
	}

	//Wrong size matrix?
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTDnew(triDiagNew);	
	Complex *Qmat_d_nev = new Complex[nEv*nEv];
	//Place data in accessible array
	for(int i=0; i<nEv; i++) {
	  for(int j=0; j<nEv; j++) {
	    Qmat_d_nev[i*nEv + j].real(eigenSolverTDnew.eigenvectors().col(i)[j]);
	    Qmat_d_nev[i*nEv + j].imag(0.0);
	    //printfQuda("(%d,%d) %f\n", i, j, Qmat_h[i][j]);
	  }
	}
	
	//Basis rotation
	for(int i=0; i<nEv; i++) {
	  *d_vecs_nev[i] = *kSpace[i];
	  blas::zero(*d_vecs_nev_out[i]);
	}
	blas::caxpy(Qmat_d_nev, d_vecs_nev, d_vecs_nev_out);
	
	//Check for convergence
	int n_conv = 0;
	//Error estimates given by ||M*q_tilde - lambda q_tilde||
	time = -clock();
	for(int i=0; i<nEv; i++) {
	    
	  //r = A * v_i
	  matVec<Float>(mat, *r, *d_vecs_nev_out[i], eig_param);	    
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  h_evals_[i] = blas::cDotProduct(*d_vecs_nev_out[i], *r)/sqrt(blas::norm2(*d_vecs_nev_out[i]));

	  //Convergence check ||A * v_i - lambda_i * v_i||
	  //--------------------------------------------------------------------------
	  blas::caxpby(h_evals_[i], *d_vecs_nev_out[i], -1.0, *r);
	  residual[i] = sqrt(blas::norm2(*r));
	  printfQuda("Residual %d = %.6e\n", i, residual[i]);
	  printfQuda("Eval %d = %.6e\n", i, h_evals_[i]);
	  printfQuda("Norm %d = %.6e\n", i, sqrt(blas::norm2(*d_vecs_nev_out[i])));

	  if(residual[i] < residual_old[i]) {
	    changed[i] = true;
	    printfQuda("Change hit %d\n", i);
	  } else changed[i] = false;
	  //Copy over residual
	  residual_old[i] = residual[i];
	  
	  //if(i < nEv && residual[i] < tol) n_conv++;
	  //if(n_conv == nEv) converged = true;
	  //--------------------------------------------------------------------------
	}
	
	for(int i=0; i<nEv; i++) {
	  if(residual[i] < tol && !locked[i]) {
	    
	    //This is a new locked eigenpair
	    num_converged++;
	      
	    //Lock it
	    locked[i] = true;
	    printf("%04d locking converged eigenvalue = %.8e resid %+.8e\n",
		   i, fabs(h_evals_[i]), residual[i]);
	  }
	}

	bool haltTest = true;
	for(int i=0; i<nEv; i++) {
	  //If no eigenvalue was improved in the last iteration,
	  //the algorithm halts. It's not going to improve any further.
	  if(changed[i] == true)
	    haltTest = false;
	}

	if(haltTest == true) {
	  printf("Orthogonality breakdown at restart iter %d "
		 "with %d eigenpairs converged.\n", restart_iter, num_converged);
	  //exit(0);
	}
	
	time += clock();
	time_c += time;
	
      }

      //Update the triDiag
      for(int i=0; i<nEv; i++) {
	alpha[i] = triDiag(i,i);
	if(i < nEv-1) beta[i] = triDiag(i,i+1);
      }
      
      restart_iter++;
    }
    
    for(int i=0; i<nKr; i++) {
      delete d_vecs_nkr[i];
      delete d_vecs_nkr_out[i];
    }
    for(int i=0; i<nEv; i++) {
      delete d_vecs_nev[i];
      delete d_vecs_nev_out[i];
    }    
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    

    //Post computation information
    //----------------------------
    printfQuda("END IRLM SOLUTION\n");
    if(!converged) {
      printf("IRLM failed to compute the requested %d vectors with a %d Krylov space in %d restart steps. "
	     "Please either increase nKr decrease nEv, or extend the number of restarts\n", nEv, nKr, max_restarts);
    } else {
      printf("IRLM computed the requested %d vectors in %d restarts steps of size %d\n", nEv, restart_iter, nKr);
      for(int i=0; i<nEv; i++) {
	printf("EigValue[%04d]: (%+.8e, %+.8e) residual %+.8e\n",
	       i, h_evals_[i].real(), h_evals_[i].imag(), residual[i]);
      }
    }
    
    Float t2 = clock() - t1;
    Float total = (time_e + time_c + time_dt + time_ls + time_mb)/CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using IRLM = %e\n", t2/CLOCKS_PER_SEC);
    printfQuda("Time spent in all the regions    = %e\n", (time_e + time_c + time_dt + time_ls + time_mb)/CLOCKS_PER_SEC);    
    printfQuda("Time spent using EIGEN           = %e  %.1f%%\n", time_e/CLOCKS_PER_SEC, 100*(time_e/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in data transfer      = %e  %.1f%%\n", time_dt/CLOCKS_PER_SEC, 100*(time_dt/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in lanczos step       = %e  %.1f%%\n", time_ls/CLOCKS_PER_SEC, 100*(time_ls/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in multi blas         = %e  %.1f%%\n", time_mb/CLOCKS_PER_SEC, 100*(time_mb/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in convergence check    %e  %.1f%%\n", time_c/CLOCKS_PER_SEC, 100*(time_c/CLOCKS_PER_SEC)/total);
    //-----------------------------

    //Local clean-up
    //--------------
    delete r;
    for(int i=0; i<nKr; i++) {
      delete kSpace[i];
    }
    //--------------
  }

  void irlmSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		 QudaEigParam *eig_param,
		 ColorSpinorParam *cpuParam){
    
    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      irlm_solve<double>(h_evecs, h_evals, mat, eig_param, cpuParam);
    } else {
      irlm_solve<float>(h_evecs, h_evals, mat, eig_param, cpuParam);
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

