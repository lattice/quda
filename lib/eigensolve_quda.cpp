#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include <quda_internal.h>
#include <quda_arpack_interface.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <Eigen/Dense>

namespace quda {

  template<typename Float>
  void matVec(const Dirac &mat,
	      ColorSpinorField &out,
	      const ColorSpinorField &in,
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
  
  template<typename Float>
  void matVecOp(const Dirac &mat,
		ColorSpinorField &out,
		const ColorSpinorField &in,
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
    blas::caxpby(d2, const_cast<ColorSpinorField&>(in), d1, out);
    if(eig_param->poly_deg == 1) return;

    
    // C_0 is the current 'in'  vector.
    // C_1 is the current 'out' vector.
       
    //Clone 'in' to two temporary vectors.
    ColorSpinorField *tmp1 = ColorSpinorField::Create(in);
    ColorSpinorField *tmp2 = ColorSpinorField::Create(in);

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

  //Orthogonalise r against V_[j]
  template<typename Float>
  void orthogonalise(std::vector<ColorSpinorField*> v,
		     std::vector<ColorSpinorField*> r,
		     int j) {
    
    std::complex<Float> s(0.0,0.0);
    for(int i=0; i<j; i++) {
      s = blas::cDotProduct(*v[i], *r[0]);
      blas::caxpy(-s, *v[i], *r[0]);      
    }
  }

  
  //Orthogonalise r against V_[j]
  template<typename Float>
  void blockOrthogonalise(std::vector<ColorSpinorField*> v,
			  std::vector<ColorSpinorField*> r,
			  int j) {

    Complex *s = new Complex[j];
    std::vector<ColorSpinorField*> d_vecs_j;
    for(int i=0; i<j; i++) {
      d_vecs_j.push_back(v[i]);
    }
    //Block dot products stored in s.
    blas::cDotProduct(s, d_vecs_j, r);
    
    //Block orthonormalise
    for(int i=0; i<j; i++) s[i] *= -1.0;
    blas::caxpy(s, d_vecs_j, r);
    
  }
    
  template<typename Float>
  void lanczosStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   std::vector<ColorSpinorField*> r,
		   QudaEigParam *eig_param,
		   Float *alpha, Float *beta, int j) {

    //Compute r = A * v_j - b_{j-i} * v_{j-1}      
    //r = A * v_j
    //DMH: maybe enforce MdagM mat vec? Defo do a polyOp.
    matVecOp<Float>(mat, *r[0], *v[j], eig_param);
    
    //r = r - b_{j-1} * v_{j-1}
    if(j>0) blas::axpy(-beta[j-1], *v[j-1], *r[0]);      
    
    //a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v[j], *r[0]);    

    //r = r - a_j * v_j
    blas::axpy(-alpha[j], *v[j], *r[0]);

    //b_j = ||r|| 
    beta[j] = sqrt(blas::norm2(*r[0]));

    //Orthogonalise
    if(beta[j] < (1.0)*sqrt(alpha[j]*alpha[j] + beta[j-1]*beta[j-1]) && j%2 == 0) {

      //The residual vector r has been deemed insufficiently
      //orthogonal to the existing Krylov space. We must
      //orthogonalise it.
      //printfQuda("orthogonalising Beta %d = %e\n", j, beta[j]);
      //orthogonalise<Float>(v, r, j);
      blockOrthogonalise<Float>(v, r, j);
      //b_j = ||r|| 
      beta[j] = sqrt(blas::norm2(*r[0]));    
    }

    //Prepare next step.
    //v_{j+1} = r / b_j
    if(j < v.size() - 1) {
      blas::zero(*v[j+1]);
      blas::axpy(1.0/beta[j], *r[0], *v[j+1]);
    }
  }
  /*
  template<typename Float>
  void arnoldiStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   ColorSpinorField &r,
		   QudaEigParam *eig_param,
		   Float *h, int j) {
    
    //Compute r = A * v_j - b_{j-i} * v_{j-1}      
    //r = A * v_j
    matVecOp<Float>(mat, r, *v[j], eig_param);
    
    std::complex<Float> s(0.0,0.0);
    for(int i=0; i<j; i++) {
      double norm = sqrt(blas::norm2(*v[i]));
      blas::ax(1.0/norm, *v[i]);
      s = blas::cDotProduct(*v[i], r);
      blas::caxpy(-s, *v[i], r);      
    }    

    //Prepare next step.
    //v_{j+1} = r / b_j
    if(j < v.size() - 1) {
      blas::zero(*v[j+1]);
      blas::axpy(1.0/beta[j], r, *v[j+1]);
    }
  }
  */
  
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
    ColorSpinorParam cudaParam(*cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    sizeof(Float) == sizeof(double) ?
      cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER :
      cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr; i++) {
      kSpace.push_back(ColorSpinorField::Create(cudaParam));
    }
    //Here we create a vector of one element. In future,
    //we may wish to make a block eigensolver, so a
    //vector structure will be needed.
    std::vector<ColorSpinorField*> r;
    r.push_back(ColorSpinorField::Create(cudaParam));
    
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
    double time;
    double time_e = 0.0;
    double time_c = 0.0;
    double time_ls = 0.0;
    double time_dt = 0.0;
    double time_mb = 0.0;
    
    while(k < nKr && !converged) {

      time = -clock();      
      lanczosStep<Float>(mat, kSpace, r, eig_param, alpha, beta, k);
      time += clock();
      time_ls += time;
      
      if((k+1)%check_interval == 0 && k > nEv) {
	
	int check = k+1;
	printfQuda("%04d converged eigenvalues at iter %04d\n", num_converged, check);
	
	time = -clock();
	//Device side linalg for Lanczos
	std::vector<ColorSpinorField*> d_vecs;
	for(int i=0; i<check; i++) {
	  d_vecs.push_back(ColorSpinorField::Create(cudaParam));
	}
	std::vector<ColorSpinorField*> d_vecs_out;
	for(int i=0; i<check; i++) {
	  d_vecs_out.push_back(ColorSpinorField::Create(cudaParam));
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

	//for(int a=0; a<check; a++) printfQuda("TD eval %d = %e\n", a, eigenSolverTD.eigenvalues()[a]);
	
	Complex *Qmat_d = new Complex[check*check];
	
	//Place data in accessible array
	for(int i=0; i<check; i++) {
	  for(int j=0; j<check; j++) {
	    Qmat_d[j*check + i].real(eigenSolverTD.eigenvectors().col(i)[j]);
	    Qmat_d[j*check + i].imag(0.0);
	    //printfQuda("(%d,%d) %f\n", i, j, Qmat_d[i][j]);
	  }
	}

	time += clock();
	time_e += time;
	
	//Use block basis rotation
	time = -clock();
	blas::caxpy(Qmat_d, d_vecs, d_vecs_out);
	time += clock();
	time_mb += time;

	//for(int a=0; a<check; a++) printfQuda("Post Rotation norm %d = %.6e\n", a, sqrt(blas::norm2(*d_vecs_out[a])));
	
	//Check for convergence
	int n_conv = 0;
	//Error estimates given by ||M*q_tilde - lambda q_tilde||
	time = -clock();
	for(int i=0; i<nEv; i++) {
	  
	  //r = A * v_i
	  matVec<Float>(mat, *r[0], *d_vecs_out[i], eig_param);
	  
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  h_evals_[i] = blas::cDotProduct(*d_vecs_out[i], *r[0])/sqrt(blas::norm2(*d_vecs_out[i]));

	  //Convergence check ||A * v_i - lambda_i * v_i||
	  blas::caxpby(h_evals_[i], *d_vecs_out[i], -1.0, *r[0]);
	  residual[i] = sqrt(blas::norm2(*r[0]));
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
    delete r[0];
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
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    Float tol = eig_param->tol;    
    bool inverse = false;
    
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
    ColorSpinorParam cudaParam(*cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    sizeof(Float) == sizeof(double) ?
      cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER :
      cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr; i++) {
      kSpace.push_back(ColorSpinorField::Create(cudaParam));
    }
    //Here we create a vector of one element. In future,
    //we may wish to make a block eigensolver, so a
    //vector structure will be needed.
    std::vector<ColorSpinorField*> r;
    r.push_back(ColorSpinorField::Create(cudaParam));

    //Device side linalg space for Lanczos
    std::vector<ColorSpinorField*> d_vecs_nev_out;
    for(int i=0; i<nEv; i++) {
      d_vecs_nev_out.push_back(ColorSpinorField::Create(cudaParam));
    }

    //Part of the spectrum to be computed.
    char *spectrum;
    spectrum = strdup("SR"); //Initialsed just to stop the compiler warning...

    if(eig_param->use_poly_acc){
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM) spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM) spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM) spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM) spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM) spectrum = strdup("LI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM) spectrum = strdup("SI");
    }
    else{
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM) spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM) spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM) spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM) spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM) spectrum = strdup("SI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM) spectrum = strdup("LI");
    }

    //Lanczos gives real eigenvales only.
    const char *L = "L";
    const char *S = "S";
    if(strncmp(L, spectrum, 1) == 0 && !eig_param->use_poly_acc) {
      inverse = true;
    } else if(strncmp(S, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      inverse = true;
    } else if(strncmp(L, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      inverse = true;
    }    
    //---------------------------------------------------------------------------



    // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------
    
    Float t1 = clock();
    bool converged = false;
    bool ortho = true;
    int num_converged = 0;
    int restart_iter = 0;
    int max_restarts = eig_param->max_restarts;
    
    int k=0;
    int check_interval = eig_param->check_interval;

    double time;
    double time_e = 0.0;   //time in Eigen (host)
    double time_c = 0.0;   //time in convergence check
    double time_ls = 0.0;  //time in Lanczos step
    double time_mb = 0.0;  //time in multiblas

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
    
    //Initial k step factorisation
    time = -clock();      
    for(k=0; k<nEv; k++) lanczosStep<Float>(mat, kSpace, r, eig_param, alpha, beta, k);
    time += clock();
    time_ls += time;

    
    //Loop over restart iterations.
    //---------------------------------------------------------------------------
    while(restart_iter < max_restarts && !converged && ortho) {
      
      time = -clock();      
      for(k=nEv; k<nKr; k++) lanczosStep<Float>(mat, kSpace, r, eig_param, alpha, beta, k);
      time += clock();
      time_ls += time;

      
      //Compute the Tridiagonal matrix T_{k,k}
      //--------------------------------------
      time = -clock();
      using Eigen::MatrixXd;
      MatrixXd triDiag = MatrixXd::Zero(nKr, nKr);
      
      for(int i=0; i<nKr; i++) {
	triDiag(i,i) = alpha[i];
	//printfQuda("alpha[%d] = %e ", i, alpha[i]);
	if(i<nKr-1) {	
	  triDiag(i+1,i) = beta[i];	
	  triDiag(i,i+1) = beta[i];
	  //printfQuda("beta[%d] = %e", i, beta[i]);
	}
	//printfQuda("\n");
      }

      //Eigensolve the T_k matrix
      Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
      Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
      eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());

      //for(int a=0; a<nKr; a++) printfQuda("TD1 eval %d = %e\n", a, eigenSolverTD.eigenvalues()[a]);
					  
      // Initialise Q
      MatrixXd Q  = MatrixXd::Identity(nKr, nKr);
      MatrixXd Qj = MatrixXd::Zero(nKr, nKr);
      MatrixXd TmMuI;
      
      //DMH: This is where the LR/SR spectrum is projected.
      //     For SR, we project out the (nKr - nEv) LARGEST eigenvalues
      //     For LR, we project out the (nKr - nEv) SMALLEST eigenvalues
      //     Eigen returns Eigenvalues from Hermitian matrices in ascending order.
      //     We take advantage of this here, but be mindful that Arnoldi
      //     type solves will require some degree of sorting.
      Float evals[nKr-nEv];
      if(inverse) for(int j=0; j<(nKr-nEv); j++) evals[j] = eigenSolverTD.eigenvalues()[j];
      else for(int j=0; j<(nKr-nEv); j++) evals[j] = eigenSolverTD.eigenvalues()[j+nEv];

      //DMH: This can be batch parallelised on the GPU.
      //     Will be usefull when (nKr - nEv) is large.
      //     The N lots of QR decomp can be batched, and
      //     the Q_0 * Q_1 * ... * Q_{N-1} product
      //     can be done in pairs in parallel, like
      //     a sum reduction.
      //     (Q_0 * Q_1) * (Q_2 * Q_3) * ... * (Q_{N-2} * Q_{N-1})
      //     =  (Q_{0*1} * Q_{2*3}) * ... * (Q_{(N-2)(N-1)})
      //     ...
      //     =  Q_{0*1*2*3*...*(N-2)(N-1)}
      for(int j=0; j<(nKr-nEv); j++) {
	
	//Apply the shift \mu_j
	TmMuI = triDiag;
	MatrixXd Id = MatrixXd::Identity(nKr, nKr);;
	TmMuI -= evals[j]*Id;
	
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
      //--------------------------------------
      

      //Block basis rotation
      //--------------------------------------
      time = -clock();
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
      
      for(int i=0; i<nEv; i++) blas::zero(*d_vecs_nev_out[i]);
      blas::caxpy(Qmat_d, kSpace, d_vecs_nev_out);
      //Copy back to Krylov space array
      for(int i=0; i<nEv; i++) {
	*kSpace[i] = *d_vecs_nev_out[i];      
	//printfQuda("Post Rotation norm %d = %.6e\n", a, sqrt(blas::norm2(*kSpace[a])));
      }
	
      //Update the residual
      //      r_{nev} = r_{nkv} * \sigma_{nev}  |  \sigma_{nev} = Q(nkv,nev)
      blas::ax(Q(nKr-1,nEv-1), *r[0]);
      //      r_{nev} += v_{nev+1} * beta_k  |  beta_k = Tm(nev+1,nev)
      blas::axpy(triDiag(nEv,nEv-1), *kSpace[nEv], *r[0]);
      
      //Construct the new starting vector
      double beta_k = sqrt(blas::norm2(*r[0]));
      blas::zero(*kSpace[nEv]);
      blas::axpy(1.0/beta_k, *r[0], *kSpace[nEv]);
      beta[nEv-1] = beta_k;
      
      time += clock();
      time_mb += time;
      //--------------------------------------


      //Convergence check
      //--------------------------------------
      if((restart_iter+1)%check_interval == 0) {

	time = -clock();
	
	//Construct the nEv,nEv Tridiagonal matrix for convergence check
	MatrixXd triDiagNew = MatrixXd::Identity(nEv, nEv);
	for(int i=0; i<nEv; i++) {
	  triDiagNew(i,i) = triDiag(i,i);
	  //printfQuda("alphaNew[%d] = %e ", i, triDiagNew(i,i));
	  if(i<nEv-1) {
	    triDiagNew(i+1,i) = triDiag(i+1,i);
	    triDiagNew(i,i+1) = triDiag(i,i+1);
	    //printfQuda("betaNew[%d] = %e", i, triDiagNew(i,i+1));
	  }
	  //printfQuda("\n");
	}
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTDnew(triDiagNew);
	
	//for(int a=0; a<nEv; a++) printfQuda("TDc eval %d = %e\n", a, eigenSolverTDnew.eigenvalues()[a]);
	
	Complex *Qmat_d_nev = new Complex[nEv*nEv];
	//Place data in accessible array
	for(int i=0; i<nEv; i++) {
	  for(int j=0; j<nEv; j++) {
	    Qmat_d_nev[j*nEv + i].real(eigenSolverTDnew.eigenvectors().col(i)[j]);
	    Qmat_d_nev[j*nEv + i].imag(0.0);
	    //printfQuda("(%d,%d) %f\n", i, j, Qmat_d_nev[i][j]);
	  }
	}
	
	time += clock();
	time_e += time;

	time = -clock();
	//Basis rotation
	std::vector<ColorSpinorField*> d_vecs_nev;
	for(int i=0; i<nEv; i++) {
	  d_vecs_nev.push_back(kSpace[i]);
	  blas::zero(*d_vecs_nev_out[i]);
	}
	
	blas::caxpy(Qmat_d_nev, d_vecs_nev, d_vecs_nev_out);
	time += clock();
	time_mb += time;
	
	//Check for convergence
	int n_conv = 0;
	//Error estimates given by ||M*q_tilde - lambda q_tilde||
	
	time = -clock();
	for(int i=0; i<nEv; i++) {
	    
	  //r = A * v_i
	  matVec<Float>(mat, *r[0], *d_vecs_nev_out[i], eig_param);	    
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  h_evals_[i] = blas::cDotProduct(*d_vecs_nev_out[i], *r[0])/sqrt(blas::norm2(*d_vecs_nev_out[i]));

	  //Convergence check ||lambda_i*v_i - A*v_i||
	  //--------------------------------------------------------------------------
	  Complex n_unit(-1.0,0.0);
	  blas::caxpby(h_evals_[i], *d_vecs_nev_out[i], n_unit, *r[0]);
	  residual[i] = sqrt(blas::norm2(*r[0]));
	  //printfQuda("Residual %d = %.6e\n", i, residual[i]);
	  //printfQuda("Eval %d = %.6e\n", i, h_evals_[i]);
	  //printfQuda("Norm %d = %.6e\n", i, sqrt(blas::norm2(*d_vecs_nev_out[i])));

	  //Copy over residual
	  residual_old[i] = residual[i];
	  //--------------------------------------------------------------------------
	}
	
	for(int i=0; i<nEv; i++) {
	  if(residual[i] < tol && !locked[i]) {
	    
	    //This is a new eigenpair.
	    //Lock it
	    locked[i] = true;
	    printf("**** locking converged eigenvalue = %.8e resid %+.8e ****\n",
		   i, fabs(h_evals_[i]), residual[i]);
	    
	    //Purge it
	    std::complex<Float> s(0.0,0.0);
	    s = blas::cDotProduct(*d_vecs_nev_out[i], *kSpace[nEv]);
	    blas::caxpy(-s, *d_vecs_nev_out[i], *kSpace[nEv]);      
	    
	    num_converged++;
	  }	  
	}

	printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter+1);
	//printf("Convergence Check at restart iter %04d\n", restart_iter+1);
	
	if(num_converged == nEv) converged = true;
	
	time += clock();
	time_c += time;
	
      }
      //--------------------------------------

      
      //Update the triDiag
      //--------------------------------------
      for(int i=0; i<nEv; i++) {
	alpha[i] = triDiag(i,i);
	if(i < nEv-1) beta[i] = triDiag(i,i+1);
      }
      triDiag(nEv-1,nEv) = beta_k;
      triDiag(nEv,nEv-1) = beta_k;
      
      restart_iter++;
    }
    //---------------------------------------------------------------------------
    

    //Post computation information
    //---------------------------------------------------------------------------
    printfQuda("END IRLM SOLUTION\n");
    if(!converged && ortho) {
      printf("IRLM failed to compute the requested %d vectors with a %d Krylov space in %d restart steps. "
	     "Please either increase nKr decrease nEv, or extend the number of restarts\n", nEv, nKr, max_restarts);
    } else if(!converged && !ortho) {
      printf("IRLM failed to compute the requested %d vectors with a %d Krylov space in %d restart steps "
	     "due to orthogonality breakdown\n", nEv, nKr, restart_iter+1);
    } else {
      printf("IRLM computed the requested %d vectors in %d restarts steps of size %d\n", nEv, restart_iter, (nKr-nEv));
      
      for(int i=0; i<nEv; i++) {
	if(inverse) {
	  printf("EigValue[%04d]: (%+.8e, %+.8e) residual %+.8e\n",
		 i, h_evals_[nEv-1-i].real(), h_evals_[nEv-1-i].imag(), residual[nEv-1-i]);
	} else {
	  printf("EigValue[%04d]: (%+.8e, %+.8e) residual %+.8e\n",
		 i, h_evals_[i].real(), h_evals_[i].imag(), residual[i]);
	}
      }
    }
    
    
    Float t2 = clock() - t1;
    Float total = (time_e + time_c + time_ls + time_mb)/CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using IRLM = %e\n", t2/CLOCKS_PER_SEC);
    printfQuda("Time spent in all the regions    = %e\n", (time_e + time_c + time_ls + time_mb)/CLOCKS_PER_SEC);    
    printfQuda("Time spent using EIGEN           = %e  %.1f%%\n", time_e/CLOCKS_PER_SEC, 100*(time_e/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in lanczos step       = %e  %.1f%%\n", time_ls/CLOCKS_PER_SEC, 100*(time_ls/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in multi blas         = %e  %.1f%%\n", time_mb/CLOCKS_PER_SEC, 100*(time_mb/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in convergence check  = %e  %.1f%%\n", time_c/CLOCKS_PER_SEC, 100*(time_c/CLOCKS_PER_SEC)/total);
    //---------------------------------------------------------------------------


    //Local clean-up
    //--------------
    delete r[0];
    for(int i=0; i<nEv; i++) {
      delete d_vecs_nev_out[i];
    }        
    for(int i=0; i<nKr; i++) {
      delete kSpace[i];
    }
    
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

  /*
  
  template<typename Float>
  void arnoldi_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		     QudaEigParam *eig_param,
		     ColorSpinorParam *cpuParam){

    // Preliminaries: Memory allocation, local variables.
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    Float tol = eig_param->tol;    

    printfQuda("\n\nSTART ARNOLDI SOLUTION\n");
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
    ColorSpinorField *r = nullptr;
    ColorSpinorParam cudaParam(*cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    sizeof(Float) == sizeof(double) ?
      cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER :
      cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;

    r = ColorSpinorField::Create(cudaParam);

    //Create Krylov space on the device
    std::vector<ColorSpinorField*> kSpace;
    for(int i=0; i<nKr; i++) {
      kSpace.push_back(ColorSpinorField::Create(cudaParam));
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



    
    // START ARNOLDI
    // ARNOLDI Method for Non-Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------

    //profile.TPSTART(QUDA_PROFILE_COMPUTE);
    Float t1 = clock();
    bool converged = false;
    int num_converged = 0;
    int k=0;
    int check_interval = eig_param->check_interval;
    double time;
    double time_e = 0.0;
    double time_c = 0.0;
    double time_ls = 0.0;
    double time_dt = 0.0;
    double time_mb = 0.0;
    
    while(k < nKr && !converged) {

      time = -clock();      
      //arnoldiStep<Float>(mat, kSpace, *r, eig_param, alpha, beta, k);
      time += clock();
      time_ls += time;
      
      if((k+1)%check_interval == 0 && k > nEv) {
	
	int check = k+1;
	printfQuda("%04d converged eigenvalues at iter %04d\n", num_converged, check);
	
	time = -clock();
	//Device side linalg for Lanczos
	std::vector<ColorSpinorField*> d_vecs;
	for(int i=0; i<check; i++) {
	  d_vecs.push_back(ColorSpinorField::Create(cudaParam));
	}
	std::vector<ColorSpinorField*> d_vecs_out;
	for(int i=0; i<check; i++) {
	  d_vecs_out.push_back(ColorSpinorField::Create(cudaParam));
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

	//for(int a=0; a<check; a++) printfQuda("TD eval %d = %e\n", a, eigenSolverTD.eigenvalues()[a]);
	
	Complex *Qmat_d = new Complex[check*check];
	
	//Place data in accessible array
	for(int i=0; i<check; i++) {
	  for(int j=0; j<check; j++) {
	    Qmat_d[j*check + i].real(eigenSolverTD.eigenvectors().col(i)[j]);
	    Qmat_d[j*check + i].imag(0.0);
	    //printfQuda("(%d,%d) %f\n", i, j, Qmat_d[i][j]);
	  }
	}

	time += clock();
	time_e += time;
	
	//Use block basis rotation
	time = -clock();
	blas::caxpy(Qmat_d, d_vecs, d_vecs_out);
	time += clock();
	time_mb += time;

	//for(int a=0; a<check; a++) printfQuda("Post Rotation norm %d = %.6e\n", a, sqrt(blas::norm2(*d_vecs_out[a])));
	
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
  
  void arnoldiSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam){
    
    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      arnoldi_solve<double>(h_evecs, h_evals, mat, eig_param, cpuParam);
    } else {
      arnoldi_solve<float>(h_evecs, h_evals, mat, eig_param, cpuParam);
    }    
  }
  */  
  
  // ARPACK INTERAFCE ROUTINES
  //--------------------------------------------------------------------------

#ifdef ARPACK_LIB
  
  void arpackErrorHelpNAUPD();
  void arpackErrorHelpNEUPD();

  template<typename Float>
  static void mergeAbs(Float *sort1, int *idx1, int n1, Float *sort2,
		       int *idx2, int n2, bool inverse) {
    int i1=0, i2=0;
    int *ord;
    Float *result;
    
    ord    = (int *)    malloc(sizeof(int)   *(n1+n2)); 
    result = (Float *) malloc(sizeof(Float)*(n1+n2)); 
    
    for(int i=0; i<(n1+n2); i++) {
      if((fabs(sort1[i1]) >= fabs(sort2[i2])) != inverse) { //LOGICAL XOR
	result[i] = sort1[i1];
	ord[i] = idx1[i1];
	i1++;
      } else {
	result[i] = sort2[i2];
	ord[i] = idx2[i2];
	i2++;
      }
      
      if(i1 == n1) {
	for(int j=i+1; j<(n1+n2); j++,i2++) {
	  result[j] = sort2[i2];
	  ord[j] = idx2[i2];
	}
	i = n1+n2;
      } else if (i2 == n2) {
	for(int j=i+1; j<(n1+n2); j++,i1++) {
	  result[j] = sort1[i1];
	  ord[j] = idx1[i1];
	}
	i = i1+i2;
      }
    }  
    for(int i=0;i<n1;i++) {
      idx1[i] = ord[i];
      sort1[i] = result[i];
    }
    
    for(int i=0;i<n2;i++) {
      idx2[i] = ord[i+n1];
      sort2[i] = result[i+n1];
    }  
    free (ord);
    free (result);
  }
  
  template<typename Float>
  static void sortAbs(Float *unsorted, int n, bool inverse, int *idx) {

    if (n <= 1)
      return;
    
    int n1,n2;
    
    n1 = n>>1;
    n2 = n-n1;
    
    Float *unsort1 = unsorted;
    Float *unsort2 = (Float *)((char*)unsorted + n1*sizeof(Float));
    int *idx1 = idx;
    int *idx2 = (int *)((char*)idx + n1*sizeof(int));
    
    sortAbs<Float>(unsort1, n1, inverse, idx1);
    sortAbs<Float>(unsort2, n2, inverse, idx2);
    
    mergeAbs<Float>(unsort1, idx1, n1, unsort2, idx2, n2, inverse);
  }
  
  void arpack_solve_double(void *h_evecs, void *h_evals,
			   const Dirac &mat,
			   QudaEigParam *eig_param,
			   ColorSpinorParam *cpuParam){
    
    //Construct parameters and memory allocation
    //---------------------------------------------------------------------------------
    
    //MPI objects
    int *fcomm_ = nullptr;
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm_ = static_cast<int*>(&mpi_comm_fort);
#endif

    //Determine local volume for memory allocations
    int local_dim[4];
    int local_vol = 1;
    for(int i = 0 ; i < 4 ; i++){
      local_dim[i] = cpuParam->x[i];
      local_vol *= local_dim[i];
    }
    
    // all FORTRAN communication uses underscored 
    int ido_ = 0; 
    int info_ = 1; //if 0, use random vector. If 1, initial residulal lives in resid_
    int *ipntr_ = (int*)malloc(14*sizeof(int));
    int *iparam_ = (int*)malloc(11*sizeof(int));
    int n_    = local_vol*4*3,
      nEv_    = eig_param->nEv,
      nKr_    = eig_param->nKr,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nKr_*nKr_ + 5*nKr_) * 2,
      rvec_   = 1;
    int max_iter = eig_param->max_restarts*(nKr_-nEv_) + nEv_;
    int *h_evals_sorted_idx = (int*)malloc(nKr_*sizeof(int));
        
    //ARPACK logfile name
    //char *arpack_logfile = eig_param->arpack_logfile;
    
    //Assign values to ARPACK params 
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
 
    //ARPACK problem type to be solved
    char howmny='P';
    char bmat = 'I';
    char *spectrum;
    spectrum = strdup("SR"); //Initialsed just to stop the compiler warning...

    if(eig_param->use_poly_acc){
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM) spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM) spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM) spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM) spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM) spectrum = strdup("LI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM) spectrum = strdup("SI");
    }
    else{
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM) spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM) spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM) spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM) spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM) spectrum = strdup("SI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM) spectrum = strdup("LI");
    }
   
    double tol_ = eig_param->tol;
    double *mod_h_evals_sorted  = (double*)malloc(nKr_*sizeof(double));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }

    //ARPACK workspace
    //Initial guess?
    //Complex I(0.0,1.0);
    Complex I(0.0,1.0);
    Complex *resid_ =
      (Complex *) malloc(ldv_*sizeof(Complex));
    
    if(info_ > 0) 
      for(int a = 0; a<ldv_; a++) {
	resid_[a] = I;
	//printfQuda("(%e , %e)\n", real(resid_[a]), imag(resid_[a]));
      }
    
    Complex sigma_ = 0.0;
    Complex *w_workd_ =
      (Complex *) malloc(3*ldv_*sizeof(Complex));
    Complex *w_workl_ =
      (Complex *) malloc(lworkl_*sizeof(Complex)); 
    Complex *w_workev_=
      (Complex *) malloc(2*nKr_*sizeof(Complex));    
    double *w_rwork_  = (double *)malloc(nKr_*sizeof(double));    
    int *select_ = (int*)malloc(nKr_*sizeof(int));

    //Alias pointers
    Complex *h_evecs_ = nullptr;
    h_evecs_ = (Complex*) (double*)(h_evecs);    
    Complex *h_evals_ = nullptr;
    h_evals_ = (Complex*) (double*)(h_evals);

    //Memory checks
    if((iparam_ == nullptr) ||
       (ipntr_ == nullptr) || 
       (resid_ == nullptr) ||  
       (w_workd_ == nullptr) || 
       (w_workl_ == nullptr) ||
       (w_workev_ == nullptr) ||
       (w_rwork_ == nullptr) || 
       (select_ == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for ARPACK workspace.\n");
    }    

    int iter_count= 0;

    bool allocate = true;
    ColorSpinorField *h_v = nullptr;
    ColorSpinorField *d_v = nullptr;    
    ColorSpinorField *h_v2 = nullptr;
    ColorSpinorField *d_v2 = nullptr;
    ColorSpinorField *resid = nullptr;    

    //ARPACK log routines
    // Code added to print the log of ARPACK  
    int arpack_log_u = 9999;
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    /*
    if ( arpack_logfile != NULL  && (comm_rank() == 0) ) {
      // correctness of this code depends on alignment in Fortran and C 
      // being the same ; if you observe crashes, disable this part 
      ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
      int msglvl0 = 0, msglvl3 = 3;
      ARPACK(pmcinitdebug)(&arpack_log_u,      //logfil
			   &msglvl3,           //mcaupd
			   &msglvl3,           //mcaup2
			   &msglvl0,           //mcaitr
			   &msglvl3,           //mceigh
			   &msglvl0,           //mcapps
			   &msglvl0,           //mcgets
			   &msglvl3            //mceupd
			   );
      
      printfQuda("eigenSolver: Log info:\n");
      printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
      printfQuda("output is directed to %s\n",arpack_logfile);
    }
#else
    if (arpack_logfile != NULL) {
      // correctness of this code depends on alignment in Fortran and C 
      // being the same ; if you observe crashes, disable this part 
      
      ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
      int msglvl0 = 0, msglvl3 = 3;
      ARPACK(mcinitdebug)(&arpack_log_u,      //logfil
			  &msglvl3,           //mcaupd
			  &msglvl3,           //mcaup2
			  &msglvl0,           //mcaitr
			  &msglvl3,           //mceigh
			  &msglvl0,           //mcapps
			  &msglvl0,           //mcgets
			  &msglvl3            //mceupd
			  );
      
      printfQuda("eigenSolver: Log info:\n");
      printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
      printfQuda("output is directed to %s\n", arpack_logfile);
    }
    */
#endif   
    
    //Start ARPACK routines
    //---------------------------------------------------------------------------------

    double t1;
    
    do {

      t1 = -((double)clock());

      //printfQuda("flag x1\n");
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pznaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nEv_, &tol_,
		      resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
      }
#else
      //printfQuda("flag x2\n");
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_, &nKr_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_, 1, 2);
      //printfQuda("flag x3\n");
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in znaupd info = %d. Exiting.",info_);
      }
#endif
      
      //If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if(allocate){

	//Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
	//less one, hence ipntr[0] - 1 to specify the correct address.

	cpuParam->location = QUDA_CPU_FIELD_LOCATION;
	cpuParam->create = QUDA_REFERENCE_FIELD_CREATE;
	cpuParam->gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	
	cpuParam->v = w_workd_ + (ipntr_[0] - 1);
	h_v = ColorSpinorField::Create(*cpuParam);
	//Adjust the position of the start of the array.
	cpuParam->v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = ColorSpinorField::Create(*cpuParam);
	
	ColorSpinorParam cudaParam(*cpuParam);
	cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
	
	d_v = ColorSpinorField::Create(cudaParam);
	d_v2 = ColorSpinorField::Create(cudaParam);
	resid = ColorSpinorField::Create(cudaParam);
	allocate = false;

      }
      //printfQuda("flag x5\n");
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {
	//printfQuda("flag x6\n");
	*d_v = *h_v;

	printfQuda("flag 1\n");
	//apply matrix-vector operation here:
	matVecOp<double>(mat, *d_v2, *d_v, eig_param);
	printfQuda("flag 2\n");
	
	*h_v2 = *d_v2;
	
      }

      t1 += clock();
	
      printfQuda("Arpack Iteration %s: %d (%e secs)\n", eig_param->use_poly_acc ? "(with poly acc) " : "", iter_count, t1/CLOCKS_PER_SEC);
      iter_count++;
      
    } while (99 != ido_ && iter_count < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors and values 
    
    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_count, info_, ido_);      
    printfQuda("Computing eigenvectors\n");
    
    //Interface to arpack routines
    //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    ARPACK(pzneupd)(fcomm_, &rvec_, &howmny, select_, h_evals_, h_evecs_,
		    &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nEv_,
		    &tol_, resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		    w_workl_, &lworkl_, w_rwork_ ,&info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in pzneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n",info_);
#else      
    ARPACK(zneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nEv_, &tol_,
		   resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif

    // Print additional convergence information.
    if( (info_) == 1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info_ == 3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing nKr.\n");
      }
    }
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    /*
    if(comm_rank() == 0){
      if (arpack_logfile != NULL){
	ARPACK(finilog)(&arpack_log_u);
      }
    }
#else
    if (arpack_logfile != NULL)
      ARPACK(finilog)(&arpack_log_u);
    */
#endif     
    
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      h_evals_sorted_idx[j] = j;
      mod_h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //Sort the eigenvalues in absolute ascending order
    t1 =  -((double)clock());
    bool inverse = true;
    const char *L = "L";
    const char *S = "S";
    if(strncmp(L, spectrum, 1) == 0 && !eig_param->use_poly_acc) {
      inverse = false;
    } else if(strncmp(S, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      inverse = false;
    } else if(strncmp(L, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      inverse = false;
    }

    sortAbs<double>(mod_h_evals_sorted, nconv, inverse, h_evals_sorted_idx);

    printfQuda("Sorted eigenvalues based on their absolute values:\n");

    //Sort the eigenvectors in absolute ascending order of the eigenvalues
    int length = 2*12*local_vol;
    double *h_evecs_sorted  = (double*)malloc(length*nEv_*sizeof(double));
    for(int a=0; a<nconv; a++) {
      memcpy(h_evecs_sorted + a*length,
	     (double*)h_evecs_ + h_evals_sorted_idx[a]*length,
	     length*sizeof(double) );
    }
    memcpy(h_evecs, h_evecs_sorted, nconv*length*sizeof(double));

    // print out the computed eigen values and their error estimates 
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    

    ColorSpinorField *h_v3 = NULL;
    for(int i =0 ; i < nconv ; i++){
      cpuParam->v = (Complex*)h_evecs_ + h_evals_sorted_idx[i]*ldv_;
      h_v3 = ColorSpinorField::Create(*cpuParam);
      
      // d_v = v
      *d_v = *h_v3;

      // d_v2 = M*v
      matVec<double>(mat, *d_v2, *d_v, eig_param);
      
      // lambda = v^dag * M*v
      h_evals_[i] = blas::cDotProduct(*d_v, *d_v2);
      
      Complex unit(1.0,0.0);
      Complex m_lambda(-real(h_evals_[i]),
				    -imag(h_evals_[i]));

      // d_v = ||M*v - lambda*v||
      blas::caxpby(unit, *d_v2, m_lambda, *d_v);
      double L2norm = blas::norm2(*d_v);
      
      printfQuda("EigValue[%04d] = %+e  %+e  Residual: %+e\n",
		 i, real(h_evals_[i]), imag(h_evals_[i]), sqrt(L2norm));
      delete h_v3;
    }
    
    t1 += clock();
    printfQuda("Eigenvalues of Dirac operator computed and sorted in: %f sec\n",
	       t1/CLOCKS_PER_SEC);
    
    
    // cleanup 
    free(ipntr_);
    free(iparam_);
    free(mod_h_evals_sorted);
    free(h_evals_sorted_idx);
    free(resid_);
    free(w_workd_);
    free(w_workl_);
    free(w_workev_);
    free(w_rwork_);
    free(select_);
    free(spectrum);
    
    delete h_v;
    delete h_v2;
    delete d_v;
    delete d_v2;
    delete resid;
    
    return;
    
  }  

  void arpackSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		   QudaEigParam *eig_param,
		   ColorSpinorParam *cpuParam){
    
    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      arpack_solve_double(h_evecs, h_evals, mat, eig_param, cpuParam);
    }
    else {
      arpack_solve_double(h_evecs, h_evals, mat, eig_param, cpuParam);
      //arpack_solve<float>(h_evecs, h_evals, mat, eig_param, cpuParam);
    }    
  }

  
  void arpackErrorHelpNAUPD() {
    printfQuda("Error help NAUPD\n");
    printfQuda("INFO Integer.  (INPUT/OUTPUT)\n");
    printfQuda("     If INFO .EQ. 0, a randomly initial residual vector is used.\n");
    printfQuda("     If INFO .NE. 0, RESID contains the initial residual vector,\n");
    printfQuda("                        possibly from a previous run.\n");
    printfQuda("     Error flag on output.\n");
    printfQuda("     =  0: Normal exit.\n");
    printfQuda("     =  1: Maximum number of iterations taken.\n");
    printfQuda("        All possible eigenvalues of OP has been found. IPARAM(5)\n");
    printfQuda("        returns the number of wanted converged Ritz values.\n");
    printfQuda("     =  2: No longer an informational error. Deprecated starting\n");
    printfQuda("        with release 2 of ARPACK.\n");
    printfQuda("     =  3: No shifts could be applied during a cycle of the\n");
    printfQuda("        Implicitly restarted Arnoldi iteration. One possibility\n");
    printfQuda("        is to increase the size of NCV relative to NEV.\n");
    printfQuda("        See remark 4 below.\n");
    printfQuda("     = -1: N must be positive.\n");
    printfQuda("     = -2: NEV must be positive.\n");
    printfQuda("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("     = -4: The maximum number of Arnoldi update iteration\n");
    printfQuda("        must be greater than zero.\n");
    printfQuda("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("     = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("     = -7: Length of private work array is not sufficient.\n");
    printfQuda("     = -8: Error return from LAPACK eigenvalue calculation;\n");
    printfQuda("     = -9: Starting vector is zero.\n");
    printfQuda("     = -10: IPARAM(7) must be 1,2,3.\n");
    printfQuda("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("     = -12: IPARAM(1) must be equal to 0 or 1.\n");
    printfQuda("     = -9999: Could not build an Arnoldi factorization.\n");
    printfQuda("        User input error highly likely.  Please\n");
    printfQuda("        check actual array dimensions and layout.\n");
    printfQuda("        IPARAM(5) returns the size of the current Arnoldi\n");
    printfQuda("        factorization.\n");
  }

  void arpackErrorHelpNEUPD() {
    printfQuda("Error help NEUPD\n");
    printfQuda("INFO Integer.  (OUTPUT)\n");
    printfQuda("     Error flag on output.\n");
    printfQuda("     =  0: Normal exit.\n");
    printfQuda("     =  1: The Schur form computed by LAPACK routine csheqr\n");
    printfQuda("        could not be reordered by LAPACK routine ztrsen.\n");
    printfQuda("        Re-enter subroutine zneupd with IPARAM(5)=NCV and\n");
    printfQuda("        increase the size of the array D to have\n");
    printfQuda("        dimension at least dimension NCV and allocate at\n");
    printfQuda("        least NCV\n");
    printfQuda("        columns for Z. NOTE: Not necessary if Z and V share\n");
    printfQuda("        the same space. Please notify the authors if this\n");
    printfQuda("        error occurs.\n");
    printfQuda("     = -1: N must be positive.\n");
    printfQuda("     = -2: NEV must be positive.\n");
    printfQuda("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("     = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("     = -7: Length of private work WORKL array is inufficient.\n");
    printfQuda("     = -8: Error return from LAPACK eigenvalue calculation.\n");
    printfQuda("        This should never happened.\n");
    printfQuda("     = -9: Error return from calculation of eigenvectors.\n");
    printfQuda("        Informational error from LAPACK routine ztrevc.\n");
    printfQuda("     = -10: IPARAM(7) must be 1,2,3\n");
    printfQuda("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("     = -12: HOWMNY = 'S' not yet implemented\n");
    printfQuda("     = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.\n");
    printfQuda("     = -14: ZNAUPD did not find any eigenvalues to sufficient\n");
    printfQuda("        accuracy.\n");
    printfQuda("     = -15: ZNEUPD got a different count of the number of\n");
    printfQuda("        converged Ritz values than ZNAUPD got. This\n");
    printfQuda("        indicates the user probably made an error in\n");
    printfQuda("        passing data from ZNAUPD to ZNEUPD or that the\n");
    printfQuda("        data was modified before entering ZNEUPD\n");
  }
  
#endif
  
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

