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

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

bool flags = false;

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
      Complex d1c(d1,0.0);
      Complex d2c(d2,0.0);
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

    Complex s(0.0,0.0);
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
  void arnoldiStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   std::vector<ColorSpinorField*> r,
		   std::vector<ColorSpinorField*> evecs,
		   bool *locked,
		   QudaEigParam *eig_param,
		   Float *alpha, Float *beta, int j) {

  }


  template<typename Float>
  void lanczosStep(const Dirac &mat,
		   std::vector<ColorSpinorField*> v,
		   std::vector<ColorSpinorField*> r,
		   std::vector<ColorSpinorField*> evecs,
		   bool *locked,
		   QudaEigParam *eig_param,
		   Float *alpha, Float *beta, int j) {

    //Compute r = A * v_j - b_{j-i} * v_{j-1}
    //r = A * v_j

    //Purge locked eigenvectors from the residual
    //DMH: This step ensures that the next mat vec
    //     does not reintroduce components of the
    //     existing eigenspace
    Complex s(0.0,0.0);
    for(unsigned int i=0; i<evecs.size() && locked[i]; i++) {
      s = blas::cDotProduct(*evecs[i], *v[j]);
      blas::caxpy(-s, *evecs[i], *v[j]);
    }

    matVecOp<Float>(mat, *r[0], *v[j], eig_param);

    //Deflate locked eigenvectors from the residual
    //DMH: This step ensures that the new Krylov vector
    //     remains orthogonal to the found eigenspace
    for(unsigned int i=0; i<evecs.size() && locked[i]; i++) {
      s = blas::cDotProduct(*evecs[i], *r[0]);
      blas::caxpy(-s, *evecs[i], *r[0]);
    }

    //r = r - b_{j-1} * v_{j-1}
    if(j>0) blas::axpy(-beta[j-1], *v[j-1], *r[0]);

    //a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v[j], *r[0]);

    //r = r - a_j * v_j
    blas::axpy(-alpha[j], *v[j], *r[0]);

    //b_j = ||r||
    beta[j] = sqrt(blas::norm2(*r[0]));

    //Orthogonalise
    if(beta[j] < (1.0)*sqrt(alpha[j]*alpha[j] + beta[j-1]*beta[j-1])) {

      //The residual vector r has been deemed insufficiently
      //orthogonal to the existing Krylov space. We must
      //orthogonalise it.
      //printfQuda("orthogonalising Beta %d = %e\n", j, beta[j]);
      orthogonalise<Float>(v, r, j);
      //blockOrthogonalise<Float>(v, r, j);
      //b_j = ||r||
      beta[j] = sqrt(blas::norm2(*r[0]));
    }

    //Prepare next step.
    //v_{j+1} = r / b_j
    if((unsigned int)j < v.size() - 1) {
      blas::zero(*v[j+1]);
      blas::axpy(1.0/beta[j], *r[0], *v[j+1]);
    }
  }

  template<typename Float>
  void computeSVD(const Dirac &mat,
		  std::vector<ColorSpinorField*> &kSpace,
		  std::vector<ColorSpinorField*> &evecs,
		  std::vector<Complex> &evals,
		  QudaEigParam *eig_param,
		  bool inverse) {

    printfQuda("Computing SVD of M\n");

    //Switch to M (or Mdag) mat vec
    eig_param->use_norm_op = QUDA_BOOLEAN_NO;
    int nConv = eig_param->nConv;
    int nEv = eig_param->nEv;

    Complex sigma_tmp[nConv/2];

    //Create a device side temp vector by cloning
    //the evecs passed to the function. We create a
    //vector of one element. In future, we may wish to
    //make a block eigensolver, so an std::vector structure
    //will be needed.

    std::vector<ColorSpinorField*> tmp;
    ColorSpinorParam csParam(*evecs[0]);
    tmp.push_back(ColorSpinorField::Create(csParam));

    for(int i=0; i <nConv/2; i++){

      int idx = i;
      //If we used poly acc, the eigenpairs are in descending order.
      if(inverse) idx = nEv-1 - i;

      //This function assumes that you have computed the eigenvectors
      //of MdagM, ie, the right SVD of M. The ith eigen vector in the array corresponds
      //to the ith right singular vector. We will sort the array as:
      //
      //     EV_array_MdagM = (Rev_0, Rev_1, ... , Rev_{n-1{)
      // to  SVD_array_M    = (Rsv_0, Lsv_0, Rsv_1, Lsv_1, ... ,
      //                       Rsv_{nEv/2-1}, Lsv_{nEv/2-1})
      //
      //We start at Rev_(n/2-1), compute Lsv_(n/2-1), then move the vectors
      //to the n-2 and n-1 positions in the array respectively.


      //As a cross check, we recompute the singular values from mat vecs rather
      //than make the direct relation (sigma_i)^2 = |lambda_i|
      //--------------------------------------------------------------------------
      Complex lambda = evals[idx];

      //M*Rev_i = M*Rsv_i = sigma_i Lsv_i
      matVec<Float>(mat, *tmp[0], *evecs[idx], eig_param);

      // sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
      Complex sigma_sq = blas::cDotProduct(*tmp[0], *tmp[0]);
      sigma_tmp[i] = Complex(sqrt(sigma_sq.real()) , sqrt(abs(sigma_sq.imag())));

      //Copy device Rsv[i] (EV[i]) to host SVD[2*i]
      *kSpace[2*i] = *evecs[idx];

      //Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
      double norm = sqrt(blas::norm2(*tmp[0]));
      blas::ax(1.0/norm, *tmp[0]);
      //Copy Lsv[i] to SVD[2*i+1]
      *kSpace[2*i+1] = *evecs[idx];

      printfQuda("Sval[%04d] = %+.16e  %+.16e   sigma - sqrt(|lambda|) = %+.16e\n",
		 i, sigma_tmp[i].real(), sigma_tmp[i].imag(),
		 sigma_tmp[i].real() - sqrt(abs(lambda.real())));
      //--------------------------------------------------------------------------

    }

    //Update the host evals array
    for(int i=0; i<nConv/2; i++) {
      evals[2*i + 0] = sigma_tmp[i];
      evals[2*i + 1] = sigma_tmp[i];
    }

    //Revert to MdagM (or MMdag) mat vec
    eig_param->use_norm_op = QUDA_BOOLEAN_YES;

    delete tmp[0];
  }

  template<typename Float>
  void lanczos_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		     QudaEigParam *eig_param,
		     ColorSpinorParam *cpuParam){


  }

  void lanczosSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam){

    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      lanczos_solve<double>(h_evecs, h_evals, mat, eig_param, cpuParam);
    } else if(eig_param->cuda_prec_ritz == QUDA_SINGLE_PRECISION) {
      lanczos_solve<float>(h_evecs, h_evals, mat, eig_param, cpuParam);
    } else {
      errorQuda("prec not supported");
    }
  }

  template<typename Float>
  void irlm_solve(std::vector<ColorSpinorField*> kSpace,
		  std::vector<Complex> &evals, const Dirac &mat,
		  QudaEigParam *eig_param){

    // Preliminaries: Memory allocation, local variables.
    //---------------------------------------------------------------------------
    int nEv = eig_param->nEv;
    int nKr = eig_param->nKr;
    int nConv = eig_param->nConv;
    Float tol = eig_param->tol;
    char *QUDA_logfile = eig_param->QUDA_logfile;
    bool inverse = false;

    Float *residual = (Float*)malloc(nKr*sizeof(Float));
    Float *residual_old = (Float*)malloc(nKr*sizeof(Float));

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

    if(flags) printfQuda("Flag 1\n");

    //Create the device side residual vector by cloning
    //the kSpace passed to the function. We create a
    //vector of one element. In future, we may wish to
    //make a block eigensolver, so a vector structure will be needed.
    std::vector<ColorSpinorField*> r;
    ColorSpinorParam csParam(*kSpace[0]);
    r.push_back(ColorSpinorField::Create(csParam));

    if(flags) printfQuda("Flag 2\n");

    //Device side space for Lanczos Ritz vector basis rotation.
    std::vector<ColorSpinorField*> d_vecs_tmp;
    for(int i=0; i<nEv; i++) {
      d_vecs_tmp.push_back(ColorSpinorField::Create(csParam));
    }

    if(flags) printfQuda("Flag 3\n");

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

    //Deduce, if using Chebyshev, whether to invert the sorting
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

    if(flags) printfQuda("Flag 4\n");

    // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
    //---------------------------------------------------------------------------

    Float t1 = clock();
    bool converged = false;

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
    double time_svd= 0.0;  //time to compute SVD

    //Ensure we are not trying to compute on a zero-field source
    //and test for an initial guess
    Float norm = sqrt(blas::norm2(*kSpace[0]));
    if (norm == 0) {
      printfQuda("Initial residual is zero. Populating with rands.\n");

      if (kSpace[0] -> Location() == QUDA_CPU_FIELD_LOCATION) {
        kSpace[0] -> Source(QUDA_RANDOM_SOURCE);
      } else {
        RNG *rng = new RNG(kSpace[0]->Volume(), 1234, kSpace[0]->X());
        rng->Init();
        spinorNoise(*kSpace[0], *rng, QUDA_NOISE_UNIFORM);
        rng->Release();
        delete rng;
      }
    }

    //Normalise initial guess
    norm = sqrt(blas::norm2(*kSpace[0]));
    blas::ax(1.0/norm, *kSpace[0]);

    printfQuda("**** START IRLM SOLUTION ****\n");
    printfQuda("Output from IRLM directed to %s\n", QUDA_logfile);
    printfQuda("nConv %d\n", nConv);
    printfQuda("nEv %d\n", nEv);
    printfQuda("nKr %d\n", nKr);
    printfQuda("polyDeg %d\n", eig_param->poly_deg);
    printfQuda("a-min %f\n", eig_param->a_min);
    printfQuda("a-max %f\n", eig_param->a_max);

    //Initial k step factorisation
    time = -clock();
    for(k=0; k<nEv; k++) {
      lanczosStep<Float>(mat, kSpace, r, d_vecs_tmp, locked, eig_param, alpha, beta, k);
      if(flags) printfQuda("Flag k=%d\n", k);
    }
    printfQuda("Initial %d step factorisation complete\n", k);
    time += clock();
    time_ls += time;

    //Loop over restart iterations.
    //---------------------------------------------------------------------------
    while(restart_iter < max_restarts && !converged) {

      time = -clock();
      for(k=nEv; k<nKr; k++) {
	lanczosStep<Float>(mat, kSpace, r, d_vecs_tmp, locked, eig_param, alpha, beta, k);
	if(flags) printfQuda("Flag k=%d\n", k);
      }
      if(flags) printfQuda("Restart %d complete\n", restart_iter+1);
      time += clock();
      time_ls += time;

      //Compute the Tridiagonal matrix T_{k,k}
      //--------------------------------------
      if(flags) printfQuda("Computing Tridiag\n");
      time = -clock();
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
      Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
      Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
      eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());

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
      Float evals_ritz[nKr-nEv];
      if(inverse) for(int j=0; j<(nKr-nEv); j++) evals_ritz[j] = eigenSolverTD.eigenvalues()[j];
      else for(int j=0; j<(nKr-nEv); j++) evals_ritz[j] = eigenSolverTD.eigenvalues()[j+nEv];

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

      if(flags) printfQuda("Computing QR\n");
      for(int j=0; j<(nKr-nEv); j++) {

	//Apply the shift \mu_j
	TmMuI = triDiag;
	MatrixXd Id = MatrixXd::Identity(nKr, nKr);;
	TmMuI -= evals_ritz[j]*Id;

	//QR decomposition of Tm - \mu_i I
	Eigen::HouseholderQR<MatrixXd> QR(TmMuI);

	//Retain Qj matrices as a product.
	Qj = QR.householderQ();
	Q = Q * Qj;

	//Update the Tridiag
	triDiag = Qj.adjoint() * triDiag;
	triDiag = triDiag * Qj;
      }
      time += clock();
      time_e += time;
      //--------------------------------------

      //Block basis rotation
      //--------------------------------------
      if(flags) printfQuda("Performing QR Rotation\n");
      time = -clock();
      Complex *Qmat_d = new Complex[nEv*nKr];
      //Place data in accessible array
      //The locked eigenvectors will not be rotated
      for(int i=0; i<nEv; i++) {
	for(int j=0; j<nKr; j++) {
	  Qmat_d[j*nEv + i].real(Q.col(i)[j]);
	  Qmat_d[j*nEv + i].imag(0.0);
	}
      }

      for(int i=0; i<nEv; i++) blas::zero(*d_vecs_tmp[i]);

      blas::caxpy(Qmat_d, kSpace, d_vecs_tmp);
      //Copy back to Krylov space array
      for(int i=0; i<nEv; i++) {
	*kSpace[i] = *d_vecs_tmp[i];
      }

      //Update the residual
      // r_{nev} = r_{nkv} * \sigma_{nev}  |  \sigma_{nev} = Q(nkv,nev)
      blas::ax(Q(nKr-1,nEv-1), *r[0]);
      // r_{nev} += v_{nev+1} * beta_k  |  beta_k = Tm(nev+1,nev)
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
      if(flags) printfQuda("Performing convergence check\n");
      if((restart_iter+1)%check_interval == 0) {

	time = -clock();

	//Construct the nEv,nEv Tridiagonal matrix for convergence check
	MatrixXd triDiagNew = MatrixXd::Identity(nEv, nEv);
	for(int i=0; i<nEv; i++) {
	  triDiagNew(i,i) = triDiag(i,i);
	  if(i<nEv-1) {
	    triDiagNew(i+1,i) = triDiag(i+1,i);
	    triDiagNew(i,i+1) = triDiag(i,i+1);
	  }
	}
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTDnew(triDiagNew);

	Complex *Qmat_d_nev = new Complex[nEv*nEv];
	//Place data in accessible array
	for(int i=0; i<nEv; i++) {
	  for(int j=0; j<nEv; j++) {
	    Qmat_d_nev[j*nEv + i].real(eigenSolverTDnew.eigenvectors().col(i)[j]);
	    Qmat_d_nev[j*nEv + i].imag(0.0);
	  }
	}

	time += clock();
	time_e += time;

	//Basis rotation
	//Pointers to the required Krylov space vectors,
	//no extra memory is allocated.
	time = -clock();
	std::vector<ColorSpinorField*> d_vecs_ptr;
	for(int i=0; i<nEv; i++) {
	  d_vecs_ptr.push_back(kSpace[i]);
	  blas::zero(*d_vecs_tmp[i]);
	}

	blas::caxpy(Qmat_d_nev, d_vecs_ptr, d_vecs_tmp);
	time += clock();
	time_mb += time;

	//Check for convergence
	//Error estimates given by ||A*vec - lambda*vec||
	time = -clock();
	for(int i=0; i<nEv; i++) {

	  //r = A * v_i
	  matVec<Float>(mat, *r[0], *d_vecs_tmp[i], eig_param);
	  //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	  evals[i] = blas::cDotProduct(*d_vecs_tmp[i], *r[0])/sqrt(blas::norm2(*d_vecs_tmp[i]));

	  //Convergence check ||lambda_i*v_i - A*v_i||
	  //--------------------------------------------------------------------------
	  Complex n_unit(-1.0,0.0);
	  blas::caxpby(evals[i], *d_vecs_tmp[i], n_unit, *r[0]);
	  residual[i] = sqrt(blas::norm2(*r[0]));

	  //Update QUDA_logfile
	  //printfQuda("Residual %d = %.6e\n", i, residual[i]);
	  //printfQuda("Eval %d = %.6e\n", i, h_evals_[i].real() );
	  //printfQuda("Norm %d = %.6e\n", i, sqrt(blas::norm2(*d_vecs_tmp[i])));

	  //Copy over residual
	  residual_old[i] = residual[i];
	  //--------------------------------------------------------------------------
	}

	for(int i=0; i<nEv; i++) {
	  if(residual[i] < tol && !locked[i]) {

	    //This is a new eigenpair, lock it
	    locked[i] = true;
	    printfQuda("**** locking vector %d, converged eigenvalue = (%+.16e,%+.16e) resid %+.16e ****\n",
		       i, evals[i].real(), evals[i].imag(), residual[i]);

	    num_converged++;
	  }
	}

	printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter+1);


	//Check that the lowest nConv Eigenpairs have converged. //FIXME
	bool all_locked = true;
	if(num_converged >= nConv) {
	  for(int i=0; i<nConv; i++) {
	    if(inverse) {
	      if(!locked[nEv-1-i]) all_locked = false;
	    } else {
	      if(!locked[i]) all_locked = false;
	    }
	  }

	  if(all_locked) {
	    //Transfer eigenvectors to kSpace array
	    for(int i=0; i<nConv; i++) *kSpace[i] =  *d_vecs_tmp[i];
	    converged = true;
	  }
	}

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
    if(!converged) {
      printfQuda("IRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d restart steps. "
		 "Please either increase nKr and nEv and/or extend the number of restarts\n", nConv, nEv, nKr, max_restarts);
    } else {
      printfQuda("IRLM computed the requested %d vectors in %d restart steps of size %d\n", nConv, restart_iter, (nKr-nEv));

      for(int i=0; i<nEv; i++) {
	printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n",
		   i, alpha[i], 0.0, beta[i]);
      }


      for(int i=0; i<nEv; i++) {
	printfQuda("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n",
		   i, evals[i].real(), evals[i].imag(), residual[i]);
      }

      //Compute SVD
      time_svd = -clock();
      if(eig_param->compute_svd) computeSVD<Float>(mat, kSpace, d_vecs_tmp, evals, eig_param, inverse);
      time_svd += clock();

    }

    Float t2 = clock() - t1;
    Float total;

    if(eig_param->compute_svd) total = (time_e + time_c + time_ls + time_mb + time_svd)/CLOCKS_PER_SEC;
    else total = (time_e + time_c + time_ls + time_mb)/CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using IRLM = %e\n", t2/CLOCKS_PER_SEC);
    printfQuda("Time spent using EIGEN           = %e  %.1f%%\n", time_e/CLOCKS_PER_SEC, 100*(time_e/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in lanczos step       = %e  %.1f%%\n", time_ls/CLOCKS_PER_SEC, 100*(time_ls/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in multi blas         = %e  %.1f%%\n", time_mb/CLOCKS_PER_SEC, 100*(time_mb/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in convergence check  = %e  %.1f%%\n", time_c/CLOCKS_PER_SEC, 100*(time_c/CLOCKS_PER_SEC)/total);
    if(eig_param->compute_svd) printfQuda("Time spent computing svd         = %e  %.1f%%\n", time_svd/CLOCKS_PER_SEC, 100*(time_svd/CLOCKS_PER_SEC)/total);
    //---------------------------------------------------------------------------

    //Local clean-up
    //--------------
    delete r[0];
    for(int i=0; i<nEv; i++) {
      delete d_vecs_tmp[i];
    }

    printfQuda("END IRLM SOLUTION\n");
  }

  void irlmSolve(std::vector<ColorSpinorField*> kSpace,
		 std::vector<Complex> &evals, const Dirac &mat,
		 QudaEigParam *eig_param){

    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      if(flags) printfQuda("DOUBLE prec IRLM\n");
      irlm_solve<double>(kSpace, evals, mat, eig_param);
    } else {
      if(flags) printfQuda("SINGLE prec IRLM\n");
      irlm_solve<float>(kSpace, evals, mat, eig_param);
    }
  }

  template<typename Float>
  void iram_solve(std::vector<ColorSpinorField*> kSpace,
		  std::vector<Complex> &evals, const Dirac &mat,
		  QudaEigParam *eig_param){


  }

  void iramSolve(std::vector<ColorSpinorField*> kSpace,
		 std::vector<Complex> &evals, const Dirac &mat,
		 QudaEigParam *eig_param){

    if(eig_param->cuda_prec_ritz == QUDA_DOUBLE_PRECISION) {
      iram_solve<double>(kSpace, evals, mat, eig_param);
    } else {
      iram_solve<float>(kSpace, evals, mat, eig_param);
    }
  }

  // ARPACK INTERAFCE ROUTINES
  //--------------------------------------------------------------------------

#ifdef ARPACK_LIB

  void arpackErrorHelpNAUPD();
  void arpackErrorHelpNEUPD();

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
#include <mpi.h>
#endif

  void arpack_solve(void *h_evecs, void *h_evals,
		    const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam){

    //ARPACK logfile name
    char *arpack_logfile = eig_param->arpack_logfile;
    printfQuda("**** START ARPACK SOLUTION ****\n");
    printfQuda("Output directed to %s\n", arpack_logfile);

    //Construct parameters and memory allocation
    //---------------------------------------------------------------------------------

    double time_ar = 0.0; //time in ARPACK
    double time_mv = 0.0; //time in QUDA mat vec + data transfer
    double time_ev = 0.0; //time in computing Eigenvectors

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

    double tol_ = eig_param->tol;
    double *mod_h_evals_sorted  = (double*)malloc(nKr_*sizeof(double));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }

    //ARPACK workspace
    //Initial guess?
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
    //int arpack_log_u = 9999;

#if (defined (QMP_COMMS) || defined (MPI_COMMS))

//     if ( arpack_logfile != NULL  && (comm_rank() == 0) ) {

//       ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
//       int msglvl0 = 0, msglvl3 = 3;
//       ARPACK(pmcinitdebug)(&arpack_log_u,      //logfil
// 			   &msglvl3,           //mcaupd
// 			   &msglvl3,           //mcaup2
// 			   &msglvl0,           //mcaitr
// 			   &msglvl3,           //mceigh
// 			   &msglvl0,           //mcapps
// 			   &msglvl0,           //mcgets
// 			   &msglvl3            //mceupd
// 			   );

//       printfQuda("eigenSolver: Log info:\n");
//       printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
//       printfQuda("output is directed to %s\n",arpack_logfile);
//     }
// #else
//     if (arpack_logfile != NULL) {

//       ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
//       int msglvl0 = 0, msglvl3 = 3;
//       ARPACK(mcinitdebug)(&arpack_log_u,      //logfil
// 			  &msglvl3,           //mcaupd
// 			  &msglvl3,           //mcaup2
// 			  &msglvl0,           //mcaitr
// 			  &msglvl3,           //mceigh
// 			  &msglvl0,           //mcapps
// 			  &msglvl0,           //mcgets
// 			  &msglvl3            //mceupd
// 			  );

//       printfQuda("eigenSolver: Log info:\n");
//       printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
//       printfQuda("output is directed to %s\n", arpack_logfile);
//     }

#endif

    //Start ARPACK routines
    //---------------------------------------------------------------------------------

    double t1, t2;

    do {

      t1 = -((double)clock());

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
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_, &nKr_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_, 1, 2);
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

      if (ido_ == 99 || info_ == 1)
	break;

      if (ido_ == -1 || ido_ == 1) {

	t2 = -clock();

	*d_v = *h_v;
	//apply matrix-vector operation here:
	matVecOp<double>(mat, *d_v2, *d_v, eig_param);
	*h_v2 = *d_v2;

	t2 +=clock();

	time_mv += t2;

      }

      t1 += clock();
      time_ar += t1;

      printfQuda("Arpack Iteration %s: %d\n", eig_param->use_poly_acc ? "(with poly acc) " : "", iter_count);
      iter_count++;

    } while (99 != ido_ && iter_count < max_iter);

    //Subspace calulated sucessfully. Compute nEv eigenvectors and values

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_count, info_, ido_);
    printfQuda("Computing eigenvectors\n");

    time_ev = -clock();

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

    //     if(comm_rank() == 0){
    //       if (arpack_logfile != NULL){
    // 	ARPACK(finilog)(&arpack_log_u);
    //       }
    //     }
    // #else
    //     if (arpack_logfile != NULL)
    //       ARPACK(finilog)(&arpack_log_u);

#endif

    printfQuda("Checking eigenvalues\n");

    int nconv = iparam_[4];

    //Sort the eigenvalues in absolute ascending order
    std::vector<std::pair<double,int>> evals_sorted;
    for(int j=0; j<nconv; j++) {
      evals_sorted.push_back( std::make_pair(h_evals_[j].real(), j));
    }

    //Sort the array by value (first in the pair)
    //and the index (second in the pair) will come along
    //for the ride.
    std::sort(evals_sorted.begin(), evals_sorted.end());
    if(inverse) std::reverse(evals_sorted.begin(), evals_sorted.end());

    // print out the computed Ritz values and their error estimates
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d] = %+.16e %+.16e Residual: %+.16e\n", j,
		 real(h_evals_[evals_sorted[j].second]),
		 imag(h_evals_[evals_sorted[j].second]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+evals_sorted[j].second)) );
    }

    //Compute Eigenvalues from Eigenvectors.
    ColorSpinorField *h_v3 = NULL;
    int idx = 0;
    for(int i=0; i<nconv; i++){
      idx = nconv-1 - evals_sorted[i].second;
      cpuParam->v = (Complex*)h_evecs_ + idx*ldv_;
      h_v3 = ColorSpinorField::Create(*cpuParam);

      // d_v = v
      *d_v = *h_v3;

      // d_v2 = M*v
      matVec<double>(mat, *d_v2, *d_v, eig_param);

      // lambda = v^dag * M*v
      h_evals_[idx] = blas::cDotProduct(*d_v, *d_v2);

      Complex unit(1.0,0.0);
      Complex m_lambda(-real(h_evals_[idx]), -imag(h_evals_[idx]));

      // d_v = ||M*v - lambda*v||
      blas::caxpby(unit, *d_v2, m_lambda, *d_v);
      double L2norm = blas::norm2(*d_v);

      printfQuda("EigValue[%04d] = %+.16e  %+.16e  Residual: %.16e\n",
		 i, real(h_evals_[idx]), imag(h_evals_[idx]), sqrt(L2norm));

      delete h_v3;
    }


    time_ev += clock();

    double total = (time_ar + time_ev)/CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using ARPACK         = %e\n", total);
    printfQuda("Time spent in ARPACK                       = %e  %.1f%%\n", (time_ar - time_mv)/CLOCKS_PER_SEC, 100*((time_ar - time_mv)/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in QUDA (M*vec + data transfer) = %e  %.1f%%\n", time_mv/CLOCKS_PER_SEC, 100*(time_mv/CLOCKS_PER_SEC)/total);
    printfQuda("Time spent in computing Eigenvectors       = %e  %.1f%%\n", time_ev/CLOCKS_PER_SEC, 100*(time_ev/CLOCKS_PER_SEC)/total);

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

#else

  void arpack_solve(void *h_evecs, void *h_evals,
		    const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam) {

    errorQuda("(P)ARPACK has not been enabled for this build");

  }

#endif

} // namespace quda




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
