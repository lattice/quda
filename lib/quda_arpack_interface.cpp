#include <quda_arpack_interface.h>
#include <tests/misc.h>
#include <vector>
#include <Eigen/Eigenvalues>

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
#include <mpi.h>
#endif

namespace quda{

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

  template<typename Float, typename Param>
  void polyOp(const Dirac &mat,
	      cudaColorSpinorField &out,
	      const cudaColorSpinorField &in,	   
	      Param *param) {
        
    Float delta,theta;
    Float sigma,sigma1,sigma_old;
    Float d1,d2,d3;

    double a = param->amin;
    double b = param->amax;

    delta = (b-a)/2.0;
    theta = (b+a)/2.0;

    sigma1 = -delta/theta;

    blas::copy(out,in);

    if(param->polyDeg == 0)
      return;
    
    d1 = sigma1/delta;
    d2 = 1.0;

    if(param->useNormOp && param->useDagger) {
      mat.MMdag(out,in);
    }
    else if(param->useNormOp && !param->useDagger) {
      mat.MdagM(out,in);
    }
    else if (!param->useNormOp && param->useDagger) {
      mat.Mdag(out,in);
    }
    else {  
      mat.M(out,in);
    }
    
    blas::caxpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);
    
    //C_1(x) = x
    if(param->polyDeg == 1 )
      return;

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
    for(int i=2; i < param->polyDeg; i++){

      sigma = 1.0/(2.0/sigma1-sigma_old);
      
      d1 = 2.0*sigma/delta;
      d2 = -d1*theta;
      d3 = -sigma*sigma_old;
      
      //mat*C_m
      if(param->useNormOp && param->useDagger) {
	mat.MMdag(out, *tmp2);
      }
      else if(param->useNormOp && !param->useDagger) {
	mat.MdagM(out, *tmp2);
      }
      else if (!param->useNormOp && param->useDagger) {
	mat.Mdag(out, *tmp2);
      }
      else {  
	mat.M(out, *tmp2);
      }
      
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

  template<typename Float, typename Param>
  void polyOpLanczos(const Dirac &mat,
		     cudaColorSpinorField &out,
		     const cudaColorSpinorField &in,	   
		     Param *param) {
    
    const double alpha = param->amin;
    const double beta  = param->amax;

    const double c1 = 2.0*(alpha+beta)/(alpha-beta); 
    const double c0 = 2.0/(alpha+beta); 

    //Clone 'in' to two temporary vectors.
    cudaColorSpinorField *tmp1 = new cudaColorSpinorField(in);
    cudaColorSpinorField *tmp2 = new cudaColorSpinorField(in);
    cudaColorSpinorField *tmp3 = new cudaColorSpinorField(in);

    blas::copy(*tmp1,in);
    blas::copy(*tmp2,in);
    
    *(tmp2) = in;
    mat.MdagM(*(tmp1), in);
    
    blas::axpby(-0.5*c1, const_cast<cudaColorSpinorField&>(in), 0.5*c0*c1, *(tmp1));
    for(int i=2; i < param->polyDeg+1; ++i) {
      mat.MdagM(out,*(tmp1));
      blas::axpby(-c1,*(tmp1),c0*c1,out);
      blas::axpy(-1.0,*(tmp2),out);
      //printfQuda("ritzMat: Ritz mat loop %d\n",i);

      if(i != param->polyDeg) {
	// tmp2 = tmp
        // tmp = out
	blas::copy(*tmp3, *tmp2);
	blas::copy(*tmp2, *tmp1);
	blas::copy(*tmp1, *tmp3);
	blas::copy(*tmp1, out);
      }
    }
    delete tmp1;
    delete tmp2;
    delete tmp3;
  }

  /*
  template<typename Float, typename Param>
  void matVec(cudaColorSpinorField &out,
	      cudaColorSpinorField &in,	      
	      const Dirac &mat,
	      Param *param) {
    
    //apply matrix-vector operation here:
    if(param->usePolyAcc) {
      polyOp<Float, param>(&mat, out, in, param);
    }
    else {
      if(param->useNormOp && param->useDagger) {
	mat.MMdag(out, in);
      }
      else if(param->useNormOp && !param->useDagger) {
	mat.MdagM(out, in);
      }
      else if (!param->useNormOp && param->useDagger) {
	mat.Mdag(out, in);
      }
      else {  
	mat.M(out, in);
      }
    }
  }
  */
  
  void arpack_solve_float(void *h_evecs, void *h_evals,
			  const Dirac &mat,
			  QudaArpackParam *arpack_param,
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
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter = arpack_param->arpackMaxiter;
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //ARPACK logfile name
    char *arpack_logfile = arpack_param->arpackLogfile;
    
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

    if(arpack_param->usePolyAcc){
      if (arpack_param->spectrum == QUDA_SR_SPECTRUM) spectrum = strdup("LR");
      else if (arpack_param->spectrum == QUDA_LR_SPECTRUM) spectrum = strdup("SR");
      else if (arpack_param->spectrum == QUDA_SM_SPECTRUM) spectrum = strdup("LM");
      else if (arpack_param->spectrum == QUDA_LM_SPECTRUM) spectrum = strdup("SM");
      else if (arpack_param->spectrum == QUDA_SI_SPECTRUM) spectrum = strdup("LI");
      else if (arpack_param->spectrum == QUDA_LI_SPECTRUM) spectrum = strdup("SI");
    }
    else{
      if (arpack_param->spectrum == QUDA_SR_SPECTRUM) spectrum = strdup("SR");
      else if (arpack_param->spectrum == QUDA_LR_SPECTRUM) spectrum = strdup("LR");
      else if (arpack_param->spectrum == QUDA_SM_SPECTRUM) spectrum = strdup("SM");
      else if (arpack_param->spectrum == QUDA_LM_SPECTRUM) spectrum = strdup("LM");
      else if (arpack_param->spectrum == QUDA_SI_SPECTRUM) spectrum = strdup("SI");
      else if (arpack_param->spectrum == QUDA_LI_SPECTRUM) spectrum = strdup("LI");
    }
    
    float tol_ = arpack_param->arpackTol;    
    float *mod_h_evals_sorted  = (float*)malloc(nkv_*sizeof(float));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //ARPACK workspace
    //Initial guess?
    std::complex<float> I(0.0,1.0);
    std::complex<float> *resid_ =
      (std::complex<float> *) malloc(ldv_*sizeof(std::complex<float>));
    
    if(info_ > 0) 
      for(int a = 0; a<ldv_; a++) {
	resid_[a] = I;
	//printfQuda("(%e , %e)\n", real(resid_[a]), imag(resid_[a]));
      }
    
    std::complex<float> sigma_ = 0.0;    
    std::complex<float> *w_workd_ =
      (std::complex<float> *) malloc(3*ldv_*sizeof(std::complex<float>));
    std::complex<float> *w_workl_ =
      (std::complex<float> *) malloc(lworkl_*sizeof(std::complex<float>)); 
    std::complex<float> *w_workev_=
      (std::complex<float> *) malloc(2*nkv_*sizeof(std::complex<float>));    
    float *w_rwork_  = (float *)malloc(nkv_*sizeof(float));    
    int *select_ = (int*)malloc(nkv_*sizeof(int));

    //Alias pointers
    std::complex<float> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<float>*) (float*)(h_evecs);    
    std::complex<float> *h_evals_ = nullptr;
    h_evals_ = (std::complex<float>*) (float*)(h_evals);

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
    
    int iter_cnt= 0;

    bool allocate = true;
    cpuColorSpinorField *h_v = nullptr;
    cpuColorSpinorField *h_v2 = nullptr;
    cudaColorSpinorField *d_v = nullptr;    
    cudaColorSpinorField *d_v2 = nullptr;
    cudaColorSpinorField *resid = nullptr;    

    //ARPACK log routines
    // Code added to print the log of ARPACK  
    int arpack_log_u = 9999;
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
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
    
#endif   
    
    //Start ARPACK routines
    //---------------------------------------------------------------------------------

    double t1;
    
    do {

      t1 = -((double)clock());
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pcnaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pcnaupd info = %d. Exiting.",info_);
      }
#else
      ARPACK(cnaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_, 1, 2);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in cnaupd info = %d. Exiting.",info_);
      }
#endif
      
      //If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if(allocate){

	//Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
	//less one, hence ipntr[0] - 1 to specify the correct address.

	cpuParam->create = QUDA_REFERENCE_FIELD_CREATE;
	cpuParam->gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	
	cpuParam->v = w_workd_ + (ipntr_[0] - 1);
	h_v = new cpuColorSpinorField(*cpuParam);
	//Adjust the position of the start of the array.
	cpuParam->v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = new cpuColorSpinorField(*cpuParam);
	
	//cpuParam->print();
	ColorSpinorParam cudaParam(*cpuParam);
	cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
	
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	resid = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  polyOp<float, QudaArpackParam>(mat, *d_v2, *d_v, arpack_param);
	}
	else {
	  if(arpack_param->useNormOp && arpack_param->useDagger) {
	    mat.MMdag(*d_v2,*d_v);
	  }
	  else if(arpack_param->useNormOp && !arpack_param->useDagger) {
	    mat.MdagM(*d_v2,*d_v);
	  }
	  else if (!arpack_param->useNormOp && arpack_param->useDagger) {
	    mat.Mdag(*d_v2,*d_v);
	  }
	  else {  
	    mat.M(*d_v2,*d_v);
	  }
	}
	*h_v2 = *d_v2;
      }

      t1 += clock();
	
      printfQuda("Arpack Iteration %s: %d (%e secs)\n", arpack_param->usePolyAcc ? "(with poly acc) " : "", iter_cnt, t1/CLOCKS_PER_SEC);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors and values 
    
    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);      
    printfQuda("Computing eigenvectors\n");
    
    //Interface to arpack routines
    //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    ARPACK(pcneupd)(fcomm_, &rvec_, &howmny, select_, h_evals_, h_evecs_,
		    &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nev_,
		    &tol_, resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		    w_workl_, &lworkl_, w_rwork_ ,&info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in pcneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pcneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(cneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in cneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in cneupd info = %d. Exiting.\n", info_);
#endif

    // Print additional convergence information.
    if( (info_) == 1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info_ == 3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    if(comm_rank() == 0){
      if (arpack_logfile != NULL){
	ARPACK(finilog)(&arpack_log_u);
      }
    }
#else
    if (arpack_logfile != NULL)
      ARPACK(finilog)(&arpack_log_u);
#endif     

    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      h_evals_sorted_idx[j] = j;
      mod_h_evals_sorted[j] = real(h_evals_[j]);
    }

    //Sort the eigenvalues in absolute ascending order
    t1 =  -((double)clock());
    bool inverse = true;
    const char *L = "L";
    const char *S = "S";
    if(strncmp(L, spectrum, 1) == 0 && !arpack_param->usePolyAcc) {
      inverse = false;
    } else if(strncmp(S, spectrum, 1) == 0 && arpack_param->usePolyAcc) {
      inverse = false;
    } else if(strncmp(L, spectrum, 1) == 0 && arpack_param->usePolyAcc) {
      inverse = false;
    }
    
    sortAbs<float>(mod_h_evals_sorted, nconv, inverse, h_evals_sorted_idx);
    
    printfQuda("Sorted eigenvalues based on their absolute values:\n");

    //Sort the eigenvectors in absolute ascending order of the eigenvalues
    int length = 2*12*local_vol;
    float *h_evecs_sorted  = (float*)malloc(length*nev_*sizeof(float));
    for(int a=0; a<nconv; a++) {
      memcpy(h_evecs_sorted + a*length,
	     (float*)h_evecs_ + h_evals_sorted_idx[a]*length,
	     length*sizeof(float) );
    }
    memcpy(h_evecs, h_evecs_sorted, nconv*length*sizeof(float));

    // print out the computed eigen values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }
    
    
    t1 = -(double)clock();
    cpuColorSpinorField *h_v3 = NULL;
    for(int i =0 ; i < nev_ ; i++){
      cpuParam->v = (std::complex<float>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_;
      h_v3 = new cpuColorSpinorField(*cpuParam);

      // d_v = v
      *d_v = *h_v3;

      // d_v2 = M*v
      if(arpack_param->useNormOp && arpack_param->useDagger) {
	mat.MMdag(*d_v2,*d_v);
      }
      else if(arpack_param->useNormOp && !arpack_param->useDagger) {
	mat.MdagM(*d_v2,*d_v);
      }
      else if (!arpack_param->useNormOp && arpack_param->useDagger) {
	mat.Mdag(*d_v2,*d_v);
      }
      else {  
	mat.M(*d_v2,*d_v);
      }

      // lambda = v^dag * M*v
      h_evals_[i] = blas::cDotProduct(*d_v, *d_v2);
      
      std::complex<float> unit(1.0,0.0);
      std::complex<float> m_lambda(-real(h_evals_[i]),
				   -imag(h_evals_[i]));

      // d_v = ||M*v - lambda*v||
      blas::caxpby(unit, *d_v2, m_lambda, *d_v);
      double L2norm = blas::norm2(*d_v);

      printfQuda("EigValue[%04d] = %+e  %+e  Residual: %+e\n",
		 i, real(h_evals_[i]), imag(h_evals_[i]), sqrt(L2norm));
      delete h_v3;
    }
    t1 += clock();
    printfQuda("Eigenvalues of Dirac operator computed in: %f sec\n",
	       t1/CLOCKS_PER_SEC);

    if(arpack_param->SVD) {
      
      //Compute SVD
      t1 = -(double)clock();
      for(int i=nev_/2 - 1; i >= 0; i--){
	
	//This function assumes that you have computed the eigenvectors
	//of MdagM, ie, the right SVD of M. The ith eigen vector in the array corresponds
	//to the ith smallest Right Singular Vector. We will sort the array as:
	//
	//     EV_array_MdagM = {Rev_0, Rev_1, Rev_2, ... , Rev_(n-1)}
	// to  SVD_array_M    = {Rsv_0, Lsv_0, Rsv_1, ... , Lsv_(n/2-1)}
	//
	//We start at Rev_(n/2-1), compute Lsv_(n/2-1), then move the vectors
	//to the n-2 and n-1 positions in the array respectively.
	
	//Copy Rev to Rsv
	memcpy((std::complex<float>*)h_evecs_ + h_evals_sorted_idx[2*i + 1]*ldv_,
	       (std::complex<float>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_,
	       2*ldv_*sizeof(float));
	
	cpuParam->v = (std::complex<float>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_;
	h_v3 = new cpuColorSpinorField(*cpuParam);      //Host Rev_i
	*d_v = *h_v3;                                   //Device Rev_i
	
	mat.M(*d_v2,*d_v);                              //  M Rev_i = M Rsv_i = sigma_i Lsv_i =   
	h_evals_[i] = blas::cDotProduct(*d_v2, *d_v2);  // (sigma_i)^2 = (sigma_i Lsv_i)^dag * (sigma_i Lsv_i)
	
	//Compute \sigma_i
	h_evals_[2*i + 1] = std::complex<float>(sqrt( real(h_evals_[i]) ) , sqrt(imag(h_evals_[i])*imag(h_evals_[i])));
	h_evals_[2*i + 0] = std::complex<float>(sqrt( real(h_evals_[i]) ) , sqrt(imag(h_evals_[i])*imag(h_evals_[i])));
	
	//Normalise the Lsv
	float L2norm = sqrt(blas::norm2(*d_v2));      
	blas::ax(1.0/L2norm, *d_v2);
	
	//Move Lsv
	*h_v3 = *d_v2;
	memcpy((std::complex<float>*)h_evecs_ + h_evals_sorted_idx[2*i]*ldv_,
	       (std::complex<float>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_,
	       2*ldv_*sizeof(float));
	
	delete h_v3;
      }
      for(int i=0; i < nev_; i++){
	printfQuda("Sval[%04d] = %+e  %+e\n",
		   i, real(h_evals_[i]), imag(h_evals_[i]));
      }
      
      t1 += clock();
      printfQuda("Left singular pairs of Dirac operator computed in: %f sec\n",
		 t1/CLOCKS_PER_SEC);
    }
    
    
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
  
  void arpack_solve_double(void *h_evecs, void *h_evals,
			   const Dirac &mat,
			   QudaArpackParam *arpack_param,
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
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter = arpack_param->arpackMaxiter;
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));
        
    //ARPACK logfile name
    char *arpack_logfile = arpack_param->arpackLogfile;

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

    if(arpack_param->usePolyAcc){
      if (arpack_param->spectrum == QUDA_SR_SPECTRUM) spectrum = strdup("LR");
      else if (arpack_param->spectrum == QUDA_LR_SPECTRUM) spectrum = strdup("SR");
      else if (arpack_param->spectrum == QUDA_SM_SPECTRUM) spectrum = strdup("LM");
      else if (arpack_param->spectrum == QUDA_LM_SPECTRUM) spectrum = strdup("SM");
      else if (arpack_param->spectrum == QUDA_SI_SPECTRUM) spectrum = strdup("LI");
      else if (arpack_param->spectrum == QUDA_LI_SPECTRUM) spectrum = strdup("SI");
    }
    else{
      if (arpack_param->spectrum == QUDA_SR_SPECTRUM) spectrum = strdup("SR");
      else if (arpack_param->spectrum == QUDA_LR_SPECTRUM) spectrum = strdup("LR");
      else if (arpack_param->spectrum == QUDA_SM_SPECTRUM) spectrum = strdup("SM");
      else if (arpack_param->spectrum == QUDA_LM_SPECTRUM) spectrum = strdup("LM");
      else if (arpack_param->spectrum == QUDA_SI_SPECTRUM) spectrum = strdup("SI");
      else if (arpack_param->spectrum == QUDA_LI_SPECTRUM) spectrum = strdup("LI");
    }
    
    double tol_ = arpack_param->arpackTol;
    double *mod_h_evals_sorted  = (double*)malloc(nkv_*sizeof(double));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }

    //ARPACK workspace
    //Initial guess?
    std::complex<double> I(0.0,1.0);
    std::complex<double> *resid_ =
      (std::complex<double> *) malloc(ldv_*sizeof(std::complex<double>));
    
    if(info_ > 0) 
      for(int a = 0; a<ldv_; a++) {
	resid_[a] = I;
	//printfQuda("(%e , %e)\n", real(resid_[a]), imag(resid_[a]));
      }
    
    std::complex<double> sigma_ = 0.0;
    std::complex<double> *w_workd_ =
      (std::complex<double> *) malloc(3*ldv_*sizeof(std::complex<double>));
    std::complex<double> *w_workl_ =
      (std::complex<double> *) malloc(lworkl_*sizeof(std::complex<double>)); 
    std::complex<double> *w_workev_=
      (std::complex<double> *) malloc(2*nkv_*sizeof(std::complex<double>));    
    double *w_rwork_  = (double *)malloc(nkv_*sizeof(double));    
    int *select_ = (int*)malloc(nkv_*sizeof(int));

    //Alias pointers
    std::complex<double> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<double>*) (double*)(h_evecs);    
    std::complex<double> *h_evals_ = nullptr;
    h_evals_ = (std::complex<double>*) (double*)(h_evals);

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

    int iter_cnt= 0;

    bool allocate = true;
    cpuColorSpinorField *h_v = nullptr;
    cudaColorSpinorField *d_v = nullptr;    
    cpuColorSpinorField *h_v2 = nullptr;
    cudaColorSpinorField *d_v2 = nullptr;
    cudaColorSpinorField *resid = nullptr;    

    //ARPACK log routines
    // Code added to print the log of ARPACK  
    int arpack_log_u = 9999;
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
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
    
#endif   
    
    //Start ARPACK routines
    //---------------------------------------------------------------------------------

    double t1;
    
    do {

      t1 = -((double)clock());
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pznaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
      }
#else
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
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

	cpuParam->create = QUDA_REFERENCE_FIELD_CREATE;
	cpuParam->gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	
	cpuParam->v = w_workd_ + (ipntr_[0] - 1);
	h_v = new cpuColorSpinorField(*cpuParam);
	//Adjust the position of the start of the array.
	cpuParam->v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = new cpuColorSpinorField(*cpuParam);
	
	//cpuParam->print();
	ColorSpinorParam cudaParam(*cpuParam);
	cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
	
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	resid = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  polyOp<double, QudaArpackParam>(mat, *d_v2, *d_v, arpack_param);
	}
	else {
	  if(arpack_param->useNormOp && arpack_param->useDagger) {
	    mat.MMdag(*d_v2,*d_v);
	  }
	  else if(arpack_param->useNormOp && !arpack_param->useDagger) {
	    mat.MdagM(*d_v2,*d_v);
	  }
	  else if (!arpack_param->useNormOp && arpack_param->useDagger) {
	    mat.Mdag(*d_v2,*d_v);
	  }
	  else {  
	    mat.M(*d_v2,*d_v);
	  }
	}

	*h_v2 = *d_v2;
	
      }

      t1 += clock();
	
      printfQuda("Arpack Iteration %s: %d (%e secs)\n", arpack_param->usePolyAcc ? "(with poly acc) " : "", iter_cnt, t1/CLOCKS_PER_SEC);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors and values 
    
    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);      
    printfQuda("Computing eigenvectors\n");
    
    //Interface to arpack routines
    //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    ARPACK(pzneupd)(fcomm_, &rvec_, &howmny, select_, h_evals_, h_evecs_,
		    &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nev_,
		    &tol_, resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		    w_workl_, &lworkl_, w_rwork_ ,&info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in pzneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n",info_);
#else      
    ARPACK(zneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
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
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
    
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    if(comm_rank() == 0){
      if (arpack_logfile != NULL){
	ARPACK(finilog)(&arpack_log_u);
      }
    }
#else
    if (arpack_logfile != NULL)
      ARPACK(finilog)(&arpack_log_u);
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
    if(strncmp(L, spectrum, 1) == 0 && !arpack_param->usePolyAcc) {
      inverse = false;
    } else if(strncmp(S, spectrum, 1) == 0 && arpack_param->usePolyAcc) {
      inverse = false;
    } else if(strncmp(L, spectrum, 1) == 0 && arpack_param->usePolyAcc) {
      inverse = false;
    }

    sortAbs<double>(mod_h_evals_sorted, nconv, inverse, h_evals_sorted_idx);

    printfQuda("Sorted eigenvalues based on their absolute values:\n");

    //Sort the eigenvectors in absolute ascending order of the eigenvalues
    int length = 2*12*local_vol;
    double *h_evecs_sorted  = (double*)malloc(length*nev_*sizeof(double));
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
    

    cpuColorSpinorField *h_v3 = NULL;
    for(int i =0 ; i < nconv ; i++){
      cpuParam->v = (std::complex<double>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_;
      h_v3 = new cpuColorSpinorField(*cpuParam);
      
      // d_v = v
      *d_v = *h_v3;

      // d_v2 = M*v
      if(arpack_param->useNormOp && arpack_param->useDagger) {
	mat.MMdag(*d_v2,*d_v);
      }
      else if(arpack_param->useNormOp && !arpack_param->useDagger) {
	mat.MdagM(*d_v2,*d_v);
      }
      else if (!arpack_param->useNormOp && arpack_param->useDagger) {
	mat.Mdag(*d_v2,*d_v);
      }
      else {  
	mat.M(*d_v2,*d_v);
      }

      // lambda = v^dag * M*v
      h_evals_[i] = blas::cDotProduct(*d_v, *d_v2);
      
      std::complex<double> unit(1.0,0.0);
      std::complex<double> m_lambda(-real(h_evals_[i]),
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
    
    if(arpack_param->SVD) {
      
      //Compute SVD
      t1 = -(double)clock();
      for(int i=nev_/2 - 1; i >= 0; i--){
	
	//This function assumes that you have computed the eigenvectors
	//of MdagM, ie, the right SVD of M. The ith eigen vector in the array corresponds
	//to the ith smallest Right Singular Vector. We will sort the array as:
	//
	//     EV_array_MdagM = {Rev_0, Rev_1, Rev_2, ... , Rev_(n-1)}
	// to  SVD_array_M    = {Rsv_0, Lsv_0, Rsv_1, ... , Lsv_(n/2-1)}
	//
	//We start at Rev_(n/2-1), compute Lsv_(n/2-1), then move the vectors
	//to the n-2 and n-1 positions in the array respectively.
	
	//Copy Rev to Rsv
	memcpy((std::complex<double>*)h_evecs_ + h_evals_sorted_idx[2*i + 1]*ldv_,
	       (std::complex<double>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_,
	       2*ldv_*sizeof(double));
	
	cpuParam->v = (std::complex<double>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_;
	h_v3 = new cpuColorSpinorField(*cpuParam);      //Host Rev_i
	*d_v = *h_v3;                                   //Device Rev_i
	
	mat.M(*d_v2,*d_v);                              //  M Rev_i = M Rsv_i = sigma_i Lsv_i =   
	h_evals_[i] = blas::cDotProduct(*d_v2, *d_v2);  // (sigma_i)^2 = (sigma_i Lsv_i)^dag * (sigma_i Lsv_i)
	
	//Compute \sigma_i
	h_evals_[2*i + 1] = std::complex<double>(sqrt( real(h_evals_[i]) ) , sqrt(imag(h_evals_[i])*imag(h_evals_[i])));
	h_evals_[2*i + 0] = std::complex<double>(sqrt( real(h_evals_[i]) ) , sqrt(imag(h_evals_[i])*imag(h_evals_[i])));
	
	//Normalise the Lsv
	double L2norm = sqrt(blas::norm2(*d_v2));      
	blas::ax(1.0/L2norm, *d_v2);
	
	//Move Lsv
	*h_v3 = *d_v2;
	memcpy((std::complex<double>*)h_evecs_ + h_evals_sorted_idx[2*i]*ldv_,
	       (std::complex<double>*)h_evecs_ + h_evals_sorted_idx[i]*ldv_,
	       2*ldv_*sizeof(double));
	
	delete h_v3;
      }
      for(int i=0; i < nev_; i++){
	printfQuda("Sval[%04d] = %+e  %+e\n",
		   i, real(h_evals_[i]), imag(h_evals_[i]));
      }
      
      t1 += clock();
      printfQuda("Left singular pairs of Dirac operator computed in: %f sec\n",
		 t1/CLOCKS_PER_SEC);
    }
    
    
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

    
  template<typename Float>
  void lanczos_solve(void *h_evecs, void *h_evals, const Dirac &mat,
		     QudaEigParam *eig_param,
		     ColorSpinorParam *cpuParam){  

    int nev = eig_param->nk;
    int nkv = eig_param->nk + eig_param->np;
    int extra = 128;
    nkv += extra;
    eig_param->usePolyAcc = QUDA_BOOLEAN_NO;
    
    //Determine local volume for memory allocations
    int local_dim[4];
    int local_vol = 1;
    for(int i = 0 ; i < 4 ; i++){
      local_dim[i] = cpuParam->x[i];
      local_vol *= local_dim[i];
    }
    int length = 2*12*local_vol;
    
    Float *mod_h_evals_sorted  = (Float*)malloc(nkv*sizeof(Float));
    Float *h_evals_resid  = (Float*)malloc(nkv*sizeof(Float));
    Float *h_evals_alt  = (Float*)malloc(nkv*sizeof(Float));
    int *h_evals_sorted_idx = (int*)malloc(nkv*sizeof(int));

    //Alias pointers
    std::complex<Float> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<Float>*) (Float*)(h_evecs);    
    std::complex<Float> *h_evals_ = nullptr;
    h_evals_ = (std::complex<Float>*) (Float*)(h_evals);
    
    //Ritz vectors
    std::vector<cpuColorSpinorField*> h_ritzVecs;
    std::complex<Float> **ritzVecs = (std::complex<Float>**)malloc((1+nkv)*sizeof(std::complex<Float*>));

    //Allocate space for the Ritz vectors.
    for(int i=0; i<nkv+1; i++) {
      ritzVecs[i] = (std::complex<Float>*)malloc(length/2*sizeof(std::complex<Float>));
      cpuParam->v = (std::complex<Float>*)ritzVecs[i];
      h_ritzVecs.emplace_back(new cpuColorSpinorField(*cpuParam));
      blas::zero(*(h_ritzVecs[i]));
    }

    //Tridiagonal matrix
    Float alpha[nkv];
    Float  beta[nkv];
    for(int i=0; i<nkv; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }
    using Eigen::MatrixXd;
    //MatrixXd triDiag = MatrixXd::Zero(nkv, nkv);   

    //Host side vectors
    cpuColorSpinorField *h_v = nullptr;
    cpuParam->create = QUDA_REFERENCE_FIELD_CREATE;
    cpuParam->gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    cpuParam->v = (Float*)h_evecs_; //Copy device data from host array (init guess)
    h_v = new cpuColorSpinorField(*cpuParam);
    
    //Device side vectors
    cudaColorSpinorField *d_v = nullptr;
    cudaColorSpinorField *d_v2 = nullptr;    
    ColorSpinorParam cudaParam(*cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    sizeof(Float) == sizeof(double) ?
      cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER :
      cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    d_v = new cudaColorSpinorField(cudaParam);
    d_v2 = new cudaColorSpinorField(cudaParam);
    
    std::vector<cudaColorSpinorField*> d_ritzVecs;
    //Allocate space for the eigenvectors and initialise.
    for(int i=0; i<nkv+1; i++) {
      d_ritzVecs.emplace_back(new cudaColorSpinorField(cudaParam));
      blas::zero(*(d_ritzVecs[i]));
    }

    //Populate source with randoms.
    printfQuda("Using random guess\n");
    d_v -> Source(QUDA_RANDOM_SOURCE);
    //*d_v = *h_v;
    
    //Ensure we are not trying to compute on a zero-field source    
    const Float norm = sqrt(blas::norm2(*d_v));
    if(norm == 0){
      errorQuda("Initial residual is zero.\n");
      return;
    }
    //Normalised initial source
    blas::ax(1.0/norm, *d_v);


    // START LANCZOS
    // Thick restart Lanczos Method for Symmetric Eigenvalue Problems
    // Kesheng Wu and Horst D. Simon
    //-----------------------------------------
    
    //Begin iteration    
    for (int k = 0; k < nkv; ++k) {          
      
      //Abbreviated Initial Step
      if(k == 0) {
	
	//q_0 = r / norm;
	*d_ritzVecs[0] = *d_v;
	
	//r = M * q_0
	//apply matrix-vector operation here:
	if(eig_param->usePolyAcc) {
	  polyOp<Float, QudaEigParam>(mat, *d_v, *(d_ritzVecs[0]), eig_param);
	  //blas::ax(1.0/sqrt(blas::norm2(*d_v)), *d_v);
	}
	else {
	  if(eig_param->useNormOp && eig_param->useDagger) {
	    mat.MMdag(*d_v, *(d_ritzVecs[0]));
	  }
	  else if(eig_param->useNormOp && !eig_param->useDagger) {
	    mat.MdagM(*d_v, *(d_ritzVecs[0]));
	  }
	  else if (!eig_param->useNormOp && eig_param->useDagger) {
	    mat.Mdag(*d_v, *(d_ritzVecs[0]));
	  }
	  else {  
	    mat.M(*d_v, *(d_ritzVecs[0]));
	  }
	}
		
	//A = r^dag * M * r
	alpha[0] = blas::reDotProduct(*(d_ritzVecs[0]), *d_v);    
	//r = r - A * q_0 
	blas::axpy(-alpha[0], *d_ritzVecs[0], *d_v);
	
	beta[0] = sqrt(blas::norm2(*d_v));    
	blas::zero(*(d_ritzVecs[1]));
	blas::axpy(1.0/beta[0], *d_v, *d_ritzVecs[1]);    
	
      } else {
      
	//q_k = r / norm;
	
	//r = M * q_k
	//apply matrix-vector operation here:
	if(eig_param->usePolyAcc) {
	  polyOp<Float, QudaEigParam>(mat, *d_v, *(d_ritzVecs[k]), eig_param);
	  //blas::ax(1.0/sqrt(blas::norm2(*d_v)), *d_v);
	}
	else {
	  if(eig_param->useNormOp && eig_param->useDagger) {
	    mat.MMdag(*d_v, *(d_ritzVecs[k]));
	  }
	  else if(eig_param->useNormOp && !eig_param->useDagger) {
	    mat.MdagM(*d_v, *(d_ritzVecs[k]));
	  }
	  else if (!eig_param->useNormOp && eig_param->useDagger) {
	    mat.Mdag(*d_v, *(d_ritzVecs[k]));
	  }
	  else {  
	    mat.M(*d_v, *(d_ritzVecs[k]));
	  }
	}

	//r = r - B_{k-1} * q_{k-1}
	blas::axpy(-beta[k-1], *d_ritzVecs[k-1], *d_v);      
	
	//A_k = r^dag * M * r
	alpha[k] = blas::reDotProduct(*(d_ritzVecs[k]), *d_v);      	

	//r = r - A_k * q_k 
	blas::axpy(-alpha[k], *d_ritzVecs[k], *d_v);      
	
	//B_k = ||r||_2
	beta[k] = sqrt(blas::norm2(*d_v));
	
	//Orthonormalise      
	for(int i=0; i<k+1; i++) {
	  Complex C = blas::cDotProduct(*(d_ritzVecs[i]), *d_v); //<i,k>
	  blas::caxpy(-C, *d_ritzVecs[i], *d_v); // k-<i,k>i
	  //check ortho
	  C = blas::cDotProduct(*(d_ritzVecs[i]), *d_v); //<i,k>
	  if(fabs(C) > 1e-8) {
	    printfQuda("init %d %d ortho (%+e,%+e)\n", k, i, real(C), imag(C));
	  }
	}

	//The final vector is the q_{m+1}
	blas::zero(*(d_ritzVecs[k+1]));
	blas::axpy(1.0/beta[k], *d_v, *(d_ritzVecs[k+1]));
	
      }
    }

    using Eigen::MatrixXd;
    MatrixXd triDiag = MatrixXd::Zero(nkv, nkv);

    
    // Compute Y_{nkv,nev}
    Float Ymat[nkv][nkv];
    using Eigen::MatrixXd;
    triDiag = MatrixXd::Zero(nkv, nkv);
    
    for(int i=0; i<nkv; i++) {
      triDiag(i,i) = alpha[i];
      if(i<nkv-1) {
	triDiag(i+1,i  ) = beta[i];
	triDiag(i,  i+1) = beta[i];
      }
      else if(i<nkv-1) {
	triDiag(i+1,i  ) = beta[i];
	triDiag(i,  i+1) = beta[i];
      }
    }

    //std::cout << "The tridiag after initial pass:" << std::endl << triDiag << std::endl << std::endl;
    
    //Eigensolve T, sort low -> high, place nev evecs in Y
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(triDiag);
    for(int i=0; i<nkv; i++)
      for(int j=0; j<nkv; j++) 
	Ymat[i][j] = eigenSolver.eigenvectors().col(i)[j];
    
    
    // Q_tilde_{nev,nkv} = Q * Y
    
    /*
    //Copy all q vectors to host for manipulation
    for(int i=0; i<nkv; i++) {
      *h_ritzVecs[i] = *d_ritzVecs[i];
    }    
    
    //loop over rows of Q
    for(int j=0; j<nkv; j++) {
      
      //put jth row of Q in temp
      std::complex<Float> tmp[nkv];
      for(int i=0; i<nkv; i++) {
	tmp[i] = ritzVecs[i][j];
	//if(j==0) printfQuda("Original row %e %e\n", real(tmp[i]), imag(tmp[i]));
      }
      
      std::complex<Float> sum = 0.0;
      //loop over columns of Y 
      for(int k=0; k<nev; k++) {
	//take product of jth row of Q and kth column of Y
	//to compute Q_tilde[k][j].
	for(int l=0; l<nkv; l++) {
	  sum += tmp[l]*Ymat[k][l];
	  //if(j==0) printfQuda("%d %d New row %e %e %e %e %e \n", k, j, real(tmp[l]), imag(tmp[l]), Ymat[k][l], real(sum), imag(sum));
	}
	ritzVecs[k][j] = sum;
	sum *= 0.0;
      }
    }
    
    //Copy all nev q vectors to device
    for(int i=0; i<nkv; i++) {
      *d_ritzVecs[i] = *h_ritzVecs[i];
    }    
    */
    
    for(int i=0; i<nkv; i++) {
      h_evals_alt[i] = eigenSolver.eigenvalues()[i];
    }
    
    for(int i=0; i<nkv; i++){
      h_evals_sorted_idx[i] = i;
      mod_h_evals_sorted[i] = h_evals_alt[i];
    }
    
    sortAbs<Float>(mod_h_evals_sorted, nkv, true, h_evals_sorted_idx);
    
    for(int i=0; i<nev; i++){
      int idx = h_evals_sorted_idx[i];
      printfQuda("EigValue[%04d] = %+e  %+e  Residual: %+e\n",		 
		 i, h_evals_alt[idx], 0.0, 0.0); //FIXME
    }
    
    free(mod_h_evals_sorted);
    free(h_evals_resid);
    free(h_evals_alt);
    free(h_evals_sorted_idx);
    delete h_v;
    delete d_v;
    delete d_v2;
    for(int i=0; i<nkv+1; i++) {
      delete h_ritzVecs[i];
      delete d_ritzVecs[i];
      delete ritzVecs[i];
    }
    delete ritzVecs;
  }
  
  void arpackSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		   QudaArpackParam *arpack_param,
		   ColorSpinorParam *cpuParam){
    
    if(arpack_param->arpackPrec == QUDA_DOUBLE_PRECISION) {
      arpack_solve_double(h_evecs, h_evals, mat, arpack_param, cpuParam);
      //arpack_solve<double>(h_evecs, h_evals, mat, arpack_param, cpuParam);
    }
    else {
      arpack_solve_float(h_evecs, h_evals, mat, arpack_param, cpuParam);
      //arpack_solve<float>(h_evecs, h_evals, mat, arpack_param, cpuParam);
    }    
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
}


