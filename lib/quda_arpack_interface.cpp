#include <quda_arpack_interface.h>
#include <tests/misc.h>

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

  
  template<typename Float>
  void polyOp(const Dirac &mat,
	      cudaColorSpinorField &out,
	      const cudaColorSpinorField &in,	   
	      QudaArpackParam *arpack_param) {
    
    Float delta,theta;
    Float sigma,sigma1,sigma_old;
    Float d1,d2,d3;
    
    Float a = arpack_param->amin;
    Float b = arpack_param->amax;
    int polyDeg = arpack_param->polyDeg;
    
    delta = (b-a)/2.0;
    theta = (b+a)/2.0;    
    sigma1 = -delta/theta;
    
    d1 =  sigma1/delta;
    d2 =  1.0;
    
    blas::copy(out,in);
    if(arpack_param->useNormOp && arpack_param->useDagger) {
      mat.MMdag(out,in);
    }
    else if(arpack_param->useNormOp && !arpack_param->useDagger) {
      mat.MdagM(out,in);
    }
    else if (!arpack_param->useNormOp && arpack_param->useDagger) {
      mat.Mdag(out,in);
    }
    else {  
      mat.M(out,in);
    }
    blas::axpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);
    
    if(polyDeg == 1 )
      return;
    
    cudaColorSpinorField *tmp1 = new cudaColorSpinorField(in);
    cudaColorSpinorField *tmp2 = new cudaColorSpinorField(in);
    
    blas::copy(*tmp1,in);
    blas::copy(*tmp2,out);
    
    sigma_old = sigma1;
    
    for(int i=2; i <= polyDeg; i++){
      sigma = 1.0/( (2.0/sigma1) - sigma_old );
    
      d1 = 2.0*sigma/delta;
      d2 = -d1*theta;
      d3 = -sigma*sigma_old;

      if(arpack_param->useNormOp && arpack_param->useDagger) {
	mat.MMdag(out, *tmp2);
      }
      else if(arpack_param->useNormOp && !arpack_param->useDagger) {
	mat.MdagM(out, *tmp2);
      }
      else if (!arpack_param->useNormOp && arpack_param->useDagger) {
	mat.Mdag(out, *tmp2);
      }
      else {  
	mat.M(out, *tmp2);
      }	  

      blas::ax(d3,*tmp1);
      std::complex<Float> d1c(d1,0);
      std::complex<Float> d2c(d2,0);
      blas::cxpaypbz(*tmp1,d2c,*tmp2,d1c,out);
      blas::copy(*tmp1,*tmp2);
      blas::copy(*tmp2,out);
      sigma_old = sigma;
    }
    
    delete tmp1;
    delete tmp2;
  }

  void matVec(cudaColorSpinorField *d_v,
	      cpuColorSpinorField *h_v,
	      cudaColorSpinorField *d_v2,	      
	      cpuColorSpinorField *h_v2,
	      const Dirac &mat,
	      QudaArpackParam *arpack_param) {
    
    *d_v = *h_v;
    
    //apply matrix-vector operation here:
    if(arpack_param->usePolyAcc) {
      polyOp<float>(mat, *d_v2, *d_v, arpack_param);
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

  
  void arpack_solve_float(void *h_evecs, void *h_evals,
			  const Dirac &mat,
			  QudaArpackParam *arpack_param,
			  ColorSpinorParam *cpuParam){

    // arpackSolveInit() START
    
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
    int info_ = 0;
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

    //Assign values to ARPACK params 
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    iparam_[7]  = arpack_param->arpackMode;


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

    // arpackSolveInit() END

    // arpackSolveEigenModesFloat/Double() START
    
    float tol_ = arpack_param->arpackTol;    
    float *mod_h_evals_sorted  = (float*)malloc(nkv_*sizeof(float));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //ARPACK workspace
    std::complex<float> sigma_ = 0.0;
    std::complex<float> *resid_ =
      (std::complex<float> *) malloc(ldv_*sizeof(std::complex<float>));
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
    cudaColorSpinorField *d_v = nullptr;    
    cpuColorSpinorField *h_v2 = nullptr;
    cudaColorSpinorField *d_v2 = nullptr;
    cudaColorSpinorField *resid = nullptr;    

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
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pcnaupd info = %d. Exiting.",info_);
      }
#else
      ARPACK(cnaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
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
	//cudaParam.print();
	
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
	  polyOp<float>(mat, *d_v2, *d_v, arpack_param);
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
		    w_workl_, &lworkl_, w_rwork_ ,&info_);
    if (info_ == -15) {
      errorQuda("\nError in pcneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pcneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(cneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in cneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in cneupd info = %d. Exiting.\n", info_);
#endif

    // arpackSolveEigenModesFloat/Double() END

    // template<typename<Float>
    // arpackSolveSortCheck() START  
    
    //Print out the computed ritz values, absolute values, and their error estimates
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      mod_h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //Sort the eigenvalues in absolute ascending order
    t1 =  -((double)clock());;
    sortAbs<float>(mod_h_evals_sorted, nconv, true, h_evals_sorted_idx);
    t1 +=  clock();

    //Print sorted evals
    printfQuda("Sorting time: %f sec\n",t1/CLOCKS_PER_SEC);
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
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    
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
      //printfQuda("local_vol = %d\n",local_vol);
    }
    
    // all FORTRAN communication uses underscored 
    int ido_; 
    int info_;
    int *ipntr_ = (int*)malloc(14*sizeof(int));
    int *iparam_ = (int*)malloc(11*sizeof(int));
    int n_    = local_vol*4*3,
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter = arpack_param->arpackMaxiter;

    double tol_ = arpack_param->arpackTol;
    
    double *mod_h_evals_sorted  = (double*)malloc(nkv_*sizeof(double));
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //ARPACK workspace
    std::complex<double> sigma_ = 0.0;
    std::complex<double> *resid_ =
      (std::complex<double> *) malloc(ldv_*sizeof(std::complex<double>));
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

    //Assign values to ARPACK params 
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    iparam_[7]  = arpack_param->arpackMode;

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
    
    int iter_cnt= 0;

    bool allocate = true;
    cpuColorSpinorField *h_v = nullptr;
    cudaColorSpinorField *d_v = nullptr;    
    cpuColorSpinorField *h_v2 = nullptr;
    cudaColorSpinorField *d_v2 = nullptr;
    cudaColorSpinorField *resid = nullptr;    

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
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
      }
#else
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
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
	cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
	//cudaParam.print();
	
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
	  polyOp<double>(mat, *d_v2, *d_v, arpack_param);
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
		    w_workl_, &lworkl_, w_rwork_ ,&info_);
    if (info_ == -15) {
      errorQuda("\nError in pzneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(zneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif

    //Print out the computed ritz values, absolute values, and their error estimates
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      mod_h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //Sort the eigenvalues in absolute ascending order
    t1 =  -((double)clock());;
    sortAbs<double>(mod_h_evals_sorted, nconv, true, h_evals_sorted_idx);
    t1 +=  clock();

    //Print sorted evals
    printfQuda("Sorting time: %f sec\n",t1/CLOCKS_PER_SEC);
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
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    
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

  void arpack_solve_init(QudaArpackParam *arpack_param,
			 arpackFortranParam *afp,
			 int local_vol){
    
    // arpackSolveInit() START

    afp->fcomm_ = nullptr;
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    afp->fcomm_ = static_cast<int*>(&mpi_comm_fort);
#endif
    
    // all FORTRAN communication uses underscored 
    afp->ido_ = 0; 
    afp->info_ = 0;
    
    afp->ipntr_    = (int*)malloc(14*sizeof(int));
    afp->iparam_   = (int*)malloc(11*sizeof(int));
    afp->n_        = local_vol*4*3;
    afp->nev_      = arpack_param->nEv;
    afp->nkv_      = arpack_param->nKv;
    afp->ldv_      = local_vol*4*3;
    afp->lworkl_   = (3 * (afp->nkv_)*(afp->nkv_) + 5*(afp->nkv_)) * 2;
    afp->rvec_     = 1;
    afp->max_iter_ = arpack_param->arpackMaxiter;
    afp->select_   = (int*)malloc(afp->nkv_*sizeof(int));
    
    //ARPACK problem type to be solved
    afp->howmany_  = 'P';
    afp->bmat_     = 'I';   
    afp->spectrum_ = strdup("SR"); //Initialsed just to stop the compiler warning...
    
    //Assign values
    afp->iparam_[0]  = 1;
    afp->iparam_[2]  = afp->max_iter_;
    afp->iparam_[3]  = 1;
    afp->iparam_[6]  = 1;
    afp->iparam_[7]  = arpack_param->arpackMode;
    
    if(arpack_param->usePolyAcc){
      if(arpack_param->spectrum == QUDA_SR_SPECTRUM) afp->spectrum_ = strdup("LR");
      else if(arpack_param->spectrum == QUDA_LR_SPECTRUM) afp->spectrum_ = strdup("SR");
      else if(arpack_param->spectrum == QUDA_SM_SPECTRUM) afp->spectrum_ = strdup("LM");
      else if(arpack_param->spectrum == QUDA_LM_SPECTRUM) afp->spectrum_ = strdup("SM");
      else if(arpack_param->spectrum == QUDA_SI_SPECTRUM) afp->spectrum_ = strdup("LI");
      else if(arpack_param->spectrum == QUDA_LI_SPECTRUM) afp->spectrum_ = strdup("SI");
    }
    else{
      if(arpack_param->spectrum == QUDA_SR_SPECTRUM) afp->spectrum_ = strdup("SR");
      else if(arpack_param->spectrum == QUDA_LR_SPECTRUM) afp->spectrum_ = strdup("LR");
      else if(arpack_param->spectrum == QUDA_SM_SPECTRUM) afp->spectrum_ = strdup("SM");
      else if(arpack_param->spectrum == QUDA_LM_SPECTRUM) afp->spectrum_ = strdup("LM");
      else if(arpack_param->spectrum == QUDA_SI_SPECTRUM) afp->spectrum_ = strdup("SI");
      else if(arpack_param->spectrum == QUDA_LI_SPECTRUM) afp->spectrum_ = strdup("LI");
    }
  }    

  void sortCheck(void *h_evecs, void *h_evals, arpackFortranParam *afp,
		 std::complex<float> *w_workl_, int local_vol) {
    
    int *h_evals_sorted_idx = (int*)malloc(afp->nkv_*sizeof(int));
    float *mod_h_evals_sorted  = (float*)malloc(afp->nkv_*sizeof(float));

    //Alias pointers
    std::complex<float> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<float>*) (float*)(h_evecs);    
    std::complex<float> *h_evals_ = nullptr;
    h_evals_ = (std::complex<float>*) (float*)(h_evals);
    
    //Memory checks
    if((mod_h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //Print out the computed ritz values, absolute values, and their error estimates
    int nconv = afp->iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(h_evals_[j]),
		 std::abs(*(w_workl_ + afp->ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      mod_h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //Sort the eigenvalues in absolute ascending order
    float t1 =  -((double)clock());;
    sortAbs<float>(mod_h_evals_sorted, nconv, true, h_evals_sorted_idx);
    t1 +=  clock();

    //Print sorted evals
    printfQuda("Sorting time: %f sec\n",t1/CLOCKS_PER_SEC);
    printfQuda("Sorted eigenvalues based on their absolute values:\n");

    //Sort the eigenvectors in absolute ascending order of the eigenvalues
    int length = 2*12*local_vol;
    float *h_evecs_sorted  = (float*)malloc(length*afp->nev_*sizeof(float));
    for(int a=0; a<nconv; a++) {
      memcpy(h_evecs_sorted + a*length,
	     (float*)h_evecs_ + h_evals_sorted_idx[a]*length,
	     length*sizeof(float) );
    }
    memcpy(h_evecs, h_evecs_sorted, nconv*length*sizeof(float));
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + afp->ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    
    // Print additional convergence information.
    if( (afp->info_) == 1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(afp->info_ == 3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }    
  }
  
  void arpack_solve_eigenmodes(QudaArpackParam *arpack_param,
			       arpackFortranParam *afp,
			       ColorSpinorParam *cpuParam,
			       void *h_evecs, void *h_evals,
			       const Dirac &mat,
			       int local_vol){
    
    float tol_ = arpack_param->arpackTol;    
    
    //ARPACK workspace
    std::complex<float> sigma_ = 0.0;
    std::complex<float> *resid_ =
      (std::complex<float> *) malloc(afp->ldv_*sizeof(std::complex<float>));
    std::complex<float> *w_workd_ =
      (std::complex<float> *) malloc(3*afp->ldv_*sizeof(std::complex<float>));
    std::complex<float> *w_workl_ =
      (std::complex<float> *) malloc(afp->lworkl_*sizeof(std::complex<float>)); 
    std::complex<float> *w_workev_=
      (std::complex<float> *) malloc(2*afp->nkv_*sizeof(std::complex<float>));    
    float *w_rwork_  = (float *)malloc(afp->nkv_*sizeof(float));    

    //Alias pointers
    std::complex<float> *h_evecs_ = nullptr;
    h_evecs_ = (std::complex<float>*) (float*)(h_evecs);    
    std::complex<float> *h_evals_ = nullptr;
    h_evals_ = (std::complex<float>*) (float*)(h_evals);

    //Memory checks
    if((resid_ == nullptr) ||  
       (w_workd_ == nullptr) || 
       (w_workl_ == nullptr) ||
       (w_workev_ == nullptr) ||
       (w_rwork_ == nullptr)) {
      errorQuda("eigenSolver: not enough memory for ARPACK workspace.\n");
    }    

    int *fcomm_  = afp->fcomm_;
    int ido_     = afp->ido_; 
    int info_    = afp->info_;
    int *ipntr_  = afp->ipntr_;
    int *iparam_ = afp->iparam_;
    int n_       = afp->n_;
    int nev_     = afp->nev_;
    int nkv_     = afp->nkv_;
    int ldv_     = afp->ldv_;
    int lworkl_  = (3 * nkv_*nkv_ + 5*nkv_) * 2;
    int rvec_    = afp->rvec_;;    
    int *select_ = afp->select_;
    
    char bmat_ = afp->bmat_;
    char howmany_ = afp->howmany_;
    char *spectrum_ = afp->spectrum_;
    
    int iter_cnt= 0;    
    double t1;
    
    do {

      t1 = -((double)clock());      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pcnaupd)(fcomm_, &ido_, &bmat_, &n_, spectrum_, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &ldv_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in pcnaupd info = %d. Exiting.",info_);
      }
#else
      ARPACK(cnaupd)(&ido_, &bmat_, &n_, spectrum_, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &ldv_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
      if (info_ != 0) {
	arpackErrorHelpNAUPD();
	errorQuda("\nError in cnaupd info = %d. Exiting.",info_);
      }
#endif

      bool allocate = true;
      cpuColorSpinorField *h_v = nullptr;
      cudaColorSpinorField *d_v = nullptr;    
      cpuColorSpinorField *h_v2 = nullptr;
      cudaColorSpinorField *d_v2 = nullptr;
      
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
	//cudaParam.print();
	
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	// MatVec() START
	matVec(d_v, h_v, d_v2, h_v2, mat, arpack_param);       	
	
      }

      t1 += clock();
	
      printfQuda("Arpack Iteration %s: %d (%e secs)\n", arpack_param->usePolyAcc ? "(with poly acc) " : "", iter_cnt, t1/CLOCKS_PER_SEC);
      iter_cnt++;

    } while (99 != ido_ && iter_cnt < afp->max_iter_);

    //Subspace calulated sucessfully. Compute nEv eigenvectors and values 
    
    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);      
    printfQuda("Computing eigenvectors\n");
    
    //Interface to arpack routines
    //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    ARPACK(pcneupd)(fcomm_, &rvec_, &howmany_, select_, h_evals_, h_evecs_,
		    &n_, &sigma_, w_workev_, &bmat_, &n_, spectrum_, &nev_,
		    &tol_, resid_, &nkv_, h_evecs_, &ldv_, iparam_, ipntr_, w_workd_,
		    w_workl_, &lworkl_, w_rwork_ ,&info_);
    if (info_ == -15) {
      errorQuda("\nError in pcneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in pcneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(cneupd)(&rvec_, &howmany_, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat_, &n_, spectrum_, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &ldv_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in cneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } else if (info_ != 0) errorQuda("\nError in cneupd info = %d. Exiting.\n", info_);
#endif

    sortCheck(h_evecs, h_evals, afp, w_workl_, local_vol);
    
  }
  
  void arpackSolveProto(void *h_evecs, void *h_evals, const Dirac &mat,
			QudaArpackParam *arpack_param,
			ColorSpinorParam *cpuParam){  
    
    
    //Construct parameters and memory allocation 
    //------------------------------------------
    
    //Determine local volume for memory allocationss
    int local_dim[4];
    int local_vol = 1;
    for(int i = 0 ; i < 4 ; i++){
      local_dim[i] = cpuParam->x[i];
      local_vol *= local_dim[i];
    }

    arpackFortranParam afp;    
    arpack_solve_init(arpack_param, &afp, local_vol);
    
    arpack_solve_eigenmodes(arpack_param, &afp, cpuParam,
			    h_evals, h_evecs, mat, local_vol);
    
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
      //arpack_solve<float>(h_evecs, h_evals, matSmooth, arpack_param, cpuParam);
    }    
  }  
  
  void arpackErrorHelpNAUPD() {
    printfQuda(" INFO Integer.  (INPUT/OUTPUT)\n");
    printfQuda("      If INFO .EQ. 0, a randomly initial residual vector is used.\n");
    printfQuda("      If INFO .NE. 0, RESID contains the initial residual vector,\n");
    printfQuda("                         possibly from a previous run.\n");
    printfQuda("      Error flag on output.\n");
    printfQuda("      =  0: Normal exit.\n");
    printfQuda("      =  1: Maximum number of iterations taken.\n");
    printfQuda("         All possible eigenvalues of OP has been found. IPARAM(5)\n");
    printfQuda("         returns the number of wanted converged Ritz values.\n");
    printfQuda("      =  2: No longer an informational error. Deprecated starting\n");
    printfQuda("         with release 2 of ARPACK.\n");
    printfQuda("      =  3: No shifts could be applied during a cycle of the\n");
    printfQuda("         Implicitly restarted Arnoldi iteration. One possibility\n");
    printfQuda("         is to increase the size of NCV relative to NEV.\n");
    printfQuda("         See remark 4 below.\n");
    printfQuda("      = -1: N must be positive.\n");
    printfQuda("      = -2: NEV must be positive.\n");
    printfQuda("      = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("      = -4: The maximum number of Arnoldi update iteration\n");
    printfQuda("         must be greater than zero.\n");
    printfQuda("      = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("      = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("      = -7: Length of private work array is not sufficient.\n");
    printfQuda("      = -8: Error return from LAPACK eigenvalue calculation;\n");
    printfQuda("      = -9: Starting vector is zero.\n");
    printfQuda("      = -10: IPARAM(7) must be 1,2,3.\n");
    printfQuda("      = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("      = -12: IPARAM(1) must be equal to 0 or 1.\n");
    printfQuda("      = -9999: Could not build an Arnoldi factorization.\n");
    printfQuda("         User input error highly likely.  Please\n");
    printfQuda("         check actual array dimensions and layout.\n");
    printfQuda("         IPARAM(5) returns the size of the current Arnoldi\n");
    printfQuda("         factorization.\n");
  }

  void arpackErrorHelpNEUPD() {
    printfQuda(" INFO    Integer.  (OUTPUT)\n");
    printfQuda("        Error flag on output.\n");
    printfQuda("        =  0: Normal exit.\n");
    printfQuda("        =  1: The Schur form computed by LAPACK routine csheqr\n");
    printfQuda("              could not be reordered by LAPACK routine ztrsen.\n");
    printfQuda("              Re-enter subroutine zneupd with IPARAM(5)=NCV and\n");
    printfQuda("              increase the size of the array D to have\n");
    printfQuda("              dimension at least dimension NCV and allocate at\n");
    printfQuda("              least NCV\n");
    printfQuda("              columns for Z. NOTE: Not necessary if Z and V share\n");
    printfQuda("              the same space. Please notify the authors if this\n");
    printfQuda("              error occurs.\n");
    printfQuda("        = -1: N must be positive.\n");
    printfQuda("        = -2: NEV must be positive.\n");
    printfQuda("        = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("        = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("        = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("        = -7: Length of private work WORKL array is inufficient.\n");
    printfQuda("        = -8: Error return from LAPACK eigenvalue calculation.\n");
    printfQuda("              This should never happened.\n");
    printfQuda("        = -9: Error return from calculation of eigenvectors.\n");
    printfQuda("              Informational error from LAPACK routine ztrevc.\n");
    printfQuda("        = -10: IPARAM(7) must be 1,2,3\n");
    printfQuda("        = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("        = -12: HOWMNY = 'S' not yet implemented\n");
    printfQuda("        = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.\n");
    printfQuda("        = -14: ZNAUPD did not find any eigenvalues to sufficient\n");
    printfQuda("               accuracy.\n");
    printfQuda("        = -15: ZNEUPD got a different count of the number of\n");
    printfQuda("               converged Ritz values than ZNAUPD got. This\n");
    printfQuda("               indicates the user probably made an error in\n");
    printfQuda("               passing data from ZNAUPD to ZNEUPD or that the\n");
    printfQuda("               data was modified before entering ZNEUPD\n");
  }
}
