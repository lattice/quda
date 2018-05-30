#include <quda_arpack_interface.h>
#include <tests/misc.h>

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
#include <mpi.h>
#endif

namespace quda{
  
  template<typename Float>
  void polyOp(Dirac &mat,
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
    mat.MdagM(out,in);    
    blas::axpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);
    
    if(polyDeg == 1 )
      return;
    
    cudaColorSpinorField *tm1 = new cudaColorSpinorField(in);
    cudaColorSpinorField *tm2 = new cudaColorSpinorField(in);
    
    blas::copy(*tm1,in);
    blas::copy(*tm2,out);
    
    sigma_old = sigma1;
    
    for(int i=2; i <= polyDeg; i++){
      sigma = 1.0/( (2.0/sigma1) - sigma_old );
    
      d1 = 2.0*sigma/delta;
      d2 = -d1*theta;
      d3 = -sigma*sigma_old;
      
      mat.MdagM(out, *tm2);
      blas::ax(d3,*tm1);
      std::complex<Float> d1c(d1,0);
      std::complex<Float> d2c(d2,0);
      blas::cxpaypbz(*tm1,d2c,*tm2,d1c,out);
      blas::copy(*tm1,*tm2);
      blas::copy(*tm2,out);
      sigma_old = sigma;
    }
    
    delete tm1;
    delete tm2;
  }

  void arpack_solve_float(void *h_evecs, void *h_evals,
			  QudaInvertParam *inv_param,
			  QudaArpackParam *arpack_param,
			  DiracParam *d_param, int *local_dim){

    //Construct parameters and memory allocation
    //---------------------------------------------------------------------------------

    //MPI objects
    int *fcomm_ = nullptr;
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm_ = static_cast<int*>(&mpi_comm_fort);
#endif

    //Determine local volume for memory allocations
    int local_vol = 1;
    for(int i = 0 ; i < 4 ; i++){ 
      local_vol *= local_dim[i];
    }
    
    // all FORTRAN communication uses underscored 
    int ido_; 
    int info_;
    int *ipntr_ = (int *) malloc(14 *sizeof(int));
    int *iparam_ = (int *) malloc(11 *sizeof(int));
    int n_    = local_vol*4*3,
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter = arpack_param->arpackMaxiter;

    float tol_ = arpack_param->arpackTol;
    
    float *h_evals_sorted  = (float*)malloc(nkv_*2*sizeof(float));
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //Memory checks
    if((h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //Construct operator.
    Dirac *mat = Dirac::create(*d_param);
    
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
      if (info_ != 0) errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
#else
      ARPACK(cnaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in znaupd info = %d. Exiting.",info_);
#endif
      
      //If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if(allocate){

	//Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
	//less one, hence ipntr[0] - 1 to specify the correct address.
	ColorSpinorParam cpuParam(w_workd_ + ipntr_[0] - 1,
				  *inv_param, local_dim , inv_param->solution_type);

	cpuParam.print();
	
	h_v = new cpuColorSpinorField(cpuParam);
	//Adjust the position of the start of the array.
	cpuParam.v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = new cpuColorSpinorField(cpuParam);

	ColorSpinorParam cudaParam(cpuParam, *inv_param);
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaParam.print();
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  polyOp<double>(*mat, *d_v2, *d_v, arpack_param);
	}
	else {
	  mat->MdagM(*d_v2,*d_v);
	}
	
	*h_v2 = *d_v2;
      }

      t1 += clock();
      
      printfQuda("Arpack Iteration : %d (%e secs)\n", iter_cnt, t1/CLOCKS_PER_SEC);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors as values 
    
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
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(cneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in cneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif
    
    // print out the computed ritz values and their error estimates 
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //SORT THE EIGENVALUES in absolute ascending order
    t1 =  -((double)clock());
    //quicksort(nconv,sorted_evals,sorted_evals_index);
    //sortAbs(sorted_evals,nconv,false,sorted_evals_index);
    //Print sorted evals
    t1 += clock();
    printfQuda("Sorting time: %f sec\n",t1/CLOCKS_PER_SEC);
    printfQuda("Sorted eigenvalues based on their absolute values:\n");
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    
    
    /*Print additional convergence information.*/
    if( (info_) == 1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info_ == 3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
    
    /* cleanup */
    //free(fcomm_);
    free(ipntr_);
    free(iparam_);
    free(h_evals_sorted);
    free(h_evals_sorted_idx);
    free(resid_);
    free(w_workd_);
    free(w_workl_);
    free(w_workev_);
    free(w_rwork_);
    free(select_);
    free(spectrum);
    
    delete mat;
    delete h_v;
    delete h_v2;
    delete d_v;
    delete d_v2;
    
    return;
  }
  
  void arpack_solve_double(void *h_evecs, void *h_evals,
			   QudaInvertParam *inv_param,
			   QudaArpackParam *arpack_param,
			   DiracParam *d_param, int *local_dim){    
    
    //Construct parameters and memory allocation
    //---------------------------------------------------------------------------------
    //MPI objects
    int *fcomm_ = nullptr;
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm_ = static_cast<int*>(&mpi_comm_fort);
#endif

    //Determine local volume for memory allocations
    int local_vol = 1;
    for(int i = 0 ; i < 4 ; i++){ 
      local_vol *= local_dim[i];
    }
    
    // all FORTRAN communication uses underscored 
    int ido_; 
    int info_;
    int *ipntr_ = (int *) malloc(14 *sizeof(int));
    int *iparam_ = (int *) malloc(11 *sizeof(int));
    int n_    = local_vol*4*3,
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter_ = arpack_param->arpackMaxiter;

    double tol_ = arpack_param->arpackTol;
    
    double *h_evals_sorted  = (double*)malloc(nkv_*2*sizeof(double));
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //Memory checks
    if((h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //Construct operator.
    Dirac *mat = Dirac::create(*d_param);
    
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
    iparam_[2]  = max_iter_;
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

    //Start ARPACK routines
    //---------------------------------------------------------------------------------
    
    do {
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pznaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
#else
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in znaupd info = %d. Exiting.",info_);
#endif
      
      //If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if(allocate){

	//Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
	//less one, hence ipntr[0] - 1 to specify the correct address.
	ColorSpinorParam cpuParam(w_workd_ + ipntr_[0] - 1,
				  *inv_param, local_dim , false);

	h_v = new cpuColorSpinorField(cpuParam);
	//Adjust the position of the start of the array.
	cpuParam.v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = new cpuColorSpinorField(cpuParam);

	ColorSpinorParam cudaParam(cpuParam, *inv_param);
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  polyOp<double>(*mat, *d_v2, *d_v, arpack_param);
	}
	else {
	  mat->MdagM(*d_v2,*d_v);
	}
	
	*h_v2 = *d_v2;
      }
      
      printfQuda("Arpack Iteration : %d\n", iter_cnt);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter_);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors as values 
    
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
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(zneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif
    
    /* print out the computed ritz values and their error estimates */
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      h_evals_sorted[j] = std::abs(h_evals_[j]);
    }

    //SORT THE EIGENVALUES in absolute ascending order
    double t1 =  -((double)clock());
    //quicksort(nconv,sorted_evals,sorted_evals_index);
    //sortAbs(sorted_evals,nconv,false,sorted_evals_index);
    //Print sorted evals
    double t2 =  -((double)clock());
    printfQuda("Sorting time: %f sec\n",t2-t1);
    printfQuda("Sorted eigenvalues based on their absolute values:\n");
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+h_evals_sorted_idx[j])) );
    }      
    
    
    /*Print additional convergence information.*/
    if( (info_) == 1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info_ == 3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
      
    /* cleanup */
    //free(fcomm_);
    free(ipntr_);
    free(iparam_);
    free(h_evals_sorted);
    free(h_evals_sorted_idx);
    free(resid_);
    free(w_workd_);
    free(w_workl_);
    free(w_workev_);
    free(w_rwork_);
    free(select_);
    free(spectrum);
    
    delete mat;
    delete h_v;
    delete h_v2;
    delete d_v;
    delete d_v2;
    
    return;
  }

  void arpack_mg_comp_solve_double(void *h_evecs, void *h_evals,
				   DiracMatrix &matSmooth,
				   QudaInvertParam *inv_param,
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
      printfQuda("local_vol = %d\n",local_vol);
    }
    
    // all FORTRAN communication uses underscored 
    int ido_; 
    int info_;
    int *ipntr_ = (int *) malloc(14 *sizeof(int));
    int *iparam_ = (int *) malloc(11 *sizeof(int));
    int n_    = local_vol*4*3,
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol*4*3,
      lworkl_ = (3 * nkv_*nkv_ + 5*nkv_) * 2,
      rvec_   = 1;
    int max_iter = arpack_param->arpackMaxiter;

    double tol_ = arpack_param->arpackTol;
    
    double *h_evals_sorted  = (double*)malloc(nkv_*2*sizeof(double));
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //Memory checks
    if((h_evals_sorted == nullptr) ||
       (h_evals_sorted_idx == nullptr) ) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }
    
    //Construct operator.
    //Dirac *mat = Dirac::create(*d_param);
    
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

    //Start ARPACK routines
    //---------------------------------------------------------------------------------
    
    do {
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pznaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
#else
      ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in znaupd info = %d. Exiting.",info_);
#endif
      
      //If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if(allocate){

	//Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
	//less one, hence ipntr[0] - 1 to specify the correct address.
	//ColorSpinorParam cpuParam(w_workd_ + ipntr_[0] - 1,
	//*inv_param, local_dim , false);

	h_v = new cpuColorSpinorField(*cpuParam);
	//Adjust the position of the start of the array.
	cpuParam->v = w_workd_ + (ipntr_[1] - 1);
	h_v2 = new cpuColorSpinorField(*cpuParam);

	ColorSpinorParam cudaParam(*cpuParam, *inv_param);
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	d_v2->PrintVector(2048);
	d_v->PrintVector(2048);
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  //polyOp<double>(*mat, *d_v2, *d_v, arpack_param);
	}
	else {
	  matSmooth.Expose()->MdagM(*d_v2,*d_v);
	}

	*h_v2 = *d_v2;
      }
      
      printfQuda("Arpack Iteration : %d\n", iter_cnt);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors as values 
    
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
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(zneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.", info_);
    } else if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif
    
    /* print out the computed ritz values and their error estimates */
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
      h_evals_sorted_idx[j] = j;
      h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //SORT THE EIGENVALUES in absolute ascending order
    double t1 =  -((double)clock());
    //quicksort(nconv,sorted_evals,sorted_evals_index);
    //sortAbs(sorted_evals,nconv,false,sorted_evals_index);
    //Print sorted evals
    double t2 =  -((double)clock());
    printfQuda("Sorting time: %f sec\n",t2-t1);
    printfQuda("Sorted eigenvalues based on their absolute values:\n");
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		   imag(h_evals_[h_evals_sorted_idx[j]]),
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
    free(h_evals_sorted);
    free(h_evals_sorted_idx);
    free(resid_);
    free(w_workd_);
    free(w_workl_);
    free(w_workev_);
    free(w_rwork_);
    free(select_);
    free(spectrum);
    
    //delete mat;
    delete h_v;
    delete h_v2;
    delete d_v;
    delete d_v2;
    
    return;

    
  }

  void arpack_mg_comp_solve_float(void *h_evecs, void *h_evals,
				  DiracMatrix &matSmooth,
				  QudaInvertParam *inv_param,
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
      printfQuda("local_vol = %d\n",local_vol);
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

    float tol_ = arpack_param->arpackTol;
    
    float *h_evals_sorted  = (float*)malloc(nkv_*2*sizeof(float));
    int *h_evals_sorted_idx = (int*)malloc(nkv_*sizeof(int));

    //Memory checks
    if((h_evals_sorted == nullptr) ||
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

    double t1,t2;
    
    do {

      t1 = -((double)clock());
      
      //Interface to arpack routines
      //----------------------------
#if (defined (QMP_COMMS) || defined (MPI_COMMS))
      ARPACK(pcnaupd)(fcomm_, &ido_, &bmat, &n_, spectrum, &nev_, &tol_,
		      resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_,
		      w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in pznaupd info = %d. Exiting.",info_);
#else
      ARPACK(cnaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_,
		     h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_,
		     w_rwork_, &info_);
      if (info_ != 0) errorQuda("\nError in znaupd info = %d. Exiting.",info_);
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

	cpuParam->print();
	//ColorSpinorParam cudaParam(*cpuParam, *inv_param);
	ColorSpinorParam cudaParam(*cpuParam);
	cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	cudaParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
	cudaParam.print();
	
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	resid = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == 99 || info_ == 1)
	break;
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;
	for(int r = 0; r < 4; r++) {
	  //d_v2->PrintVector(r);
	  //d_v->PrintVector(r);
	}
	//apply matrix-vector operation here:
	if(arpack_param->usePolyAcc) {
	  //polyOp<float>((matSmooth.Expose()), *d_v2, *d_v, arpack_param);
	  //exit(0);
	}
	else {
	  matSmooth.Expose()->MdagM(*d_v2,*d_v);
	  //matSmooth(*d_v2,*d_v,*resid);
	}
	for(int r = 0; r < 4; r++) {
	  //d_v2->PrintVector(r);
	  //d_v->PrintVector(r);
	}
	*h_v2 = *d_v2;
	//exit(0);
      }

      t1 += clock();
	
      printfQuda("Arpack Iteration : %d (%e secs)\n", iter_cnt, t1/CLOCKS_PER_SEC);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);
    
    //Subspace calulated sucessfully. Compute nEv eigenvectors as values 
    
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
    } else if (info_ != 0) errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else      
    ARPACK(cneupd)(&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_,
		   w_workev_, &bmat, &n_, spectrum, &nev_, &tol_,
		   resid_, &nkv_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
		   w_workl_, &lworkl_, w_rwork_, &info_);
    if (info_ == -15) {
      errorQuda("\nError in cneupd info = %d. You likely need to\n"
		"increase the maximum ARPACK iterations. Exiting.\n", info_);
    } if (info_ != 0) errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif
    
    /* print out the computed ritz values and their error estimates */
    int nconv = iparam_[4];
    for(int j=0; j<nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[j]),
		 imag(h_evals_[j]),
		 std::abs(*(w_workl_ + ipntr_[10]-1+j)));
	h_evals_sorted_idx[j] = j;
	h_evals_sorted[j] = std::abs(h_evals_[j]);
    }
    
    //SORT THE EIGENVALUES in absolute ascending order
    t1 =  -((double)clock());;
    //quicksort(nconv,sorted_evals,sorted_evals_index);
    //sortAbs(sorted_evals,nconv,false,sorted_evals_index);
    //Print sorted evals
    t2 =  -((double)clock());;
    printfQuda("Sorting time: %f sec\n",t2-t1);
    printfQuda("Sorted eigenvalues based on their absolute values:\n");
    
    // print out the computed ritz values and their error estimates 
    for(int j=0; j< nconv; j++){
      printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		 real(h_evals_[h_evals_sorted_idx[j]]),
		 imag(h_evals_[h_evals_sorted_idx[j]]),
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
    free(h_evals_sorted);
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
  
  
  void arpackSolve(void *h_evecs, void *h_evals,
		   QudaInvertParam *inv_param,
		   QudaArpackParam *arpack_param,
		   DiracParam *d_param, int *local_dim){    

    if(arpack_param->arpackPrec == QUDA_DOUBLE_PRECISION) {
      arpack_solve_double(h_evecs,h_evals,inv_param,arpack_param,d_param,local_dim);
    }
    else {
      arpack_solve_float(h_evecs,h_evals,inv_param,arpack_param,d_param,local_dim);
    }
  }

  void arpackMGComparisonSolve(void *h_evecs, void *h_evals, DiracMatrix &matSmooth,
			       QudaInvertParam *inv_param,
			       QudaArpackParam *arpack_param,
			       ColorSpinorParam *cpuParam){

    if(arpack_param->arpackPrec == QUDA_DOUBLE_PRECISION) {
      arpack_mg_comp_solve_double(h_evecs,h_evals,matSmooth,
				  inv_param,arpack_param,
				  cpuParam);
    }
    else {
      arpack_mg_comp_solve_float(h_evecs,h_evals,matSmooth,
				 inv_param,arpack_param,
				 cpuParam);
    }
    
  }  
}


