#include <quda_arpack_interface.h>
#include <tests/misc.h>

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
#include <mpi.h>
#endif

//-------------------------------------------------------------
template<typename Float> void arpack_naupd(int &ido, char &bmat, int &n, char *which,
					   int &nev, Float &tol,
					   std::complex<Float> *resid, int &ncv,
					   std::complex<Float> *v, int &ldv,
					   int *iparam, int *ipntr,
					   std::complex<Float> *workd,
					   std::complex<Float> *workl, int &lworkl,
					   Float *rwork, int &info, int *fcomm) {
#ifdef ARPACK_LIB
  if(sizeof(Float) == sizeof(float)) {
    float _tol  = static_cast<float>(tol);
#ifdef MULTI_GPU
    ARPACK(pcnaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol,
		    reinterpret_cast<std::complex<float> *>(resid), &ncv,
		    reinterpret_cast<std::complex<float> *>(v),
		    &ldv, iparam, ipntr,
		    reinterpret_cast<std::complex<float> *>(workd),
		    reinterpret_cast<std::complex<float> *>(workl), &lworkl,
		    reinterpret_cast<float*>(rwork), &info);
#else
    ARPACK(cnaupd)(&ido, &bmat, &n, which, &nev, &_tol,
		   reinterpret_cast<std::complex<float> *>(resid), &ncv,
		   reinterpret_cast<std::complex<float> *>(v),
		   &ldv, iparam, ipntr,
		   reinterpret_cast<std::complex<float> *>(workd),
		   reinterpret_cast<std::complex<float> *>(workl), &lworkl,
		   reinterpret_cast<float*>(rwork), &info);
#endif //MULTI_GPU
  }
  else {
    double _tol = static_cast<double>(tol);
#ifdef MULTI_GPU
    ARPACK(pznaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol,
		    reinterpret_cast<std::complex<double> *>(resid), &ncv,
		    reinterpret_cast<std::complex<double> *>(v),
		    &ldv, iparam, ipntr,
		    reinterpret_cast<std::complex<double> *>(workd),
		    reinterpret_cast<std::complex<double> *>(workl), &lworkl,
		    reinterpret_cast<double*>(rwork), &info);
#else
    ARPACK(znaupd)(&ido, &bmat, &n, which, &nev, &_tol,
		   reinterpret_cast<std::complex<double> *>(resid), &ncv,
		   reinterpret_cast<std::complex<double> *>(v),
		   &ldv, iparam, ipntr,
		   reinterpret_cast<std::complex<double> *>(workd),
		   reinterpret_cast<std::complex<double> *>(workl), &lworkl,
		   reinterpret_cast<double*>(rwork), &info);
#endif //MULTI_GPU
  }
#endif //ARPACK_LIB
  return;
}

template<typename Float> void arpack_neupd (int &comp_evecs, char howmny, int *select,
					    std::complex<Float>* evals,
					    std::complex<Float>* v, int &ldv,
					    std::complex<Float> sigma,
					    std::complex<Float>* workev, 
					    char bmat, int &n, char *which, int &nev,
					    Float tol,
					    std::complex<Float>* resid, int &ncv,
					    std::complex<Float>* v1, int &ldv1,
					    int *iparam, int *ipntr, 
					    std::complex<Float>* workd,
					    std::complex<Float>* workl,
					    int &lworkl, Float* rwork,
					    int &info, int *fcomm) {
#ifdef ARPACK_LIB
  if(sizeof(Float) == sizeof(float)) {   
    float _tol = static_cast<float>(tol);
    std::complex<float> _sigma = static_cast<std::complex<float> >(sigma);
#ifdef MULTI_GPU
    ARPACK(pcneupd)(fcomm, &comp_evecs, &howmny, select,
		    reinterpret_cast<std::complex<float> *>(evals),
		    reinterpret_cast<std::complex<float> *>(v), &ldv, &_sigma,
		    reinterpret_cast<std::complex<float> *>(workev),
		    &bmat, &n, which, &nev, &_tol,
		    reinterpret_cast<std::complex<float> *>(resid), &ncv,
		    reinterpret_cast<std::complex<float> *>(v1),
		    &ldv1, iparam, ipntr,
		    reinterpret_cast<std::complex<float> *>(workd),
		    reinterpret_cast<std::complex<float> *>(workl), &lworkl,
		    reinterpret_cast<float *>(rwork), &info);
#else    
    ARPACK(cneupd)(&comp_evecs, &howmny, select,
		   reinterpret_cast<std::complex<float> *>(evals),
		   reinterpret_cast<std::complex<float> *>(v), &ldv, &_sigma,
		   reinterpret_cast<std::complex<float> *>(workev),
		   &bmat, &n, which, &nev, &_tol,
		   reinterpret_cast<std::complex<float> *>(resid), &ncv,
		   reinterpret_cast<std::complex<float> *>(v1),
		   &ldv1, iparam, ipntr,
		   reinterpret_cast<std::complex<float> *>(workd),
		   reinterpret_cast<std::complex<float> *>(workl), &lworkl,
		   reinterpret_cast<float *>(rwork), &info); 
#endif //MULTI_GPU
  }
  else {
    double _tol = static_cast<double>(tol);
    std::complex<double> _sigma = static_cast<std::complex<double> >(sigma);
#ifdef MULTI_GPU
    ARPACK(pzneupd)(fcomm, &comp_evecs, &howmny, select,
		    reinterpret_cast<std::complex<double> *>(evals),
		    reinterpret_cast<std::complex<double> *>(v), &ldv, &_sigma,
		    reinterpret_cast<std::complex<double> *>(workev),
		    &bmat, &n, which, &nev, &_tol,
		    reinterpret_cast<std::complex<double> *>(resid), &ncv,
		    reinterpret_cast<std::complex<double> *>(v1),
		    &ldv1, iparam, ipntr,
		    reinterpret_cast<std::complex<double> *>(workd),
		    reinterpret_cast<std::complex<double> *>(workl), &lworkl,
		    reinterpret_cast<double *>(rwork), &info);
#else
    ARPACK(zneupd)(&comp_evecs, &howmny, select,
		   reinterpret_cast<std::complex<double> *>(evals),
		   reinterpret_cast<std::complex<double> *>(v), &ldv, &_sigma,
		   reinterpret_cast<std::complex<double> *>(workev),
		   &bmat, &n, which, &nev, &_tol,
		   reinterpret_cast<std::complex<double> *>(resid), &ncv,
		   reinterpret_cast<std::complex<double> *>(v1),
		   &ldv1, iparam, ipntr,
		   reinterpret_cast<std::complex<double> *>(workd),
		   reinterpret_cast<std::complex<double> *>(workl), &lworkl,
		   reinterpret_cast<double *>(rwork), &info);
#endif //MULTI_GPU
  }
#endif //ARPACK_LIB
  return;
}

namespace quda{

  template<typename Float>
  void polyOp(Dirac &mat,
	      cudaColorSpinorField &out,
	      const cudaColorSpinorField &in,	   
	      QudaArpackParam *arpack_param) {
    
    double delta,theta;
    double sigma,sigma1,sigma_old;
    double d1,d2,d3;
    
    double a = arpack_param->amin;
    double b = arpack_param->amax;
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
      std::complex<double> d1c(d1,0);
      std::complex<double> d2c(d2,0);
      blas::cxpaypbz(*tm1,d2c,*tm2,d1c,out);
      blas::copy(*tm1,*tm2);
      blas::copy(*tm2,out);
      sigma_old = sigma;
    }
    
    delete tm1;
    delete tm2;
  }
  
  template<typename Float>
  void arpack_solve(void *h_evecs, void *h_evals,
		    QudaInvertParam *inv_param,
		    QudaArpackParam *arpack_param,
		    Dirac &mat){

    int *fcomm = nullptr;
#ifdef MULTI_GPU
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm = static_cast<int*>(&mpi_comm_fort);
#endif

    int local_vol = 10;
    
    int max_iter = arpack_param->arpackMaxiter;

    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_    = local_vol,
      nev_    = arpack_param->nEv,
      nkv_    = arpack_param->nKv,
      ldv_    = local_vol,
      lworkl_ = (3 * nkv_ * nkv_ + 5 * nkv_) * 2,
      rvec_   = 1;
    std::complex<Float> sigma_ = 0.0;
    Float tol_ = arpack_param->arpackTol;
    
    std::complex<Float> *resid_      = new std::complex<Float>[ldv_];
    std::complex<Float> *w_workd_    = new std::complex<Float>[3*ldv_];
    std::complex<Float> *w_workl_    = new std::complex<Float>[lworkl_];
    Float *w_rwork_                  = new Float[nkv_];
    
    /* __neupd-only workspace */
    std::complex<Float> *w_workev_   = new std::complex<Float>[ 2 * nkv_];
    int *select_                     = new int[ nkv_];

    //Alias pointers
    std::complex<Float> *h_evecs_ = NULL;
    h_evecs_ = (std::complex<Float>*) &(h_evecs);
    
    std::complex<Float> *h_evals_ = NULL;
    h_evals_ = (std::complex<Float>*) &(h_evals);
    
    // cnaupd cycle 
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    iparam_[7] = arpack_param->arpackMode;
    
    char howmny='P';
    char bmat = 'I';

    char *spectrum;
    spectrum = strdup("SR"); //Just to stop the compiler warning...
    
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
    const int localL = 10;
    cpuColorSpinorField *h_v = NULL;
    cudaColorSpinorField *d_v = NULL;    
    cpuColorSpinorField *h_v2 = NULL;
    cudaColorSpinorField *d_v2 = NULL;    
    
    do {
      //interface to arpack routines
      arpack_naupd<Float>(ido_, bmat, n_, spectrum, nev_, tol_,
			  resid_, nkv_, h_evals_, ldv_, iparam_, ipntr_,
			  w_workd_, w_workl_, lworkl_, w_rwork_, info_, fcomm);
      
      if (info_ != 0) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);
      
      iter_cnt++;

      if(allocate){
	ColorSpinorParam cpuParam(w_workd_ + ipntr_[0] - 1,
				  *inv_param, &localL, !arpack_param->useFullOp);
	h_v = new cpuColorSpinorField(cpuParam);
	cpuParam.v = w_workd_ + ipntr_[1] - 1;
	h_v2 = new cpuColorSpinorField(cpuParam);
	
	ColorSpinorParam cudaParam(cpuParam, *inv_param);
	cudaParam.create = QUDA_ZERO_FIELD_CREATE;
	d_v = new cudaColorSpinorField(cudaParam);
	d_v2 = new cudaColorSpinorField(cudaParam);
	allocate = false;
      }
      
      if (ido_ == -1 || ido_ == 1) {

	*d_v = *h_v;	
	//apply matrix vector here:
	if(arpack_param->usePolyAcc) polyOp<Float>(mat, *d_v2, *d_v, arpack_param);
	else mat.MdagM(*d_v2,*d_v);
	
	*h_v2= *d_v2;
      }
      
      if(iter_cnt % 50 == 0) printfQuda("\nArpack Iteration : %d\n", iter_cnt);
      iter_cnt++;
      
    } while (99 != ido_ && iter_cnt < max_iter);

    if ( info_ < 0 ){
      errorQuda("Error in _naupd, info = %d. Exiting.", info_);
    } else {
      
      printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);      
      printfQuda("Computing eigenvectors\n");    
      arpack_neupd<Float>(rvec_, howmny, select_, h_evals_, h_evecs_, ldv_, sigma_,
			  w_workev_, bmat, n_, spectrum, nev_, tol_,
			  resid_, nkv_, h_evecs_, ldv_, iparam_, ipntr_, w_workd_,
			  w_workl_, lworkl_, w_rwork_, info_, fcomm);
      
      if (info_ != 0)
	errorQuda("\nError in _neupd, info = %d. Exiting.\n", info_);    
      
    }
    
    /* cleanup */
    if (w_workl_  != nullptr) delete [] w_workl_;
    if (w_rwork_  != nullptr) delete [] w_rwork_;
    if (w_workev_ != nullptr) delete [] w_workev_;
    if (select_   != nullptr) delete [] select_;

    if (w_workd_  != nullptr) delete [] w_workd_;
    if (resid_    != nullptr) delete [] resid_;
    
    return;
  }
  
  //ARPACK SOLVER//

  void arpackSolve(void *h_evecs, void *h_evals,
		   QudaInvertParam *inv_param,
		   QudaArpackParam *arpack_param,
		   Dirac &mat){
    
    if(arpack_param->arpackPrec == QUDA_DOUBLE_PRECISION) 
      arpack_solve<double>(h_evecs, h_evals, inv_param, arpack_param, mat);
    else
      arpack_solve<float>(h_evecs, h_evals, inv_param, arpack_param, mat);
  }  
}



