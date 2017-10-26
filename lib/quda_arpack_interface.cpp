#include <quda_arpack_interface.h>

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
#include <mpi.h>
#endif


//using namespace quda ;

  struct SortEvals{
    double _val;
    int    _idx;

    static bool small_values;

    SortEvals(double val, int idx) : _val(val), _idx(idx) {};

    static bool SelectSmall (SortEvals v1, SortEvals v2) { return (v1._val < v2._val);}
    static bool SelectLarge (SortEvals v1, SortEvals v2) { return (v1._val > v2._val);}

  };

  bool SortEvals::small_values = true;

  template<typename Float> void arpack_naupd(int &ido, char &bmat, int &n, char *which, int &nev, Float &tol,  std::complex<Float> *resid, int &ncv, std::complex<Float> *v, int &ldv,
                    int *iparam, int *ipntr, std::complex<Float> *workd, std::complex<Float> *workl, int &lworkl, Float *rwork, int &info, int *fcomm)
  {
#ifdef ARPACK_LIB
    if(sizeof(Float) == sizeof(float))
    {
       float _tol  = static_cast<float>(tol);
#ifdef MULTI_GPU
       ARPACK(pcnaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl), &lworkl, reinterpret_cast<float*>(rwork), &info);
#else
       ARPACK(cnaupd)(&ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl), &lworkl, reinterpret_cast<float*>(rwork), &info);
#endif //MULTI_GPU
    }
    else
    {
       double _tol = static_cast<double>(tol);
#ifdef MULTI_GPU
       ARPACK(pznaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl), &lworkl, reinterpret_cast<double*>(rwork), &info);
#else
       ARPACK(znaupd)(&ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl), &lworkl, reinterpret_cast<double*>(rwork), &info);
#endif //MULTI_GPU
    }
#endif //ARPACK_LIB
    return;
  }

  template<typename Float> void arpack_neupd (int &comp_evecs, char howmny, int *select, std::complex<Float>* evals, std::complex<Float>* v, int &ldv, std::complex<Float> sigma, std::complex<Float>* workev, 
		       char bmat, int &n, char *which, int &nev, Float tol,  std::complex<Float>* resid, int &ncv, std::complex<Float>* v1, int &ldv1, int *iparam, int *ipntr, 
                       std::complex<Float>* workd, std::complex<Float>* workl, int &lworkl, Float* rwork, int &info, int *fcomm)
  {
#ifdef ARPACK_LIB
    if(sizeof(Float) == sizeof(float))
    {   
       float _tol = static_cast<float>(tol);
       std::complex<float> _sigma = static_cast<std::complex<float> >(sigma);
#ifdef MULTI_GPU
       ARPACK(pcneupd)(fcomm, &comp_evecs, &howmny, select, reinterpret_cast<std::complex<float> *>(evals),
                     reinterpret_cast<std::complex<float> *>(v), &ldv, &_sigma, reinterpret_cast<std::complex<float> *>(workev), &bmat, &n, which,
                     &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v1),
                     &ldv1, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl),
                     &lworkl, reinterpret_cast<float *>(rwork), &info);
#else

       ARPACK(cneupd)(&comp_evecs, &howmny, select, reinterpret_cast<std::complex<float> *>(evals),
                     reinterpret_cast<std::complex<float> *>(v), &ldv, &_sigma, reinterpret_cast<std::complex<float> *>(workev), &bmat, &n, which,
                     &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v1),
                     &ldv1, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl),
                     &lworkl, reinterpret_cast<float *>(rwork), &info); 
#endif //MULTI_GPU
    }
    else
    {
       double _tol = static_cast<double>(tol);
       std::complex<double> _sigma = static_cast<std::complex<double> >(sigma);
#ifdef MULTI_GPU
       ARPACK(pzneupd)(fcomm, &comp_evecs, &howmny, select, reinterpret_cast<std::complex<double> *>(evals),
                     reinterpret_cast<std::complex<double> *>(v), &ldv, &_sigma, reinterpret_cast<std::complex<double> *>(workev), &bmat, &n, which,
                     &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v1),
                     &ldv1, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl),
                     &lworkl, reinterpret_cast<double *>(rwork), &info);
#else
       ARPACK(zneupd)(&comp_evecs, &howmny, select, reinterpret_cast<std::complex<double> *>(evals),
                     reinterpret_cast<std::complex<double> *>(v), &ldv, &_sigma, reinterpret_cast<std::complex<double> *>(workev), &bmat, &n, which,
                     &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v1),
                     &ldv1, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl),
                     &lworkl, reinterpret_cast<double *>(rwork), &info);
#endif //MULTI_GPU
    }
#endif //ARPACK_LIB
    return;
  }

namespace quda{

  template<typename Float> class QudaMatvec {

    protected:

      /**Problem matrix **/
      DiracMatrix &matEigen;
      /**Matrix vector precision (may not coincide with arpack IRA routines precision) **/
      QudaPrecision matPrecision;
      /*Vector quda gpu fields required for the operation*/
      ColorSpinorField *cuda_in ;
      ColorSpinorField *cuda_out;

    public:

      QudaMatvec(DiracMatrix &matEigen, QudaPrecision prec, ColorSpinorField &meta) : matEigen(matEigen), matPrecision(prec) { 
         ColorSpinorParam csParam(meta);
         //we need an explicit fieldOrder setup here since meta is a none-native field. 
         csParam.fieldOrder = (meta.Nspin() == 1 || matPrecision == QUDA_DOUBLE_PRECISION) ? QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
         csParam.location   = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
         csParam.create     = QUDA_ZERO_FIELD_CREATE;
         csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
         csParam.setPrecision(matPrecision);

         cuda_in = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
         cuda_out = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
      }

      virtual ~QudaMatvec() { 
        delete cuda_in ; 
        delete cuda_out;
      }

      void operator()(std::complex<Float> *out, std::complex<Float> *in);
  };

  template<typename Float>
  void QudaMatvec<Float>::operator() (std::complex<Float> *out, std::complex<Float> *in)
  {
      ColorSpinorParam csParam(*cuda_in);
      csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;

      csParam.v      = static_cast<void*>(in);
      cpuColorSpinorField *cpu_in = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

      csParam.v      = static_cast<void*>(out);
      cpuColorSpinorField *cpu_out = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

      *cuda_in = *cpu_in;
      //
      matEigen(*cuda_out, *cuda_in);
      //
      *cpu_out = *cuda_out;

      delete cpu_in ;
      delete cpu_out;

      return;
  }

  template<typename Float>  class ArpackArgs {

    private:
      //main setup:
      QudaMatvec<Float>                       &matvec;
      std::vector<ColorSpinorField*>          &evecs ; //container of spinor fields

      //arpack objects:
      size_t clen;
      size_t cldn;
      std::complex<Float> *w_d_; //just keep eigenvalues
      std::complex<Float> *w_v_; //continuous buffer to keep eigenvectors

      /**spectrum info**/
      int nev;//number of eigenvecs to be comupted
      int ncv;//search subspace dimension (note that 1 <= NCV-NEV and NCV <= N) 
      char *lanczos_which;// ARPACK which="{S,L}{R,I,M}
      /**general arpack library parameters**/	
      Float tol;
      int   info;

    public:

      ArpackArgs(QudaMatvec<Float> &matvec, std::vector<ColorSpinorField*> &evecs, std::complex<Float> *evals, int nev, int ncv, char *which, Float tol) : matvec(matvec), evecs(evecs), w_d_(evals), nev(nev), ncv(ncv), lanczos_which(which), tol(tol), info(0) 
      {
         clen = evecs[0]->Length() >> 1;//complex length
         cldn = clen;

         w_v_ = new std::complex<Float>[cldn*ncv];     /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */ 
      }       

      virtual ~ArpackArgs() {  delete w_v_; }
 
      //Main IRA algorithm driver:
      void apply();     
      //save computed eigenmodes to the user defined arrays:
      void save();
  };


//copy fields:
  template<typename Float> 
  void ArpackArgs<Float>::save()
  {
    printfQuda("\nLoad eigenvectors..\n");

    std::vector<SortEvals> sorted_evals;
    sorted_evals.reserve(nev);

    ColorSpinorParam csParam(*evecs[0]);

    csParam.create = QUDA_REFERENCE_FIELD_CREATE;  
    //cpuParam.extendDimensionality();5-dim field
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);

    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    std::string arpack_which(lanczos_which);

    if (arpack_which.compare(std::string("SM"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(std::norm(w_d_[e]), e));
    } else if (arpack_which.compare(std::string("SI"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(w_d_[e].imag(), e));
    } else if (arpack_which.compare(std::string("SR"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(w_d_[e].real(), e));
    } else if (arpack_which.compare(std::string("LM"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(std::norm(w_d_[e]), e));
      SortEvals::small_values = false;
    } else if (arpack_which.compare(std::string("LI"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(w_d_[e].imag(), e));
      SortEvals::small_values = false;
    } else if (arpack_which.compare(std::string("LR"))) {
      for(int e = 0; e < nev; e++) sorted_evals.push_back( SortEvals(w_d_[e].real(), e));
      SortEvals::small_values = false;
    } else {
      errorQuda("\nSorting option is not supported.\n");
    }

    if(SortEvals::small_values) std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortEvals::SelectSmall );
    else                        std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortEvals::SelectLarge );

    cpuColorSpinorField *cpu_tmp = nullptr;
    int ev_id = 0;

    for(std::vector<ColorSpinorField*>::iterator vec = evecs.begin() ; vec != evecs.end(); ++vec) {
      int sorted_id =  sorted_evals[ev_id++]._idx;

      printfQuda("%d ,Re= %le, Im= %le\n", sorted_id, w_d_[sorted_id].real(), w_d_[sorted_id].imag());

      std::complex<Float>* tmp_buffer   =  &w_v_[sorted_id*cldn];
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*vec);

      csParam.v = static_cast<void*>(tmp_buffer);
      cpu_tmp = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

      *curr_nullvec = *cpu_tmp;//this does not work for different precision (usual memcpy)?

      delete cpu_tmp;
    }

    return;
  }

  template<typename Float>
  void ArpackArgs<Float>::apply( )
  {
    int *fcomm = nullptr;
#ifdef MULTI_GPU
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm = static_cast<int*>(&mpi_comm_fort);
#endif

    int   max_iter = 4000;

    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_      = clen,
        nev_    = nev,
        ncv_    = ncv,
        ldv_    = cldn,
        lworkl_ = (3 * ncv_ * ncv_ + 5 * ncv_),
        rvec_   = 1;
    std::complex<Float> sigma_ = 0.0;
    Float tol_ = tol;

    std::complex<Float> *resid_      = new std::complex<Float>[cldn];
    std::complex<Float> *w_workd_    = new std::complex<Float>[(cldn * 3)];
    std::complex<Float> *w_workl_    = new std::complex<Float>[lworkl_];
    Float *w_rwork_                  = new Float[ncv_];
    
    /* __neupd-only workspace */
    std::complex<Float> *w_workev_   = new std::complex<Float>[ 2 * ncv_];
    int *select_                     = new int[ ncv_];

    /* cnaupd cycle */
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;

    char howmny='A';//'P'
    char bmat = 'I';

    int iter_cnt= 0;

    do {
      //interface to arpack routines
      arpack_naupd<Float>(ido_, bmat, n_, lanczos_which, nev_, tol, resid_, ncv_, w_v_, ldv_, iparam_, ipntr_, w_workd_, w_workl_, lworkl_, w_rwork_, info_, fcomm);
  
      if (info_ != 0) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);

      iter_cnt++;
        
      if (ido_ == -1 || ido_ == 1) {
         //apply matrix vector here:
         matvec(&(w_workd_[(ipntr_[1]-1)]), &(w_workd_[(ipntr_[0]-1)]));

         if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
      } 

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);

    //int conv_cnt = iparam_[4];

    /* for howmny="P", no additional space is required */
    arpack_neupd<Float>(rvec_, howmny, select_, w_d_, w_v_, ldv_, sigma_, w_workev_, bmat, n_, lanczos_which,
                        nev_, tol_, resid_, ncv_, w_v_, ldv_, iparam_, ipntr_, w_workd_, w_workl_, lworkl_, w_rwork_, info_, fcomm);

    if (info_ != 0) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    

    /* cleanup */
    if (w_workl_ != nullptr)   delete [] w_workl_;
    if (w_rwork_ != nullptr)   delete [] w_rwork_;
    if (w_workev_ != nullptr)  delete [] w_workev_;
    if (select_ != nullptr)    delete [] select_;

    if (w_workd_ != nullptr)   delete [] w_workd_;
    if (resid_   != nullptr)   delete [] resid_;

    return;
  }

///////////////////////////////////////////////////ARPACK SOLVER////////////////////////////////////////////////////////
 template<typename Float>
 void arpack_solve( std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision arpackPrec, double tol, int nev, int ncv, char *target)
 {
   QudaMatvec<Float> mv(matEigen, matPrec, *B[0]);
   ArpackArgs<Float> arg(mv, B, static_cast<std::complex<Float> *> (evals), nev , ncv, target, (Float)tol);
   arg.apply();
   arg.save();

   return;
 }

 void arpackSolve( std::vector<ColorSpinorField*> &B, void* evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision arpackPrec, double tol, int nev, int ncv, char *target)
 {
#ifdef ARPACK_LIB
   if((nev <= 0) or (nev > static_cast<int>(B[0]->Length())))      errorQuda("Wrong number of the requested eigenvectors.\n");
   if(((ncv-nev) < 2) or (ncv > static_cast<int>(B[0]->Length()))) errorQuda("Wrong size of the IRAM work subspace.\n");
   if(arpackPrec == QUDA_DOUBLE_PRECISION) arpack_solve<double>(B, evals, matEigen, matPrec, arpackPrec, tol, nev , ncv, target);
   else                                    arpack_solve<float> (B, evals, matEigen, matPrec, arpackPrec, tol, nev , ncv, target);
#else
   errorQuda("Arpack library was not built.\n");
 #endif
   return;
 }

}


