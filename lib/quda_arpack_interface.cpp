#include <quda_arpack_interface.h>
#include <color_spinor_field_order.h>

using namespace quda ;

  struct SortEvals{
    double _val;
    int    _idx;

    SortEvals(double val, int idx) : _val(val), _idx(idx) {};

    static bool CmpEigenNrms (SortEvals v1, SortEvals v2) { return (v1._val < v2._val);}
  };

//#define RUN_DP_ARPACK

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertFrom2DVector(cpuColorSpinorField &out, std::complex<Float> *in) {
     if(out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", out.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> outOrder(static_cast<ColorSpinorField&>(out));//fineColor =3 here!

     blas::zero(out);

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < out.VolumeCB(); x_cb++) {

         int i = parity*out.VolumeCB() + x_cb;
         int xx[4] = {0};
         out.LatticeIndex(xx, i);

         int _2d_idx = (xx[0] + xx[1]*out.X(0))*fineSpin*reducedColor;

         if( xx[2] == 0 && xx[3] == 0 ) for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) outOrder(parity, x_cb, s, c) = in[_2d_idx+s*reducedColor+c];
       }
     }

     return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertTo2DVector(std::complex<Float> *out, cpuColorSpinorField &in) {
     if(in.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", in.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> inOrder(static_cast<ColorSpinorField&>(in));

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < in.VolumeCB(); x_cb++) {

         int i = parity*in.VolumeCB() + x_cb;
         int xx[4] = {0};
         in.LatticeIndex(xx, i);

         int _2d_idx = (xx[0] + xx[1]*in.X(0))*fineSpin*reducedColor;

         if( xx[2] == 0 && xx[3] == 0 ) for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) out[_2d_idx+s*reducedColor+c] = inOrder(parity, x_cb, s, c);
       }
     }

     return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertFrom4DVector(cpuColorSpinorField &out, std::complex<Float> *in) {
     if(out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", out.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> outOrder(static_cast<ColorSpinorField&>(out));//fineColor =3 here!

     blas::zero(out);

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < out.VolumeCB(); x_cb++) {

         int i = parity*out.VolumeCB() + x_cb;
   
         int _4d_idx = i*fineSpin*reducedColor;

         for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) outOrder(parity, x_cb, s, c) = in[_4d_idx+s*reducedColor+c];//SU2
       }
     }

     return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertTo4DVector(std::complex<Float> *out, cpuColorSpinorField &in) {
     if(in.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", in.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> inOrder(static_cast<ColorSpinorField&>(in));

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < in.VolumeCB(); x_cb++) {

         int i = parity*in.VolumeCB() + x_cb;

         int _4d_idx = i*fineSpin*reducedColor;

         for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) out[_4d_idx+s*reducedColor+c] = inOrder(parity, x_cb, s, c);//SU2
       }
     }

     return;
  }


//call from multigrid routines!
  template<typename Float, int fineSpin, int fineColor, int reducedColor >
  int arpack_2d_solve( char * lanczos_which, std::vector<ColorSpinorField*> &B, void *evals,   DiracMatrix &mat, QudaPrecision mat_precision, Float tol, int nev, int ncv )
  {
    ColorSpinorParam csParam(*B[0]);

    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    
    if(sizeof(Float) == B[0]->Precision())
      csParam.setPrecision(B[0]->Precision()); 
    else
      csParam.setPrecision(QUDA_DOUBLE_PRECISION); 

    cpuColorSpinorField *t = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

    csParam.location = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.setPrecision(mat_precision);
    //
    ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
    //
    ColorSpinorField *mx = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    Dirac &dirac  = const_cast<Dirac&>(*(mat.Expose()));
    double _mass = dirac.Mass();
    dirac.setMass((0.1));

    int n_iters = 0;           /* return the iteration count */
    int nconv = 0;             /* number of converged evals/evecs */

    //Float tol      = ( sizeof(Float) == sizeof(double) ) ?  5e-2 : 1e-3;//5e-6

    int   max_iter = 240000;

    size_t clen = t->X(0)*t->X(1)*t->Nspin()*reducedColor;//no need for spin dof
    size_t cldn = clen;
    const size_t wbytes = cldn*sizeof(std::complex<Float>);
    printfQuda("\nclen = %lu, wbytes = %lu\n", clen, wbytes);
//32: 8,16-256; 64-256, mass 0.1
//64: 16 -128 64-128 48-128  128-256 256-512 mass 0.1
    //int nev = 24;//B.size();
    //int ncv = 128;//64*nev;//or N*nev;

    //void *evals        = malloc(wbytes*(nev+1)); /* return buffer for evalues,  [nev + 1] */
    void *arpack_evecs = malloc(wbytes*ncv); /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */

    void *tmp[3];
  
    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_      = clen,
        nev_    = nev,
        ncv_    = ncv,
        ldv_    = cldn,
        lworkl_ = (3 * ncv_ * ncv_ + 5 * ncv_) * 2,
        rvec_   = 1;
    void *sigma_ = malloc(sizeof(std::complex<Float>));
    Float tol_ = tol;

    void *w_d_         = evals;
    void *w_v_         = arpack_evecs;

    void *resid_      = malloc(wbytes);
    void *w_workd_    = malloc(wbytes * 3);
    void *w_workl_    = malloc(sizeof(std::complex<Float>) * lworkl_);
    void *w_rwork_    = malloc(sizeof(Float) *ncv_);
    
    /* __neupd-only workspace */
    void *w_workev_   = malloc(sizeof(std::complex<Float>) * 2 * ncv_);
    int *select_                     = (int*)malloc(sizeof(int) * ncv_);

    if(resid_ == nullptr||
           w_workd_ == nullptr||
           w_workl_ == nullptr||
           w_rwork_ == nullptr||
           w_workev_ == nullptr||
           select_ == nullptr)    errorQuda("Could not allocate memory..");

    memset(sigma_, 0, sizeof(std::complex<Float>));
    memset(resid_, 0, wbytes);
    memset(w_workd_, 0, wbytes * 3);

    for(int i = 0; i < 3; i++) tmp[i] = (void*)((char*)w_workd_ + i*wbytes);

    /* cnaupd cycle */
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    int iter_cnt= 0;
    char bmat = 'I';
    do {
         if(sizeof(Float) == sizeof(float))
           ARPACK(cnaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (float*)&tol, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                          &lworkl_, (float*)w_rwork_, &info_);
         else
           ARPACK(znaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (double*)&tol, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                          &lworkl_, (double*)w_rwork_, &info_);
  
        if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);

        iter_cnt++;
        
        if (99 == ido_ || 1 == info_)
            break;

        if (-1 == ido_ || 1 == ido_) {
           const int input_idx = (ipntr_[0]-1) / clen;
           if(input_idx < 0 || input_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", input_idx);
           convertFrom2DVector<Float, fineSpin, fineColor, reducedColor> (*t, static_cast<std::complex<Float> *>(tmp[input_idx]));//convert!!!
           *x = *t;
           //
           mat(*mx, *x);
           //
           const int output_idx = (ipntr_[1]-1) / clen;
           if(output_idx < 0 || output_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", output_idx);
           *t = *mx;
           convertTo2DVector<Float, fineSpin, fineColor, reducedColor> (static_cast<std::complex<Float> *>(tmp[output_idx]), *t);//convert!!!

           if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
        } else 
        {errorQuda("\nError detected!\n");}  

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);


    int conv_cnt = 0;
    if (info_ == 0) {
        //assert(iparam_[4] == nev);
        conv_cnt = iparam_[4];
    } else if (info_ == 1) {
        conv_cnt = iparam_[4];
    } 

    char howmny='P';

    /* for howmny="P", no additional space is required */
    if(sizeof(Float) == sizeof(float))
       ARPACK(cneupd)(&rvec_, &howmny, select_, (_Complex float *)w_d_,
                     (_Complex float *)w_v_, &ldv_, (_Complex float *)sigma_, (_Complex float *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (float*)&tol_, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                     &lworkl_, (float *)w_rwork_, &info_);
    else
       ARPACK(zneupd)(&rvec_, &howmny, select_, (_Complex double *)w_d_,
                     (_Complex double *)w_v_, &ldv_, (_Complex double *)sigma_, (_Complex double *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (double*)&tol_, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                     &lworkl_, (double *)w_rwork_, &info_);

    if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    


    for(int i = 0; i < nev; i++)
    {
      tmp[0] = (std::complex<Float>*)((char*)w_v_ + i*wbytes);//note that this is a parity field
      convertFrom2DVector<Float, fineSpin, fineColor, reducedColor>(*t, static_cast<std::complex<Float> *>(tmp[0]));//convert!

      *x = *t;
      //
      mat(*mx, *x);
      std::complex<Float> tmp_eval = static_cast<std::complex<Float>*>(w_d_)[i];
      std::complex<double> cl = std::complex<double>(tmp_eval.real(),tmp_eval.imag()); 

      printfQuda("\nCheck norm2 (Re = %le, Im = %le) : %le\n", cl.real(), cl.imag(), sqrt( blas::caxpyNorm(-cl, *x, *mx)) / sqrt( blas::norm2(*x) ));
    }
//copy fields:

    std::vector<SortEvals> sorted_evals_cntr;
    sorted_evals_cntr.reserve(nev);

//    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( abs( w_d_[e].imag() ), e ));
    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( ( (static_cast<std::complex<Float>* >(w_d_))[e].imag() ), e ));
    std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);

    int ev_id = 0;
    for(std::vector<ColorSpinorField*>::iterator nullvec = B.begin() ; nullvec != B.end(); ++nullvec) {
      int sorted_id =  sorted_evals_cntr[ev_id]._idx;//does not work!!!
      //tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + ev_id*wbytes);//prvious version
      tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + sorted_id*wbytes);
      convertFrom2DVector<Float, fineSpin, fineColor, reducedColor>(*t, static_cast<std::complex<Float> *>(tmp[0]));
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*nullvec);
      *x = *t;
      *curr_nullvec = *x;
      //convertFrom2DVector (*curr_nullvec, tmp[0], z_slice, t_slice, _2dNc);
      ev_id += 1;
    }

    printfQuda("\ndone..\n");

    /* cleanup */
    if (w_workl_ != nullptr)   free(w_workl_);
    if (w_rwork_ != nullptr)   free(w_rwork_);
    if (w_workev_ != nullptr)  free(w_workev_);
    if (select_ != nullptr)    free(select_);

    n_iters    = iter_cnt;
    nconv      = conv_cnt;

    dirac.setMass(_mass);

    delete t;
    delete x;
    delete mx;

    free(arpack_evecs);

    if (w_workd_ != nullptr)   free(w_workd_);
    if (resid_   != nullptr)   free(resid_);
    return 0;
  }

//call from multigrid routines:

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  int arpack_4d_reduced_color_solve( char * lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision mat_precision, Float tol, int nev, int ncv)
  {
    ColorSpinorParam csParam(*B[0]);

    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    
    if(sizeof(Float) == B[0]->Precision())
      csParam.setPrecision(B[0]->Precision()); 
    else
      csParam.setPrecision(QUDA_DOUBLE_PRECISION); 

    cpuColorSpinorField *t = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

    csParam.location = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.setPrecision(mat_precision);
    //
    ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
    //
    ColorSpinorField *mx = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    Dirac &dirac  = const_cast<Dirac&>(*(mat.Expose()));
    double _mass = dirac.Mass();
    dirac.setMass((0.1));

    int n_iters = 0;           /* return the iteration count */
    int nconv = 0;             /* number of converged evals/evecs */

    int   max_iter = 240000;

    size_t clen = t->X(0)*t->X(1)*t->X(2)*t->X(3)*t->Nspin()*reducedColor;
    size_t cldn = clen;
    const size_t wbytes = cldn*sizeof(std::complex<Float>);
    //printfQuda("\nclen = %lu, wbytes = %lu\n", clen, wbytes);
    //nt nev = 24;//B.size();
    //int ncv = 64*nev;//or N*nev;

    //void *evals        = malloc(wbytes*(nev+1)); /* return buffer for evalues,  [nev + 1] */
    void *arpack_evecs = malloc(wbytes*ncv);     /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */

    void *tmp[3];
  
    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_      = clen,
        nev_    = nev,
        ncv_    = ncv,
        ldv_    = cldn,
        lworkl_ = (3 * ncv_ * ncv_ + 5 * ncv_) * 2,
        rvec_   = 1;
    void *sigma_ = malloc(sizeof(std::complex<Float>));
    Float tol_ = tol;

    void *w_d_         = evals;
    void *w_v_         = arpack_evecs;

    void *resid_      = malloc(wbytes);
    void *w_workd_    = malloc(wbytes * 3);
    void *w_workl_    = malloc(sizeof(std::complex<Float>) * lworkl_);
    void *w_rwork_    = malloc(sizeof(Float) *ncv_);
    
    /* __neupd-only workspace */
    void *w_workev_   = malloc(sizeof(std::complex<Float>) * 2 * ncv_);
    int *select_                     = (int*)malloc(sizeof(int) * ncv_);

    if(resid_ == nullptr||
           w_workd_ == nullptr||
           w_workl_ == nullptr||
           w_rwork_ == nullptr||
           w_workev_ == nullptr||
           select_ == nullptr)    errorQuda("Could not allocate memory..");

    memset(sigma_, 0, sizeof(std::complex<Float>));
    memset(resid_, 0, wbytes);
    memset(w_workd_, 0, wbytes * 3);

    for(int i = 0; i < 3; i++) tmp[i] = (void*)((char*)w_workd_ + i*wbytes);

    /* cnaupd cycle */
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    int iter_cnt= 0;
    char bmat = 'I';
    do {
         if(sizeof(Float) == sizeof(float))
           ARPACK(cnaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (float*)&tol, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                          &lworkl_, (float*)w_rwork_, &info_);
         else
           ARPACK(znaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (double*)&tol, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                          &lworkl_, (double*)w_rwork_, &info_);
  
        if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);

        iter_cnt++;
        
        if (99 == ido_ || 1 == info_)
            break;

        if (-1 == ido_ || 1 == ido_) {
           const int input_idx = (ipntr_[0]-1) / clen;
           if(input_idx < 0 || input_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", input_idx);
           convertFrom4DVector<Float, fineSpin, fineColor, reducedColor> (*t, static_cast<std::complex<Float> *>(tmp[input_idx]));//convert!!!
           *x = *t;
           //
           mat(*mx, *x);
           //
           const int output_idx = (ipntr_[1]-1) / clen;
           if(output_idx < 0 || output_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", output_idx);
           *t = *mx;
           convertTo4DVector<Float, fineSpin, fineColor, reducedColor> (static_cast<std::complex<Float> *>(tmp[output_idx]), *t);//convert to arpack field

           if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
        } else 
        {errorQuda("\nError detected!\n");}  

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);


    int conv_cnt = 0;
    if (info_ == 0) {
        //assert(iparam_[4] == nev);
        conv_cnt = iparam_[4];
    } else if (info_ == 1) {
        conv_cnt = iparam_[4];
    } 

    char howmny='P';

    /* for howmny="P", no additional space is required */
    if(sizeof(Float) == sizeof(float))
       ARPACK(cneupd)(&rvec_, &howmny, select_, (_Complex float *)w_d_,
                     (_Complex float *)w_v_, &ldv_, (_Complex float *)sigma_, (_Complex float *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (float*)&tol_, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                     &lworkl_, (float *)w_rwork_, &info_);
    else
       ARPACK(zneupd)(&rvec_, &howmny, select_, (_Complex double *)w_d_,
                     (_Complex double *)w_v_, &ldv_, (_Complex double *)sigma_, (_Complex double *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (double*)&tol_, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                     &lworkl_, (double *)w_rwork_, &info_);

    if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    

    for(int i = 0; i < nev; i++)
    {
      tmp[0] = (std::complex<Float>*)((char*)w_v_ + i*wbytes);//note that this is a parity field
      convertFrom4DVector<Float, fineSpin, fineColor, reducedColor>(*t, static_cast<std::complex<Float> *>(tmp[0]));//convert!

      *x = *t;
      //
      mat(*mx, *x);
 
      std::complex<Float> tmp_eval = static_cast<std::complex<Float>*>(w_d_)[i];
      std::complex<double> cl = std::complex<double>(tmp_eval.real(),tmp_eval.imag());

      printfQuda("\nCheck norm2 (Re = %le, Im = %le) : %le\n", cl.real(), cl.imag(), sqrt( blas::caxpyNorm(-cl, *x, *mx)) / sqrt( blas::norm2(*x) ));
    }
//copy fields:

    std::vector<SortEvals> sorted_evals_cntr;
    sorted_evals_cntr.reserve(nev);

//    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( abs( w_d_[e].imag() ), e ));
    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( ( (static_cast<std::complex<Float>* >(w_d_))[e].imag() ), e ));
    std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);

    int ev_id = 0;
    for(std::vector<ColorSpinorField*>::iterator nullvec = B.begin() ; nullvec != B.end(); ++nullvec) {
      int sorted_id =  sorted_evals_cntr[ev_id]._idx;//does not work!!!
      //tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + ev_id*wbytes);//prvious version
      tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + sorted_id*wbytes);
      convertFrom4DVector<Float, fineSpin, fineColor, reducedColor>(*t, static_cast<std::complex<Float> *>(tmp[0]));
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*nullvec);
      *x = *t;
      *curr_nullvec = *x;
      ev_id += 1;
    }

    printfQuda("\ndone..\n");

    /* cleanup */
    if (w_workl_ != nullptr)   free(w_workl_);
    if (w_rwork_ != nullptr)   free(w_rwork_);
    if (w_workev_ != nullptr)  free(w_workev_);
    if (select_ != nullptr)    free(select_);

    n_iters    = iter_cnt;
    nconv      = conv_cnt;

    dirac.setMass(_mass);

    delete t;
    delete x;
    delete mx;

    free(arpack_evecs);

    if (w_workd_ != nullptr)   free(w_workd_);
    if (resid_   != nullptr)   free(resid_);
    return 0;
  }


  template<typename Float, int fineSpin, int fineColor>
  int arpack_4d_solve( char *lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision mat_precision, Float tol, int nev, int ncv)
  {
    int n_iters = 0;           /* return the iteration count */
    int nconv = 0;             /* number of converged evals/evecs */

    int   max_iter = 240000;

    size_t clen = B[0]->X(0)*B[0]->X(1)*B[0]->X(2)*B[0]->X(3)*B[0]->Nspin()*B[0]->Ncolor();
    size_t cldn = clen;
    const size_t wbytes = cldn*sizeof(std::complex<Float>);
    //printfQuda("\nclen = %lu, wbytes = %lu\n", clen, wbytes);
    //nt nev = 24;//B.size();
    //int ncv = 64*nev;//or N*nev;

    //void *evals        = malloc(wbytes*(nev+1)); /* return buffer for evalues,  [nev + 1] */
    void *arpack_evecs = malloc(wbytes*ncv);     /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */

    void *tmp[3];

    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_      = clen,
        nev_    = nev,
        ncv_    = ncv,
        ldv_    = cldn,
        lworkl_ = (3 * ncv_ * ncv_ + 5 * ncv_) * 2,
        rvec_   = 1;
    void *sigma_ = malloc(sizeof(std::complex<Float>));
    Float tol_ = tol;

    void *w_d_         = evals;
    void *w_v_         = arpack_evecs;

    void *resid_      = malloc(wbytes);
    void *w_workd_    = malloc(wbytes * 3);
    void *w_workl_    = malloc(sizeof(std::complex<Float>) * lworkl_);
    void *w_rwork_    = malloc(sizeof(Float) *ncv_);
    
    /* __neupd-only workspace */
    void *w_workev_   = malloc(sizeof(std::complex<Float>) * 2 * ncv_);
    int *select_                     = (int*)malloc(sizeof(int) * ncv_);

    if(resid_ == nullptr||
           w_workd_ == nullptr||
           w_workl_ == nullptr||
           w_rwork_ == nullptr||
           w_workev_ == nullptr||
           select_ == nullptr)    errorQuda("Could not allocate memory..");

    memset(sigma_, 0, sizeof(std::complex<Float>));
    memset(resid_, 0, wbytes);
    memset(w_workd_, 0, wbytes * 3);

    for(int i = 0; i < 3; i++) tmp[i] = (void*)((char*)w_workd_ + i*wbytes);

//problem specific setup
    ColorSpinorParam csParam(*B[0]);

    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    
    if(sizeof(Float) == B[0]->Precision())
      csParam.setPrecision(B[0]->Precision()); 
    else
      csParam.setPrecision(QUDA_DOUBLE_PRECISION); 

    cpuColorSpinorField *t[3];

    for(int i = 0; i < 3; i++)
    {
      csParam.v = tmp[i];
      t[i] = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
    }

    csParam.location = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.setPrecision(mat_precision);
    //
    ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
    //
    ColorSpinorField *mx = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    Dirac &dirac  = const_cast<Dirac&>(*(mat.Expose()));
    double _mass = dirac.Mass();
    dirac.setMass((0.1));//

    /* cnaupd cycle */
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;
    int iter_cnt= 0;
    char bmat = 'I';
    do {
         if(sizeof(Float) == sizeof(float))
           ARPACK(cnaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (float*)&tol, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                          &lworkl_, (float*)w_rwork_, &info_);
         else
           ARPACK(znaupd)(&ido_, &bmat, &n_, lanczos_which,
                          &nev_, (double*)&tol, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                          &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                          &lworkl_, (double*)w_rwork_, &info_);
  
        if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);

        iter_cnt++;
        
        if (99 == ido_ || 1 == info_)
            break;

        if (-1 == ido_ || 1 == ido_) {
           const int input_idx = (ipntr_[0]-1) / clen;
           if(input_idx < 0 || input_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", input_idx);
           *x = *t[input_idx];
           //
           mat(*mx, *x);
           //
           const int output_idx = (ipntr_[1]-1) / clen;
           if(output_idx < 0 || output_idx > 3 ) errorQuda("\nFailed to compute input index (%d)\n", output_idx);
           *t[output_idx] = *mx;

           if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
        } else 
        {errorQuda("\nError detected!\n");}  

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);


    int conv_cnt = 0;
    if (info_ == 0) {
        //assert(iparam_[4] == nev);
        conv_cnt = iparam_[4];
    } else if (info_ == 1) {
        conv_cnt = iparam_[4];
    } 

    char howmny='P';

    /* for howmny="P", no additional space is required */
    if(sizeof(Float) == sizeof(float))
       ARPACK(cneupd)(&rvec_, &howmny, select_, (_Complex float *)w_d_,
                     (_Complex float *)w_v_, &ldv_, (_Complex float *)sigma_, (_Complex float *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (float*)&tol_, (_Complex float *)resid_, &ncv_, (_Complex float *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex float *)w_workd_, (_Complex float *)w_workl_,
                     &lworkl_, (float *)w_rwork_, &info_);
    else
       ARPACK(zneupd)(&rvec_, &howmny, select_, (_Complex double *)w_d_,
                     (_Complex double *)w_v_, &ldv_, (_Complex double *)sigma_, (_Complex double *)w_workev_, "I", &n_, lanczos_which,
                     &nev_, (double*)&tol_, (_Complex double *)resid_, &ncv_, (_Complex double *)w_v_,
                     &ldv_, iparam_, ipntr_, (_Complex double *)w_workd_, (_Complex double *)w_workl_,
                     &lworkl_, (double *)w_rwork_, &info_);

    if (info_ < 0 || 1 < info_) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    

//copy fields:

    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    
    if(sizeof(Float) == B[0]->Precision())
      csParam.setPrecision(B[0]->Precision()); 
    else
      csParam.setPrecision(QUDA_DOUBLE_PRECISION); 

    std::vector<SortEvals> sorted_evals_cntr;
    sorted_evals_cntr.reserve(nev);

//    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( abs( w_d_[e].imag() ), e ));
    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals( ( (static_cast<std::complex<Float>*>(w_d_))[e].imag() ), e ));
    std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);

    int ev_id = 0;
    for(std::vector<ColorSpinorField*>::iterator nullvec = B.begin() ; nullvec != B.end(); ++nullvec) {
      int sorted_id =  sorted_evals_cntr[ev_id]._idx;//does not work!!!
      //tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + ev_id*wbytes);//prvious version
      tmp[0] = (std::complex<Float>*)((char*)arpack_evecs + sorted_id*wbytes);
      csParam.v = tmp[0];
      cpuColorSpinorField *arpa_eignvec = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*nullvec);
      *curr_nullvec = *arpa_eignvec;
      ev_id += 1;
      delete arpa_eignvec;
    }

    printfQuda("\ndone..\n");

    /* cleanup */
    if (w_workl_ != nullptr)   free(w_workl_);
    if (w_rwork_ != nullptr)   free(w_rwork_);
    if (w_workev_ != nullptr)  free(w_workev_);
    if (select_ != nullptr)    free(select_);

    n_iters    = iter_cnt;
    nconv      = conv_cnt;

    dirac.setMass(_mass);

    for(int i = 0; i < 3; i++) delete t[i];
    delete x;
    delete mx;

    free(arpack_evecs);

    if (w_workd_ != nullptr)   free(w_workd_);
    if (resid_   != nullptr)   free(resid_);

    return 0;
  }



///////////////////////////////////////////////////ARPACK SOLVER////////////////////////////////////////////////////////


 void ArpackArgs::EigenSolver( )
 {
   const int fineSpin  = 1;
   const int fineColor = 3;

   if(_2d_field)
   {
     if(reducedColors == 1)
     {
        if(use_full_prec_arpack)
        {
           arpack_2d_solve<double, fineSpin, fineColor, 1>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv );
        }
        else
        {
           arpack_2d_solve<float, fineSpin, fineColor, 1>( lanczos_which, B, evals, matEigen, mat_precision, (float)tol, nev , ncv  );
        }
     }
     else
     {
        errorQuda("\nUnsupported colors.\n");
     }
   }
   else
   {
     if(reducedColors == 2)
     {
        if(use_full_prec_arpack)
        {
           arpack_4d_reduced_color_solve<double, fineSpin, fineColor, 2>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev, ncv  );
        }
        else
        {
           arpack_4d_reduced_color_solve<float, fineSpin, fineColor, 2>( lanczos_which, B, evals, matEigen, mat_precision, (float)tol, nev, ncv  );
        }
     }
     else
     {
        if(use_full_prec_arpack)
        {
           arpack_4d_solve<double, fineSpin, fineColor>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev, ncv  );
        }
        else
        {
           arpack_4d_solve<float, fineSpin, fineColor>( lanczos_which, B, evals, matEigen, mat_precision, (float)tol, nev, ncv  );
        }
     }
   }
 
   return;
 }

