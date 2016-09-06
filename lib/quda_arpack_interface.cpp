#include <quda_arpack_interface.h>
#include <color_spinor_field_order.h>

using namespace quda ;

  struct SortEvals{
    double _val;
    int    _idx;

    SortEvals(double val, int idx) : _val(val), _idx(idx) {};

    static bool Cmp (SortEvals v1, SortEvals v2) { return (v1._val < v2._val);}
  };

  template<typename Float> void arpack_naupd(int &ido, char &bmat, int &n, char *which, int &nev, Float &tol,  std::complex<Float> *resid, int &ncv, std::complex<Float> *v, int &ldv,
                    int *iparam, int *ipntr, std::complex<Float> *workd, std::complex<Float> *workl, int &lworkl, Float *rwork, int &info, int *fcomm)
  {
    if(sizeof(Float) == sizeof(float))
    {
       float _tol  = static_cast<float>(tol);
#if (defined(MPI_COMMS) || defined(QMP_COMMS))
       ARPACK(pcnaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl), &lworkl, reinterpret_cast<float*>(rwork), &info);
#else
       ARPACK(cnaupd)(&ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<float> *>(resid), &ncv, reinterpret_cast<std::complex<float> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<float> *>(workd), reinterpret_cast<std::complex<float> *>(workl), &lworkl, reinterpret_cast<float*>(rwork), &info);
#endif
    }
    else
    {
       double _tol = static_cast<double>(tol);
#if (defined(MPI_COMMS) || defined(QMP_COMMS))
       ARPACK(pznaupd)(fcomm, &ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl), &lworkl, reinterpret_cast<double*>(rwork), &info);
#else
       ARPACK(znaupd)(&ido, &bmat, &n, which, &nev, &_tol, reinterpret_cast<std::complex<double> *>(resid), &ncv, reinterpret_cast<std::complex<double> *>(v),
                       &ldv, iparam, ipntr, reinterpret_cast<std::complex<double> *>(workd), reinterpret_cast<std::complex<double> *>(workl), &lworkl, reinterpret_cast<double*>(rwork), &info);
#endif
    }

    return;
  }

  template<typename Float> void arpack_neupd (int &comp_evecs, char howmny, int *select, std::complex<Float>* evals, std::complex<Float>* v, int &ldv, std::complex<Float> sigma, std::complex<Float>* workev, 
		       char bmat, int &n, char *which, int &nev, Float tol,  std::complex<Float>* resid, int &ncv, std::complex<Float>* v1, int &ldv1, int *iparam, int *ipntr, 
                       std::complex<Float>* workd, std::complex<Float>* workl, int &lworkl, Float* rwork, int &info, int *fcomm)
  {
    if(sizeof(Float) == sizeof(float))
    {   
       float _tol = static_cast<float>(tol);
       std::complex<float> _sigma = static_cast<std::complex<float> >(sigma);
#if (defined(MPI_COMMS) || defined(QMP_COMMS))
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
#endif
    }
    else
    {
       double _tol = static_cast<double>(tol);
       std::complex<double> _sigma = static_cast<std::complex<double> >(sigma);
#if (defined(MPI_COMMS) || defined(QMP_COMMS))
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
#endif
    }

    return;
  }


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

  template<typename cpuFloat, typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertFrom2DVector_v2(cpuColorSpinorField &out, std::complex<Float> *in) {
     if(out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", out.FieldOrder() );
     quda::colorspinor::FieldOrderCB<cpuFloat,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> outOrder(static_cast<ColorSpinorField&>(out));//fineColor =3 here!

     blas::zero(out);

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < out.VolumeCB(); x_cb++) {

         int i = parity*out.VolumeCB() + x_cb;
         int xx[4] = {0};
         out.LatticeIndex(xx, i);

         int _2d_idx = (xx[0] + xx[1]*out.X(0))*fineSpin*reducedColor;

         if( xx[2] == 0 && xx[3] == 0 ) for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) outOrder(parity, x_cb, s, c) = std::complex<cpuFloat>( in[_2d_idx+s*reducedColor+c].real(), in[_2d_idx+s*reducedColor+c].imag() );
       }
     }

     return;
  }


  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation>
  void arpack_matvec(std::complex<Float> *out, std::complex<Float> *in,  DiracMatrix &matEigen, QudaPrecision matPrecision, ColorSpinorField &meta)
  {
    ColorSpinorParam csParam(meta);

    csParam.create = QUDA_ZERO_FIELD_CREATE;  
    //cpuParam.extendDimensionality();5-dim field
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    if(!do_2d_emulation) 
    {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      csParam.v      = static_cast<void*>(in);
    }

    cpuColorSpinorField *cpu_tmp1 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
    
    if(!do_2d_emulation)
    {
      csParam.v      = static_cast<void*>(out);
    }
    else
    {
      convertFrom2DVector<Float, fineSpin, fineColor, reducedColor>(*cpu_tmp1, in);
    }
    
    cpuColorSpinorField *cpu_tmp2 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.location   = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
    csParam.create     = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(matPrecision);

    ColorSpinorField *cuda_tmp1 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(*cpu_tmp1, csParam));
    //
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField *cuda_tmp2 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    matEigen(*cuda_tmp2, *cuda_tmp1);

    *cpu_tmp2 = *cuda_tmp2;
    if(do_2d_emulation) convertTo2DVector<Float, fineSpin, fineColor, reducedColor>(out, *cpu_tmp2);

    delete cpu_tmp1;
    delete cpu_tmp2;

    delete cuda_tmp1;
    delete cuda_tmp2;

    return;
  }

//copy fields:
  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation> 
  void copy_eigenvectors(std::vector<ColorSpinorField*> &B, std::complex<Float> *arpack_evecs, std::complex<Float> *arpack_evals, const int cldn, const int nev, char *which)
  {
    printfQuda("\nLoad eigenvectors..\n");

    std::vector<SortEvals> sorted_evals_cntr;
    sorted_evals_cntr.reserve(nev);

    ColorSpinorParam csParam(*B[0]);

    csParam.create = do_2d_emulation ? QUDA_ZERO_FIELD_CREATE : QUDA_REFERENCE_FIELD_CREATE;  
    //cpuParam.extendDimensionality();5-dim field
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    if(!do_2d_emulation) csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    std::string arpack_which(which);

    for(int e = 0; e < nev; e++) 
    {
      if     (arpack_which.compare(std::string("SM")))    sorted_evals_cntr.push_back( SortEvals(std::norm(arpack_evals[e]), e ));
      else if(arpack_which.compare(std::string("SI")))    sorted_evals_cntr.push_back( SortEvals(arpack_evals[e].imag(), e ));
      else if(arpack_which.compare(std::string("SR")))    sorted_evals_cntr.push_back( SortEvals(arpack_evals[e].real(), e ));
      else
          errorQuda("\nSorting option is not supported.\n");
    }

    std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::Cmp);

    cpuColorSpinorField *cpu_tmp = nullptr;
    int ev_id = 0;

    for(std::vector<ColorSpinorField*>::iterator vec = B.begin() ; vec != B.end(); ++vec) {
      int sorted_id =  sorted_evals_cntr[ev_id++]._idx;

      printfQuda("%d ,Re= %le, Im= %le\n", sorted_id, arpack_evals[sorted_id].real(), arpack_evals[sorted_id].imag());

      std::complex<Float>* tmp_buffer =  &arpack_evecs[sorted_id*cldn];
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*vec);

      if(do_2d_emulation)
      {
        cpu_tmp = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

        if      (csParam.precision == QUDA_DOUBLE_PRECISION)
          convertFrom2DVector_v2<double, Float, fineSpin, fineColor, reducedColor>(*cpu_tmp, tmp_buffer);
        else if (csParam.precision == QUDA_SINGLE_PRECISION)
          convertFrom2DVector_v2<float, Float, fineSpin, fineColor, reducedColor>(*cpu_tmp, tmp_buffer);
        else
          errorQuda("\nUnsupported precision.\n"); 
      }
      else
      {
        csParam.v = static_cast<void*>(tmp_buffer);
        cpu_tmp = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
      }

      *curr_nullvec = *cpu_tmp;//this does not work for different precision (usual memcpy)?

      delete cpu_tmp;
    }

    printfQuda("\n..done.\n");

    return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation>
  int arpack_solve( char *lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision matPrecision, Float tol, int nev, int ncv)
  {
    int *fcomm = nullptr;
#if (defined(MPI_COMMS) || defined(QMP_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm = static_cast<int*>(&mpi_comm_fort);
#endif

    int   max_iter = 4000;

    size_t clen = B[0]->X(0)*B[0]->X(1)*( do_2d_emulation ? reducedColor : (B[0]->X(2)*B[0]->X(3))*fineColor)*fineSpin;
    size_t cldn = clen;

    std::complex<Float> *arpack_evecs = new std::complex<Float>[cldn*ncv];     /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */

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

    std::complex<Float> *w_d_  = static_cast<std::complex<Float> *>(evals);
    std::complex<Float> *w_v_  = arpack_evecs;

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
         arpack_matvec<Float, fineSpin, fineColor, reducedColor, do_2d_emulation> (&(w_workd_[(ipntr_[1]-1)]), &(w_workd_[(ipntr_[0]-1)]),  mat, matPrecision, *B[0]) ;

         if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
      } 

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);

    //int conv_cnt = iparam_[4];

    /* for howmny="P", no additional space is required */
    arpack_neupd<Float>(rvec_, howmny, select_, w_d_, w_v_, ldv_, sigma_, w_workev_, bmat, n_, lanczos_which,
                        nev_, tol_, resid_, ncv_, w_v_, ldv_, iparam_, ipntr_, w_workd_, w_workl_, lworkl_, w_rwork_, info_, fcomm);

    if (info_ != 0) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    
//copy fields:
    copy_eigenvectors<Float, fineSpin, fineColor, reducedColor, do_2d_emulation>(B, arpack_evecs, w_d_, cldn, nev, lanczos_which);

    printfQuda("\ndone..\n");

    /* cleanup */
    if (w_workl_ != nullptr)   delete [] w_workl_;
    if (w_rwork_ != nullptr)   delete [] w_rwork_;
    if (w_workev_ != nullptr)  delete [] w_workev_;
    if (select_ != nullptr)    delete [] select_;

    //n_iters    = iter_cnt;
    //nconv      = conv_cnt;
    delete [] arpack_evecs;

    if (w_workd_ != nullptr)   delete [] w_workd_;
    if (resid_   != nullptr)   delete [] resid_;

    return 0;
  }

  template <typename Float, int fineSpin, int reducedColor, bool do_2d_emulation>
  void arpack_solve( char *lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision matPrecision, Float tol, int nev, int ncv )
  {
    if(B[0]->Ncolor() == 3)
    {
      const int fineColor  = 3;
      arpack_solve<Float, fineSpin,fineColor,reducedColor, do_2d_emulation>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv);
    }
    else if(B[0]->Ncolor() == 4)
    {
      const int fineColor  = 4;
      arpack_solve<Float, fineSpin,fineColor,reducedColor, do_2d_emulation>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv);
    }
    else 
    {
      errorQuda("\nfineColor = %d\n is not supported, please revisit quda_arpack_interface to add support.", B[0]->Ncolor() );
    }
  }

  template <typename Float, int reducedColor, bool do_2d_emulation>
  void arpack_solve( char *lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision matPrecision, Float tol, int nev, int ncv )
  {
    if(B[0]->Nspin() == 1)
    {
      const int fineSpin  = 1;
      arpack_solve<Float, fineSpin, reducedColor, do_2d_emulation>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv);
    }
    else if(B[0]->Nspin() == 2)
    {
      const int fineSpin  = 2;
      arpack_solve<Float, fineSpin, reducedColor, do_2d_emulation>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv);
    }
    else if(B[0]->Nspin() == 4)
    {
      const int fineSpin  = 4;
      arpack_solve<Float, fineSpin, reducedColor, do_2d_emulation>( lanczos_which, B, evals, matEigen, mat_precision, tol, nev , ncv);
    }
    else 
    {
      errorQuda("\nfineSpin = %d\n is not supported, please revisit quda_arpack_interface to add support.", B[0]->Nspin() );
    }
  }

///////////////////////////////////////////////////ARPACK SOLVER////////////////////////////////////////////////////////


 void ArpackArgs::operator()( std::vector<ColorSpinorField*> &B, std::complex<double> *evals )
 {
#ifdef ARPACK_LIB
   if(_2d_field)
   {
     warningQuda("\nSolving 2d eigen-problem\n");
     if(reducedColors == 1)
     {
        if(use_full_prec_arpack)   arpack_solve<double, 1, true>( lanczos_which, B, (void*)evals, matEigen, mat_precision, tol, nev , ncv  );
        else                       arpack_solve<float, 1, true>( lanczos_which, B, (void*)evals, matEigen, mat_precision, (float)tol, nev , ncv  );
     }
     else errorQuda("\nUnsupported colors.\n");
   }
   else 
   {
     //Warning: reduced colors not used here:   
     if(use_full_prec_arpack)   arpack_solve<double, 3, false>( lanczos_which, B, (void*)evals, matEigen, mat_precision, tol, nev , ncv );
     else                       arpack_solve<float, 3, false>( lanczos_which, B, (void*)evals, matEigen, mat_precision, (float)tol, nev , ncv  );
   }
 #endif
   return;
 }

