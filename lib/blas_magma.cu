#include <blas_magma.h>
#include <string.h>

#include <util_quda.h>
#include <quda_internal.h>

#ifdef MAGMA_LIB

#include <magma.h>

#define _cV  MagmaVec
#define _cU  MagmaUpper
#define _cR  MagmaRight
#define _cL  MagmaLeft
#define _cC  MagmaConjTrans
#define _cN  MagmaNoTrans
#define _cNV MagmaNoVec

#endif

#ifdef MAGMA_LIB


  template<typename magmaFloat> void magma_gesv(void *sol, const int ldn, const int n, void *Mat, const int ldm)
  {
    cudaPointerAttributes ptr_attr;
    if(cudaPointerGetAttributes(&ptr_attr, Mat) == cudaErrorInvalidValue) errorQuda("In magma_gesv, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

    magma_int_t *ipiv;
    magma_int_t err, info;

    magma_imalloc_pinned(&ipiv, n);

    void *tmp;

    magma_malloc_pinned((void**)&tmp, ldm*n*sizeof(magmaFloat));
    memcpy(tmp, Mat, ldm*n*sizeof(magmaFloat));

    if ( ptr_attr.memoryType == cudaMemoryTypeDevice ) {
      if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {  
         err = magma_cgesv_gpu(n, 1, static_cast<magmaFloatComplex* >(tmp), ldm, ipiv, static_cast<magmaFloatComplex* >(sol), ldn, &info);
         if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_cgesv_gpu), exit ...\n");
      }
      else
      {
         err = magma_zgesv_gpu(n, 1, static_cast<magmaDoubleComplex*>(tmp), ldm, ipiv, static_cast<magmaDoubleComplex*>(sol), ldn, &info);
         if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_zgesv_gpu), exit ...\n");
      }
    }  else if ( ptr_attr.memoryType == cudaMemoryTypeHost ) {

      if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {  
         err = magma_cgesv(n, 1, static_cast<magmaFloatComplex* >(tmp), ldm, ipiv, static_cast<magmaFloatComplex* >(sol), ldn, &info);
         if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_cgesv), exit ...\n");
      }
      else
      {
         err = magma_zgesv(n, 1, static_cast<magmaDoubleComplex*>(tmp), ldm, ipiv, static_cast<magmaDoubleComplex*>(sol), ldn, &info);
         if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_zgesv), exit ...\n");
      }
    }

    magma_free_pinned(ipiv);
    magma_free_pinned(tmp);

    return;
  }


/////

  template<typename magmaFloat> void magma_geev(void *Mat, const int m, const int ldm, void *vr, void *evalues, const int ldv)
  {
    cudaPointerAttributes ptr_attr;
    if(cudaPointerGetAttributes(&ptr_attr, Mat) == cudaErrorInvalidValue) errorQuda("In magma_geev, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

    magma_int_t err, info;

    void *work_  = nullptr, *rwork_ = nullptr;

    if ( ptr_attr.memoryType == cudaMemoryTypeDevice ) {
      errorQuda("\nGPU version is not supported.\n");
    }  else if ( ptr_attr.memoryType == cudaMemoryTypeHost ) {

      if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {
        magmaFloatComplex qwork;

        magmaFloatComplex *work = static_cast<magmaFloatComplex*>(work_);
        float *rwork = static_cast<float*>(rwork_);

        err = magma_cgeev(_cNV, _cV, m, nullptr, ldm, nullptr, nullptr, ldv, nullptr, ldv, &qwork, -1, nullptr, &info);
        if( err != 0 ) errorQuda( "Error: CGEEVX, info %d\n",info);

        magma_int_t lwork = static_cast<magma_int_t>( MAGMA_C_REAL(qwork));

        magma_smalloc_pinned(&rwork, 2*m);
        magma_cmalloc_pinned(&work, lwork);

        err = magma_cgeev(_cNV, _cV, m, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<magmaFloatComplex*>(evalues), nullptr, ldv, static_cast<magmaFloatComplex*>(vr), ldv, work, lwork, rwork, &info);
        if( err != 0 ) errorQuda( "Error: CGEEVX, info %d\n",info);

      }
      else
      {
        magmaDoubleComplex qwork;

        magmaDoubleComplex *work = static_cast<magmaDoubleComplex*>(work_);
        double *rwork = static_cast<double*>(rwork_);
        err = magma_zgeev(_cNV, _cV, m, nullptr, ldm, nullptr, nullptr, ldv, nullptr, ldv, &qwork, -1, nullptr, &info);
        if( err != 0 ) errorQuda( "Error: ZGEEVX, info %d\n",info);

        magma_int_t lwork = static_cast<magma_int_t>( MAGMA_Z_REAL(qwork));

        magma_dmalloc_pinned(&rwork, 2*m);
        magma_zmalloc_pinned(&work, lwork);

        err = magma_zgeev(_cNV, _cV, m, static_cast<magmaDoubleComplex*>(Mat), ldm, static_cast<magmaDoubleComplex*>(evalues), nullptr, ldv, static_cast<magmaDoubleComplex*>(vr), ldv, work, lwork, rwork, &info);
        if( err != 0 ) errorQuda( "Error: ZGEEVX, info %d\n",info);
      }
    }

    if(rwork_)  magma_free_pinned(rwork_);
    if(work_ )  magma_free_pinned(work_);

    return;
  }


/////

  template<typename  magmaFloat> void magma_gels(void *Mat, void *c, int rows, int cols, int ldm)
  {
    cudaPointerAttributes ptr_attr;
    if(cudaPointerGetAttributes(&ptr_attr, Mat) == cudaErrorInvalidValue) errorQuda("In magma_gels, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

    magma_int_t err, info, lwork;
    void *hwork_ = nullptr;

    if ( ptr_attr.memoryType == cudaMemoryTypeDevice )
    {
      if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {
        magma_int_t nb = magma_get_cgeqrf_nb( rows, cols );
        lwork = std::max( cols*nb, 2*nb*nb );

        magmaFloatComplex *hwork = static_cast<magmaFloatComplex*>(hwork_);
        magma_cmalloc_cpu( &hwork, lwork);

        err = magma_cgels_gpu( _cN, rows, cols, 1, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<magmaFloatComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_cgels_gpu, %d, exit ...\n", info);
      } else {
        magma_int_t nb = magma_get_zgeqrf_nb( rows, cols );

        lwork = std::max( cols*nb, 2*nb*nb );
        magmaDoubleComplex *hwork = static_cast<magmaDoubleComplex*>(hwork_);
        magma_zmalloc_cpu( &hwork, lwork);

        err = magma_zgels_gpu( _cN, rows, cols, 1, static_cast<magmaDoubleComplex*>(Mat), ldm, static_cast<magmaDoubleComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_zgels_gpu, %d, exit ...\n", info);
      }
    }  else if ( ptr_attr.memoryType == cudaMemoryTypeHost ) {


     if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {
        magma_int_t nb = magma_get_cgeqrf_nb( rows, cols );

        lwork = std::max( cols*nb, 2*nb*nb );
        magmaFloatComplex *hwork = static_cast<magmaFloatComplex*>(hwork_);
        magma_cmalloc_cpu( &hwork, lwork);

        err = magma_cgels( _cN, rows, cols, 1, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<magmaFloatComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_cgels_cpu, %d, exit ...\n", info);
      } else {
        magma_int_t nb = magma_get_zgeqrf_nb( rows, cols );

        lwork = std::max( cols*nb, 2*nb*nb );
        magmaDoubleComplex *hwork = static_cast<magmaDoubleComplex*>(hwork_);
        magma_zmalloc_cpu( &hwork, lwork);

        err = magma_zgels( _cN, rows, cols, 1, static_cast<magmaDoubleComplex*>(Mat), ldm, static_cast<magmaDoubleComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_zgels_cpu, %d, exit ...\n", info);
      }
    }

    if(hwork_) magma_free_cpu(hwork_);

    return;
  }


 template<typename magmaFloat> void magma_heev(void *Mat, const int m, const int ldm, void *evalues)
  {
    cudaPointerAttributes ptr_attr;
    if(cudaPointerGetAttributes(&ptr_attr, Mat) == cudaErrorInvalidValue) errorQuda("In magma_heev, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

    magma_int_t err, info;

    void *work_  = nullptr, *rwork_ = nullptr;
    int *iwork   = nullptr;
    int qiwork;

    if ( ptr_attr.memoryType == cudaMemoryTypeDevice ) {
      errorQuda("\nGPU version is not supported.\n");
    }  else if ( ptr_attr.memoryType == cudaMemoryTypeHost ) {
      if(sizeof(magmaFloat) == sizeof(magmaFloatComplex))
      {
        magmaFloatComplex qwork;
        float qrwork;

        magmaFloatComplex *work = static_cast<magmaFloatComplex*>(work_);
        float *rwork = static_cast<float*>(rwork_);

        err = magma_cheevd(_cV, _cU, m, nullptr, ldm, nullptr, &qwork, -1, &qrwork, -1, &qiwork, -1, &info);
        if( err != 0 ) errorQuda( "Error: CHEEVD, info %d\n",info);

        magma_int_t lwork  = static_cast<magma_int_t>( MAGMA_C_REAL(qwork));
        magma_int_t lrwork = static_cast<magma_int_t>( qrwork );
        magma_int_t liwork = static_cast<magma_int_t>( qiwork );

        magma_cmalloc_pinned(&work,  lwork);
        magma_smalloc_pinned(&rwork, lrwork);
        magma_imalloc_pinned(&iwork, liwork);

        err = magma_cheevd(_cV, _cU, m, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<float*>(evalues), work, lwork, rwork, lrwork, iwork, liwork, &info);
        if( err != 0 ) errorQuda( "Error: CHEEVD, info %d\n",info);
      } else  {
        magmaDoubleComplex qwork;
        double qrwork;

        magmaDoubleComplex *work = static_cast<magmaDoubleComplex*>(work_);
        double *rwork = static_cast<double*>(rwork_);

        err = magma_zheevd(_cV, _cU, m, nullptr, ldm, nullptr, &qwork, -1, &qrwork, -1, &qiwork, -1, &info);
        if( err != 0 ) errorQuda( "Error: ZHEEVD, info %d\n",info);

        magma_int_t lwork  = static_cast<magma_int_t>( MAGMA_Z_REAL(qwork));
        magma_int_t lrwork = static_cast<magma_int_t>( qrwork );
        magma_int_t liwork = static_cast<magma_int_t>( qiwork );

        magma_zmalloc_pinned(&work,  lwork);
        magma_dmalloc_pinned(&rwork, lrwork);
        magma_imalloc_pinned(&iwork, liwork);

        err = magma_zheevd(_cV, _cU, m, static_cast<magmaDoubleComplex*>(Mat), ldm, static_cast<double*>(evalues), work, lwork, rwork, lrwork, iwork, liwork, &info);
        if( err != 0 ) errorQuda( "Error: ZHEEVD, info %d\n",info);
      }
    }

    if(rwork_)  magma_free_pinned(rwork_);
    if(work_ )  magma_free_pinned(work_);
    if(iwork )  magma_free_pinned(iwork);

    return;
  }

#endif // MAGMA_LIB

 void magma_Xgesv(void* sol, const int ldn, const int n, void* Mat, const int ldm, const int prec)
  {
#ifdef MAGMA_LIB
    if      (prec == sizeof(std::complex< double >)) magma_gesv<magmaDoubleComplex>(sol, ldn, n, Mat, ldm);
    else if (prec == sizeof(std::complex< float  >)) magma_gesv<magmaFloatComplex >(sol, ldn, n, Mat, ldm);
    else errorQuda("\nPrecision is not supported.\n");
#endif
    return;
  }

  void magma_Xgeev(void *Mat, const int m, const int ldm, void *vr, void *evalues, const int ldv, const int prec)
  {
#ifdef MAGMA_LIB
    if      (prec == sizeof(std::complex< double >)) magma_geev<magmaDoubleComplex>(Mat, m, ldm, vr, evalues, ldv);
    else if (prec == sizeof(std::complex< float  >)) magma_geev<magmaFloatComplex >(Mat, m, ldm, vr, evalues, ldv);
    else errorQuda("\nPrecision is not supported.\n");
#endif
    return;
  }


  void magma_Xgels(void *Mat, void *c, int rows, int cols, int ldm, const int prec)
  {
#ifdef MAGMA_LIB
    if      (prec == sizeof(std::complex< double >)) magma_gels<magmaDoubleComplex>(Mat, c, rows, cols, ldm);
    else if (prec == sizeof(std::complex< float  >)) magma_gels<magmaFloatComplex >(Mat, c, rows, cols, ldm);
    else errorQuda("\nPrecision is not supported.\n");
#endif
    return;
  }

  void magma_Xheev(void *Mat, const int m, const int ldm, void *evalues, const int prec)
  {
#ifdef MAGMA_LIB
    if      (prec == sizeof(std::complex< double >)) magma_heev<magmaDoubleComplex>(Mat, m, ldm, evalues);
    else if (prec == sizeof(std::complex< float  >)) magma_heev<magmaFloatComplex >(Mat, m, ldm, evalues);
    else errorQuda("\nPrecision is not supported.\n");
#endif
    return;
  }


  void OpenMagma(){
#ifdef MAGMA_LIB
    magma_int_t err = magma_init();

    if(err != MAGMA_SUCCESS) errorQuda("\nError: cannot initialize MAGMA library\n");

    int major, minor, micro;

    magma_version( &major, &minor, &micro);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("\nMAGMA library version: %d.%d\n\n", major,  minor);
#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif
    return;
  }

  void CloseMagma(){
#ifdef MAGMA_LIB
    if(magma_finalize() != MAGMA_SUCCESS) errorQuda("\nError: cannot close MAGMA library\n");
#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif
    return;
  }


#ifdef MAGMA_LIB

#undef _cV
#undef _cU
#undef _cR
#undef _cL
#undef _cC
#undef _cN
#undef _cNV

#endif
