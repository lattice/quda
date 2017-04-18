#include <blas_magma.h>
#include <string.h>

#include <vector>
#include <algorithm>

#include <util_quda.h>
#include <quda_internal.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

//#define MAGMA_2X //default version version of the MAGMA library
#define MAGMA_17

#ifdef MAGMA_LIB
#include <magma.h>

#ifdef MAGMA_14

#define _cV 'V'
#define _cU 'U'

#define _cR 'R'
#define _cL 'L'

#define _cC 'C'
#define _cN 'N'

#define _cNV 'N'

#else

#define _cV MagmaVec
#define _cU MagmaUpper

#define _cR MagmaRight
#define _cL MagmaLeft

#define _cC MagmaConjTrans
#define _cN MagmaNoTrans

#define _cNV MagmaNoVec

#endif

#endif

//Column major format: Big matrix times Little matrix.

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
#ifndef MAGMA_2X
        magma_int_t nb = magma_get_cgeqrf_nb( rows );
#else
        magma_int_t nb = magma_get_cgeqrf_nb( rows, cols );
#endif
        lwork = std::max( cols*nb, 2*nb*nb );

        magmaFloatComplex *hwork = static_cast<magmaFloatComplex*>(hwork_);
        magma_cmalloc_cpu( &hwork, lwork);

        err = magma_cgels_gpu( _cN, rows, cols, 1, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<magmaFloatComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_cgels_gpu, %d, exit ...\n", info);
      } else {
#ifndef MAGMA_2X
        magma_int_t nb = magma_get_zgeqrf_nb( rows );
#else
        magma_int_t nb = magma_get_zgeqrf_nb( rows, cols );
#endif
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
#ifndef MAGMA_2X
        magma_int_t nb = magma_get_cgeqrf_nb( rows );
#else
        magma_int_t nb = magma_get_cgeqrf_nb( rows, cols );
#endif

        lwork = std::max( cols*nb, 2*nb*nb );
        magmaFloatComplex *hwork = static_cast<magmaFloatComplex*>(hwork_);
        magma_cmalloc_cpu( &hwork, lwork);

        err = magma_cgels( _cN, rows, cols, 1, static_cast<magmaFloatComplex*>(Mat), ldm, static_cast<magmaFloatComplex*>(c),
                             ldm, hwork, lwork, &info );
        if (err != 0)  errorQuda("\nError in magma_cgels_cpu, %d, exit ...\n", info);
      } else {
#ifndef MAGMA_2X
        magma_int_t nb = magma_get_zgeqrf_nb( rows );
#else
        magma_int_t nb = magma_get_zgeqrf_nb( rows, cols );
#endif
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

#endif

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
    printfQuda("\nMAGMA library version: %d.%d\n\n", major,  minor);
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


#define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1. ) + (n_)) + (2. / 3.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1. ) + (m_)) + (2. / 3.) * (n_)) )
#define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1. / 6.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1. / 6.) * (n_)) )

#define FLOPS_ZGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )
#define FLOPS_CGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )

#define FMULS_GETRI(n_) ( (n_) * ((5. / 6.) + (n_) * ((2. / 3.) * (n_) + 0.5)) )
#define FADDS_GETRI(n_) ( (n_) * ((5. / 6.) + (n_) * ((2. / 3.) * (n_) - 1.5)) )

#define FLOPS_ZGETRI(n_) (6. * FMULS_GETRI((double)(n_)) + 2.0 * FADDS_GETRI((double)(n_)) )
#define FLOPS_CGETRI(n_) (6. * FMULS_GETRI((double)(n_)) + 2.0 * FADDS_GETRI((double)(n_)) )

void magma_batchInvertMatrix(void *Ainv_h, void* A_h, const int n, const int batch, const int prec)
{
#ifdef MAGMA_LIB
  printfQuda("%s with n=%d and batch=%d\n", __func__, n, batch);

  magma_queue_t queue = 0;

  size_t size = 2*n*n*prec*batch;
  void *A_d = device_malloc(size);
  void *Ainv_d = device_malloc(size);
  qudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

  magma_int_t **dipiv_array = static_cast<magma_int_t**>(device_malloc(batch*sizeof(magma_int_t*)));
  magma_int_t *dipiv_tmp = static_cast<magma_int_t*>(device_malloc(batch*n*sizeof(magma_int_t)));
#ifndef MAGMA_2X
  set_ipointer(dipiv_array, dipiv_tmp, 1, 0, 0, n, batch, queue);
#else
  magma_iset_pointer(dipiv_array, dipiv_tmp, 1, 0, 0, n, batch, queue);
#endif

  magma_int_t *dinfo_array = static_cast<magma_int_t*>(device_malloc(batch*sizeof(magma_int_t)));
  magma_int_t *info_array = static_cast<magma_int_t*>(safe_malloc(batch*sizeof(magma_int_t)));
  magma_int_t err;

  // FIXME do this in pipelined fashion to reduce memory overhead.
  if (prec == 4) {
    magmaFloatComplex **A_array = static_cast<magmaFloatComplex**>(device_malloc(batch*sizeof(magmaFloatComplex*)));
    magmaFloatComplex **Ainv_array = static_cast<magmaFloatComplex**>(device_malloc(batch*sizeof(magmaFloatComplex*)));
#ifndef MAGMA_2X
    cset_pointer(A_array, static_cast<magmaFloatComplex*>(A_d), n, 0, 0, n*n, batch, queue);
    cset_pointer(Ainv_array, static_cast<magmaFloatComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);
#else
    magma_cset_pointer(A_array, static_cast<magmaFloatComplex*>(A_d), n, 0, 0, n*n, batch, queue);
    magma_cset_pointer(Ainv_array, static_cast<magmaFloatComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);
#endif
    double magma_time = magma_sync_wtime(queue);
    err = magma_cgetrf_batched(n, n, A_array, n, dipiv_array, dinfo_array, batch, queue);
    //err = magma_cgetrf_nopiv_batched(n, n, A_array, n, dinfo_array, batch, queue); (no getri support for nopiv?)
    magma_time = magma_sync_wtime(queue) - magma_time;
    printfQuda("LU factorization completed in %f seconds with GFLOPS = %f\n",
	       magma_time, 1e-9 * batch * FLOPS_CGETRF(n,n) / magma_time);

    if(err != 0) errorQuda("\nError in LU decomposition (magma_cgetrf), error code = %d\n", err);

    qudaMemcpy(info_array, dinfo_array, batch*sizeof(magma_int_t), cudaMemcpyDeviceToHost);
    for (int i=0; i<batch; i++) {
      if (info_array[i] < 0) {
	errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
      } else if (info_array[i] > 0) {
	errorQuda("%d factorization completed but the factor U is exactly singular", i);
      }
    }

    magma_time = magma_sync_wtime(queue);
    err = magma_cgetri_outofplace_batched(n, A_array, n, dipiv_array, Ainv_array, n, dinfo_array, batch, queue);
    magma_time = magma_sync_wtime(queue) - magma_time;
    printfQuda("Matrix inversion completed in %f seconds with GFLOPS = %f\n",
	       magma_time, 1e-9 * batch * FLOPS_CGETRI(n) / magma_time);

    if(err != 0) errorQuda("\nError in matrix inversion (magma_cgetri), error code = %d\n", err);

    qudaMemcpy(info_array, dinfo_array, batch*sizeof(magma_int_t), cudaMemcpyDeviceToHost);

    for (int i=0; i<batch; i++) {
      if (info_array[i] < 0) {
	errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
      } else if (info_array[i] > 0) {
	errorQuda("%d factorization completed but the factor U is exactly singular", i);
      }
    }

    device_free(Ainv_array);
    device_free(A_array);
  } else if (prec == 8) {
    magmaDoubleComplex **A_array    = static_cast<magmaDoubleComplex**>(device_malloc(batch*sizeof(magmaDoubleComplex*)));
    magmaDoubleComplex **Ainv_array = static_cast<magmaDoubleComplex**>(device_malloc(batch*sizeof(magmaDoubleComplex*)));

#ifndef MAGMA_2X
    zset_pointer(A_array, static_cast<magmaDoubleComplex*>(A_d), n, 0, 0, n*n, batch, queue);
    zset_pointer(Ainv_array, static_cast<magmaDoubleComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);
#else
    magma_zset_pointer(A_array, static_cast<magmaDoubleComplex*>(A_d), n, 0, 0, n*n, batch, queue);
    magma_zset_pointer(Ainv_array, static_cast<magmaDoubleComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);
#endif

    double magma_time = magma_sync_wtime(queue);
    err = magma_zgetrf_batched(n, n, A_array, n, dipiv_array, dinfo_array, batch, queue);
    magma_time = magma_sync_wtime(queue) - magma_time;
    printfQuda("LU factorization completed in %f seconds with GFLOPS = %f\n",
	       magma_time, 1e-9 * batch * FLOPS_ZGETRF(n,n) / magma_time);

    if(err != 0) errorQuda("\nError in LU decomposition (magma_zgetrf), error code = %d\n", err);

    qudaMemcpy(info_array, dinfo_array, batch*sizeof(magma_int_t), cudaMemcpyDeviceToHost);
    for (int i=0; i<batch; i++) {
      if (info_array[i] < 0) {
	errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
      } else if (info_array[i] > 0) {
	errorQuda("%d factorization completed but the factor U is exactly singular", i);
      }
    }

    magma_time = magma_sync_wtime(queue);
    err = magma_zgetri_outofplace_batched(n, A_array, n, dipiv_array, Ainv_array, n, dinfo_array, batch, queue);
    magma_time = magma_sync_wtime(queue) - magma_time;
    printfQuda("Matrix inversion completed in %f seconds with GFLOPS = %f\n",
	       magma_time, 1e-9 * batch * FLOPS_ZGETRI(n) / magma_time);

    if(err != 0) errorQuda("\nError in matrix inversion (magma_cgetri), error code = %d\n", err);

    qudaMemcpy(info_array, dinfo_array, batch*sizeof(magma_int_t), cudaMemcpyDeviceToHost);

    for (int i=0; i<batch; i++) {
      if (info_array[i] < 0) {
	errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
      } else if (info_array[i] > 0) {
	errorQuda("%d factorization completed but the factor U is exactly singular", i);
      }
    }

    device_free(Ainv_array);
    device_free(A_array);
  } else {
    errorQuda("%s not implemented for precision=%d", __func__, prec);
  }

  qudaMemcpy(Ainv_h, Ainv_d, size, cudaMemcpyDeviceToHost);

  device_free(dipiv_tmp);
  device_free(dipiv_array);
  device_free(dinfo_array);
  host_free(info_array);
  device_free(Ainv_d);
  device_free(A_d);

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
