#include <blas_magma.h>
#include <string.h>

#include <util_quda.h>
#include <quda_internal.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

#define MAGMA_17 //default version version of the MAGMA library

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

void BlasMagmaArgs::BatchInvertMatrix(void *Ainv_h, void* A_h, const int n, const int batch, const int prec)
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
  set_ipointer(dipiv_array, dipiv_tmp, 1, 0, 0, n, batch, queue);

  magma_int_t *no_piv_array = static_cast<magma_int_t*>(safe_malloc(batch*n*sizeof(magma_int_t)));

  for (int i=0; i<batch; i++) {
    for (int j=0; j<n; j++) {
      no_piv_array[i*n + j] = j+1;
    }
  }
  qudaMemcpy(dipiv_tmp, no_piv_array, batch*n*sizeof(magma_int_t), cudaMemcpyHostToDevice);

  host_free(no_piv_array);

  magma_int_t *dinfo_array = static_cast<magma_int_t*>(device_malloc(batch*sizeof(magma_int_t)));
  magma_int_t *info_array = static_cast<magma_int_t*>(safe_malloc(batch*sizeof(magma_int_t)));
  magma_int_t err;

  // FIXME do this in pipelined fashion to reduce memory overhead.
  if (prec == 4) {
    magmaFloatComplex **A_array = static_cast<magmaFloatComplex**>(device_malloc(batch*sizeof(magmaFloatComplex*)));
    magmaFloatComplex **Ainv_array = static_cast<magmaFloatComplex**>(device_malloc(batch*sizeof(magmaFloatComplex*)));

    cset_pointer(A_array, static_cast<magmaFloatComplex*>(A_d), n, 0, 0, n*n, batch, queue);
    cset_pointer(Ainv_array, static_cast<magmaFloatComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);

    double magma_time = magma_sync_wtime(queue);
    //err = magma_cgetrf_batched(n, n, A_array, n, dipiv_array, dinfo_array, batch, queue);
    err = magma_cgetrf_nopiv_batched(n, n, A_array, n, dinfo_array, batch, queue);
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
    magmaDoubleComplex **A_array = static_cast<magmaDoubleComplex**>(device_malloc(batch*sizeof(magmaDoubleComplex*)));
    zset_pointer(A_array, static_cast<magmaDoubleComplex*>(A_d), n, 0, 0, n*n, batch, queue);

    magmaDoubleComplex **Ainv_array = static_cast<magmaDoubleComplex**>(device_malloc(batch*sizeof(magmaDoubleComplex*)));
    zset_pointer(Ainv_array, static_cast<magmaDoubleComplex*>(Ainv_d), n, 0, 0, n*n, batch, queue);

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
