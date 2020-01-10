#ifdef CUBLAS_LIB
#include <blas_cublas.h>
#include <cublas_v2.h>
#endif
#include <malloc_quda.h>

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

namespace quda {

  namespace cublas { 

#ifdef CUBLAS_LIB
    static cublasHandle_t handle;
#endif

    void init() {
#ifdef CUBLAS_LIB
      cublasStatus_t error = cublasCreate(&handle);
      if (error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
#endif
    }

    void destroy() {
#ifdef CUBLAS_LIB
      cublasStatus_t error = cublasDestroy(handle);
      if (error != CUBLAS_STATUS_SUCCESS) errorQuda("\nError indestroying cublas context, error code = %d\n", error);
#endif
    }

    // mini kernel to set the array of pointers needed for batched cublas
    template<typename T>
    __global__ void set_pointer(T **output_array_a, T *input_a, T **output_array_b, T *input_b, int batch_offset)
    {
      output_array_a[blockIdx.x] = input_a + blockIdx.x * batch_offset;
      output_array_b[blockIdx.x] = input_b + blockIdx.x * batch_offset;
    }

    // FIXME do this in pipelined fashion to reduce memory overhead.
    long long BatchInvertMatrix(void *Ainv, void* A, const int n, const int batch, QudaPrecision prec, QudaFieldLocation location)
    {
      long long flops = 0;
#ifdef CUBLAS_LIB
      timeval start, stop;
      gettimeofday(&start, NULL);

      size_t size = 2*n*n*prec*batch;
      void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
      void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
      if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

      int *dipiv = static_cast<int*>(pool_device_malloc(batch*n*sizeof(int)));
      int *dinfo_array = static_cast<int*>(pool_device_malloc(batch*sizeof(int)));
      int *info_array = static_cast<int*>(pool_pinned_malloc(batch*sizeof(int)));

      if (prec == QUDA_SINGLE_PRECISION) {
	typedef cuFloatComplex C;
	C **A_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));
	C **Ainv_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));

	set_pointer<C><<<batch,1>>>(A_array, (C*)A_d, Ainv_array, (C*)Ainv_d, n*n);

	cublasStatus_t error = cublasCgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
	flops += batch*FLOPS_CGETRF(n,n);

	if (error != CUBLAS_STATUS_SUCCESS)
	  errorQuda("\nError in LU decomposition (cublasCgetrfBatched), error code = %d\n", error);

	qudaMemcpy(info_array, dinfo_array, batch*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<batch; i++) {
	  if (info_array[i] < 0) {
	    errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
	  } else if (info_array[i] > 0) {
	    errorQuda("%d factorization completed but the factor U is exactly singular", i);
	  }
	}
    
	error = cublasCgetriBatched(handle, n, (const C**)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
	flops += batch*FLOPS_CGETRI(n);

	if (error != CUBLAS_STATUS_SUCCESS)
	  errorQuda("\nError in matrix inversion (cublasCgetriBatched), error code = %d\n", error);

	qudaMemcpy(info_array, dinfo_array, batch*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<batch; i++) {
	  if (info_array[i] < 0) {
	    errorQuda("%d argument had an illegal value or another error occured, such as memory allocation failed", i);
	  } else if (info_array[i] > 0) {
	    errorQuda("%d factorization completed but the factor U is exactly singular", i);
	  }
	}

	pool_device_free(Ainv_array);
	pool_device_free(A_array);

      } else {
	errorQuda("%s not implemented for precision=%d", __func__, prec);
      }

      if (location == QUDA_CPU_FIELD_LOCATION) {
	qudaMemcpy(Ainv, Ainv_d, size, cudaMemcpyDeviceToHost);
	pool_device_free(Ainv_d);
	pool_device_free(A_d);
      }

      pool_device_free(dipiv);
      pool_device_free(dinfo_array);
      pool_pinned_free(info_array);

      qudaDeviceSynchronize();
      gettimeofday(&stop, NULL);
      long ds = stop.tv_sec - start.tv_sec;
      long dus = stop.tv_usec - start.tv_usec;
      double time = ds + 0.000001*dus;

      if (getVerbosity() >= QUDA_VERBOSE)
	printfQuda("Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
#endif // CUBLAS_LIB

      return flops;
    }

  } // namespace cublas

} // namespace quda
