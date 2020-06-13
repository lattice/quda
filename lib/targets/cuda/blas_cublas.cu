#ifdef NATIVE_BLAS_LIB
#include <blas_lapack.h>
#include <cublas_v2.h>
#include <malloc_quda.h>

namespace quda {

  namespace blas_lapack { 

      
    static cublasHandle_t handle;

    void init() {
      cublasStatus_t error = cublasCreate(&handle);
      if (error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
    }

    void destroy() {
      cublasStatus_t error = cublasDestroy(handle);
      if (error != CUBLAS_STATUS_SUCCESS) errorQuda("\nError indestroying cublas context, error code = %d\n", error);
    }

    // mini kernel to set the array of pointers needed for batched cublas
    template<typename T>
    __global__ void set_pointer(T **output_array_a, T *input_a, T **output_array_b, T *input_b, int batch_offset)
    {
      output_array_a[blockIdx.x] = input_a + blockIdx.x * batch_offset;
      output_array_b[blockIdx.x] = input_b + blockIdx.x * batch_offset;
    }

    // FIXME do this in pipelined fashion to reduce memory overhead.
    long long BatchInvertMatrix(void *Ainv, void* A, const int n, const uint64_t batch, QudaPrecision prec, QudaFieldLocation location)
    {
      if (getVerbosity() >= QUDA_SUMMARIZE)
	printfQuda("BatchInvertMatrixNATIVE: Nc = %d, batch = %lu\n", n, batch);
      long long flops = 0;      
      timeval start, stop;
      gettimeofday(&start, NULL);

      size_t size = 2*n*n*prec*batch;
      void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
      void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
      if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

      int *dipiv = static_cast<int*>(pool_device_malloc(batch*n*sizeof(int)));
      int *dinfo_array = static_cast<int*>(pool_device_malloc(batch*sizeof(int)));
      int *info_array = static_cast<int*>(pool_pinned_malloc(batch*sizeof(int)));
      memset(info_array, '0', batch*sizeof(int)); // silence memcheck warnings

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
	for (uint64_t i=0; i<batch; i++) {
	  if (info_array[i] < 0) {
	    errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed", i);
	  } else if (info_array[i] > 0) {
	    errorQuda("%lu factorization completed but the factor U is exactly singular", i);
	  }
	}
    
	error = cublasCgetriBatched(handle, n, (const C**)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
	flops += batch*FLOPS_CGETRI(n);

	if (error != CUBLAS_STATUS_SUCCESS)
	  errorQuda("\nError in matrix inversion (cublasCgetriBatched), error code = %d\n", error);

	qudaMemcpy(info_array, dinfo_array, batch*sizeof(int), cudaMemcpyDeviceToHost);

	for (uint64_t i=0; i<batch; i++) {
	  if (info_array[i] < 0) {
	    errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed", i);
	  } else if (info_array[i] > 0) {
	    errorQuda("%lu factorization completed but the factor U is exactly singular", i);
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
      
      return flops;
    }
  } // namespace blas_lapack
} // namespace quda
#endif //NATIVE_BLAS_LIB
