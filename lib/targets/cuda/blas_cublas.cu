#include <blas_lapack.h>
#ifdef NATIVE_LAPACK_LIB
#include <cublas_v2.h>
#include <malloc_quda.h>
#endif
namespace quda {

  namespace native_lapack { 
    
#ifdef NATIVE_LAPACK_LIB
    static cublasHandle_t handle;
#endif
    static bool cublas_init = false;
      
    void init() {
      if(!cublas_init) {
#ifdef NATIVE_LAPACK_LIB
	cublasStatus_t error = cublasCreate(&handle);
	if (error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
	cublas_init = true;
#endif
      }
    }

    void destroy() {
      if(cublas_init) {
#ifdef NATIVE_LAPACK_LIB
	cublasStatus_t error = cublasDestroy(handle);
	if (error != CUBLAS_STATUS_SUCCESS) errorQuda("\nError indestroying cublas context, error code = %d\n", error);
	cublas_init = false;
#endif
      }
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
#ifdef NATIVE_LAPACK_LIB
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
      
#if 0
      // Debug code
      std::complex<float> *A_h = static_cast<std::complex<float>*>(pool_pinned_malloc(size));
      std::complex<float> *Ainv_h = static_cast<std::complex<float>*>(pool_pinned_malloc(size));
      
      qudaMemcpy((void*)Ainv_h, Ainv_d, size, cudaMemcpyDeviceToHost);
      qudaMemcpy((void*)A_h, A_d, size, cudaMemcpyDeviceToHost);
      
      double residual2 = 0.0, residual2_max = 0.0;
      for(int i=0;i<batch;i++) {
	std::vector<std::complex<float>> unit(n*n);
	for(int j=0; j<n*n; j++)  unit[j]=0.0;
	for(int j=0; j<n; j++)
	  for(int k=0; k<n; k++)
	    for(int l=0; l<n; l++) {
	      //I_{k,j} = Ainv_{k,l} * A_{l,j}
	      unit[j*n + k] += Ainv_h[i*n*n + l*n + k] * A_h[i*n*n + j*n + l];
	    }
	
	for(int j=0;j<n;j++)
	  for(int k=0;k<n;k++) {
	    
	    if(j==k) unit[j*n+k] -= 1.0;

	    double tmp=std::norm(unit[j*n+k]);
	    if(tmp>residual2) residual2_max = tmp;
	    residual2 += tmp;
	    
	  }	
      }
      printf("Ainv*A GPU Check: batch=%lu n=%d average diff=%15.7f, max diff=%15.7f\n", batch, n, sqrt(residual2/(n*n*batch)), sqrt(residual2_max));	
      
      pool_pinned_free(Ainv_h);
      pool_pinned_free(A_h);	     
#endif
  
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
#else
      errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
      return 0; // Stops a compiler warning
#endif
    }
  } // namespace native_lapack
} // namespace quda

