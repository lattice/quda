#include <ccomplex>
#include <blas_lapack.h>
#ifdef NATIVE_LAPACK_LIB
#include <cublas_v2.h>
#include <malloc_quda.h>
#endif

//#define _DEBUG

#ifdef _DEBUG
#include <Eigen/LU>
using namespace Eigen;
#endif

namespace quda
{

  namespace blas_lapack
  {

    namespace native
    {

#ifdef NATIVE_LAPACK_LIB
      static cublasHandle_t handle;
#endif
      static bool cublas_init = false;

      void init()
      {
        if (!cublas_init) {
#ifdef NATIVE_LAPACK_LIB
          cublasStatus_t error = cublasCreate(&handle);
          if (error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
          cublas_init = true;
#endif
        }
      }

      void destroy()
      {
        if (cublas_init) {
#ifdef NATIVE_LAPACK_LIB
          cublasStatus_t error = cublasDestroy(handle);
          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError indestroying cublas context, error code = %d\n", error);
          cublas_init = false;
#endif
        }
      }

#ifdef _DEBUG
      template <typename EigenMatrix, typename Float>
      __host__ void checkEigen(std::complex<Float> *A_h, std::complex<Float> *Ainv_h, int n, uint64_t batch)
      {
        EigenMatrix A = EigenMatrix::Zero(n, n);
        EigenMatrix Ainv = EigenMatrix::Zero(n, n);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            A(k, j) = A_h[batch * n * n + j * n + k];
            Ainv(k, j) = Ainv_h[batch * n * n + j * n + k];
          }
        }

        // Check result:
        EigenMatrix unit = EigenMatrix::Identity(n, n);
        EigenMatrix prod = A * Ainv;
        Float L2norm = ((prod - unit).norm() / (n * n));
        printfQuda("cuBLAS: Norm of (A * Ainv - I) batch %lu = %e\n", batch, L2norm);
      }
#endif

      // FIXME do this in pipelined fashion to reduce memory overhead.
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision prec,
                                  QudaFieldLocation location)
      {
#ifdef NATIVE_LAPACK_LIB
        init();
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (native - cuBLAS): Nc = %d, batch = %lu\n", n, batch);

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        size_t size = 2 * n * n * prec * batch;
        void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
        void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
        if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

#ifdef _DEBUG
        // Debug code: Copy original A matrix to host
        std::complex<float> *A_h
          = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<float> *>(pool_pinned_malloc(size)) :
                                                    static_cast<std::complex<float> *>(A_d));
        if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, cudaMemcpyDeviceToHost);
#endif

        int *dipiv = static_cast<int *>(pool_device_malloc(batch * n * sizeof(int)));
        int *dinfo_array = static_cast<int *>(pool_device_malloc(batch * sizeof(int)));
        int *info_array = static_cast<int *>(pool_pinned_malloc(batch * sizeof(int)));
        memset(info_array, '0', batch * sizeof(int)); // silence memcheck warnings

        if (prec == QUDA_SINGLE_PRECISION) {
          typedef cuFloatComplex C;
          C **A_array = static_cast<C **>(pool_device_malloc(2 * batch * sizeof(C *)));
          C **Ainv_array = A_array + batch;
          C **A_array_h = static_cast<C **>(pool_pinned_malloc(2 * batch * sizeof(C *)));
          C **Ainv_array_h = A_array_h + batch;
          for (uint64_t i = 0; i < batch; i++) {
            A_array_h[i] = static_cast<C *>(A_d) + i * n * n;
            Ainv_array_h[i] = static_cast<C *>(Ainv_d) + i * n * n;
          }
          qudaMemcpy(A_array, A_array_h, 2 * batch * sizeof(C *), cudaMemcpyHostToDevice);

          cublasStatus_t error = cublasCgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
          flops += batch * FLOPS_CGETRF(n, n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in LU decomposition (cublasCgetrfBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), cudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          error = cublasCgetriBatched(handle, n, (const C **)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
          flops += batch * FLOPS_CGETRI(n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in matrix inversion (cublasCgetriBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), cudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          pool_device_free(A_array);
          pool_pinned_free(A_array_h);

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          std::complex<float> *Ainv_h = static_cast<std::complex<float> *>(pool_pinned_malloc(size));
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, cudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcf, float>(A_h, Ainv_h, n, i); }
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
#endif
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
        double time = ds + 0.000001 * dus;

        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);

        return flops;
#else
        errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
        return 0; // Stops a compiler warning
#endif
      }

      long long BatchGEMM(void *A_data, void* B_data, void* C_data, QudaCublasParam cublas_param, QudaFieldLocation location)
      {
	long long flops = 0;
#ifdef NATIVE_LAPACK_LIB
	timeval start, stop;
	gettimeofday(&start, NULL);
	
	const uint64_t batch = cublas_param.batch_count;
	uint64_t data_size = (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_S ||
			      cublas_param.data_type == QUDA_CUBLAS_DATATYPE_C) ? 4 : 8;
	
	if(cublas_param.data_type == QUDA_CUBLAS_DATATYPE_C ||
	   cublas_param.data_type == QUDA_CUBLAS_DATATYPE_Z) {
	  data_size *= 2;
	}
	
	// Swap A and B if in row order
	if (cublas_param.data_order == QUDA_CUBLAS_DATAORDER_ROW) {
	  std::swap(cublas_param.m, cublas_param.n);
	  std::swap(cublas_param.lda, cublas_param.ldb);
	  std::swap(cublas_param.trans_a, cublas_param.trans_b);
	  std::swap(cublas_param.a_offset, cublas_param.b_offset);
	  std::swap(A_data, B_data);
	}
	
	// Number of data between batches      
	unsigned int A_batch_size = cublas_param.lda * cublas_param.k;
	if (cublas_param.trans_a != QUDA_CUBLAS_OP_N)
	  A_batch_size = cublas_param.lda * cublas_param.m; 
	unsigned int B_batch_size = cublas_param.ldb * cublas_param.n;
	if (cublas_param.trans_b != QUDA_CUBLAS_OP_N)
	  B_batch_size = cublas_param.ldb * cublas_param.k; 
	unsigned int C_batch_size = cublas_param.ldc * cublas_param.n;       
	
	// Data size of the entire array 
	size_t sizeAarr = A_batch_size * data_size * batch;
	size_t sizeBarr = B_batch_size * data_size * batch;
	size_t sizeCarr = C_batch_size * data_size * batch;
	
	// If already on the device, just use the given pointer. If the data is on
	// the host, allocate device memory and transfer
	void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A_data : pool_device_malloc(sizeAarr);
	void *B_d = location == QUDA_CUDA_FIELD_LOCATION ? B_data : pool_device_malloc(sizeBarr);
	void *C_d = location == QUDA_CUDA_FIELD_LOCATION ? C_data : pool_device_malloc(sizeCarr);
	if (location == QUDA_CPU_FIELD_LOCATION) {
	  qudaMemcpy(A_d, A_data, sizeAarr, cudaMemcpyHostToDevice);
	  qudaMemcpy(B_d, B_data, sizeBarr, cudaMemcpyHostToDevice);
	  qudaMemcpy(C_d, C_data, sizeCarr, cudaMemcpyHostToDevice);
	}
	
	cublasOperation_t trans_a = CUBLAS_OP_N;
	switch(cublas_param.trans_a) {
	case QUDA_CUBLAS_OP_N: trans_a = CUBLAS_OP_N; break;
	case QUDA_CUBLAS_OP_T: trans_a = CUBLAS_OP_T; break;
	case QUDA_CUBLAS_OP_C: trans_a = CUBLAS_OP_C; break;
	default : errorQuda("Unknown QUDA_CUBLAS_OP type %d\n", cublas_param.trans_a);
	}
	
	cublasOperation_t trans_b = CUBLAS_OP_N;
	switch(cublas_param.trans_b) {
	case QUDA_CUBLAS_OP_N: trans_b = CUBLAS_OP_N; break;
	case QUDA_CUBLAS_OP_T: trans_b = CUBLAS_OP_T; break;
	case QUDA_CUBLAS_OP_C: trans_b = CUBLAS_OP_C; break;
	default : errorQuda("Unknown QUDA_CUBLAS_OP type %d\n", cublas_param.trans_b);
	}
	
	if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_Z) {
#if 0
	  typedef cuDoubleComplex Z ;
	  
	  const Z alpha = make_double2((double)creal(cublas_param.alpha),
				       (double)cimag(cublas_param.alpha));
	  
	  const Z beta  = make_double2((double)creal(cublas_param.beta),
				       (double)cimag(cublas_param.beta));
	  
	  cublasStatus_t error;
	  if(batch > 1) {
	    Z **A_ptr_array = static_cast<Z**>(pool_device_malloc(batch*sizeof(Z*)));
	    Z **B_ptr_array = static_cast<Z**>(pool_device_malloc(batch*sizeof(Z*)));
	    Z **C_ptr_array = static_cast<Z**>(pool_device_malloc(batch*sizeof(Z*)));
	    
	    set_pointer_gemm<Z><<<batch,1>>>(A_ptr_array, (Z*)A_d + cublas_param.a_offset, A_batch_size,
					     B_ptr_array, (Z*)B_d + cublas_param.b_offset, B_batch_size,
					     C_ptr_array, (Z*)C_d + cublas_param.c_offset, C_batch_size);
	    
	    error = cublasZgemmBatched(handle, trans_a, trans_b, cublas_param.m,
				       cublas_param.n, cublas_param.k, &alpha,
				       A_ptr_array, cublas_param.lda,
				       B_ptr_array, cublas_param.ldb, &beta,
				       C_ptr_array, cublas_param.ldc, batch);
	    
	    pool_device_free(A_ptr_array);
	    pool_device_free(B_ptr_array);
	    pool_device_free(C_ptr_array);
	    
	  } else {
	    error = cublasZgemm(handle, trans_a, trans_b, cublas_param.m,
				cublas_param.n, cublas_param.k, &alpha,
				(Z*)A_d + cublas_param.a_offset, cublas_param.lda,
				(Z*)B_d + cublas_param.b_offset, cublas_param.ldb, &beta,
				(Z*)C_d + cublas_param.c_offset, cublas_param.ldc);
	  }
	  
	  //flops += batch*FLOPS_CGETRF(n,n);
	  if (error != CUBLAS_STATUS_SUCCESS)
	    errorQuda("\nError in cuBLASZGEMMBatched), error code = %d\n", error);
#endif	  
	} else if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_C) {
	  
	  typedef cuComplex C;

	  const C alpha  = make_float2((float)creal(cublas_param.alpha),
				       (float)cimag(cublas_param.alpha));	  
	  
	  const C beta  = make_float2((float)creal(cublas_param.beta),
				      (float)cimag(cublas_param.beta));
	  
	  cublasStatus_t error;
	  if(batch > 1) {
	    C **A_ptr_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));
	    C **B_ptr_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));
	    C **C_ptr_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));
	    
	    for (uint64_t i = 0; i < batch; i++) {
	      A_ptr_array[i] = static_cast<C *>(A_d) + cublas_param.a_offset + i * A_batch_size;
	      B_ptr_array[i] = static_cast<C *>(B_d) + cublas_param.b_offset + i * B_batch_size;
	      C_ptr_array[i] = static_cast<C *>(C_d) + cublas_param.c_offset + i * C_batch_size;
	    }
	    
	    error = cublasCgemmBatched(handle, trans_a, trans_b, cublas_param.m,
				       cublas_param.n, cublas_param.k, &alpha,
				       A_ptr_array, cublas_param.lda,
				       B_ptr_array, cublas_param.ldb, &beta,
				       C_ptr_array, cublas_param.ldc, batch);
	    
	    pool_device_free(A_ptr_array);
	    pool_device_free(B_ptr_array);
	    pool_device_free(C_ptr_array);
	    
	  } else {
	    error = cublasCgemm(handle, trans_a, trans_b, cublas_param.m,
				cublas_param.n, cublas_param.k, &alpha,
				(C*)A_d + cublas_param.a_offset, cublas_param.lda,
				(C*)B_d + cublas_param.b_offset, cublas_param.ldb, &beta,
				(C*)C_d + cublas_param.c_offset, cublas_param.ldc);
	    
	  }
	  
	  //flops += batch*FLOPS_CGETRF(n,n);
	  if (error != CUBLAS_STATUS_SUCCESS)
	    errorQuda("\nError in cuBLASCGEMMBatched), error code = %d\n", error);
	  
	  
	} else if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_D) {
#if 0
	  typedef double D;
	  
	  const D alpha = (D)creal(cublas_param.alpha);	
	  const D beta  = (D)creal(cublas_param.beta);
	  
	  cublasStatus_t error;
	  if(batch > 1) {
	    D **A_ptr_array = static_cast<D**>(pool_device_malloc(batch*sizeof(D*)));
	    D **B_ptr_array = static_cast<D**>(pool_device_malloc(batch*sizeof(D*)));
	    D **C_ptr_array = static_cast<D**>(pool_device_malloc(batch*sizeof(D*)));
	    
	    set_pointer_gemm<D><<<batch,1>>>(A_ptr_array, (D*)A_d + cublas_param.a_offset, A_batch_size,
					     B_ptr_array, (D*)B_d + cublas_param.b_offset, B_batch_size,
					     C_ptr_array, (D*)C_d + cublas_param.c_offset, C_batch_size);
	    
	    error = cublasDgemmBatched(handle, trans_a, trans_b, cublas_param.m,
				       cublas_param.n, cublas_param.k, &alpha,
				       A_ptr_array, cublas_param.lda,
				       B_ptr_array, cublas_param.ldb, &beta,
				       C_ptr_array, cublas_param.ldc, batch);
	    
	    pool_device_free(A_ptr_array);
	    pool_device_free(B_ptr_array);
	    pool_device_free(C_ptr_array);
	    
	  } else {
	    error = cublasDgemm(handle, trans_a, trans_b, cublas_param.m,
				cublas_param.n, cublas_param.k, &alpha,
				(D*)A_d + cublas_param.a_offset, cublas_param.lda,
				(D*)B_d + cublas_param.b_offset, cublas_param.ldb, &beta,
				(D*)C_d + cublas_param.c_offset, cublas_param.ldc);
	    
	  }
	  
	  //flops += batch*FLOPS_CGETRF(n,n);
	  if (error != CUBLAS_STATUS_SUCCESS)
	    errorQuda("\nError in cuBLASDGEMMBatched), error code = %d\n", error);
#endif	  
	} else if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_S) {
#if 0	  
	  typedef float S;
	  
	  const S alpha = (S)creal(cublas_param.alpha);	
	  const S beta  = (S)creal(cublas_param.beta);
	  
	  cublasStatus_t error;
	  if(batch > 1) {
	    S **A_ptr_array = static_cast<S**>(pool_device_malloc(batch*sizeof(S*)));
	    S **B_ptr_array = static_cast<S**>(pool_device_malloc(batch*sizeof(S*)));
	    S **C_ptr_array = static_cast<S**>(pool_device_malloc(batch*sizeof(S*)));
	    
	    set_pointer_gemm<S><<<batch,1>>>(A_ptr_array, (S*)A_d + cublas_param.a_offset, A_batch_size,
					     B_ptr_array, (S*)B_d + cublas_param.b_offset, B_batch_size,
					     C_ptr_array, (S*)C_d + cublas_param.c_offset, C_batch_size);
	    
	    error = cublasSgemmBatched(handle, trans_a, trans_b, cublas_param.m,
				       cublas_param.n, cublas_param.k, &alpha,
				       A_ptr_array, cublas_param.lda,
				       B_ptr_array, cublas_param.ldb, &beta,
				       C_ptr_array, cublas_param.ldc, batch);
	    
	    pool_device_free(A_ptr_array);
	    pool_device_free(B_ptr_array);
	    pool_device_free(C_ptr_array);
	    
	  } else {
	    error = cublasSgemm(handle, trans_a, trans_b, cublas_param.m,
				cublas_param.n, cublas_param.k, &alpha,
				(S*)A_d + cublas_param.a_offset, cublas_param.lda,
				(S*)B_d + cublas_param.b_offset, cublas_param.ldb, &beta,
				(S*)C_d + cublas_param.c_offset, cublas_param.ldc);
	    
	  }
	  
	  //flops += batch*FLOPS_CGETRF(n,n);
	  if (error != CUBLAS_STATUS_SUCCESS)
	    errorQuda("\nError in cuBLASSGEMMBatched), error code = %d\n", error);	
#endif	    	  
	} else {
	  errorQuda("cublasGEMM type %d not implemented\n", cublas_param.data_type);  	
	}
	
	// Restore A and B if in row order
	if (cublas_param.data_order == QUDA_CUBLAS_DATAORDER_ROW) {
	  std::swap(cublas_param.m, cublas_param.n);
	  std::swap(cublas_param.lda, cublas_param.ldb);
	  std::swap(cublas_param.trans_a, cublas_param.trans_b);
	  std::swap(cublas_param.a_offset, cublas_param.b_offset);
	  std::swap(A_data, B_data);
	}
	
	if (location == QUDA_CPU_FIELD_LOCATION) {
	  qudaMemcpy(C_data, C_d, sizeCarr, cudaMemcpyDeviceToHost);
	  pool_device_free(A_d);
	  pool_device_free(B_d);
	  pool_device_free(C_d);
	}
	
	qudaDeviceSynchronize();
	gettimeofday(&stop, NULL);
	long ds = stop.tv_sec - start.tv_sec;
	long dus = stop.tv_usec - start.tv_usec;
	double time = ds + 0.000001*dus;
	if (getVerbosity() >= QUDA_VERBOSE)
	  printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
	
#endif // CUBLAS_LIB
	
	return flops;
      }
    } // namespace native
  }   // namespace blas_lapack
} // namespace quda
