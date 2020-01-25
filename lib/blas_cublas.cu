#include "hip/hip_runtime.h"
#ifdef CUBLAS_LIB
#include <blas_cublas.h>
#include <hipblas.h>
#endif
#include <malloc_quda.h>

#include <Eigen/LU>

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
    static hipblasHandle_t handle;
#endif

    void init() {
#ifdef CUBLAS_LIB
      hipblasStatus_t error = hipblasCreate(&handle);
      if (error != HIPBLAS_STATUS_SUCCESS) errorQuda("hipblasCreate failed with error %d", error);
#endif
    }

    void destroy() {
#ifdef CUBLAS_LIB
      hipblasStatus_t error = hipblasDestroy(handle);
      if (error != HIPBLAS_STATUS_SUCCESS) errorQuda("\nError indestroying cublas context, error code = %d\n", error);
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
    long long BatchInvertMatrix(void *Ainv, void* A, const int n, const uint64_t batch, QudaPrecision prec, QudaFieldLocation location)
    {
      long long flops = 0;
      timeval start, stop;
      size_t size = 2*n*n*prec*batch;
/*#ifdef CUBLAS_LIB
      gettimeofday(&start, NULL);

      void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
      void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
      if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, hipMemcpyHostToDevice);

      int *dipiv = static_cast<int*>(pool_device_malloc(batch*n*sizeof(int)));
      int *dinfo_array = static_cast<int*>(pool_device_malloc(batch*sizeof(int)));
      int *info_array = static_cast<int*>(pool_pinned_malloc(batch*sizeof(int)));
      memset(info_array, '0', batch*sizeof(int)); // silence memcheck warnings

      if (prec == QUDA_SINGLE_PRECISION) {
	typedef hipFloatComplex C;
	C **A_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));
	C **Ainv_array = static_cast<C**>(pool_device_malloc(batch*sizeof(C*)));

	set_pointer<C><<<batch,1>>>(A_array, (C*)A_d, Ainv_array, (C*)Ainv_d, n*n);

	hipblasStatus_t error = hipblasCgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
	flops += batch*FLOPS_CGETRF(n,n);

	if (error != HIPBLAS_STATUS_SUCCESS)
	  errorQuda("\nError in LU decomposition (hipblasCgetrfBatched), error code = %d\n", error);

	qudaMemcpy(info_array, dinfo_array, batch*sizeof(int), hipMemcpyDeviceToHost);
	for (uint64_t i=0; i<batch; i++) {
	  if (info_array[i] < 0) {
	    errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed", i);
	  } else if (info_array[i] > 0) {
	    errorQuda("%lu factorization completed but the factor U is exactly singular", i);
	  }
	}
    
	error = hipblasCgetriBatched(handle, n, (const C**)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
	flops += batch*FLOPS_CGETRI(n);

	if (error != HIPBLAS_STATUS_SUCCESS)
	  errorQuda("\nError in matrix inversion (hipblasCgetriBatched), error code = %d\n", error);

	qudaMemcpy(info_array, dinfo_array, batch*sizeof(int), hipMemcpyDeviceToHost);

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
	qudaMemcpy(Ainv, Ainv_d, size, hipMemcpyDeviceToHost);
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
#endif */ // CUBLAS_LIB

      gettimeofday(&start, NULL);
      void *A_h = location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size): A;
      void *Ainv_h = pool_pinned_malloc(size);
      if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy(A_h, A, size,hipMemcpyDeviceToHost);

      if (prec == QUDA_SINGLE_PRECISION) {

        std::complex<float> *A_eig=(std::complex<float> *)A_h;
        std::complex<float> *Ainv_eig=(std::complex<float> *)Ainv_h;

#pragma omp parallel for
        for(int i=0;i<batch;i++)
        {
                Eigen::MatrixXcd res=Eigen::MatrixXcd::Zero(n,n),inv;
                for(int j=0;j<n;j++)
                for(int k=0;k<n;k++)
                        res(j,k)=A_eig[i*n*n+j*n+k];
                inv=res.inverse();
                for(int j=0;j<n;j++)
                for(int k=0;k<n;k++)
                        Ainv_eig[i*n*n+j*n+k]=inv(j,k);
        }
      }
      else {
        errorQuda("%s not implemented for precision=%d", __func__, prec);
      }

     gettimeofday(&stop, NULL);
     long ds = stop.tv_sec - start.tv_sec;
     long dus = stop.tv_usec - start.tv_usec;
     double time = ds + 0.000001*dus;

        printfQuda("CPU: Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * batch*FLOPS_CGETRI(n) / time);

      if (location == QUDA_CUDA_FIELD_LOCATION) {
        qudaMemcpy(Ainv, Ainv_h, size, hipMemcpyHostToDevice);
        pool_pinned_free(Ainv_h);
        pool_pinned_free(A_h);
      }

      return flops;
    }

  } // namespace cublas

} // namespace quda
