#include <blas_cublas.h>
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
  
    using namespace Eigen;

    void init() {}
    
    void destroy() {}
    
    template<typename EigenMatrix, typename Float>
    void invertEigen(std::complex<Float> *A_eig, std::complex<Float> *Ainv_eig, int n, uint64_t batch) {
      
      EigenMatrix res = EigenMatrix::Zero(n,n);
      EigenMatrix inv;
      for(int j = 0; j<n; j++) {
	for(int k = 0; k<n; k++) {
	  res(j,k) = A_eig[batch*n*n + j*n + k];
	}
      }
      
      inv = res.inverse();
      
      for(int j=0; j<n; j++) {
	for(int k=0; k<n; k++) {
	  Ainv_eig[batch*n*n + j*n + k] = inv(j,k);
	}
      }
    }
    
    
    // FIXME do this in pipelined fashion to reduce memory overhead.
    long long BatchInvertMatrix(void *Ainv, void* A, const int n, const uint64_t batch, QudaPrecision prec, QudaFieldLocation location)
    {
      long long flops = 0;
      printfQuda("BatchInvertMatrix: Nc = %d, batch = %lu\n", n, batch);
      timeval start, stop;
      size_t size = 2*n*n*prec*batch;
      
      gettimeofday(&start, NULL);
      void *A_h = location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size): A;
      void *Ainv_h = pool_pinned_malloc(size);
      if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy(A_h, A, size,cudaMemcpyDeviceToHost);
      
      if (prec == QUDA_SINGLE_PRECISION) {
	
	std::complex<float> *A_eig = (std::complex<float> *)A_h;
	std::complex<float> *Ainv_eig = (std::complex<float> *)Ainv_h;
	
#pragma omp parallel for
	for(uint64_t i=0; i<batch; i++) {
	  invertEigen<MatrixXcf, float>(A_eig, Ainv_eig, n, batch);
	}
      }
      else if (prec == QUDA_DOUBLE_PRECISION) {
	std::complex<double> *A_eig = (std::complex<double> *)A_h;
	std::complex<double> *Ainv_eig = (std::complex<double> *)Ainv_h;
	
#pragma omp parallel for
	for(uint64_t i=0; i<batch; i++) {
	  invertEigen<MatrixXcd, double>(A_eig, Ainv_eig, n, batch);
	}	
      } else {
	errorQuda("%s not implemented for precision=%d", __func__, prec);
      }
      
      gettimeofday(&stop, NULL);
      long dsh = stop.tv_sec - start.tv_sec;
      long dush = stop.tv_usec - start.tv_usec;
      double timeh = dsh + 0.000001*dush;
      
      printfQuda("CPU: Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", timeh, 1e-9 * batch*FLOPS_CGETRI(n) / timeh);
      
      //Optional check
#if 0
      void *Ainv_ref_h = location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size): Ainv;
      if (location == QUDA_CUDA_FIELD_LOCATION) cudaMemcpy(Ainv_ref_h, Ainv, size,cudaMemcpyDeviceToHost);
      
      double res=0.0,res2=0.0,norm=0.0;
      float *Ainv_c=(float *)Ainv_h, *Ainv_g=(float *)Ainv_ref_h;
      for(int i=0;i<2*n*n*batch;i++)
	{
	  res+=pow(Ainv_c[i]-Ainv_g[i],2);
	  res2+=pow(Ainv_c[i]-Ainv_g[i],2)/pow(Ainv_g[i],2);
	  norm+=pow(Ainv_g[i],2);
	}
      printfQuda("BatchInvertMatrix: relative difference=%e, diff2 = %e\n",
		 sqrt(res/norm),sqrt(res2/2*n*n*batch));
#endif
      
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	qudaMemcpy(Ainv, Ainv_h, size, cudaMemcpyHostToDevice);
	pool_pinned_free(Ainv_h);
	pool_pinned_free(A_h);
      }
      return flops;
    }
  } // namespace cublas
} // namespace quda
