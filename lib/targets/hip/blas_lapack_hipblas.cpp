#include <blas_lapack.h>
#ifdef NATIVE_LAPACK_LIB
#include <cublas_v2.h>
#include <malloc_quda.h>
#endif

//#define _DEBUG

#ifdef _DEBUG
#include <eigen_helper.h>
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
        if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, qudaMemcpyHostToDevice);

#ifdef _DEBUG
        // Debug code: Copy original A matrix to host
        std::complex<float> *A_h
          = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<float> *>(pool_pinned_malloc(size)) :
                                                    static_cast<std::complex<float> *>(A_d));
        if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, qudaMemcpyDeviceToHost);
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
          qudaMemcpy(A_array, A_array_h, 2 * batch * sizeof(C *), qudaMemcpyHostToDevice);

          cublasStatus_t error = cublasCgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
          flops += batch * FLOPS_CGETRF(n, n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in LU decomposition (cublasCgetrfBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);
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

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);

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
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcf, float>(A_h, Ainv_h, n, i); }
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
#endif
        } else {
          errorQuda("%s not implemented for precision=%d", __func__, prec);
        }

        if (location == QUDA_CPU_FIELD_LOCATION) {
          qudaMemcpy(Ainv, Ainv_d, size, qudaMemcpyDeviceToHost);
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

    } // namespace native
  }   // namespace blas_lapack
} // namespace quda
