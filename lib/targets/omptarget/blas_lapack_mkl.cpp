#include <complex.h>
#include <blas_lapack.h>
#include <timer.h>
#ifdef NATIVE_LAPACK_LIB
#include <mkl.h>
#include <mkl_omp_offload.h>
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

      void init() {}
      void destroy() {}

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

#ifdef NATIVE_LAPACK_LIB
      // FIXME do this in pipelined fashion to reduce memory overhead.
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision prec,
                                  QudaFieldLocation location)
      {
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (native - MKL): Nc = %d, batch = %lu\n", n, batch);

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        size_t size = 2 * n * n * prec * batch;
        void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
        void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
        if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, qudaMemcpyHostToDevice);

#ifdef _DEBUG
        // Debug code: Copy original A matrix to host
        if (prec == QUDA_SINGLE_PRECISION) {
          std::complex<float> *A_h
            = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<float> *>(pool_pinned_malloc(size)) :
                                                      static_cast<std::complex<float> *>(A_d));
          if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, qudaMemcpyDeviceToHost);
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          std::complex<double> *A_h
            = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<double> *>(pool_pinned_malloc(size)) :
                                                      static_cast<std::complex<double> *>(A_d));
          if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, qudaMemcpyDeviceToHost);
        } else {
          errorQuda("%s not implemented for precision=%d", __func__, prec);
        }
#endif

        MKL_INT *dipiv = static_cast<MKL_INT *>(pool_device_malloc(batch * n * sizeof(MKL_INT)));
        MKL_INT *dinfo_array = static_cast<MKL_INT *>(pool_device_malloc(batch * sizeof(MKL_INT)));
        MKL_INT *info_array = static_cast<MKL_INT *>(pool_pinned_malloc(batch * sizeof(MKL_INT)));
        memset(info_array, '0', batch * sizeof(MKL_INT)); // silence memcheck warnings

        MKL_INT n_array = n;
        MKL_INT stride_array = n * n;
        MKL_INT batch_size = batch;

        if (prec == QUDA_SINGLE_PRECISION) {
          typedef MKL_Complex8 C;
          #pragma omp target variant dispatch use_device_ptr(A_d, dipiv, dinfo_array)
          {
            cgetrf_batch_strided(&n_array, &n_array, (C *)A_d, &n_array, &stride_array, dipiv, &n_array, &batch_size, dinfo_array);
          }
          flops += batch * FLOPS_CGETRF(n, n);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(MKL_INT), qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          #pragma omp target variant dispatch use_device_ptr(A_d, Ainv_d, dipiv, dinfo_array)
          {
            cgetri_oop_batch_strided(&n_array, (C *)A_d, &n_array, &stride_array, dipiv, &n_array, (C *)Ainv_d, &n_array, &stride_array, &batch_size, dinfo_array);
          }
          flops += batch * FLOPS_CGETRI(n);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(MKL_INT), qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          std::complex<float> *Ainv_h = static_cast<std::complex<float> *>(pool_pinned_malloc(size));
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcf, float>(A_h, Ainv_h, n, i); }
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
#endif
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          typedef MKL_Complex16 Z;
          #pragma omp target variant dispatch use_device_ptr(A_d, dipiv, dinfo_array)
          {
            zgetrf_batch_strided(&n_array, &n_array, (Z *)A_d, &n_array, &stride_array, dipiv, &n_array, &batch_size, dinfo_array);
          }
          flops += batch * FLOPS_ZGETRF(n, n);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(MKL_INT), qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          #pragma omp target variant dispatch use_device_ptr(A_d, Ainv_d, dipiv, dinfo_array)
          {
            zgetri_oop_batch_strided(&n_array, (Z *)A_d, &n_array, &stride_array, dipiv, &n_array, (Z *)Ainv_d, &n_array, &stride_array, &batch_size, dinfo_array);
          }
          flops += batch * FLOPS_CGETRI(n);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(MKL_INT), qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          std::complex<double> *Ainv_h = static_cast<std::complex<double> *>(pool_pinned_malloc(size));
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcd, double>(A_h, Ainv_h, n, i); }
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
      }
#else
      long long BatchInvertMatrix(void *, void *, const int, const uint64_t, QudaPrecision, QudaFieldLocation)
      {
        errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
        return 0; // Stops a compiler warning
      }
#endif

#ifdef NATIVE_LAPACK_LIB
      long long stridedBatchGEMM(void *A_data, void *B_data, void *C_data, QudaBLASParam blas_param,
                                 QudaFieldLocation location)
      {
        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        // Sanity checks on parameters
        //-------------------------------------------------------------------------
        // If the user passes non positive M,N, or K, we error out
        int min_dim = std::min(blas_param.m, std::min(blas_param.n, blas_param.k));
        if (min_dim <= 0) {
          errorQuda("BLAS dims must be positive: m=%d, n=%d, k=%d", blas_param.m, blas_param.n, blas_param.k);
        }

        // If the user passes a negative stride, we error out as this has no meaning.
        int min_stride = std::min(std::min(blas_param.a_stride, blas_param.b_stride), blas_param.c_stride);
        if (min_stride < 0) {
          errorQuda("BLAS strides must be positive or zero: a_stride=%d, b_stride=%d, c_stride=%d", blas_param.a_stride,
                    blas_param.b_stride, blas_param.c_stride);
        }

        // If the user passes a negative offset, we error out as this has no meaning.
        int min_offset = std::min(std::min(blas_param.a_offset, blas_param.b_offset), blas_param.c_offset);
        if (min_offset < 0) {
          errorQuda("BLAS offsets must be positive or zero: a_offset=%d, b_offset=%d, c_offset=%d", blas_param.a_offset,
                    blas_param.b_offset, blas_param.c_offset);
        }

        // If the batch value is non-positve, we error out
        if (blas_param.batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param.batch_count); }

        // Leading dims are dependendent on the matrix op type.
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
          if (blas_param.trans_a == QUDA_BLAS_OP_N) {
            if (blas_param.lda < std::max(1, blas_param.m))
              errorQuda("lda=%d must be >= max(1,m=%d)", blas_param.lda, blas_param.m);
          } else {
            if (blas_param.lda < std::max(1, blas_param.k))
              errorQuda("lda=%d must be >= max(1,k=%d)", blas_param.lda, blas_param.k);
          }

          if (blas_param.trans_b == QUDA_BLAS_OP_N) {
            if (blas_param.ldb < std::max(1, blas_param.k))
              errorQuda("ldb=%d must be >= max(1,k=%d)", blas_param.ldb, blas_param.k);
          } else {
            if (blas_param.ldb < std::max(1, blas_param.n))
              errorQuda("ldb=%d must be >= max(1,n=%d)", blas_param.ldb, blas_param.n);
          }
          if (blas_param.ldc < std::max(1, blas_param.m))
            errorQuda("ldc=%d must be >= max(1,m=%d)", blas_param.ldc, blas_param.m);
        } else {
          if (blas_param.trans_a == QUDA_BLAS_OP_N) {
            if (blas_param.lda < std::max(1, blas_param.k))
              errorQuda("lda=%d must be >= max(1,k=%d)", blas_param.lda, blas_param.k);
          } else {
            if (blas_param.lda < std::max(1, blas_param.m))
              errorQuda("lda=%d must be >= max(1,m=%d)", blas_param.lda, blas_param.m);
          }
          if (blas_param.trans_b == QUDA_BLAS_OP_N) {
            if (blas_param.ldb < std::max(1, blas_param.n))
              errorQuda("ldb=%d must be >= max(1,n=%d)", blas_param.ldb, blas_param.n);
          } else {
            if (blas_param.ldb < std::max(1, blas_param.k))
              errorQuda("ldb=%d must be >= max(1,k=%d)", blas_param.ldb, blas_param.k);
          }
          if (blas_param.ldc < std::max(1, blas_param.n))
            errorQuda("ldc=%d must be >= max(1,n=%d)", blas_param.ldc, blas_param.n);
        }
        //-------------------------------------------------------------------------

        // Parse parameters for CUBLAS
        //-------------------------------------------------------------------------
        // Swap A and B if in row order
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_ROW) {
          std::swap(blas_param.m, blas_param.n);
          std::swap(blas_param.lda, blas_param.ldb);
          std::swap(blas_param.trans_a, blas_param.trans_b);
          std::swap(blas_param.a_offset, blas_param.b_offset);
          std::swap(blas_param.a_stride, blas_param.b_stride);
          std::swap(A_data, B_data);
        }

        // Get maximum stride length to deduce the number of batches in the
        // computation
        int max_stride = std::max(std::max(blas_param.a_stride, blas_param.b_stride), blas_param.c_stride);

        // If the user gives strides of 0 for all arrays, we are essentially performing
        // a GEMM on the first matrices in the array N_{batch} times.
        // Give them what they ask for, YMMV...
        // If the strides have not been set, we are just using strides of 1.
        if (max_stride == 0) max_stride = 1;

        // The number of GEMMs to compute
        const uint64_t batch = blas_param.batch_count / max_stride;

        uint64_t data_size
          = (blas_param.data_type == QUDA_BLAS_DATATYPE_S || blas_param.data_type == QUDA_BLAS_DATATYPE_C) ? 4 : 8;

        if (blas_param.data_type == QUDA_BLAS_DATATYPE_C || blas_param.data_type == QUDA_BLAS_DATATYPE_Z) {
          data_size *= 2;
        }

        // Number of data between batches
        unsigned int A_batch_size = blas_param.lda * blas_param.k;
        if (blas_param.trans_a != QUDA_BLAS_OP_N) A_batch_size = blas_param.lda * blas_param.m;
        unsigned int B_batch_size = blas_param.ldb * blas_param.n;
        if (blas_param.trans_b != QUDA_BLAS_OP_N) B_batch_size = blas_param.ldb * blas_param.k;
        unsigned int C_batch_size = blas_param.ldc * blas_param.n;

        // Strides in the cublas param are defaulted to -1. If that remains unchanged,
        // the stride will be the regular batch size, else the user specified value
        // is used.
        unsigned int a_stride = blas_param.a_stride == 0 ? A_batch_size : A_batch_size * blas_param.a_stride;
        unsigned int b_stride = blas_param.b_stride == 0 ? B_batch_size : B_batch_size * blas_param.b_stride;
        unsigned int c_stride = blas_param.c_stride == 0 ? C_batch_size : C_batch_size * blas_param.c_stride;

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
          qudaMemcpy(A_d, A_data, sizeAarr, qudaMemcpyHostToDevice);
          qudaMemcpy(B_d, B_data, sizeBarr, qudaMemcpyHostToDevice);
          qudaMemcpy(C_d, C_data, sizeCarr, qudaMemcpyHostToDevice);
        }

        CBLAS_TRANSPOSE trans_a = CblasNoTrans;
        switch (blas_param.trans_a) {
        case QUDA_BLAS_OP_N: trans_a = CblasNoTrans; break;
        case QUDA_BLAS_OP_T: trans_a = CblasTrans; break;
        case QUDA_BLAS_OP_C: trans_a = CblasConjTrans; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_a);
        }

        CBLAS_TRANSPOSE trans_b = CblasNoTrans;
        switch (blas_param.trans_b) {
        case QUDA_BLAS_OP_N: trans_b = CblasNoTrans; break;
        case QUDA_BLAS_OP_T: trans_b = CblasTrans; break;
        case QUDA_BLAS_OP_C: trans_b = CblasConjTrans; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_b);
        }
        //-------------------------------------------------------------------------

        // Call CUBLAS
        //-------------------------------------------------------------------------
        if (blas_param.data_type == QUDA_BLAS_DATATYPE_Z) {

          typedef MKL_Complex16 Z;
          static_assert(sizeof(Z)==sizeof(double2), "MKL_Complex16 and double2 must be the same.");

          const double2 alpha = make_double2((double)(static_cast<std::complex<double>>(blas_param.alpha).real()),
                                             (double)(static_cast<std::complex<double>>(blas_param.alpha).imag()));

          const double2 beta = make_double2((double)(static_cast<std::complex<double>>(blas_param.beta).real()),
                                            (double)(static_cast<std::complex<double>>(blas_param.beta).imag()));

          if (batch > 1) {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_zgemm_batch_strided(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k,
                                        &alpha, (Z *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
                                        (Z *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, &beta,
                                        (Z *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
            }
          } else {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_zgemm(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
                          (Z *)A_d + blas_param.a_offset, blas_param.lda, (Z *)B_d + blas_param.b_offset,
                          blas_param.ldb, &beta, (Z *)C_d + blas_param.c_offset, blas_param.ldc);
            }
          }
          flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_C) {

          typedef MKL_Complex8 C;
          static_assert(sizeof(C)==sizeof(float2), "MKL_Complex8 and float2 must be the same.");

          const float2 alpha = make_float2((float)(static_cast<std::complex<double>>(blas_param.alpha).real()),
                                           (float)(static_cast<std::complex<double>>(blas_param.alpha).imag()));

          const float2 beta = make_float2((float)(static_cast<std::complex<double>>(blas_param.beta).real()),
                                          (float)(static_cast<std::complex<double>>(blas_param.beta).imag()));

          if (batch > 1) {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_cgemm_batch_strided(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k,
                                         &alpha, (C *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
                                         (C *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, &beta,
                                         (C *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
            }
          } else {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_cgemm(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
                          (C *)A_d + blas_param.a_offset, blas_param.lda, (C *)B_d + blas_param.b_offset,
                          blas_param.ldb, &beta, (C *)C_d + blas_param.c_offset, blas_param.ldc);
            }
          }
          flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_D) {

          typedef double D;

          const D alpha = (D)(static_cast<std::complex<double>>(blas_param.alpha).real());
          const D beta = (D)(static_cast<std::complex<double>>(blas_param.beta).real());

          if (batch > 1) {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_dgemm_batch_strided(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k,
                                        alpha, (D *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
                                        (D *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
                                        (D *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
            }
          } else {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_dgemm(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
                          (D *)A_d + blas_param.a_offset, blas_param.lda, (D *)B_d + blas_param.b_offset,
                          blas_param.ldb, beta, (D *)C_d + blas_param.c_offset, blas_param.ldc);
            }
          }
          flops += batch * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_S) {

          typedef float S;

          const S alpha = (S)(static_cast<std::complex<float>>(blas_param.alpha).real());
          const S beta = (S)(static_cast<std::complex<float>>(blas_param.beta).real());

          if (batch > 1) {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_sgemm_batch_strided(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k,
                                        alpha, (S *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
                                        (S *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
                                        (S *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
            }
          } else {
            #pragma omp target variant dispatch use_device_ptr(A_d, B_d, C_d)
            {
              cblas_sgemm(CblasColMajor, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
                          (S *)A_d + blas_param.a_offset, blas_param.lda, (S *)B_d + blas_param.b_offset,
                          blas_param.ldb, beta, (S *)C_d + blas_param.c_offset, blas_param.ldc);
            }
          }
          flops += batch * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else {
          errorQuda("MKL GEMM type %d not implemented\n", blas_param.data_type);
        }
        //-------------------------------------------------------------------------

        // Clean up
        //-------------------------------------------------------------------------
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_ROW) {
          std::swap(blas_param.m, blas_param.n);
          std::swap(blas_param.lda, blas_param.ldb);
          std::swap(blas_param.trans_a, blas_param.trans_b);
          std::swap(blas_param.a_offset, blas_param.b_offset);
          std::swap(blas_param.a_stride, blas_param.b_stride);
          std::swap(A_data, B_data);
        }

        if (location == QUDA_CPU_FIELD_LOCATION) {
          qudaMemcpy(C_data, C_d, sizeCarr, qudaMemcpyDeviceToHost);
          pool_device_free(A_d);
          pool_device_free(B_d);
          pool_device_free(C_d);
        }

        qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        long ds = stop.tv_sec - start.tv_sec;
        long dus = stop.tv_usec - start.tv_usec;
        double time = ds + 0.000001 * dus;
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
        //-------------------------------------------------------------------------

        return flops;
      }
#else
      long long stridedBatchGEMM(void *, void *, void *, QudaBLASParam, QudaFieldLocation)
      {
        errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
        return 0;
      }
#endif

    } // namespace native
  }   // namespace blas_lapack
} // namespace quda
