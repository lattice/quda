#include <timer.h>
#include <blas_lapack.h>
#include <eigen_helper.h>

//#define _DEBUG

namespace quda
{
  namespace blas_lapack
  {

    // whether we are using the native blas-lapack library
    static bool native_blas_lapack = true;
    bool use_native() { return native_blas_lapack; }
    void set_native(bool native) { native_blas_lapack = native; }

    namespace generic
    {

      void init() {}

      void destroy() {}

      // Batched inversion ckecking
      //---------------------------------------------------
      template <typename EigenMatrix, typename Float>
      void invertEigen(std::complex<Float> *A_eig, std::complex<Float> *Ainv_eig, int n, uint64_t batch)
      {
        EigenMatrix res = EigenMatrix::Zero(n, n);
        EigenMatrix inv = EigenMatrix::Zero(n, n);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) { res(k, j) = A_eig[batch * n * n + j * n + k]; }
        }

        inv = res.inverse();

        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) { Ainv_eig[batch * n * n + j * n + k] = inv(k, j); }
        }

        // Check result:
#ifdef _DEBUG
        EigenMatrix unit = EigenMatrix::Identity(n, n);
        EigenMatrix prod = res * inv;
        Float L2norm = ((prod - unit).norm() / (n * n));
        printfQuda("Eigen: Norm of (A * Ainv - I) batch %lu = %e\n", batch, L2norm);
#endif
      }
      //---------------------------------------------------

      // Batched Inversions
      //---------------------------------------------------
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision prec,
                                  QudaFieldLocation location)
      {
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (generic - Eigen): Nc = %d, batch = %lu\n", n, batch);

        size_t size = 2 * n * n * batch * prec;
        void *A_h = (location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size) : A);
        void *Ainv_h = (location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size) : Ainv);
        if (location == QUDA_CUDA_FIELD_LOCATION) { qudaMemcpy(A_h, A, size, qudaMemcpyDeviceToHost); }

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        if (prec == QUDA_SINGLE_PRECISION) {
          std::complex<float> *A_eig = (std::complex<float> *)A_h;
          std::complex<float> *Ainv_eig = (std::complex<float> *)Ainv_h;

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (uint64_t i = 0; i < batch; i++) { invertEigen<MatrixXcf, float>(A_eig, Ainv_eig, n, i); }
          flops += batch * FLOPS_CGETRF(n, n);
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          std::complex<double> *A_eig = (std::complex<double> *)A_h;
          std::complex<double> *Ainv_eig = (std::complex<double> *)Ainv_h;

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (uint64_t i = 0; i < batch; i++) { invertEigen<MatrixXcd, double>(A_eig, Ainv_eig, n, i); }
          flops += batch * FLOPS_ZGETRF(n, n);
        } else {
          errorQuda("%s not implemented for precision = %d", __func__, prec);
        }

        gettimeofday(&stop, NULL);
        long dsh = stop.tv_sec - start.tv_sec;
        long dush = stop.tv_usec - start.tv_usec;
        double timeh = dsh + 0.000001 * dush;

        if (getVerbosity() >= QUDA_VERBOSE) {
          int threads = 1;
#ifdef _OPENMP
          threads = omp_get_num_threads();
#endif
          printfQuda("CPU: Batched matrix inversion completed in %f seconds using %d threads with GFLOPS = %f\n", timeh,
                     threads, 1e-9 * flops / timeh);
        }

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
          qudaMemcpy((void *)Ainv, Ainv_h, size, qudaMemcpyHostToDevice);
        }

        return flops;
      }

      // Srided Batched GEMM helpers
      //--------------------------------------------------------------------------
      template <typename EigenMat, typename T>
      void fillArray(EigenMat &EigenArr, T *arr, int rows, int cols, int ld, int offset, bool fill_eigen)
      {
        int counter = offset;
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            if (fill_eigen)
              EigenArr(i, j) = arr[counter];
            else
              arr[counter] = EigenArr(i, j);
            counter++;
          }
          counter += (ld - cols);
        }
      }

      template <typename EigenMat, typename T>
      void GEMM(void *A_h, void *B_h, void *C_h, T alpha, T beta, int max_stride, QudaBLASParam &blas_param)
      {
        // Problem parameters
        int m = blas_param.m;
        int n = blas_param.n;
        int k = blas_param.k;
        int lda = blas_param.lda;
        int ldb = blas_param.ldb;
        int ldc = blas_param.ldc;

        // If the user did not set any stride values, we default them to 1
        // as batch size 0 is an option.
        int a_stride = blas_param.a_stride == 0 ? 1 : blas_param.a_stride;
        int b_stride = blas_param.b_stride == 0 ? 1 : blas_param.b_stride;
        int c_stride = blas_param.c_stride == 0 ? 1 : blas_param.c_stride;
        int a_offset = blas_param.a_offset;
        int b_offset = blas_param.b_offset;
        int c_offset = blas_param.c_offset;
        int batches = blas_param.batch_count;

        // Number of data between batches
        unsigned int A_batch_size = blas_param.lda * blas_param.k;
        if (blas_param.trans_a != QUDA_BLAS_OP_N) A_batch_size = blas_param.lda * blas_param.m;
        unsigned int B_batch_size = blas_param.ldb * blas_param.n;
        if (blas_param.trans_b != QUDA_BLAS_OP_N) B_batch_size = blas_param.ldb * blas_param.k;
        unsigned int C_batch_size = blas_param.ldc * blas_param.n;

        T *A_ptr = (T *)(&A_h)[0];
        T *B_ptr = (T *)(&B_h)[0];
        T *C_ptr = (T *)(&C_h)[0];

        // Eigen objects to store data
        EigenMat Amat = EigenMat::Zero(m, k);
        EigenMat Bmat = EigenMat::Zero(k, n);
        EigenMat Cmat = EigenMat::Zero(m, n);

        for (int batch = 0; batch < batches; batch += max_stride) {

          // Populate Eigen objects
          fillArray<EigenMat, T>(Amat, A_ptr, m, k, lda, a_offset, true);
          fillArray<EigenMat, T>(Bmat, B_ptr, k, n, ldb, b_offset, true);
          fillArray<EigenMat, T>(Cmat, C_ptr, m, n, ldc, c_offset, true);

          // Apply op(A) and op(B)
          switch (blas_param.trans_a) {
          case QUDA_BLAS_OP_T: Amat.transposeInPlace(); break;
          case QUDA_BLAS_OP_C: Amat.adjointInPlace(); break;
          case QUDA_BLAS_OP_N: break;
          default: errorQuda("Unknown blas op type %d", blas_param.trans_a);
          }

          switch (blas_param.trans_b) {
          case QUDA_BLAS_OP_T: Bmat.transposeInPlace(); break;
          case QUDA_BLAS_OP_C: Bmat.adjointInPlace(); break;
          case QUDA_BLAS_OP_N: break;
          default: errorQuda("Unknown blas op type %d", blas_param.trans_b);
          }

          // Perform GEMM using Eigen
          Cmat = alpha * Amat * Bmat + beta * Cmat;

          // Write back to the C array
          fillArray<EigenMat, T>(Cmat, C_ptr, m, n, ldc, c_offset, false);

          a_offset += A_batch_size * a_stride;
          b_offset += B_batch_size * b_stride;
          c_offset += C_batch_size * c_stride;
        }
      }
      //---------------------------------------------------

      // Strided Batched GEMM
      //---------------------------------------------------
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

        // Parse parameters for Eigen
        //-------------------------------------------------------------------------
        // Swap A and B if in column order
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
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
        // If this evaluates to -1, the user did not set any strides.
        if (max_stride <= 0) max_stride = 1;

        // Then number of GEMMs to compute
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

        // Data size of the entire array
        size_t sizeAarr = A_batch_size * data_size * batch;
        size_t sizeBarr = B_batch_size * data_size * batch;
        size_t sizeCarr = C_batch_size * data_size * batch;

        // If already on the host, just use the given pointer. If the data is on
        // the device, allocate host memory and transfer
        void *A_h = location == QUDA_CPU_FIELD_LOCATION ? A_data : pool_pinned_malloc(sizeAarr);
        void *B_h = location == QUDA_CPU_FIELD_LOCATION ? B_data : pool_pinned_malloc(sizeBarr);
        void *C_h = location == QUDA_CPU_FIELD_LOCATION ? C_data : pool_pinned_malloc(sizeCarr);
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          qudaMemcpy(A_h, A_data, sizeAarr, qudaMemcpyDeviceToHost);
          qudaMemcpy(B_h, B_data, sizeBarr, qudaMemcpyDeviceToHost);
          qudaMemcpy(C_h, C_data, sizeCarr, qudaMemcpyDeviceToHost);
        }

        if (blas_param.data_type == QUDA_BLAS_DATATYPE_Z) {

          typedef std::complex<double> Z;
          const Z alpha = blas_param.alpha;
          const Z beta = blas_param.beta;
          GEMM<MatrixXcd, Z>(A_h, B_h, C_h, alpha, beta, max_stride, blas_param);
          flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);

        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_C) {

          typedef std::complex<float> C;
          const C alpha = blas_param.alpha;
          const C beta = blas_param.beta;
          GEMM<MatrixXcf, C>(A_h, B_h, C_h, alpha, beta, max_stride, blas_param);
          flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);

        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_D) {

          typedef double D;
          const D alpha = (D)(static_cast<std::complex<double>>(blas_param.alpha).real());
          const D beta = (D)(static_cast<std::complex<double>>(blas_param.beta).real());
          GEMM<MatrixXd, D>(A_h, B_h, C_h, alpha, beta, max_stride, blas_param);
          flops += batch * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);

        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_S) {

          typedef float S;
          const S alpha = (S)(static_cast<std::complex<float>>(blas_param.alpha).real());
          const S beta = (S)(static_cast<std::complex<float>>(blas_param.beta).real());
          GEMM<MatrixXf, S>(A_h, B_h, C_h, alpha, beta, max_stride, blas_param);
          flops += batch * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);

        } else {
          errorQuda("blasGEMM type %d not implemented\n", blas_param.data_type);
        }

        // Restore the blas parameters to their original values
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
          std::swap(blas_param.m, blas_param.n);
          std::swap(blas_param.lda, blas_param.ldb);
          std::swap(blas_param.trans_a, blas_param.trans_b);
          std::swap(blas_param.a_offset, blas_param.b_offset);
          std::swap(blas_param.a_stride, blas_param.b_stride);
          std::swap(A_data, B_data);
        }

        // Transfer data
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          qudaMemcpy(C_data, C_h, sizeCarr, qudaMemcpyHostToDevice);
          pool_pinned_free(A_h);
          pool_pinned_free(B_h);
          pool_pinned_free(C_h);
        }

        qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        long ds = stop.tv_sec - start.tv_sec;
        long dus = stop.tv_usec - start.tv_usec;
        double time = ds + 0.000001 * dus;
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);

        return flops;
      }
    } // namespace generic
  }   // namespace blas_lapack
} // namespace quda
