#include <blas_lapack.h>
#include <timer.h>

#ifdef NATIVE_LAPACK_LIB
#include <quda_sycl_api.h>
//#include <complex>
#include <oneapi/mkl.hpp>
#include <malloc_quda.h>
using namespace oneapi::mkl;
using namespace oneapi::mkl::lapack;
#endif

#define _DEBUG

#ifdef _DEBUG
#include <eigen_helper.h>
#endif

namespace quda
{

  namespace blas_lapack
  {

    namespace native
    {

      static bool native_init = false;

      void init()
      {
        if (!native_init) {
          native_init = true;
#ifndef NATIVE_LAPACK_LIB
	  quda::blas_lapack::generic::init();
#endif
        }
      }

      void destroy()
      {
        if (native_init) {
          native_init = false;
#ifndef NATIVE_LAPACK_LIB
	  quda::blas_lapack::generic::destroy();
#endif
        }
      }

#ifdef _DEBUG
      template <typename EigenMatrix, typename Float>
      __host__ void checkEigen(std::complex<Float> *A_h, std::complex<Float> *Ainv_h,
			       int n, uint64_t batch)
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
        printfQuda("oneMKL: Norm of (A * Ainv - I) batch %lu = %e\n", batch, L2norm);
      }
#endif

#ifdef NATIVE_LAPACK_LIB
      // FIXME do this in pipelined fashion to reduce memory overhead.
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch,
				  QudaPrecision prec, QudaFieldLocation location)
      {
        init();
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (native - oneMKL): Nc = %d, batch = %lu\n", n, batch);

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

	std::int64_t stride_a = n * n;
        size_t size = 2 * n * n * prec * batch;
	std::int64_t *dipiv = static_cast<std::int64_t *>(pool_device_malloc(batch * n * sizeof(std::int64_t)));
	auto q = device::defaultQueue();

        if (prec == QUDA_SINGLE_PRECISION) {
          typedef std::complex<float> C;

	  C *A_d = static_cast<C *>(A);
	  C *Ainv_d = static_cast<C *>(Ainv);
	  if(location == QUDA_CPU_FIELD_LOCATION) {
	    A_d = static_cast<C *>(pool_device_malloc(size));
	    Ainv_d = static_cast<C *>(pool_device_malloc(size));
	    qudaMemcpy(A_d, A, size, qudaMemcpyHostToDevice);
	  }

#ifdef _DEBUG
	  // Debug code: Copy original A matrix to host
	  C *A_h = static_cast<C *>(pool_pinned_malloc(size));
	  qudaMemcpy(A_h, A_d, size, qudaMemcpyDeviceToHost);
#endif

	  try {
	    std::int64_t getrf_scratchpad_size = getrf_batch_scratchpad_size<C>(q, n, n, n, stride_a, n, batch);
	    C *getrf_scratchpad = static_cast<C *>(pool_device_malloc(getrf_scratchpad_size*sizeof(C)));
	    auto getrf_event = getrf_batch(q, n, n, A_d, n, stride_a, dipiv, n, batch, getrf_scratchpad, getrf_scratchpad_size);
	    flops += batch * FLOPS_CGETRF(n, n);

	    std::int64_t getri_scratchpad_size = getri_batch_scratchpad_size<C>(q, n, n, n, n, batch);
	    C *getri_scratchpad = static_cast<C *>(pool_device_malloc(getri_scratchpad_size*sizeof(C)));
	    auto getri_event = getri_batch(q, n, A_d, n, stride_a, dipiv, n, Ainv_d, n, stride_a, batch, getri_scratchpad, getri_scratchpad_size, {getrf_event});
	    flops += batch * FLOPS_CGETRI(n);
	    getri_event.wait_and_throw();
	    pool_device_free(getrf_scratchpad);
	    pool_device_free(getri_scratchpad);
	  } catch(oneapi::mkl::lapack::exception const& e) {
	    // Handle LAPACK related exceptions happened during synchronous call
	    errorQuda("Unexpected exception caught during synchronous call to LAPACK API:\nreason: %s\ninfo: %ld", e.what(), e.info());
	  } catch(sycl::exception const& e) {
	    // Handle not LAPACK related exceptions happened during synchronous call
	    errorQuda("Unexpected exception caught during synchronous call to SYCL API:\n %s", e.what());
	  }

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          C *Ainv_h = static_cast<C *>(pool_pinned_malloc(size));
          qudaMemcpy(Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
	    checkEigen<MatrixXcf, float>(A_h, Ainv_h, n, i);
	  }
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
#endif
	  if (location == QUDA_CPU_FIELD_LOCATION) {
	    qudaMemcpy(Ainv, Ainv_d, size, qudaMemcpyDeviceToHost);
	    pool_device_free(Ainv_d);
	    pool_device_free(A_d);
	  }
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          typedef std::complex<double> C;

	  C *A_d = static_cast<C *>(A);
	  C *Ainv_d = static_cast<C *>(Ainv);
	  if(location == QUDA_CPU_FIELD_LOCATION) {
	    A_d = static_cast<C *>(pool_device_malloc(size));
	    Ainv_d = static_cast<C *>(pool_device_malloc(size));
	    qudaMemcpy(A_d, A, size, qudaMemcpyHostToDevice);
	  }

#ifdef _DEBUG
	  // Debug code: Copy original A matrix to host
	  C *A_h = static_cast<C *>(pool_pinned_malloc(size));
	  qudaMemcpy(A_h, A_d, size, qudaMemcpyDeviceToHost);
#endif

	  try {
	    std::int64_t getrf_scratchpad_size = getrf_batch_scratchpad_size<C>(q, n, n, n, stride_a, n, batch);
	    C *getrf_scratchpad = static_cast<C *>(pool_device_malloc(getrf_scratchpad_size*sizeof(C)));
	    auto getrf_event = getrf_batch(q, n, n, A_d, n, stride_a, dipiv, n, batch, getrf_scratchpad, getrf_scratchpad_size);
	    flops += batch * FLOPS_CGETRF(n, n);

	    std::int64_t getri_scratchpad_size = getri_batch_scratchpad_size<C>(q, n, n, n, n, batch);
	    C *getri_scratchpad = static_cast<C *>(pool_device_malloc(getri_scratchpad_size*sizeof(C)));
	    auto getri_event = getri_batch(q, n, A_d, n, stride_a, dipiv, n, Ainv_d, n, stride_a, batch, getri_scratchpad, getri_scratchpad_size, {getrf_event});
	    flops += batch * FLOPS_CGETRI(n);
	    getri_event.wait_and_throw();
	    pool_device_free(getrf_scratchpad);
	    pool_device_free(getri_scratchpad);
	  } catch(oneapi::mkl::lapack::exception const& e) {
	    // Handle LAPACK related exceptions happened during synchronous call
	    errorQuda("Unexpected exception caught during synchronous call to LAPACK API:\nreason: %s\ninfo: %ld", e.what(), e.info());
	  } catch(sycl::exception const& e) {
	    // Handle not LAPACK related exceptions happened during synchronous call
	    errorQuda("Unexpected exception caught during synchronous call to SYCL API:\n %s", e.what());
	  }

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          C *Ainv_h = static_cast<C *>(pool_pinned_malloc(size));
          qudaMemcpy(Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
	    checkEigen<MatrixXcf, double>(A_h, Ainv_h, n, i);
	  }
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
#endif
	  if (location == QUDA_CPU_FIELD_LOCATION) {
	    qudaMemcpy(Ainv, Ainv_d, size, qudaMemcpyDeviceToHost);
	    pool_device_free(Ainv_d);
	    pool_device_free(A_d);
	  }
        } else {
          errorQuda("%s not implemented for precision=%d", __func__, prec);
        }

        pool_device_free(dipiv);

        //qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        long ds = stop.tv_sec - start.tv_sec;
        long dus = stop.tv_usec - start.tv_usec;
        double time = ds + 0.000001 * dus;

        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);

        return flops;
      }
#else
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch,
				  QudaPrecision prec, QudaFieldLocation location)
      {
	return quda::blas_lapack::generic::BatchInvertMatrix(Ainv, A, n, batch,
							     prec, location);
      }
#endif

#ifdef NATIVE_LAPACK_LIB
      long long stridedBatchGEMM(void *A_data, void *B_data, void *C_data,
				 QudaBLASParam blas_param, QudaFieldLocation location)
      {
	warningQuda("using mkl stridedBatchGEMM");
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

        transpose trans_a = transpose::N;
        switch (blas_param.trans_a) {
        case QUDA_BLAS_OP_N: trans_a = transpose::N; break;
        case QUDA_BLAS_OP_T: trans_a = transpose::T; break;
        case QUDA_BLAS_OP_C: trans_a = transpose::C; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_a);
        }

        transpose trans_b = transpose::N;
        switch (blas_param.trans_b) {
        case QUDA_BLAS_OP_N: trans_b = transpose::N; break;
        case QUDA_BLAS_OP_T: trans_b = transpose::T; break;
        case QUDA_BLAS_OP_C: trans_b = transpose::C; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_b);
        }
        //-------------------------------------------------------------------------

        // Call CUBLAS
        //-------------------------------------------------------------------------
        if (blas_param.data_type == QUDA_BLAS_DATATYPE_Z) {
          typedef std::complex<double> Z;

          const Z alpha(static_cast<std::complex<double>>(blas_param.alpha));
	  const Z beta(static_cast<std::complex<double>>(blas_param.beta));

	  auto q = device::defaultQueue();
	  sycl::event evnt;
          if (batch > 1) {
	    evnt = blas::column_major::gemm_batch
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (Z *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
	       (Z *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
	       (Z *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
	    evnt.wait();
          } else {
	    evnt = blas::column_major::gemm
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (Z *)A_d + blas_param.a_offset, blas_param.lda,
	       (Z *)B_d + blas_param.b_offset, blas_param.ldb, beta,
	       (Z *)C_d + blas_param.c_offset, blas_param.ldc);
	    evnt.wait();
	  }
	  flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
	} else if (blas_param.data_type == QUDA_BLAS_DATATYPE_C) {
          typedef std::complex<float> C;

          const C alpha(static_cast<std::complex<double>>(blas_param.alpha));
	  const C beta(static_cast<std::complex<double>>(blas_param.beta));

	  auto q = device::defaultQueue();
	  sycl::event evnt;
          if (batch > 1) {
	    evnt = blas::column_major::gemm_batch
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (C *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
	       (C *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
	       (C *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
	    evnt.wait();
          } else {
	    evnt = blas::column_major::gemm
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (C *)A_d + blas_param.a_offset, blas_param.lda,
	       (C *)B_d + blas_param.b_offset, blas_param.ldb, beta,
	       (C *)C_d + blas_param.c_offset, blas_param.ldc);
	    evnt.wait();
          }
	  flops += batch * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_D) {
          typedef double D;

          const D alpha = (D) static_cast<std::complex<double>>(blas_param.alpha).real();
          const D beta = (D) static_cast<std::complex<double>>(blas_param.beta).real();

	  auto q = device::defaultQueue();
	  sycl::event evnt;
          if (batch > 1) {
	    evnt = blas::column_major::gemm_batch
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (D *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
	       (D *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
	       (D *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
	    evnt.wait();
          } else {
	    evnt = blas::column_major::gemm
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (D *)A_d + blas_param.a_offset, blas_param.lda,
	       (D *)B_d + blas_param.b_offset, blas_param.ldb, beta,
	       (D *)C_d + blas_param.c_offset, blas_param.ldc);
	    evnt.wait();
          }
	  flops += batch * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
        } else if (blas_param.data_type == QUDA_BLAS_DATATYPE_S) {
          typedef float S;

          const S alpha = (S) static_cast<std::complex<float>>(blas_param.alpha).real();
          const S beta = (S) static_cast<std::complex<float>>(blas_param.beta).real();

	  auto q = device::defaultQueue();
	  sycl::event evnt;
          if (batch > 1) {
	    evnt = blas::column_major::gemm_batch
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (S *)A_d + blas_param.a_offset, blas_param.lda, a_stride,
	       (S *)B_d + blas_param.b_offset, blas_param.ldb, b_stride, beta,
	       (S *)C_d + blas_param.c_offset, blas_param.ldc, c_stride, batch);
	    evnt.wait();
          } else {
	    evnt = blas::column_major::gemm
	      (q, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, alpha,
	       (S *)A_d + blas_param.a_offset, blas_param.lda,
	       (S *)B_d + blas_param.b_offset, blas_param.ldb, beta,
	       (S *)C_d + blas_param.c_offset, blas_param.ldc);
	    evnt.wait();
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
        //if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
        //-------------------------------------------------------------------------

        return flops;
      }
#else
      long long stridedBatchGEMM(void *A_data, void *B_data, void *C_data,
				 QudaBLASParam blas_param, QudaFieldLocation location)
      {
	warningQuda("using generic stridedBatchGEMM");
	return quda::blas_lapack::generic::stridedBatchGEMM(A_data, B_data, C_data,
							    blas_param, location);
      }
#endif

    } // namespace native
  }   // namespace blas_lapack
} // namespace quda
