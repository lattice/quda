#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>
#include <inttypes.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include "misc.h"

using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

void fillEigenArrayColMaj(MatrixXcd &EigenArr, complex<double> *arr, int rows, int cols, int ld, int offset)
{
  int counter = offset;
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      EigenArr(i, j) = arr[counter];
      counter++;
    }
    counter += (ld - rows);
  }
}

void fillEigenArrayRowMaj(MatrixXcd &EigenArr, complex<double> *arr, int rows, int cols, int ld, int offset)
{
  int counter = offset;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      EigenArr(i, j) = arr[counter];
      counter++;
    }
    counter += (ld - cols);
  }
}

void blasGEMMEigenVerify(void *arrayA, void *arrayB, void *arrayCcopy, void *arrayC, uint64_t refA_size,
                         uint64_t refB_size, uint64_t refC_size, QudaBLASParam *blas_param)
{

  // Problem parameters
  int m = blas_param->m;
  int n = blas_param->n;
  int k = blas_param->k;

  int lda = blas_param->lda;
  int ldb = blas_param->ldb;
  int ldc = blas_param->ldc;

  int a_stride = blas_param->strideA;
  int b_stride = blas_param->strideB;
  int c_stride = blas_param->strideC;

  int a_offset = blas_param->a_offset;
  int b_offset = blas_param->b_offset;
  int c_offset = blas_param->c_offset;

  int batches = blas_param->batch_count;

  complex<double> alpha = blas_param->alpha;
  complex<double> beta = blas_param->beta;

  // Eigen objects to store data
  MatrixXcd A = MatrixXd::Zero(m, k);
  MatrixXcd B = MatrixXd::Zero(k, n);
  MatrixXcd C_eigen = MatrixXd::Zero(m, n);
  MatrixXcd C_gpu = MatrixXd::Zero(m, n);
  MatrixXcd C_resid = MatrixXd::Zero(m, n);

  // Pointers to data
  complex<double> *A_ptr = (complex<double> *)(&arrayA)[0];
  complex<double> *B_ptr = (complex<double> *)(&arrayB)[0];
  complex<double> *C_ptr = (complex<double> *)(&arrayC)[0];
  complex<double> *Ccopy_ptr = (complex<double> *)(&arrayCcopy)[0];

  // Sanity checks on parameters
  //-------------------------------------------------------------------------
  // If the user passes non positive M,N, or K, we error out
  int min_dim = std::min(m, std::min(n, k));
  if (min_dim <= 0) { errorQuda("BLAS dims must be positive: m=%d, n=%d, k=%d", m, n, k); }

  // If the user passes a negative stride, we error out as this has no meaning.
  int min_stride = std::min(std::min(a_stride, b_stride), c_stride);
  if (min_stride < 0) {
    errorQuda("BLAS strides must be positive or zero: strideA=%d, strideB=%d, strideC=%d", a_stride, b_stride, c_stride);
  }

  // If the user passes a negative offset, we error out as this has no meaning.
  int min_offset = std::min(std::min(a_offset, b_offset), c_offset);
  if (min_offset < 0) {
    errorQuda("BLAS offsets must be positive or zero: a_offset=%d, b_offset=%d, c_offset=%d", a_offset, b_offset,
              c_offset);
  }

  // If the batch value is non-positve, we error out
  if (batches <= 0) { errorQuda("Batches must be positive: batches=%d", batches); }

  // Leading dims are dependendent on the matrix op type.
  if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
    if (blas_param->trans_a == QUDA_BLAS_OP_N) {
      if (lda < std::max(1, m)) errorQuda("lda=%d must be >= max(1,m=%d)", lda, m);
    } else {
      if (lda < std::max(1, k)) errorQuda("lda=%d must be >= max(1,k=%d)", lda, k);
    }
    if (blas_param->trans_b == QUDA_BLAS_OP_N) {
      if (ldb < std::max(1, k)) errorQuda("ldb=%d must be >= max(1,k=%d)", ldb, k);
    } else {
      if (ldb < std::max(1, n)) errorQuda("ldb=%d must be >= max(1,n=%d)", ldb, n);
    }
    if (ldc < std::max(1, m)) errorQuda("ldc=%d must be >= max(1,m=%d)", ldc, m);
  } else {
    if (blas_param->trans_a == QUDA_BLAS_OP_N) {
      if (lda < std::max(1, k)) errorQuda("lda=%d must be >= max(1,k=%d)", lda, k);
    } else {
      if (lda < std::max(1, m)) errorQuda("lda=%d must be >= max(1,m=%d)", lda, m);
    }
    if (blas_param->trans_b == QUDA_BLAS_OP_N) {
      if (ldb < std::max(1, n)) errorQuda("ldb=%d must be >= max(1,n=%d)", ldb, n);
    } else {
      if (ldb < std::max(1, k)) errorQuda("ldb=%d must be >= max(1,k=%d)", ldb, k);
    }
    if (ldc < std::max(1, n)) errorQuda("ldc=%d must be >= max(1,n=%d)", ldc, n);
  }
  //-------------------------------------------------------------------------

  // Get maximum stride length to deduce the number of batches in the
  // computation
  int max_stride = std::max(std::max(a_stride, b_stride), c_stride);

  // If the user gives strides of 0 for all arrays, we are essentially performing
  // a GEMM on the first matrices in the array N_{batch} times.
  // Give them what they ask for, YMMV...
  // If the strides have not been set, we are just using strides of 1.
  if (max_stride <= 0) max_stride = 1;

  printfQuda("Computing Eigen matrix opertaion a * A_{%lu,%lu} * B_{%lu,%lu} + b * C_{%lu,%lu} = C_{%lu,%lu}\n",
             A.rows(), A.cols(), B.rows(), B.cols(), C_eigen.rows(), C_eigen.cols(), C_eigen.rows(), C_eigen.cols());

  for (int batch = 0; batch < batches; batch += max_stride) {

    // Populate Eigen objects
    if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
      fillEigenArrayColMaj(A, A_ptr, m, k, lda, a_offset);
      fillEigenArrayColMaj(B, B_ptr, k, n, ldb, b_offset);
      fillEigenArrayColMaj(C_eigen, Ccopy_ptr, m, n, ldc, c_offset);
      fillEigenArrayColMaj(C_gpu, C_ptr, m, n, ldc, c_offset);
    } else {
      fillEigenArrayRowMaj(A, A_ptr, m, k, lda, a_offset);
      fillEigenArrayRowMaj(B, B_ptr, k, n, ldb, b_offset);
      fillEigenArrayRowMaj(C_eigen, Ccopy_ptr, m, n, ldc, c_offset);
      fillEigenArrayRowMaj(C_gpu, C_ptr, m, n, ldc, c_offset);
    }

    // Apply op(A) and op(B)
    switch (blas_param->trans_a) {
    case QUDA_BLAS_OP_T: A.transposeInPlace(); break;
    case QUDA_BLAS_OP_C: A.adjointInPlace(); break;
    case QUDA_BLAS_OP_N: break;
    default: errorQuda("Unknown blas op type %d", blas_param->trans_a);
    }

    switch (blas_param->trans_b) {
    case QUDA_BLAS_OP_T: B.transposeInPlace(); break;
    case QUDA_BLAS_OP_C: B.adjointInPlace(); break;
    case QUDA_BLAS_OP_N: break;
    default: errorQuda("Unknown blas op type %d", blas_param->trans_b);
    }

    // Perform GEMM using Eigen
    C_eigen = alpha * A * B + beta * C_eigen;

    // Check Eigen result against blas
    C_resid = C_gpu - C_eigen;

    printfQuda("batch %d: (C_host - C_gpu) Frobenius norm = %e. Relative deviation = %e\n", batch, C_resid.norm(),
               C_resid.norm() / (C_resid.rows() * C_resid.cols()));

    a_offset += refA_size * a_stride;
    b_offset += refB_size * b_stride;
    c_offset += refC_size * c_stride;
  }
}

void blasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayC, void *arrayCcopy, uint64_t refA_size,
                        uint64_t refB_size, uint64_t refC_size, int re_im, size_t data_size, QudaBLASParam *blas_param)
{

  int batches = blas_param->batch_count;
  // Copy data from problem sized array to reference sized array.
  // Include A and B to ensure no data corruption occurred
  void *checkA = pinned_malloc(refA_size * re_im * data_size * batches);
  void *checkB = pinned_malloc(refB_size * re_im * data_size * batches);
  void *checkC = pinned_malloc(refC_size * re_im * data_size * batches);
  void *checkCcopy = pinned_malloc(refC_size * re_im * data_size * batches);

  memset(checkA, 0, batches * refA_size * re_im * data_size);
  memset(checkB, 0, batches * refB_size * re_im * data_size);
  memset(checkC, 0, batches * refC_size * re_im * data_size);
  memset(checkCcopy, 0, batches * refC_size * re_im * data_size);

  switch (blas_param->data_type) {
  case QUDA_BLAS_DATATYPE_S:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)checkA)[i] = ((float *)arrayA)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)checkB)[i] = ((float *)arrayB)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((double *)checkC)[i] = ((float *)arrayC)[i / 2];
      ((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i / 2];
    }
    break;
  case QUDA_BLAS_DATATYPE_D:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)checkA)[i] = ((double *)arrayA)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)checkB)[i] = ((double *)arrayB)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((double *)checkC)[i] = ((double *)arrayC)[i / 2];
      ((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i / 2];
    }
    break;
  case QUDA_BLAS_DATATYPE_C:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((double *)checkA)[i] = ((float *)arrayA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((double *)checkB)[i] = ((float *)arrayB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((double *)checkC)[i] = ((float *)arrayC)[i];
      ((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i];
    }
    break;
  case QUDA_BLAS_DATATYPE_Z:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((double *)checkA)[i] = ((double *)arrayA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((double *)checkB)[i] = ((double *)arrayB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((double *)checkC)[i] = ((double *)arrayC)[i];
      ((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i];
    }
    break;
  default: errorQuda("Unrecognised data type %d\n", blas_param->data_type);
  }

  blasGEMMEigenVerify(checkA, checkB, checkCcopy, checkC, refA_size, refB_size, refC_size, blas_param);

  host_free(checkA);
  host_free(checkB);
  host_free(checkC);
  host_free(checkCcopy);
}
