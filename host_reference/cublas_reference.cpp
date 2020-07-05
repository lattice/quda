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

void cublasGEMMEigenVerify(void *arrayA, void *arrayB, void *arrayCcopy, void *arrayC, uint64_t refA_size,
                           uint64_t refB_size, uint64_t refC_size, QudaCublasParam *cublas_param)
{

  // Problem parameters
  int m = cublas_param->m;
  int n = cublas_param->n;
  int k = cublas_param->k;
  int lda = cublas_param->lda;
  int ldb = cublas_param->ldb;
  int ldc = cublas_param->ldc;
  int a_offset = cublas_param->a_offset;
  int b_offset = cublas_param->b_offset;
  int c_offset = cublas_param->c_offset;
  int batches = cublas_param->batch_count;
  complex<double> alpha = cublas_param->alpha;
  complex<double> beta = cublas_param->beta;

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

  printfQuda("Computing Eigen matrix opertaion a * A_{%lu,%lu} * B_{%lu,%lu} + b * C_{%lu,%lu} = C_{%lu,%lu}\n",
             A.rows(), A.cols(), B.rows(), B.cols(), C_eigen.rows(), C_eigen.cols(), C_eigen.rows(), C_eigen.cols());

  for (int batch = 0; batch < batches; batch++) {

    // Populate Eigen objects
    if (cublas_param->data_order == QUDA_CUBLAS_DATAORDER_COL) {
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
    switch (cublas_param->trans_a) {
    case QUDA_CUBLAS_OP_T: A.transposeInPlace(); break;
    case QUDA_CUBLAS_OP_C: A.adjointInPlace(); break;
    case QUDA_CUBLAS_OP_N: break;
    default: errorQuda("Unknown cuBLAS op type %d", cublas_param->trans_a);
    }

    switch (cublas_param->trans_b) {
    case QUDA_CUBLAS_OP_T: B.transposeInPlace(); break;
    case QUDA_CUBLAS_OP_C: B.adjointInPlace(); break;
    case QUDA_CUBLAS_OP_N: break;
    default: errorQuda("Unknown cuBLAS op type %d", cublas_param->trans_b);
    }

    // Perform GEMM using Eigen
    C_eigen = alpha * A * B + beta * C_eigen;

    // Check Eigen result against cuBLAS
    C_resid = C_gpu - C_eigen;

    printfQuda("batch %d: (C_host - C_gpu) Frobenius norm = %e. Relative deviation = %e\n", batch, C_resid.norm(),
               C_resid.norm() / (C_resid.rows() * C_resid.cols()));

    a_offset += refA_size;
    b_offset += refB_size;
    c_offset += refC_size;
  }
}

void cublasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayC, void *arrayCcopy, uint64_t refA_size,
                          uint64_t refB_size, uint64_t refC_size, int re_im, size_t data_size,
                          QudaCublasParam *cublas_param)
{

  int batches = cublas_param->batch_count;
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

  switch (cublas_param->data_type) {
  case QUDA_CUBLAS_DATATYPE_S:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)checkA)[i] = ((float *)arrayA)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)checkB)[i] = ((float *)arrayB)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((double *)checkC)[i] = ((float *)arrayC)[i / 2];
      ((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i / 2];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_D:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)checkA)[i] = ((double *)arrayA)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)checkB)[i] = ((double *)arrayB)[i / 2]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((double *)checkC)[i] = ((double *)arrayC)[i / 2];
      ((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i / 2];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_C:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((double *)checkA)[i] = ((float *)arrayA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((double *)checkB)[i] = ((float *)arrayB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((double *)checkC)[i] = ((float *)arrayC)[i];
      ((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_Z:
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((double *)checkA)[i] = ((double *)arrayA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((double *)checkB)[i] = ((double *)arrayB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((double *)checkC)[i] = ((double *)arrayC)[i];
      ((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i];
    }
    break;
  default: errorQuda("Unrecognised data type %d\n", cublas_param->data_type);
  }

  cublasGEMMEigenVerify(checkA, checkB, checkCcopy, checkC, refA_size, refB_size, refC_size, cublas_param);

  host_free(checkA);
  host_free(checkB);
  host_free(checkC);
  host_free(checkCcopy);
}
