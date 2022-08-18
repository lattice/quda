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

#include <eigen_helper.h>

template <typename T> using complex = std::complex<T>;

void fillEigenArray(MatrixXcd &EigenArr, complex<double> *arr, int rows, int cols, int ld, int offset)
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

void prepare_ref_array(void *array, int batches, uint64_t array_size, size_t data_size, QudaBLASDataType data_type)
{
  memset(array, 0, batches * array_size * data_size);
  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * array_size * batches; i += 2) { ((double *)array)[i] = rand() / (double)RAND_MAX; }
  // Populate the imaginary part with rands if needed
  if (data_type == QUDA_BLAS_DATATYPE_C || data_type == QUDA_BLAS_DATATYPE_Z) {
    for (uint64_t i = 1; i < 2 * array_size * batches; i += 2) { ((double *)array)[i] = rand() / (double)RAND_MAX; }
  }
}

void copy_array(void *array_out, void *array_in, int batches, uint64_t array_size, size_t data_out_size,
                QudaBLASDataType data_type)
{
  // Copy the real part only
  if (data_type == QUDA_BLAS_DATATYPE_S || data_type == QUDA_BLAS_DATATYPE_D) {
    for (uint64_t i = 0; i < 2 * array_size * batches; i += 2) {
      if (data_out_size == sizeof(float))
        ((float *)array_out)[i / 2] = ((double *)array_in)[i];
      else if (data_out_size == sizeof(double))
        ((double *)array_out)[i / 2] = ((double *)array_in)[i];
      else
        errorQuda("Unsupported data out size %lu", data_out_size);
    }
  }
  // Copy both the real and the imaginary parts
  if (data_type == QUDA_BLAS_DATATYPE_C || data_type == QUDA_BLAS_DATATYPE_Z) {
    for (uint64_t i = 0; i < 2 * array_size * batches; i++) {
      if (data_out_size == sizeof(float))
        ((float *)array_out)[i] = ((double *)array_in)[i];
      else if (data_out_size == sizeof(double))
        ((double *)array_out)[i] = ((double *)array_in)[i];
      else
        errorQuda("Unsupported data out size %lu", data_out_size);
    }
  }
}

double blasGEMMEigenVerify(void *A_data, void *B_data, void *C_data_copy, void *C_data, uint64_t refA_size,
                           uint64_t refB_size, uint64_t refC_size, QudaBLASParam *blas_param)
{
  // Sanity checks on parameters
  //-------------------------------------------------------------------------
  // If the user passes non positive M,N, or K, we error out
  int min_dim = std::min(blas_param->m, std::min(blas_param->n, blas_param->k));
  if (min_dim <= 0) {
    errorQuda("BLAS dims must be positive: m=%d, n=%d, k=%d", blas_param->m, blas_param->n, blas_param->k);
  }

  // If the user passes a negative stride, we error out as this has no meaning.
  int min_stride = std::min(std::min(blas_param->a_stride, blas_param->b_stride), blas_param->c_stride);
  if (min_stride < 0) {
    errorQuda("BLAS strides must be positive or zero: a_stride=%d, b_stride=%d, c_stride=%d", blas_param->a_stride,
              blas_param->b_stride, blas_param->c_stride);
  }

  // If the user passes a negative offset, we error out as this has no meaning.
  int min_offset = std::min(std::min(blas_param->a_offset, blas_param->b_offset), blas_param->c_offset);
  if (min_offset < 0) {
    errorQuda("BLAS offsets must be positive or zero: a_offset=%d, b_offset=%d, c_offset=%d", blas_param->a_offset,
              blas_param->b_offset, blas_param->c_offset);
  }

  // If the batch value is non-positve, we error out
  if (blas_param->batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param->batch_count); }

  // Leading dims are dependendent on the matrix op type.
  if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
    if (blas_param->trans_a == QUDA_BLAS_OP_N) {
      if (blas_param->lda < std::max(1, blas_param->m))
        errorQuda("lda=%d must be >= max(1,m=%d)", blas_param->lda, blas_param->m);
    } else {
      if (blas_param->lda < std::max(1, blas_param->k))
        errorQuda("lda=%d must be >= max(1,k=%d)", blas_param->lda, blas_param->k);
    }

    if (blas_param->trans_b == QUDA_BLAS_OP_N) {
      if (blas_param->ldb < std::max(1, blas_param->k))
        errorQuda("ldb=%d must be >= max(1,k=%d)", blas_param->ldb, blas_param->k);
    } else {
      if (blas_param->ldb < std::max(1, blas_param->n))
        errorQuda("ldb=%d must be >= max(1,n=%d)", blas_param->ldb, blas_param->n);
    }
    if (blas_param->ldc < std::max(1, blas_param->m))
      errorQuda("ldc=%d must be >= max(1,m=%d)", blas_param->ldc, blas_param->m);
  } else {
    if (blas_param->trans_a == QUDA_BLAS_OP_N) {
      if (blas_param->lda < std::max(1, blas_param->k))
        errorQuda("lda=%d must be >= max(1,k=%d)", blas_param->lda, blas_param->k);
    } else {
      if (blas_param->lda < std::max(1, blas_param->m))
        errorQuda("lda=%d must be >= max(1,m=%d)", blas_param->lda, blas_param->m);
    }
    if (blas_param->trans_b == QUDA_BLAS_OP_N) {
      if (blas_param->ldb < std::max(1, blas_param->n))
        errorQuda("ldb=%d must be >= max(1,n=%d)", blas_param->ldb, blas_param->n);
    } else {
      if (blas_param->ldb < std::max(1, blas_param->k))
        errorQuda("ldb=%d must be >= max(1,k=%d)", blas_param->ldb, blas_param->k);
    }
    if (blas_param->ldc < std::max(1, blas_param->n))
      errorQuda("ldc=%d must be >= max(1,n=%d)", blas_param->ldc, blas_param->n);
  }
  //-------------------------------------------------------------------------

  // Parse parameters for Eigen
  //-------------------------------------------------------------------------
  // Swap A and B if in column order
  if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
    std::swap(blas_param->m, blas_param->n);
    std::swap(blas_param->lda, blas_param->ldb);
    std::swap(blas_param->trans_a, blas_param->trans_b);
    std::swap(blas_param->a_offset, blas_param->b_offset);
    std::swap(blas_param->a_stride, blas_param->b_stride);
    std::swap(A_data, B_data);
  }

  // Problem parameters
  int m = blas_param->m;
  int n = blas_param->n;
  int k = blas_param->k;

  int lda = blas_param->lda;
  int ldb = blas_param->ldb;
  int ldc = blas_param->ldc;

  int a_stride = blas_param->a_stride;
  int b_stride = blas_param->b_stride;
  int c_stride = blas_param->c_stride;

  int a_offset = blas_param->a_offset;
  int b_offset = blas_param->b_offset;
  int c_offset = blas_param->c_offset;

  int batches = blas_param->batch_count;

  complex<double> alpha = blas_param->alpha;
  complex<double> beta = blas_param->beta;
  if (blas_param->data_type == QUDA_BLAS_DATATYPE_S || blas_param->data_type == QUDA_BLAS_DATATYPE_D) {
    alpha.imag(0.0);
    beta.imag(0.0);
  }

  // Eigen objects to store data
  MatrixXcd A = MatrixXd::Zero(m, k);
  MatrixXcd B = MatrixXd::Zero(k, n);
  MatrixXcd C_eigen = MatrixXd::Zero(m, n);
  MatrixXcd C_dev = MatrixXd::Zero(m, n);
  MatrixXcd C_resid = MatrixXd::Zero(m, n);

  // Pointers to data
  complex<double> *A_ptr = (complex<double> *)(&A_data)[0];
  complex<double> *B_ptr = (complex<double> *)(&B_data)[0];
  complex<double> *C_ptr = (complex<double> *)(&C_data)[0];
  complex<double> *Ccopy_ptr = (complex<double> *)(&C_data_copy)[0];

  // Get maximum stride length to deduce the number of batches in the
  // computation
  int max_stride = std::max(std::max(a_stride, b_stride), c_stride);

  // If the user gives strides of 0 for all arrays, we are essentially performing
  // a GEMM on the first matrices in the array N_{batch} times.
  // Give them what they ask for, YMMV...
  // If the strides have not been set, we are just using strides of 1.
  if (max_stride <= 0) max_stride = 1;

  printfQuda("Computing Eigen matrix operation a * A_{%lu,%lu} * B_{%lu,%lu} + b * C_{%lu,%lu} = C_{%lu,%lu}\n",
             A.rows(), A.cols(), B.rows(), B.cols(), C_eigen.rows(), C_eigen.cols(), C_eigen.rows(), C_eigen.cols());

  double max_relative_deviation = 0.0;
  for (int batch = 0; batch < batches; batch += max_stride) {

    // Populate Eigen objects
    fillEigenArray(A, A_ptr, m, k, lda, a_offset);
    fillEigenArray(B, B_ptr, k, n, ldb, b_offset);
    fillEigenArray(C_eigen, Ccopy_ptr, m, n, ldc, c_offset);
    fillEigenArray(C_dev, C_ptr, m, n, ldc, c_offset);

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
    C_resid = C_dev - C_eigen;
    double deviation = C_resid.norm();
    double relative_deviation = deviation / C_eigen.norm();
    max_relative_deviation = std::max(max_relative_deviation, relative_deviation);

    printfQuda("batch %d: (C_host - C_dev) Frobenius norm = %e. Relative deviation = %e\n", batch, deviation,
               relative_deviation);

    a_offset += refA_size * a_stride;
    b_offset += refB_size * b_stride;
    c_offset += refC_size * c_stride;
  }

  // Restore the blas parameters to their original values
  if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
    std::swap(blas_param->m, blas_param->n);
    std::swap(blas_param->lda, blas_param->ldb);
    std::swap(blas_param->trans_a, blas_param->trans_b);
    std::swap(blas_param->a_offset, blas_param->b_offset);
    std::swap(blas_param->a_stride, blas_param->b_stride);
    std::swap(A_data, B_data);
  }

  return max_relative_deviation;
}

double blasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayC, void *arrayCcopy, uint64_t refA_size,
                          uint64_t refB_size, uint64_t refC_size, QudaBLASParam *blas_param)
{
  // Reference data is always in complex double
  size_t data_size = sizeof(double);
  int re_im = 2;
  data_size *= re_im;

  // If the user passes non-zero offsets, add one extra
  // matrix to the test data.
  int batches_extra = 0;
  if (blas_param->a_offset + blas_param->b_offset + blas_param->c_offset > 0) { batches_extra++; }
  int batches = blas_param->batch_count + batches_extra;

  // Copy data from problem sized array to reference sized array.
  // Include A and B to ensure no data corruption occurred
  void *checkA = pinned_malloc(refA_size * data_size * batches);
  void *checkB = pinned_malloc(refB_size * data_size * batches);
  void *checkC = pinned_malloc(refC_size * data_size * batches);
  void *checkCcopy = pinned_malloc(refC_size * data_size * batches);

  memset(checkA, 0, batches * refA_size * data_size);
  memset(checkB, 0, batches * refB_size * data_size);
  memset(checkC, 0, batches * refC_size * data_size);
  memset(checkCcopy, 0, batches * refC_size * data_size);

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

  auto deviation = blasGEMMEigenVerify(checkA, checkB, checkCcopy, checkC, refA_size, refB_size, refC_size, blas_param);

  host_free(checkA);
  host_free(checkB);
  host_free(checkC);
  host_free(checkCcopy);

  return deviation;
}

double blasLUInvEigenVerify(void *ref_array, void *dev_inv_array, uint64_t array_size, QudaBLASParam *blas_param)
{

  // Sanity checks on parameters
  //-------------------------------------------------------------------------
  // If the batch value is non-positve, we error out
  if (blas_param->batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param->batch_count); }
  //-------------------------------------------------------------------------

  // Parse parameters for Eigen
  //-------------------------------------------------------------------------
  // Problem parameters
  int offset = 0;
  int batches = blas_param->batch_count;
  int mat_rank = blas_param->inv_mat_size;

  // Eigen objects to store data
  MatrixXcd ref = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd ref_inv = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd dev_inv = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd inv_resid = MatrixXd::Zero(mat_rank, mat_rank);

  // Pointers to data
  complex<double> *ref_ptr = (complex<double> *)(&ref_array)[0];
  complex<double> *dev_inv_ptr = (complex<double> *)(&dev_inv_array)[0];

  printfQuda("Computing Eigen matrix operation ref_inv{%lu,%lu} = ref{%lu,%lu}^(-1)\n", ref.rows(), ref.cols(),
             ref_inv.rows(), ref_inv.cols());

  double max_relative_deviation = 0.0;
  for (int batch = 0; batch < batches; batch++) {

    // Populate Eigen objects
    fillEigenArray(ref, ref_ptr, mat_rank, mat_rank, mat_rank, offset);
    fillEigenArray(dev_inv, dev_inv_ptr, mat_rank, mat_rank, mat_rank, offset);

    // Check Eigen result against blas
    ref_inv = ref.inverse();
    inv_resid = dev_inv - ref_inv;

    double deviation = inv_resid.norm();
    double relative_deviation = deviation / ref_inv.norm();
    max_relative_deviation = std::max(max_relative_deviation, relative_deviation);

    printfQuda("batch %d: (ref_inv - dev_inv) Frobenius norm = %e. Relative deviation = %e\n", batch, deviation,
               relative_deviation);

    offset += array_size;
  }

  return max_relative_deviation;
}

double blasLUInvQudaVerify(void *ref_array, void *dev_array_inv, uint64_t array_size, QudaBLASParam *blas_param)
{
  // Reference data is always in complex double, but real data may
  // be supported in the future.
  size_t data_out_size = sizeof(double);
  int re_im = 2;

  int batches = blas_param->batch_count;

  // Copy data from problem sized array to reference sized array.
  void *dev_array_inv_copy = pinned_malloc(array_size * re_im * data_out_size * batches);

  size_t data_in_size = 0;
  switch (blas_param->data_type) {
  case QUDA_BLAS_DATATYPE_C: data_in_size = sizeof(float); break;
  case QUDA_BLAS_DATATYPE_Z: data_in_size = sizeof(double); break;
  case QUDA_BLAS_DATATYPE_S:
  case QUDA_BLAS_DATATYPE_D:
  default: errorQuda("Unsupported data type %d\n", blas_param->data_type);
  }

  for (uint64_t i = 0; i < 2 * array_size * batches; i++) {
    if (data_in_size == sizeof(float))
      ((double *)dev_array_inv_copy)[i] = ((float *)dev_array_inv)[i];
    else if (data_in_size == sizeof(double))
      ((double *)dev_array_inv_copy)[i] = ((double *)dev_array_inv)[i];
    else
      errorQuda("Unsupported data-in size %lu", data_in_size);
  }

  auto deviation = blasLUInvEigenVerify(ref_array, dev_array_inv_copy, array_size, blas_param);

  host_free(dev_array_inv_copy);
  return deviation;
}
