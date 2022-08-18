#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <tune_quda.h>

using namespace quda;

// Forward declarations for profiling and parameter checking
// The helper functions are defined in interface_quda.cpp
TimeProfile &getProfileBLAS();
void checkBLASParam(QudaBLASParam &param);

void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean use_native, QudaBLASParam *blas_param)
{
  getProfileBLAS().TPSTART(QUDA_PROFILE_TOTAL);
  checkBLASParam(*blas_param);

  // cuBLAS works exclusively in column major order. If the input data is in
  // row major order, we may treat the A and B and C arrays as A^T, B^T, and C^T.
  // We swap the order of the A * B multiplication and swap the
  // operation types and other data to recover the the desired result in the
  // desired order.
  // E.g: in row major, the operation,
  // C = a * A^T * B + b * C
  //
  // will become the column major operation
  // C^T = a * B^T * A + b * C^T
  //
  // By inspection, one can see that transposition of the above column major
  // operation will result in the desired row major answer:
  //
  // (C^T)^T = a * (B^T * A)^T + b * (C^T)^T
  //  -->  C = a *  A^T * B    + b *  C
  //
  // We must also swap around some parameters. The Row major indices,
  // A_{m, lda}, B_{k, ldb}, C_{m, ldc}
  // become
  // A^T_{lda, m}, B^T_{ldb, k}, C^T_{ldc, m}.
  // so the leading dimensions remain the same. However, we must change the actual
  // matrix dims m,n,k to reflect the change to column major.
  // m_{col} = n_{row}
  // n_{col} = m_{row}
  // k_{col} = k_{row}
  // And because we are swapping the A and B arrays, we must also swap their
  // leading dim values and any offsets. All this is done behind the scenes in the
  // BatchGEMM function, and before function exit all pointers and values are
  // restored to the values they had on entry.

  if (use_native == QUDA_BOOLEAN_FALSE) {
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
    blas_lapack::generic::stridedBatchGEMM(arrayA, arrayB, arrayC, *blas_param, QUDA_CPU_FIELD_LOCATION);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  } else {
    getProfileBLAS().TPSTART(QUDA_PROFILE_INIT);

    // The data in the arrays is on the host. We transfer the data to the device here
    // for timing purposes. One can pass host pointers to the BatchGEMM function
    // and it will handle the data movement for the user.

    // Extract data from the param struct for device malloc
    uint64_t arrayA_size = 0, arrayB_size = 0, arrayC_size = 0;
    if (blas_param->data_order == QUDA_BLAS_DATAORDER_COL) {
      // leading dimension is in terms of consecutive data
      // elements in a column, multiplied by number of rows
      if (blas_param->trans_a == QUDA_BLAS_OP_N) {
        arrayA_size = blas_param->lda * blas_param->k; // A_mk
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array A_{%d, %d}\n", blas_param->lda, blas_param->k);
      } else {
        arrayA_size = blas_param->lda * blas_param->m; // A_km
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array A_{%d, %d}\n", blas_param->lda, blas_param->m);
      }

      if (blas_param->trans_b == QUDA_BLAS_OP_N) {
        arrayB_size = blas_param->ldb * blas_param->n; // B_kn
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array B_{%d, %d}\n", blas_param->ldb, blas_param->n);
      } else {
        arrayB_size = blas_param->ldb * blas_param->k; // B_nk
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array B_{%d, %d}\n", blas_param->ldb, blas_param->k);
      }
      arrayC_size = blas_param->ldc * blas_param->n; // C_mn
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array C_{%d, %d}\n", blas_param->ldc, blas_param->n);
    } else {
      // leading dimension is in terms of consecutive data
      // elements in a row, multiplied by number of columns.
      if (blas_param->trans_a == QUDA_BLAS_OP_N) {
        arrayA_size = blas_param->lda * blas_param->m; // A_mk
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array A_{%d, %d}\n", blas_param->m, blas_param->lda);
      } else {
        arrayA_size = blas_param->lda * blas_param->k; // A_km
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array A_{%d, %d}\n", blas_param->k, blas_param->lda);
      }
      if (blas_param->trans_b == QUDA_BLAS_OP_N) {
        arrayB_size = blas_param->ldb * blas_param->k; // B_nk
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array B_{%d, %d}\n", blas_param->k, blas_param->ldb);
      } else {
        arrayB_size = blas_param->ldb * blas_param->n; // B_kn
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array B_{%d, %d}\n", blas_param->n, blas_param->ldb);
      }
      arrayC_size = blas_param->ldc * blas_param->m; // C_mn
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("array C_{%d, %d}\n", blas_param->m, blas_param->ldc);
    }

    size_t data_size = (blas_param->data_type == QUDA_BLAS_DATATYPE_D || blas_param->data_type == QUDA_BLAS_DATATYPE_Z) ?
      sizeof(double) :
      sizeof(float);
    int re_im = 1;
    if (blas_param->data_type == QUDA_BLAS_DATATYPE_C || blas_param->data_type == QUDA_BLAS_DATATYPE_Z) { re_im *= 2; }

    // If the user passes non-zero offsets, add one extra
    // matrix to the device array to accomodate it.
    int batches_extra = 0;
    if (blas_param->a_offset + blas_param->b_offset + blas_param->c_offset > 0) { batches_extra++; }
    int batches = blas_param->batch_count + batches_extra;

    size_t A_bytes = batches * arrayA_size * re_im * data_size;
    size_t B_bytes = batches * arrayB_size * re_im * data_size;
    size_t C_bytes = batches * arrayC_size * re_im * data_size;
    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("A_Gbtyes = %f, B_Gbtyes = %f, C_Gbtyes = %f\n", 1.0 * A_bytes / std::pow(1024, 3),
                 1.0 * B_bytes / std::pow(1024, 3), 1.0 * C_bytes / std::pow(1024, 3));
    void *A_d = pool_device_malloc(A_bytes);
    void *B_d = pool_device_malloc(B_bytes);
    void *C_d = pool_device_malloc(C_bytes);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("QUDA: arrays allocated successfully.\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_INIT);

    // Transfer host data to device
    getProfileBLAS().TPSTART(QUDA_PROFILE_H2D);
    qudaMemcpy(A_d, arrayA, A_bytes, qudaMemcpyHostToDevice);
    qudaMemcpy(B_d, arrayB, B_bytes, qudaMemcpyHostToDevice);
    qudaMemcpy(C_d, arrayC, C_bytes, qudaMemcpyHostToDevice);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("QUDA: arrays copied successfully.\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_H2D);

    // Compute Batched GEMM
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);

    blas_lapack::native::stridedBatchGEMM(A_d, B_d, C_d, *blas_param, QUDA_CUDA_FIELD_LOCATION);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("BatchGEMM success!\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);

    // Copy device C array back to host
    getProfileBLAS().TPSTART(QUDA_PROFILE_D2H);
    qudaMemcpy(arrayC, C_d, C_bytes, qudaMemcpyDeviceToHost);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_D2H);

    // Clean up
    getProfileBLAS().TPSTART(QUDA_PROFILE_FREE);
    pool_device_free(A_d);
    pool_device_free(B_d);
    pool_device_free(C_d);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_FREE);
  }

  getProfileBLAS().TPSTOP(QUDA_PROFILE_TOTAL);
  saveTuneCache();
}

void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *blas_param)
{
  getProfileBLAS().TPSTART(QUDA_PROFILE_TOTAL);
  checkBLASParam(*blas_param);

  getProfileBLAS().TPSTART(QUDA_PROFILE_INIT);
  const int n = blas_param->inv_mat_size;
  const uint64_t batches = blas_param->batch_count;
  QudaPrecision prec = QUDA_INVALID_PRECISION;
  switch (blas_param->data_type) {
  case QUDA_BLAS_DATATYPE_Z: prec = QUDA_DOUBLE_PRECISION; break;
  case QUDA_BLAS_DATATYPE_C: prec = QUDA_SINGLE_PRECISION; break;
  case QUDA_BLAS_DATATYPE_D:
  case QUDA_BLAS_DATATYPE_S:
  default: errorQuda("LU inversion not supported for data type %d", blas_param->data_type);
  }
  getProfileBLAS().TPSTOP(QUDA_PROFILE_INIT);

  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
  if (use_native == QUDA_BOOLEAN_FALSE)
    blas_lapack::generic::BatchInvertMatrix(Ainv, A, n, batches, prec, QUDA_CPU_FIELD_LOCATION);
  else
    blas_lapack::native::BatchInvertMatrix(Ainv, A, n, batches, prec, QUDA_CPU_FIELD_LOCATION);

  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  getProfileBLAS().TPSTOP(QUDA_PROFILE_TOTAL);
  saveTuneCache();
}
