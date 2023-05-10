#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>

#include <inttypes.h>

#include <test.h>
#include <blas_reference.h>
#include <misc.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// if "--enable-testing true" is passed, we run the tests defined in here
#include <blas_interface_test_gtest.hpp>

QudaBLASDataType blas_data_type = QUDA_BLAS_DATATYPE_C;
QudaBLASDataOrder blas_data_order = QUDA_BLAS_DATAORDER_COL;
QudaBLASType blas_test_type = QUDA_BLAS_GEMM;
int blas_batch = 16;

QudaBLASOperation blas_gemm_trans_a = QUDA_BLAS_OP_N;
QudaBLASOperation blas_gemm_trans_b = QUDA_BLAS_OP_N;
std::array<int, 3> blas_gemm_mnk = {64, 64, 64};
std::array<int, 3> blas_gemm_leading_dims = {128, 128, 128};
std::array<int, 3> blas_gemm_offsets = {0, 0, 0};
std::array<int, 3> blas_gemm_strides = {1, 1, 1};
std::array<double, 2> blas_gemm_alpha_re_im = {M_PI, M_E};
std::array<double, 2> blas_gemm_beta_re_im = {M_LN2, M_LN10};

int blas_lu_inv_mat_size = 128;

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info(QudaBLASParam &param)
{
  printfQuda("running the following test:\n");
  printfQuda("BLAS interface %s test\n", get_blas_type_str(param.blas_type));
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void setBLASParam(QudaBLASParam &blas_param)
{
  blas_param.trans_a = blas_gemm_trans_a;
  blas_param.trans_b = blas_gemm_trans_b;
  blas_param.m = blas_gemm_mnk[0];
  blas_param.n = blas_gemm_mnk[1];
  blas_param.k = blas_gemm_mnk[2];
  blas_param.lda = blas_gemm_leading_dims[0];
  blas_param.ldb = blas_gemm_leading_dims[1];
  blas_param.ldc = blas_gemm_leading_dims[2];
  blas_param.a_offset = blas_gemm_offsets[0];
  blas_param.b_offset = blas_gemm_offsets[1];
  blas_param.c_offset = blas_gemm_offsets[2];
  blas_param.a_stride = blas_gemm_strides[0];
  blas_param.b_stride = blas_gemm_strides[1];
  blas_param.c_stride = blas_gemm_strides[2];
  memcpy(&blas_param.alpha, blas_gemm_alpha_re_im.data(), sizeof(__complex__ double));
  memcpy(&blas_param.beta, blas_gemm_beta_re_im.data(), sizeof(__complex__ double));
  blas_param.data_order = blas_data_order;
  blas_param.data_type = blas_data_type;
  blas_param.batch_count = blas_batch;
  blas_param.blas_type = blas_test_type;
  blas_param.inv_mat_size = blas_lu_inv_mat_size;
}

double gemm_test(test_t test_param)
{
  QudaBLASParam blas_param = newQudaBLASParam();
  blas_data_type = ::testing::get<1>(test_param);
  blas_test_type = ::testing::get<0>(test_param);
  setBLASParam(blas_param);

  display_test_info(blas_param);

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

  // If the batch value is non-positve, we error out
  if (blas_param.batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param.batch_count); }
  //-------------------------------------------------------------------------

  // Reference data is always in complex double
  size_t data_in_size = sizeof(double);

  // If the user passes non-zero offsets, add one extra
  // matrix to the test data.
  int batches_extra = 0;
  if (blas_param.a_offset + blas_param.b_offset + blas_param.c_offset > 0) { batches_extra++; }
  int batches = blas_param.batch_count + batches_extra;
  uint64_t refA_size = 0, refB_size = 0, refC_size = 0;
  if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
    // leading dimension is in terms of consecutive data
    // elements in a column, multiplied by number of rows
    if (blas_param.trans_a == QUDA_BLAS_OP_N) {
      refA_size = blas_param.lda * blas_param.k; // A_mk
    } else {
      refA_size = blas_param.lda * blas_param.m; // A_km
    }

    if (blas_param.trans_b == QUDA_BLAS_OP_N) {
      refB_size = blas_param.ldb * blas_param.n; // B_kn
    } else {
      refB_size = blas_param.ldb * blas_param.k; // B_nk
    }
    refC_size = blas_param.ldc * blas_param.n; // C_mn
  } else {
    // leading dimension is in terms of consecutive data
    // elements in a row, multiplied by number of columns.
    if (blas_param.trans_a == QUDA_BLAS_OP_N) {
      refA_size = blas_param.lda * blas_param.m; // A_mk
    } else {
      refA_size = blas_param.lda * blas_param.k; // A_km
    }
    if (blas_param.trans_b == QUDA_BLAS_OP_N) {
      refB_size = blas_param.ldb * blas_param.k; // B_nk
    } else {
      refB_size = blas_param.ldb * blas_param.n; // B_kn
    }
    refC_size = blas_param.ldc * blas_param.m; // C_mn
  }

  void *refA = pinned_malloc(batches * refA_size * 2 * data_in_size);
  void *refB = pinned_malloc(batches * refB_size * 2 * data_in_size);
  void *refC = pinned_malloc(batches * refC_size * 2 * data_in_size);
  void *refCcopy = pinned_malloc(batches * refC_size * 2 * data_in_size);

  prepare_ref_array(refA, batches, refA_size, data_in_size, blas_data_type);
  prepare_ref_array(refB, batches, refB_size, data_in_size, blas_data_type);
  prepare_ref_array(refC, batches, refC_size, data_in_size, blas_data_type);
  prepare_ref_array(refCcopy, batches, refC_size, data_in_size, blas_data_type);

  // Create new arrays appropriate for the requested problem, and copy over the data.
  void *arrayA = nullptr;
  void *arrayB = nullptr;
  void *arrayC = nullptr;
  void *arrayCcopy = nullptr;
  size_t data_out_size = 0;
  // Reference data is always complex, but test data can be either real or complex
  int re_im = 0;

  switch (blas_data_type) {
  case QUDA_BLAS_DATATYPE_S:
    data_out_size = sizeof(float);
    re_im = 1;
    break;
  case QUDA_BLAS_DATATYPE_D:
    data_out_size = sizeof(double);
    re_im = 1;
    break;
  case QUDA_BLAS_DATATYPE_C:
    data_out_size = sizeof(float);
    re_im = 2;
    break;
  case QUDA_BLAS_DATATYPE_Z:
    data_out_size = sizeof(double);
    re_im = 2;
    break;
  default: errorQuda("Unrecognised data type %d\n", blas_data_type);
  }

  arrayA = pinned_malloc(batches * refA_size * re_im * data_out_size);
  arrayB = pinned_malloc(batches * refB_size * re_im * data_out_size);
  arrayC = pinned_malloc(batches * refC_size * re_im * data_out_size);
  arrayCcopy = pinned_malloc(batches * refC_size * re_im * data_out_size);

  copy_array(arrayA, refA, batches, refA_size, data_out_size, blas_data_type);
  copy_array(arrayB, refB, batches, refB_size, data_out_size, blas_data_type);
  copy_array(arrayC, refC, batches, refC_size, data_out_size, blas_data_type);
  copy_array(arrayCcopy, refC, batches, refC_size, data_out_size, blas_data_type);

  // Perform device GEMM Blas operation
  blasGEMMQuda(arrayA, arrayB, arrayC, native_blas_lapack ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE, &blas_param);

  double deviation = 0.0;
  if (verify_results) {
    deviation = blasGEMMQudaVerify(arrayA, arrayB, arrayC, arrayCcopy, refA_size, refB_size, refC_size, &blas_param);
  }

  host_free(refA);
  host_free(refB);
  host_free(refC);
  host_free(refCcopy);

  host_free(arrayA);
  host_free(arrayB);
  host_free(arrayC);
  host_free(arrayCcopy);

  return deviation;
}

double lu_inv_test(test_t test_param)
{
  QudaBLASParam blas_param = newQudaBLASParam();
  blas_data_type = ::testing::get<1>(test_param);
  blas_test_type = ::testing::get<0>(test_param);
  setBLASParam(blas_param);

  display_test_info(blas_param);

  // Sanity checks on parameters
  //-------------------------------------------------------------------------
  // Leading dims, strides, and offsets are irrelevant for LU inversions.

  // If the batch value is non-positve, we error out
  if (blas_param.batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param.batch_count); }
  //-------------------------------------------------------------------------

  // Reference data is always in complex double
  size_t data_in_size = sizeof(double);

  int batches = blas_param.batch_count;
  uint64_t array_size = blas_param.inv_mat_size * blas_param.inv_mat_size;

  // Create host data reference arrays
  void *ref_array = pinned_malloc(batches * array_size * 2 * data_in_size);
  void *ref_array_inv = pinned_malloc(batches * array_size * 2 * data_in_size);
  prepare_ref_array(ref_array, batches, array_size, data_in_size, blas_data_type);

  // Create device array appropriate for the requested problem.
  void *dev_array = nullptr;
  void *dev_array_inv = nullptr;
  size_t data_out_size = 0;
  // For now, data is always complex for LU inversion.
  int re_im = 2;

  switch (blas_data_type) {
  case QUDA_BLAS_DATATYPE_C: data_out_size = sizeof(float); break;
  case QUDA_BLAS_DATATYPE_Z: data_out_size = sizeof(double); break;
  case QUDA_BLAS_DATATYPE_S:
  case QUDA_BLAS_DATATYPE_D:
  default: errorQuda("Unsupported data type %d\n", blas_data_type);
  }

  dev_array = pinned_malloc(batches * array_size * re_im * data_out_size);
  dev_array_inv = pinned_malloc(batches * array_size * re_im * data_out_size);

  copy_array(dev_array, ref_array, batches, array_size, data_out_size, blas_data_type);

  // Perform device LU inversion
  blasLUInvQuda(dev_array_inv, dev_array, native_blas_lapack ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE, &blas_param);

  double deviation = 0.0;
  if (verify_results) { deviation = blasLUInvQudaVerify(ref_array, dev_array_inv, array_size, &blas_param); }

  host_free(ref_array);
  host_free(ref_array_inv);
  host_free(dev_array);
  host_free(dev_array_inv);

  return deviation;
}

struct blas_interface_test : quda_test {

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);

    CLI::TransformPairs<QudaBLASDataType> blas_dt_map {
      {"C", QUDA_BLAS_DATATYPE_C}, {"Z", QUDA_BLAS_DATATYPE_Z}, {"S", QUDA_BLAS_DATATYPE_S}, {"D", QUDA_BLAS_DATATYPE_D}};

    CLI::TransformPairs<QudaBLASDataOrder> blas_data_order_map {{"row", QUDA_BLAS_DATAORDER_ROW},
                                                                {"col", QUDA_BLAS_DATAORDER_COL}};
    CLI::TransformPairs<QudaBLASOperation> blas_op_map {
      {"N", QUDA_BLAS_OP_N}, {"T", QUDA_BLAS_OP_T}, {"C", QUDA_BLAS_OP_C}};

    CLI::TransformPairs<QudaBLASType> blas_type_map {{"gemm", QUDA_BLAS_GEMM}, {"lu-inv", QUDA_BLAS_LU_INV}};

    // Option group for BLAS test related options
    auto opgroup = app->add_option_group("BLAS Interface", "Options controlling BLAS interface tests");

    opgroup
      ->add_option("--blas-data-type", blas_data_type,
                   "Whether to use single(S), double(D), and/or complex(C/Z) data types (default C)")
      ->transform(CLI::QUDACheckedTransformer(blas_dt_map));

    opgroup
      ->add_option("--blas-test-type", blas_test_type,
                   "Whether to perform the GEMM test or LU Inversion test (default GEMM)")
      ->transform(CLI::QUDACheckedTransformer(blas_type_map));

    opgroup
      ->add_option("--blas-data-order", blas_data_order,
                   "Whether data is in row major or column major order (default row)")
      ->transform(CLI::QUDACheckedTransformer(blas_data_order_map));

    opgroup
      ->add_option(
        "--blas-gemm-trans-a", blas_gemm_trans_a,
        "Whether to leave the A GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
      ->transform(CLI::QUDACheckedTransformer(blas_op_map));

    opgroup
      ->add_option(
        "--blas-gemm-trans-b", blas_gemm_trans_b,
        "Whether to leave the B GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
      ->transform(CLI::QUDACheckedTransformer(blas_op_map));

    opgroup
      ->add_option("--blas-gemm-alpha", blas_gemm_alpha_re_im,
                   "Set the complex value of alpha for GEMM (default {1.0,0.0}")
      ->expected(2);

    opgroup
      ->add_option("--blas-gemm-beta", blas_gemm_beta_re_im, "Set the complex value of beta for GEMM (default {1.0,0.0}")
      ->expected(2);

    opgroup
      ->add_option("--blas-gemm-mnk", blas_gemm_mnk,
                   "Set the dimensions of the A, B, and C matrices GEMM (default 128 128 128)")
      ->expected(3);

    opgroup
      ->add_option("--blas-gemm-leading-dims", blas_gemm_leading_dims,
                   "Set the leading dimensions A, B, and C matrices GEMM (default 128 128 128) ")
      ->expected(3);

    opgroup
      ->add_option("--blas-gemm-offsets", blas_gemm_offsets,
                   "Set the offsets for GEMM matrices A, B, and C (default 0 0 0)")
      ->expected(3);

    opgroup
      ->add_option("--blas-gemm-strides", blas_gemm_strides,
                   "Set the strides for GEMM matrices A, B, and C (default 1 1 1)")
      ->expected(3);

    opgroup->add_option("--blas-batch", blas_batch, "Set the number of batches for GEMM or LU inversion (default 16)");

    opgroup->add_option("--blas-lu-inv-mat-size", blas_lu_inv_mat_size,
                        "Set the size of the square matrix to invert via LU (default 128)");
  }

  blas_interface_test(int argc, char **argv) : quda_test("BLAS Interface Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  blas_interface_test test(argc, argv);
  test.init();

  int result = 0;
  if (enable_testing) {
    result = test.execute();
    if (result) warningQuda("Google tests for QUDA BLAS failed.");
  } else {
    // Perform the BLAS op specified by the command line
    switch (blas_test_type) {
    case QUDA_BLAS_GEMM: {
      switch (blas_data_type) {
      case QUDA_BLAS_DATATYPE_S: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_S}); break;
      case QUDA_BLAS_DATATYPE_D: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_D}); break;
      case QUDA_BLAS_DATATYPE_C: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_C}); break;
      case QUDA_BLAS_DATATYPE_Z: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_Z}); break;
      default: errorQuda("Undefined QUDA BLAS data type %d\n", blas_data_type);
      }
      break;
    }
    case QUDA_BLAS_LU_INV: {
      switch (blas_data_type) {
      case QUDA_BLAS_DATATYPE_C: lu_inv_test(test_t {QUDA_BLAS_LU_INV, QUDA_BLAS_DATATYPE_C}); break;
      case QUDA_BLAS_DATATYPE_Z: lu_inv_test(test_t {QUDA_BLAS_LU_INV, QUDA_BLAS_DATATYPE_Z}); break;
      default: errorQuda("QUDA BLAS data type %d not supported for LU Inversion\n", blas_data_type);
      }
    } break;
    default: errorQuda("Unknown QUDA BLAS test type %d\n", blas_test_type);
    }
  }

  return result;
}
