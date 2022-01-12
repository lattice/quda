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
#include "blas_reference.h"
#include "misc.h"

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *data_type_str[] = {
  "realSingle",
  "realDouble",
  "complexSingle",
  "complexDouble",
};

QudaBLASOperation blas_trans_a = QUDA_BLAS_OP_N;
QudaBLASOperation blas_trans_b = QUDA_BLAS_OP_N;
QudaBLASDataType blas_data_type = QUDA_BLAS_DATATYPE_C;
QudaBLASDataOrder blas_data_order = QUDA_BLAS_DATAORDER_COL;

std::array<int, 3> blas_mnk = {64, 64, 64};
auto &blas_m = blas_mnk[0];
auto &blas_n = blas_mnk[1];
auto &blas_k = blas_mnk[2];

std::array<int, 3> blas_leading_dims = {128, 128, 128};
auto &blas_lda = blas_leading_dims[0];
auto &blas_ldb = blas_leading_dims[1];
auto &blas_ldc = blas_leading_dims[2];

std::array<int, 3> blas_offsets = {0, 0, 0};
auto &blas_a_offset = blas_offsets[0];
auto &blas_b_offset = blas_offsets[1];
auto &blas_c_offset = blas_offsets[2];

std::array<int, 3> blas_strides = {1, 1, 1};
auto &blas_a_stride = blas_strides[0];
auto &blas_b_stride = blas_strides[1];
auto &blas_c_stride = blas_strides[2];

std::array<double, 2> blas_alpha_re_im = {M_PI, M_E};
std::array<double, 2> blas_beta_re_im = {M_LN2, M_LN10};
int blas_batch = 16;

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("BLAS interface test\n");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

double test(int data_type)
{
  QudaBLASDataType test_data_type = QUDA_BLAS_DATATYPE_INVALID;
  switch (data_type) {
  case 0: test_data_type = QUDA_BLAS_DATATYPE_S; break;
  case 1: test_data_type = QUDA_BLAS_DATATYPE_D; break;
  case 2: test_data_type = QUDA_BLAS_DATATYPE_C; break;
  case 3: test_data_type = QUDA_BLAS_DATATYPE_Z; break;
  default: errorQuda("Undefined QUDA BLAS data type %d\n", data_type);
  }

  QudaBLASParam blas_param = newQudaBLASParam();
  blas_param.trans_a = blas_trans_a;
  blas_param.trans_b = blas_trans_b;
  blas_param.m = blas_mnk[0];
  blas_param.n = blas_mnk[1];
  blas_param.k = blas_mnk[2];
  blas_param.lda = blas_leading_dims[0];
  blas_param.ldb = blas_leading_dims[1];
  blas_param.ldc = blas_leading_dims[2];
  blas_param.a_offset = blas_offsets[0];
  blas_param.b_offset = blas_offsets[1];
  blas_param.c_offset = blas_offsets[2];
  blas_param.a_stride = blas_strides[0];
  blas_param.b_stride = blas_strides[1];
  blas_param.c_stride = blas_strides[2];
  memcpy(&blas_param.alpha, blas_alpha_re_im.data(), sizeof(__complex__ double));
  memcpy(&blas_param.beta, blas_beta_re_im.data(), sizeof(__complex__ double));
  blas_param.data_order = blas_data_order;
  blas_param.data_type = test_data_type;
  blas_param.batch_count = blas_batch;

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
  size_t data_size = sizeof(double);
  int re_im = 2;
  data_size *= re_im;

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

  void *refA = pinned_malloc(batches * refA_size * data_size);
  void *refB = pinned_malloc(batches * refB_size * data_size);
  void *refC = pinned_malloc(batches * refC_size * data_size);
  void *refCcopy = pinned_malloc(batches * refC_size * data_size);

  memset(refA, 0, batches * refA_size * data_size);
  memset(refB, 0, batches * refB_size * data_size);
  memset(refC, 0, batches * refC_size * data_size);
  memset(refCcopy, 0, batches * refC_size * data_size);

  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)refA)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)refB)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
    ((double *)refC)[i] = rand() / (double)RAND_MAX;
    ((double *)refCcopy)[i] = ((double *)refC)[i];
  }

  // Populate the imaginary part with rands if needed
  if (test_data_type == QUDA_BLAS_DATATYPE_C || test_data_type == QUDA_BLAS_DATATYPE_Z) {
    for (uint64_t i = 1; i < 2 * refA_size * batches; i += 2) { ((double *)refA)[i] = rand() / (double)RAND_MAX; }
    for (uint64_t i = 1; i < 2 * refB_size * batches; i += 2) { ((double *)refB)[i] = rand() / (double)RAND_MAX; }
    for (uint64_t i = 1; i < 2 * refC_size * batches; i += 2) {
      ((double *)refC)[i] = rand() / (double)RAND_MAX;
      ((double *)refCcopy)[i] = ((double *)refC)[i];
    }
  }

  // Create new arrays appropriate for the requested problem, and copy over the data.
  void *arrayA = nullptr;
  void *arrayB = nullptr;
  void *arrayC = nullptr;
  void *arrayCcopy = nullptr;

  switch (test_data_type) {
  case QUDA_BLAS_DATATYPE_S:
    arrayA = pinned_malloc(batches * refA_size * sizeof(float));
    arrayB = pinned_malloc(batches * refB_size * sizeof(float));
    arrayC = pinned_malloc(batches * refC_size * sizeof(float));
    arrayCcopy = pinned_malloc(batches * refC_size * sizeof(float));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((float *)arrayA)[i / 2] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((float *)arrayB)[i / 2] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((float *)arrayC)[i / 2] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i / 2] = ((double *)refC)[i];
    }
    break;
  case QUDA_BLAS_DATATYPE_D:
    arrayA = pinned_malloc(batches * refA_size * sizeof(double));
    arrayB = pinned_malloc(batches * refB_size * sizeof(double));
    arrayC = pinned_malloc(batches * refC_size * sizeof(double));
    arrayCcopy = pinned_malloc(batches * refC_size * sizeof(double));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)arrayA)[i / 2] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)arrayB)[i / 2] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
      ((double *)arrayC)[i / 2] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i / 2] = ((double *)refC)[i];
    }
    break;
  case QUDA_BLAS_DATATYPE_C:
    arrayA = pinned_malloc(batches * refA_size * 2 * sizeof(float));
    arrayB = pinned_malloc(batches * refB_size * 2 * sizeof(float));
    arrayC = pinned_malloc(batches * refC_size * 2 * sizeof(float));
    arrayCcopy = pinned_malloc(batches * refC_size * 2 * sizeof(float));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((float *)arrayA)[i] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((float *)arrayB)[i] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((float *)arrayC)[i] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i] = ((double *)refC)[i];
    }
    break;
  case QUDA_BLAS_DATATYPE_Z:
    arrayA = pinned_malloc(batches * refA_size * 2 * sizeof(double));
    arrayB = pinned_malloc(batches * refB_size * 2 * sizeof(double));
    arrayC = pinned_malloc(batches * refC_size * 2 * sizeof(double));
    arrayCcopy = pinned_malloc(batches * refC_size * 2 * sizeof(double));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size * batches; i++) { ((double *)arrayA)[i] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size * batches; i++) { ((double *)arrayB)[i] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size * batches; i++) {
      ((double *)arrayC)[i] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i] = ((double *)refC)[i];
    }
    break;
  default: errorQuda("Unrecognised data type %d\n", test_data_type);
  }

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

// The following tests gets each BLAS type and precision using google testing framework
using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class BLASTest : public ::testing::TestWithParam<int>
{
protected:
  int param;

public:
  virtual ~BLASTest() { }
  virtual void SetUp() { param = GetParam(); }
};

// Sets up the Google test
TEST_P(BLASTest, verify)
{
  auto data_type = GetParam();
  auto deviation = test(data_type);
  decltype(deviation) tol;
  switch (data_type) {
  case 0:
  case 2: tol = 10 * std::numeric_limits<float>::epsilon(); break;
  case 1:
  case 3: tol = 10 * std::numeric_limits<double>::epsilon(); break;
  }
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

// Helper function to construct the test name
std::string getBLASName(testing::TestParamInfo<int> param)
{
  int data_type = param.param;
  std::string str(data_type_str[data_type]);
  return str;
}

// Instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, BLASTest, Range(0, 4), getBLASName);

void add_blas_interface_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  CLI::TransformPairs<QudaBLASDataType> blas_dt_map {
    {"C", QUDA_BLAS_DATATYPE_C}, {"Z", QUDA_BLAS_DATATYPE_Z}, {"S", QUDA_BLAS_DATATYPE_S}, {"D", QUDA_BLAS_DATATYPE_D}};

  CLI::TransformPairs<QudaBLASDataOrder> blas_data_order_map {{"row", QUDA_BLAS_DATAORDER_ROW},
                                                              {"col", QUDA_BLAS_DATAORDER_COL}};

  CLI::TransformPairs<QudaBLASOperation> blas_op_map {{"N", QUDA_BLAS_OP_N}, {"T", QUDA_BLAS_OP_T}, {"C", QUDA_BLAS_OP_C}};

  auto opgroup = quda_app->add_option_group("BLAS Interface", "Options controlling BLAS interface tests");
  opgroup
    ->add_option("--blas-data-type", blas_data_type,
                 "Whether to use single(S), double(D), and/or complex(C/Z) data types (default C)")
    ->transform(CLI::QUDACheckedTransformer(blas_dt_map));

  opgroup
    ->add_option("--blas-data-order", blas_data_order, "Whether data is in row major or column major order (default row)")
    ->transform(CLI::QUDACheckedTransformer(blas_data_order_map));

  opgroup
    ->add_option(
      "--blas-trans-a", blas_trans_a,
      "Whether to leave the A GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
    ->transform(CLI::QUDACheckedTransformer(blas_op_map));

  opgroup
    ->add_option(
      "--blas-trans-b", blas_trans_b,
      "Whether to leave the B GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
    ->transform(CLI::QUDACheckedTransformer(blas_op_map));

  opgroup->add_option("--blas-alpha", blas_alpha_re_im, "Set the complex value of alpha for GEMM (default {1.0,0.0}")
    ->expected(2);

  opgroup->add_option("--blas-beta", blas_beta_re_im, "Set the complex value of beta for GEMM (default {1.0,0.0}")
    ->expected(2);

  opgroup
    ->add_option("--blas-mnk", blas_mnk, "Set the dimensions of the A, B, and C matrices GEMM (default 128 128 128)")
    ->expected(3);

  opgroup
    ->add_option("--blas-leading-dims", blas_leading_dims,
                 "Set the leading dimensions A, B, and C matrices GEMM (default 128 128 128) ")
    ->expected(3);

  opgroup->add_option("--blas-offsets", blas_offsets, "Set the offsets for matrices A, B, and C (default 0 0 0)")
    ->expected(3);

  opgroup->add_option("--blas-strides", blas_strides, "Set the strides for matrices A, B, and C (default 1 1 1)")
    ->expected(3);

  opgroup->add_option("--blas-batch", blas_batch, "Set the number of batches for GEMM (default 16)");
}

int main(int argc, char **argv)
{
  // Start Google Test Suite
  //-----------------------------------------------------------------------------
  ::testing::InitGoogleTest(&argc, argv);

  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  auto app = make_app();
  add_blas_interface_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // call srand() with a rank-dependent seed
  initRand();
  setQudaPrecisions();
  display_test_info();
  setVerbosity(verbosity);

  // initialize the QUDA library
  initQuda(device_ordinal);
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);
  //-----------------------------------------------------------------------------

  int result = 0;
  if (verify_results) {
    // Run full set of test if we're doing a verification run
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
    if (result) warningQuda("Google tests for QUDA BLAS failed.");
  } else {
    // Perform the BLAS op specified by the command line
    switch (blas_data_type) {
    case QUDA_BLAS_DATATYPE_S: test(0); break;
    case QUDA_BLAS_DATATYPE_D: test(1); break;
    case QUDA_BLAS_DATATYPE_C: test(2); break;
    case QUDA_BLAS_DATATYPE_Z: test(3); break;
    default: errorQuda("Undefined QUDA BLAS data type %d\n", blas_data_type);
    }
  }

  //-----------------------------------------------------------------------------

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return result;
}
