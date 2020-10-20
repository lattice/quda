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

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec\n");
  printfQuda("%s   %s\n", get_prec_str(prec), get_prec_str(prec_sloppy));

  printfQuda("BLAS interface test\n");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{

  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();
  setQudaPrecisions();
  display_test_info();
  setVerbosity(verbosity);
  
  // initialize the QUDA library
  initQuda(device_ordinal);
  //-----------------------------------------------------------------------------

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
  blas_param.strideA = blas_strides[0];
  blas_param.strideB = blas_strides[1];
  blas_param.strideC = blas_strides[2];
  blas_param.alpha = (__complex__ double)blas_alpha_re_im[0];
  blas_param.beta = (__complex__ double)blas_beta_re_im[0];
  blas_param.data_order = blas_data_order;
  blas_param.data_type = blas_data_type;
  blas_param.batch_count = blas_batch;

  // Reference data is always in complex double
  size_t data_size = sizeof(double);
  int re_im = 2;
  int batches = blas_param.batch_count;
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

  void *refA = pinned_malloc(batches * refA_size * re_im * data_size);
  void *refB = pinned_malloc(batches * refB_size * re_im * data_size);
  void *refC = pinned_malloc(batches * refC_size * re_im * data_size);
  void *refCcopy = pinned_malloc(batches * refC_size * re_im * data_size);

  memset(refA, 0, batches * refA_size * re_im * data_size);
  memset(refB, 0, batches * refB_size * re_im * data_size);
  memset(refC, 0, batches * refC_size * re_im * data_size);
  memset(refCcopy, 0, batches * refC_size * re_im * data_size);

  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * refA_size * batches; i += 2) { ((double *)refA)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refB_size * batches; i += 2) { ((double *)refB)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refC_size * batches; i += 2) {
    ((double *)refC)[i] = rand() / (double)RAND_MAX;
    ((double *)refCcopy)[i] = ((double *)refC)[i];
  }

  // Populate the imaginary part with rands if needed
  if (blas_param.data_type == QUDA_BLAS_DATATYPE_C || blas_param.data_type == QUDA_BLAS_DATATYPE_Z) {
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

  switch (blas_param.data_type) {
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
  default: errorQuda("Unrecognised data type %d\n", blas_param.data_type);
  }

  // Perform GPU GEMM Blas operation
  blasGEMMQuda(arrayA, arrayB, arrayC, &blas_param);

  if (verify_results) {
    blasGEMMQudaVerify(arrayA, arrayB, arrayC, arrayCcopy, refA_size, refB_size, refC_size, re_im, data_size,
		       &blas_param);
  }

  host_free(refA);
  host_free(refB);
  host_free(refC);
  host_free(refCcopy);

  host_free(arrayA);
  host_free(arrayB);
  host_free(arrayC);
  host_free(arrayCcopy);

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
