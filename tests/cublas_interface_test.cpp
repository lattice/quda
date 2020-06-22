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
#include "cublas_reference.h"
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

  printfQuda("cuBLAS interface test\n");
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

  // initialize the QUDA library
  initQuda(device);
  //-----------------------------------------------------------------------------

  QudaCublasParam cublas_param = newQudaCublasParam();
  cublas_param.trans_a = cublas_trans_a;
  cublas_param.trans_b = cublas_trans_b;
  cublas_param.m = cublas_mnk[0];
  cublas_param.n = cublas_mnk[1];
  cublas_param.k = cublas_mnk[2];
  cublas_param.lda = cublas_leading_dims[0];
  cublas_param.ldb = cublas_leading_dims[1];
  cublas_param.ldc = cublas_leading_dims[2];
  cublas_param.a_offset = cublas_offsets[0];
  cublas_param.b_offset = cublas_offsets[1];
  cublas_param.c_offset = cublas_offsets[2];
  cublas_param.alpha = (__complex__ double)cublas_alpha_re_im[0];
  cublas_param.beta = (__complex__ double)cublas_beta_re_im[0];
  cublas_param.data_order = cublas_data_order;
  cublas_param.data_type = cublas_data_type;
  cublas_param.batch_count = cublas_batch;

  // Reference data is always in complex double
  size_t data_size = sizeof(double);
  int re_im = 2;
  uint64_t refA_size = 0, refB_size = 0, refC_size = 0;
  if (cublas_param.data_order == QUDA_CUBLAS_DATAORDER_COL) {
    // leading dimension is in terms of consecutive data
    // elements in a column, multiplied by number of rows
    if (cublas_param.trans_a == QUDA_CUBLAS_OP_N) {
      refA_size = cublas_param.lda * cublas_param.k; // A_mk
    } else {
      refA_size = cublas_param.lda * cublas_param.m; // A_km
    }

    if (cublas_param.trans_b == QUDA_CUBLAS_OP_N) {
      refB_size = cublas_param.ldb * cublas_param.n; // B_kn
    } else {
      refB_size = cublas_param.ldb * cublas_param.k; // B_nk
    }
    refC_size = cublas_param.ldc * cublas_param.n; // C_mn
  } else {
    // leading dimension is in terms of consecutive data
    // elements in a row, multiplied by number of columns.
    if (cublas_param.trans_a == QUDA_CUBLAS_OP_N) {
      refA_size = cublas_param.lda * cublas_param.m; // A_mk
    } else {
      refA_size = cublas_param.lda * cublas_param.k; // A_km
    }
    if (cublas_param.trans_b == QUDA_CUBLAS_OP_N) {
      refB_size = cublas_param.ldb * cublas_param.k; // B_nk
    } else {
      refB_size = cublas_param.ldb * cublas_param.n; // B_kn
    }
    refC_size = cublas_param.ldc * cublas_param.m; // C_mn
  }

  void *refA = pinned_malloc(refA_size * re_im * data_size);
  void *refB = pinned_malloc(refB_size * re_im * data_size);
  void *refC = pinned_malloc(refC_size * re_im * data_size);
  void *refCcopy = pinned_malloc(refC_size * re_im * data_size);

  memset(refA, 0, refA_size * re_im * data_size);
  memset(refB, 0, refB_size * re_im * data_size);
  memset(refC, 0, refC_size * re_im * data_size);
  memset(refCcopy, 0, refC_size * re_im * data_size);

  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * refA_size; i += 2) { ((double *)refA)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refB_size; i += 2) { ((double *)refB)[i] = rand() / (double)RAND_MAX; }
  for (uint64_t i = 0; i < 2 * refC_size; i += 2) {
    ((double *)refC)[i] = rand() / (double)RAND_MAX;
    ((double *)refCcopy)[i] = ((double *)refC)[i];
  }

  // Populate the imaginary part with rands if needed
  if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_C || cublas_param.data_type == QUDA_CUBLAS_DATATYPE_Z) {
    for (uint64_t i = 1; i < 2 * refA_size; i += 2) { ((double *)refA)[i] = rand() / (double)RAND_MAX; }
    for (uint64_t i = 1; i < 2 * refB_size; i += 2) { ((double *)refB)[i] = rand() / (double)RAND_MAX; }
    for (uint64_t i = 1; i < 2 * refC_size; i += 2) {
      ((double *)refC)[i] = rand() / (double)RAND_MAX;
      ((double *)refCcopy)[i] = ((double *)refC)[i];
    }
  }

  // Create new arrays appropriate for the requested problem, and copy over the data.
  void *arrayA = nullptr;
  void *arrayB = nullptr;
  void *arrayC = nullptr;
  void *arrayCcopy = nullptr;

  switch (cublas_param.data_type) {
  case QUDA_CUBLAS_DATATYPE_S:
    arrayA = pinned_malloc(refA_size * sizeof(float));
    arrayB = pinned_malloc(refB_size * sizeof(float));
    arrayC = pinned_malloc(refC_size * sizeof(float));
    arrayCcopy = pinned_malloc(refC_size * sizeof(float));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size; i += 2) { ((float *)arrayA)[i / 2] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size; i += 2) { ((float *)arrayB)[i / 2] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size; i += 2) {
      ((float *)arrayC)[i / 2] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i / 2] = ((double *)refC)[i];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_D:
    arrayA = pinned_malloc(refA_size * sizeof(double));
    arrayB = pinned_malloc(refB_size * sizeof(double));
    arrayC = pinned_malloc(refC_size * sizeof(double));
    arrayCcopy = pinned_malloc(refC_size * sizeof(double));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size; i += 2) { ((double *)arrayA)[i / 2] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size; i += 2) { ((double *)arrayB)[i / 2] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size; i += 2) {
      ((double *)arrayC)[i / 2] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i / 2] = ((double *)refC)[i];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_C:
    arrayA = pinned_malloc(refA_size * 2 * sizeof(float));
    arrayB = pinned_malloc(refB_size * 2 * sizeof(float));
    arrayC = pinned_malloc(refC_size * 2 * sizeof(float));
    arrayCcopy = pinned_malloc(refC_size * 2 * sizeof(float));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size; i++) { ((float *)arrayA)[i] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size; i++) { ((float *)arrayB)[i] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size; i++) {
      ((float *)arrayC)[i] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i] = ((double *)refC)[i];
    }
    break;
  case QUDA_CUBLAS_DATATYPE_Z:
    arrayA = pinned_malloc(refA_size * 2 * sizeof(double));
    arrayB = pinned_malloc(refB_size * 2 * sizeof(double));
    arrayC = pinned_malloc(refC_size * 2 * sizeof(double));
    arrayCcopy = pinned_malloc(refC_size * 2 * sizeof(double));
    // Populate
    for (uint64_t i = 0; i < 2 * refA_size; i++) { ((double *)arrayA)[i] = ((double *)refA)[i]; }
    for (uint64_t i = 0; i < 2 * refB_size; i++) { ((double *)arrayB)[i] = ((double *)refB)[i]; }
    for (uint64_t i = 0; i < 2 * refC_size; i++) {
      ((double *)arrayC)[i] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i] = ((double *)refC)[i];
    }
    break;
  default: errorQuda("Unrecognised data type %d\n", cublas_param.data_type);
  }

  // Perform GPU GEMM Blas operation
  cublasGEMMQuda(arrayA, arrayB, arrayC, &cublas_param);

  if (verify_results) {
    if (cublas_param.batch_count != 1) errorQuda("Testing with batched arrays not yet supported.");
    cublasGEMMQudaVerify(arrayA, arrayB, arrayC, arrayCcopy, refA_size, refB_size, refC_size, re_im, data_size,
                         &cublas_param);
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
