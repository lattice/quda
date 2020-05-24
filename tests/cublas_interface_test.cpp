#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <contract_reference.h>
#include "misc.h"

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda
{
  extern void setTransferGPU(bool);
}

void cublasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayCcopy, void*arrayC,
			  QudaCublasParam *cublas_param){

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
  cublas_param.alpha  = (__complex__ double)cublas_alpha_re_im[0];  
  cublas_param.beta  = (__complex__ double)cublas_beta_re_im[0];
  cublas_param.batch_count = cublas_batch;
  cublas_param.type = (prec == QUDA_DOUBLE_PRECISION) ? QUDA_CUBLAS_DATATYPE_Z : QUDA_CUBLAS_DATATYPE_C;

  printQudaCublasParam(&cublas_param);
  
  unsigned int arrayA_size = cublas_param.m * cublas_param.k; //A_mk
  unsigned int arrayB_size = cublas_param.k * cublas_param.n; //B_kn
  unsigned int arrayC_size = cublas_param.m * cublas_param.n; //C_mn

  size_t data_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  if(cublas_param.type == QUDA_CUBLAS_DATATYPE_C || cublas_param.type == QUDA_CUBLAS_DATATYPE_Z) {
    data_size *= 2;
  }
  
  void *arrayA = malloc(arrayA_size * data_size * cublas_param.batch_count);
  void *arrayB = malloc(arrayB_size * data_size * cublas_param.batch_count);
  void *arrayC = malloc(arrayC_size * data_size * cublas_param.batch_count);
  void *arrayCcopy = malloc(arrayC_size * data_size * cublas_param.batch_count);
  
  if (prec == QUDA_SINGLE_PRECISION) {
    for (unsigned int i = 0; i < arrayA_size * cublas_param.batch_count; i++) {
      ((float *)arrayA)[i] = rand() / (float)RAND_MAX;
    }
    for (unsigned int i = 0; i < arrayB_size * cublas_param.batch_count; i++) {
      ((float *)arrayB)[i] = rand() / (float)RAND_MAX;
    }
    for (unsigned int i = 0; i < arrayC_size * cublas_param.batch_count; i++) {
      ((float *)arrayC)[i] = rand() / (float)RAND_MAX;
      ((float *)arrayCcopy)[i] = ((float *)arrayC)[i];
    }
  } else {
    for (unsigned int i = 0; i < arrayA_size * cublas_param.batch_count; i++) {
      ((double *)arrayA)[i] = rand() / (double)RAND_MAX;
    }
    for (unsigned int i = 0; i < arrayB_size * cublas_param.batch_count; i++) {
      ((double *)arrayB)[i] = rand() / (double)RAND_MAX;
    }
    for (unsigned int i = 0; i < arrayC_size * cublas_param.batch_count; i++) {
      ((double *)arrayC)[i] = rand() / (double)RAND_MAX;
      ((double *)arrayCcopy)[i] = ((double *)arrayC)[i];
    }
  }
  
  // Perform GPU GEMM Blas operation
  cublasGEMMQuda(arrayA, arrayB, arrayC, &cublas_param);

  if(verify_results) {
    cublasGEMMQudaVerify(arrayA, arrayB, arrayCcopy, arrayC, &cublas_param);
  }
  
  free(arrayA);
  free(arrayB);
  free(arrayC);

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
