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

#include <Eigen/Dense>
using namespace Eigen;

void fillEigenArrayColMaj(MatrixXcd &EigenArr, complex<double>* arr, int rows, int cols, int counter = 0){
  for(int j=0; j<cols; j++) {
    for(int i=0; i<rows; i++) {
      EigenArr(i,j) = arr[counter];
      counter++;
    }
  }
}

void fillEigenArrayRowMaj(MatrixXcd &EigenArr, complex<double>* arr, int rows, int cols, int counter = 0){
  for(int i=0; i<rows; i++) {
    for(int j=0; j<cols; j++) {
      EigenArr(i,j) = arr[counter];
      counter++;
    }
  }
}

void cublasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayCcopy, void*arrayC,
			  QudaCublasParam *cublas_param){

  // Problem parameters
  int m = cublas_param->m;
  int n = cublas_param->n;
  int k = cublas_param->k;
  int lda = cublas_param->lda;
  int ldb = cublas_param->ldb;
  int ldc = cublas_param->ldc;  
  complex<double> alpha = cublas_param->alpha;
  complex<double> beta = cublas_param->beta;

  // Eigen objects to store data
  uint64_t A_r, A_c, B_r, B_c, C_r, C_c;
  if(cublas_param->data_order == QUDA_CUBLAS_DATAORDER_COL) {
    A_r = lda; A_c = k;
    B_r = ldb; B_c = n;
    C_r = ldc; C_c = n;
  } else {
    A_r = m; A_c = lda;
    B_r = k; B_c = ldb;
    C_r = m; C_c = ldc;
  }
  
  MatrixXcd A = MatrixXd::Zero(A_r, A_c);
  MatrixXcd B = MatrixXd::Zero(B_r, B_c);
  MatrixXcd C_eigen = MatrixXd::Zero(C_r, C_c);
  MatrixXcd C_gpu = MatrixXd::Zero(C_r, C_c);
  MatrixXcd C_resid = MatrixXd::Zero(C_r, C_c);
  
  // Pointers to data
  complex<double>* A_ptr = (complex<double>*)(&arrayA)[0];
  complex<double>* B_ptr = (complex<double>*)(&arrayB)[0];
  complex<double>* C_ptr = (complex<double>*)(&arrayC)[0];
  complex<double>* Ccopy_ptr = (complex<double>*)(&arrayCcopy)[0];

  // Populate Eigen objects
  if(cublas_param->data_order == QUDA_CUBLAS_DATAORDER_COL) {
    fillEigenArrayColMaj(A, A_ptr, A_r, A_c);
    fillEigenArrayColMaj(B, B_ptr, B_r, B_c);
    fillEigenArrayColMaj(C_eigen, Ccopy_ptr, C_r, C_c);
    fillEigenArrayColMaj(C_gpu, C_ptr, C_r, C_c);
  }
  else {
    fillEigenArrayRowMaj(A, A_ptr, A_r, A_c);
    fillEigenArrayRowMaj(B, B_ptr, B_r, B_c);
    fillEigenArrayRowMaj(C_eigen, Ccopy_ptr, C_r, C_c);
    fillEigenArrayRowMaj(C_gpu, C_ptr, C_r, C_c);
  }

  // Apply the matrix operation types to A and B
  switch(cublas_param->trans_a) {
  case QUDA_CUBLAS_OP_T : A.transposeInPlace(); break;
  case QUDA_CUBLAS_OP_C : A.adjointInPlace(); break;
  case QUDA_CUBLAS_OP_N : break;
  default :
    errorQuda("Unknown cuBLAS op type %d", cublas_param->trans_a);
  }

  switch(cublas_param->trans_b) {
  case QUDA_CUBLAS_OP_T : B.transposeInPlace(); break;
  case QUDA_CUBLAS_OP_C : B.adjointInPlace(); break;
  case QUDA_CUBLAS_OP_N : break;
  default :
    errorQuda("Unknown cuBLAS op type %d", cublas_param->trans_b);
  }
  
  // Perform GEMM using Eigen
  printfQuda("a * A_{%lu,%lu} * B_{%lu,%lu} + b * C_{%lu,%lu} = C_{%lu,%lu}\n",
	     A_r, A_c, B_r, B_c, C_r, C_c, C_r, C_c);

  C_eigen = alpha * A * B + beta * C_eigen;
  
  // Check Eigen result against cuBLAS
  C_resid = C_gpu - C_eigen;
  
  printfQuda("(C_host - C_gpu) Frobenius norm = %e. Relative deviation = %e\n", C_resid.norm(), C_resid.norm()/(C_resid.rows() * C_resid.cols()));
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

  int dCount;
  cudaGetDeviceCount( &dCount );  
  int device;
  for (device = 0; device < dCount; ++device ) {
    
    cudaDeviceProp deviceProp;      
    cudaGetDeviceProperties( &deviceProp, device );
    printfQuda( "%d - name:                    %s\n" ,device ,deviceProp.name );
    printfQuda( "%d - totalGlobalMem:          %d bytes ( %.2f Gbytes)\n" ,device ,deviceProp.totalGlobalMem , deviceProp.totalGlobalMem / (float)( 1024 * 1024 * 1024)  );
    printfQuda( "%d - sharedMemPerBlock:       %d bytes ( %.2f Kbytes)\n" ,device ,deviceProp.sharedMemPerBlock ,deviceProp.sharedMemPerBlock / (float)1024  );
    printfQuda( "%d - regsPerBlock:            %d\n" ,device ,deviceProp.regsPerBlock );
    printfQuda( "%d - warpSize:                %d\n" ,device ,deviceProp.warpSize );
    printfQuda( "%d - memPitch:                %d\n" ,device ,deviceProp.memPitch );
    printfQuda( "%d - maxThreadsPerBlock:      %d\n" ,device ,deviceProp.maxThreadsPerBlock );
    printfQuda( "%d - maxThreadsDim[0]:        %d\n" ,device ,deviceProp.maxThreadsDim[0] );
    printfQuda( "%d - maxThreadsDim[1]:        %d\n" ,device ,deviceProp.maxThreadsDim[1] );
    printfQuda( "%d - maxThreadsDim[2]:        %d\n" ,device ,deviceProp.maxThreadsDim[2] );
    printfQuda( "%d - maxGridSize[0]:          %d\n" ,device ,deviceProp.maxGridSize[0] );
    printfQuda( "%d - maxGridSize[1]:          %d\n" ,device ,deviceProp.maxGridSize[1] );
    printfQuda( "%d - maxGridSize[2]:          %d\n" ,device ,deviceProp.maxGridSize[2] );
    printfQuda( "%d - totalConstMem:           %d bytes ( %.2f Kbytes)\n" ,device ,deviceProp.totalConstMem ,deviceProp.totalConstMem / (float) 1024 );
    printfQuda( "%d - compute capability:      %d.%d\n" ,device ,deviceProp.major ,deviceProp.minor);
    printfQuda( "%d - clockRate                %d kilohertz\n" ,device ,deviceProp.clockRate );
    printfQuda( "%d - textureAlignment         %d\n\n" ,device ,deviceProp.textureAlignment );
  }
  
  
  QudaCublasParam cublas_param = newQudaCublasParam();
  cublas_param.trans_a = cublas_trans_a;
  cublas_param.trans_b = cublas_trans_b;
  cublas_param.m = cublas_mnk[0];
  cublas_param.n = cublas_mnk[1];
  cublas_param.k = cublas_mnk[2];
  cublas_param.lda = cublas_leading_dims[0];
  cublas_param.ldb = cublas_leading_dims[1];
  cublas_param.ldc = cublas_leading_dims[2];
  cublas_param.alpha = (__complex__ double)cublas_alpha_re_im[0];  
  cublas_param.beta  = (__complex__ double)cublas_beta_re_im[0];
  cublas_param.data_order = cublas_data_order;
  cublas_param.data_type = cublas_data_type;

  // Testing for batch not yet supported.
  cublas_param.batch_count = cublas_batch;

  // Reference data is always in complex double
  size_t data_size = sizeof(double);
  int re_im = 2;
  uint64_t refA_size = 0, refB_size = 0, refC_size = 0;
  if(cublas_param.data_order == QUDA_CUBLAS_DATAORDER_COL) {
    // leading dimension is in terms of consecutive data
    // elements in a column, multiplied by number of rows
    refA_size = cublas_param.lda * cublas_param.k; //A_mk
    refB_size = cublas_param.ldb * cublas_param.n; //B_kn
    refC_size = cublas_param.ldc * cublas_param.n; //C_mn
  } else {
    // leading dimension is in terms of consecutive data
    // elements in a row, multiplied by number of columns
    refA_size = cublas_param.m * cublas_param.lda; //A_mk
    refB_size = cublas_param.k * cublas_param.ldb; //B_kn
    refC_size = cublas_param.m * cublas_param.ldc; //C_mn
  }
  
  void *refA = malloc(refA_size * re_im * data_size);
  void *refB = malloc(refB_size * re_im * data_size);
  void *refC = malloc(refC_size * re_im * data_size);
  void *refCcopy = malloc(refC_size * re_im * data_size);

  memset(refA, 0, refA_size * re_im * data_size);
  memset(refB, 0, refB_size * re_im * data_size);
  memset(refC, 0, refC_size * re_im * data_size);
  memset(refCcopy, 0, refC_size * re_im * data_size);
  
  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * refA_size; i+=2) {
    ((double *)refA)[i] = rand() / (double)RAND_MAX;
  }
  for (uint64_t i = 0; i < 2 * refB_size; i+=2) {
    ((double *)refB)[i] = rand() / (double)RAND_MAX;
  }
  for (uint64_t i = 0; i < 2 * refC_size; i+=2) {
    ((double *)refC)[i] = rand() / (double)RAND_MAX;
    ((double *)refCcopy)[i] = ((double *)refC)[i];
  }

  // Populate the imaginary part with rands
  if (cublas_param.data_type == QUDA_CUBLAS_DATATYPE_C ||
      cublas_param.data_type == QUDA_CUBLAS_DATATYPE_Z) {
    for (uint64_t i = 1; i < 2 * refA_size; i+=2) {
      ((double *)refA)[i] = rand() / (double)RAND_MAX;
    }
    for (uint64_t i = 1; i < 2 * refB_size; i+=2) {
      ((double *)refB)[i] = rand() / (double)RAND_MAX;
    }
    for (uint64_t i = 1; i < 2 * refC_size; i+=2) {
      ((double *)refC)[i] = rand() / (double)RAND_MAX;
      ((double *)refCcopy)[i] = ((double *)refC)[i];
    }    
  }

  // Create new arrays appropriate for the requested problem, and copy the data.
  void *arrayA;
  void *arrayB;
  void *arrayC;
  void *arrayCcopy;

  switch (cublas_param.data_type) {
  case QUDA_CUBLAS_DATATYPE_S :
    arrayA = malloc(refA_size * sizeof(float));
    arrayB = malloc(refB_size * sizeof(float));
    arrayC = malloc(refC_size * sizeof(float));
    arrayCcopy = malloc(refC_size * sizeof(float));
    // Populate 
    for (uint64_t i = 0; i < 2 * refA_size; i+=2) {
      ((float *)arrayA)[i/2] = ((double *)refA)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refB_size; i+=2) {
      ((float *)arrayB)[i/2] = ((double *)refB)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refC_size; i+=2) {
      ((float *)arrayC)[i/2] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i/2] = ((double *)refC)[i]; 
    }
    break;
  case QUDA_CUBLAS_DATATYPE_D :
    arrayA = malloc(refA_size * sizeof(double));
    arrayB = malloc(refB_size * sizeof(double));
    arrayC = malloc(refC_size * sizeof(double));
    arrayCcopy = malloc(refC_size * sizeof(double));
    // Populate 
    for (uint64_t i = 0; i < 2 * refA_size; i+=2) {
      ((double *)arrayA)[i/2] = ((double *)refA)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refB_size; i+=2) {
      ((double *)arrayB)[i/2] = ((double *)refB)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refC_size; i+=2) {
      ((double *)arrayC)[i/2] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i/2] = ((double *)refC)[i]; 
    }
    break;
  case QUDA_CUBLAS_DATATYPE_C :
    arrayA = malloc(refA_size * 2 * sizeof(float));
    arrayB = malloc(refB_size * 2 * sizeof(float));
    arrayC = malloc(refC_size * 2 * sizeof(float));
    arrayCcopy = malloc(refC_size * 2 * sizeof(float));
    // Populate 
    for (uint64_t i = 0; i < 2 * refA_size; i++) {
      ((float *)arrayA)[i] = ((double *)refA)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refB_size; i++) {
      ((float *)arrayB)[i] = ((double *)refB)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refC_size; i++) {
      ((float *)arrayC)[i] = ((double *)refC)[i];
      ((float *)arrayCcopy)[i] = ((double *)refC)[i]; 
    }
    break;
  case QUDA_CUBLAS_DATATYPE_Z :
    arrayA = malloc(refA_size * 2 * sizeof(double));
    arrayB = malloc(refB_size * 2 * sizeof(double));
    arrayC = malloc(refC_size * 2 * sizeof(double));
    arrayCcopy = malloc(refC_size * 2 * sizeof(double));
    // Populate 
    for (uint64_t i = 0; i < 2 * refA_size; i++) {
      ((double *)arrayA)[i] = ((double *)refA)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refB_size; i++) {
      ((double *)arrayB)[i] = ((double *)refB)[i]; 
    }
    for (uint64_t i = 0; i < 2 * refC_size; i++) {
      ((double *)arrayC)[i] = ((double *)refC)[i];
      ((double *)arrayCcopy)[i] = ((double *)refC)[i]; 
    }
    break;
  default :
    errorQuda("Unrecognised data type %d\n", cublas_param.data_type);
  }
  
  // Perform GPU GEMM Blas operation
  cublasGEMMQuda(arrayA, arrayB, arrayC, &cublas_param);
  
  if(verify_results) {

    // Copy data from problem sized array to reference sized array.
    void *checkA = malloc(refA_size * re_im * data_size);
    void *checkB = malloc(refB_size * re_im * data_size);
    void *checkC = malloc(refC_size * re_im * data_size);
    void *checkCcopy = malloc(refC_size * re_im * data_size);

    memset(checkA, 0, refA_size * re_im * data_size);
    memset(checkB, 0, refB_size * re_im * data_size);
    memset(checkC, 0, refC_size * re_im * data_size);
    memset(checkCcopy, 0, refC_size * re_im * data_size);
    
    switch (cublas_param.data_type) {
    case QUDA_CUBLAS_DATATYPE_S :
      for (uint64_t i = 0; i < 2 * refA_size; i+=2) {
	((double *)checkA)[i] = ((float *)arrayA)[i/2]; 
      }
      for (uint64_t i = 0; i < 2 * refB_size; i+=2) {
	((double *)checkB)[i] = ((float *)arrayB)[i/2]; 
      }
      for (uint64_t i = 0; i < 2 * refC_size; i+=2) {
	((double *)checkC)[i] = ((float *)arrayC)[i/2]; 
	((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i/2]; 
      }
      break;      
    case QUDA_CUBLAS_DATATYPE_D :
      for (uint64_t i = 0; i < 2 * refA_size; i+=2) {
	((double *)checkA)[i] = ((double *)arrayA)[i/2]; 
      }
      for (uint64_t i = 0; i < 2 * refB_size; i+=2) {
	((double *)checkB)[i] = ((double *)arrayB)[i/2]; 
      }
      for (uint64_t i = 0; i < 2 * refC_size; i+=2) {
	((double *)checkC)[i] = ((double *)arrayC)[i/2]; 
	((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i/2]; 
      }
      break;      
    case QUDA_CUBLAS_DATATYPE_C :
      for (uint64_t i = 0; i < 2 * refA_size; i++) {
	((double *)checkA)[i] = ((float *)arrayA)[i]; 
      }
      for (uint64_t i = 0; i < 2 * refB_size; i++) {
	((double *)checkB)[i] = ((float *)arrayB)[i]; 
      }
      for (uint64_t i = 0; i < 2 * refC_size; i++) {
	((double *)checkC)[i] = ((float *)arrayC)[i];
	((double *)checkCcopy)[i] = ((float *)arrayCcopy)[i]; 
      }
      break;
    case QUDA_CUBLAS_DATATYPE_Z :
      for (uint64_t i = 0; i < 2 * refA_size; i++) {
	((double *)checkA)[i] = ((double *)arrayA)[i]; 
      }
      for (uint64_t i = 0; i < 2 * refB_size; i++) {
	((double *)checkB)[i] = ((double *)arrayB)[i]; 
      }
      for (uint64_t i = 0; i < 2 * refC_size; i++) {
	((double *)checkC)[i] = ((double *)arrayC)[i];
	((double *)checkCcopy)[i] = ((double *)arrayCcopy)[i]; 
      }
      break;
    default :
      errorQuda("Unrecognised data type %d\n", cublas_param.data_type);
    }
    
    cublasGEMMQudaVerify(checkA, checkB, checkCcopy, checkC, &cublas_param);
    
    free(checkA);
    free(checkB);
    free(checkC);
    free(checkCcopy);
  }

  free(refA);
  free(refB);
  free(refC);
  free(refCcopy);
  
  free(arrayA);
  free(arrayB);
  free(arrayC);
  free(arrayCcopy);

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
