#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <blas_quda.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 1;

QudaInvertParam inv_param;

ParitySpinor x, y, z, w, v;

int nIters;

void init() {

  int X[4];

  X[0] = 32;
  X[1] = 32;
  X[2] = 32;
  X[3] = 32;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_SINGLE_PRECISION;
  inv_param.verbosity = QUDA_VERBOSE;

  invert_param = &inv_param;

  int dev = 0;
  initQuda(dev);

  // need single parity dimensions
  X[0] /= 2;
  v = allocateParitySpinor(X, inv_param.cuda_prec);
  w = allocateParitySpinor(X, inv_param.cuda_prec);
  x = allocateParitySpinor(X, inv_param.cuda_prec);
  y = allocateParitySpinor(X, inv_param.cuda_prec);
  z = allocateParitySpinor(X, inv_param.cuda_prec);

}

void end() {
  // release memory
  freeParitySpinor(v);
  freeParitySpinor(w);
  freeParitySpinor(x);
  freeParitySpinor(y);
  freeParitySpinor(z);
  endQuda();
}

double benchmark(int kernel) {

  double a, b;
  double2 a2, b2;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  for (int i=0; i < nIters; ++i) {
    switch (kernel) {

    case 0:
      axpbyCuda(a, x, b, y);
      break;

    case 1:
      xpyCuda(x, y);
      break;

    case 2:
      axpyCuda(a, x, y);
      break;

    case 3:
      xpayCuda(x, a, y);
      break;

    case 4:
      mxpyCuda(x, y);
      break;

    case 5:
      axCuda(a, x);
      break;

    case 6:
      caxpyCuda(a2, x, y);
      break;

    case 7:
      caxpbyCuda(a2, x, b2, y);
      break;

    case 8:
      cxpaypbzCuda(x, a2, y, b2, z);
      break;

    case 9:
      axpyZpbxCuda(a, x, y, z, b);
      break;

    case 10:
      caxpbypzYmbwCuda(a2, x, b2, y, z, w);
      break;
      
      // double
    case 11:
      sumCuda(x);
      break;

    case 12:
      normCuda(x);
      break;

    case 13:
      reDotProductCuda(x, y);
      break;

    case 14:
      axpyNormCuda(a, x, y);
      break;

    case 15:
      xmyNormCuda(x, y);
      break;
      
      // double2
    case 16:
      cDotProductCuda(x, y);
      break;

    case 17:
      xpaycDotzyCuda(x, a, y, z);
      break;
      
      // double3
    case 18:
      cDotProductNormACuda(x, y);
      break;

    case 19:
      cDotProductNormBCuda(x, y);
      break;

    case 20:
      caxpbypzYmbwcDotProductWYNormYQuda(a2, x, b2, y, z, w, v);
      break;
      
    default:
      printf("Undefined blas kernel %d\n", kernel);
      exit(1);
    }
  }
  
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  double secs = runTime / 1000;
  return secs;
}


int main(int argc, char** argv) {
  init();

  int kernels[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  char names[][100] = {
      "axpbyCuda(a, x, b, y):                                     ",
      "xpyCuda(x, y):                                             ",
      "axpyCuda(a, x, y):                                         ",
      "xpayCuda(x, a, y):                                         ",
      "mxpyCuda(x, y):                                            ",
      "axCuda(a, x):                                              ",
      "caxpyCuda(a2, x, y):                                       ",
      "caxpbyCuda(a2, x, b2, y):                                  ",
      "cxpaypbzCuda(x, a2, y, b2, z):                             ",
      "axpyZpbxCuda(a, x, y, z, b):                               ",
      "caxpbypzYmbwCuda(a2, x, b2, y, z, w):                      ",
      "sumCuda(x):                                                ",
      "normCuda(x):                                               ",
      "reDotProductCuda(x, y):                                    ",
      "axpyNormCuda(a, x, y):                                     ",
      "xmyNormCuda(x, y):                                         ",
      "cDotProductCuda(x, y):                                     ",
      "xpaycDotzyCuda(x, a, y, z):                                ",
      "cDotProductNormACuda(x, y):                                ",
      "cDotProductNormBCuda(x, y):                                ",
      "caxpbypzYmbwcDotProductWYNormYQuda(a2, x, b2, y, z, w, v): "
  };

  nIters = 1;
  // first do warmup run
  for (int i = 0; i <= 20; i++) {
    benchmark(kernels[i]);
  }

  nIters = 1000;
  for (int i = 0; i <= 20; i++) {
    blas_quda_flops = 0;
    blas_quda_bytes = 0;
    double secs = benchmark(kernels[i]);
    double flops = blas_quda_flops;
    double bytes = blas_quda_bytes;
    printf("%s %f s, flops = %e, Gflops/s = %f, GiB/s = %f\n\n", 
	   names[i], secs, flops, (flops*1e-9)/(secs), bytes/(secs*(1<<30)));
    //printf("Bandwidth:    %f GiB/s\n\n", GiB / secs);
  }
}


