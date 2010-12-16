#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <test_util.h>

// volume per GPU
const int LX = 24;
const int LY = 24;
const int LZ = 24;
const int LT = 24;

// corresponds to 10 iterations for V=24^4 at half precision
const int Niter = 10 * 331776 / (LX * LY * LZ * LT);

const int Nkernels = 24;
const int ThreadMin = 32;
const int ThreadMax = 1024;
const int GridMin = 1;
const int GridMax = 65536;

cudaColorSpinorField *x, *y, *z, *w, *v, *h, *l;

// defines blas_threads[][] and blas_blocks[][]
#include "../lib/blas_param.h"


void setPrec(ColorSpinorParam &param, const QudaPrecision precision)
{
  param.precision = precision;
  if (precision == QUDA_DOUBLE_PRECISION) {
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  } else {
    param.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
  }
}


// returns true if the specified kernel performs a reduction
bool isReduction(int kernel)
{
  return (kernel > 13);
}


void initFields(int prec)
{
  // precisions used for the source field in the copyCuda() benchmark
  QudaPrecision high_aux_prec;
  QudaPrecision low_aux_prec;

  ColorSpinorParam param;
  param.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  param.nColor = 3;
  param.nSpin = 4; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  param.nDim = 4; // number of spacetime dimensions
  param.x[0] = LX;
  param.x[1] = LY;
  param.x[2] = LZ;
  param.x[3] = LT;

  param.pad = LX*LY*LZ/2;
  param.siteSubset = QUDA_PARITY_SITE_SUBSET;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  
  param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  param.create = QUDA_NULL_FIELD_CREATE;

  switch(prec) {
  case 0:
    setPrec(param, QUDA_HALF_PRECISION);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    low_aux_prec = QUDA_SINGLE_PRECISION;
    break;
  case 1:
    setPrec(param, QUDA_SINGLE_PRECISION);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    low_aux_prec = QUDA_HALF_PRECISION;
    break;
  case 2:
    setPrec(param, QUDA_DOUBLE_PRECISION);
    high_aux_prec = QUDA_SINGLE_PRECISION;
    low_aux_prec = QUDA_HALF_PRECISION;
    break;
  }

  v = new cudaColorSpinorField(param);
  checkCudaError();

  w = new cudaColorSpinorField(param);
  x = new cudaColorSpinorField(param);
  y = new cudaColorSpinorField(param);
  z = new cudaColorSpinorField(param);

  setPrec(param, high_aux_prec);
  h = new cudaColorSpinorField(param);

  setPrec(param, low_aux_prec);
  l = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  // turn off error checking in blas kernels
  setBlasTuning(1);
}


void freeFields()
{
  // release memory
  delete v;
  delete w;
  delete x;
  delete y;
  delete z;
  delete h;
  delete l;
}


double benchmark(int kernel, int niter) {

  double a, b, c;
  Complex a2, b2;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i=0; i < niter; ++i) {

    switch (kernel) {

    case 0:
      copyCuda(*y, *h);
      break;

    case 1:
      copyCuda(*y, *l);
      break;
      
    case 2:
      axpbyCuda(a, *x, b, *y);
      break;

    case 3:
      xpyCuda(*x, *y);
      break;

    case 4:
      axpyCuda(a, *x, *y);
      break;

    case 5:
      xpayCuda(*x, a, *y);
      break;

    case 6:
      mxpyCuda(*x, *y);
      break;

    case 7:
      axCuda(a, *x);
      break;

    case 8:
      caxpyCuda(a2, *x, *y);
      break;

    case 9:
      caxpbyCuda(a2, *x, b2, *y);
      break;

    case 10:
      cxpaypbzCuda(*x, a2, *y, b2, *z);
      break;

    case 11:
      axpyBzpcxCuda(a, *x, *y, b, *z, c);
      break;

    case 12:
      axpyZpbxCuda(a, *x, *y, *z, b);
      break;

    case 13:
      caxpbypzYmbwCuda(a2, *x, b2, *y, *z, *w);
      break;
      
    // double
    case 14:
      sumCuda(*x);
      break;

    case 15:
      normCuda(*x);
      break;

    case 16:
      reDotProductCuda(*x, *y);
      break;

    case 17:
      axpyNormCuda(a, *x, *y);
      break;

    case 18:
      xmyNormCuda(*x, *y);
      break;
      
    // double2
    case 19:
      cDotProductCuda(*x, *y);
      break;

    case 20:
      xpaycDotzyCuda(*x, a, *y, *z);
      break;
      
    // double3
    case 21:
      cDotProductNormACuda(*x, *y);
      break;

    case 22:
      cDotProductNormBCuda(*x, *y);
      break;

    case 23:
      caxpbypzYmbwcDotProductWYNormYCuda(a2, *x, b2, *y, *z, *w, *v);
      break;

    default:
      errorQuda("Undefined blas kernel %d\n", kernel);
    }
  }
  
  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000;
  return secs;
}


void write(char *names[], int threads[][3], int blocks[][3])
{
  printf("\nWriting optimal parameters to blas_param.h\n");

  FILE *fp = fopen("blas_param.h", "w");
  fprintf(fp, "//\n// Auto-tuned blas CUDA parameters, generated by blas_test\n//\n\n");

  fprintf(fp, "static int blas_threads[%d][3] = {\n", Nkernels);

  for (int i=0; i<Nkernels; i++) {
    fprintf(fp, "  {%4d, %4d, %4d}%c  // Kernel %2d: %s\n", threads[i][0], threads[i][1], threads[i][2],
	    ((i == Nkernels-1) ? ' ' : ','), i, names[i]);
  }
  fprintf(fp, "};\n\n");

  fprintf(fp, "static int blas_blocks[%d][3] = {\n", Nkernels);

  for (int i=0; i<Nkernels; i++) {
    fprintf(fp, "  {%5d, %5d, %5d}%c  // Kernel %2d: %s\n", blocks[i][0], blocks[i][1], blocks[i][2],
	    ((i == Nkernels-1) ? ' ' : ','), i, names[i]);
  }
  fprintf(fp, "};\n");

  fclose(fp);
}


int main(int argc, char** argv)
{
  int dev = 0;
  if (argc == 2) dev = atoi(argv[1]);
  initQuda(dev);

  char *names[] = {
    "copyCuda (high source precision)",
    "copyCuda (low source precision)",
    "axpbyCuda",
    "xpyCuda",
    "axpyCuda",
    "xpayCuda",
    "mxpyCuda",
    "axCuda",
    "caxpyCuda",
    "caxpbyCuda",
    "cxpaypbzCuda",
    "axpyBzpcxCuda",
    "axpyZpbxCuda",
    "caxpbypzYmbwCuda",
    "sumCuda",
    "normCuda",
    "reDotProductCuda",
    "axpyNormCuda",
    "xmyNormCuda",
    "cDotProductCuda",
    "xpaycDotzyCuda",
    "cDotProductNormACuda",
    "cDotProductNormBCuda",
    "caxpbypzYmbwcDotProductWYNormYCuda"
  };

  char *prec_str[] = {"half", "single", "double"};
  
  // Only benchmark double precision if supported
#if (__CUDA_ARCH__ >= 130)
  int Nprec = 3;
#else
  int Nprec = 2;
#endif

  int niter = Niter;

  for (int prec = 0; prec < Nprec; prec++) {

    printf("\nBenchmarking %s precision with %d iterations...\n\n", prec_str[prec], niter);
    initFields(prec);

    for (int kernel = 0; kernel < Nkernels; kernel++) {

      double gflops_max = 0.0;
      double gbytes_max = 0.0;
      int threads_max = 0; 
      int blocks_max = 0;

      cudaError_t error;

      // only benchmark "high precision" copyCuda() if double is supported
      if ((Nprec < 3) && (kernel == 0)) continue;

      for (unsigned int thread = ThreadMin; thread <= ThreadMax; thread+=32) {

	// for reduction kernels, the number of threads must be a power of two
	if (isReduction(kernel) && (thread & (thread-1))) continue;

	for (unsigned int grid = GridMin; grid <= GridMax; grid *= 2) {
	  setBlasParam(kernel, prec, thread, grid);
	  
	  // first do warmup run
	  benchmark(kernel, 1);
	  
	  blas_quda_flops = 0;
	  blas_quda_bytes = 0;

	  double secs = benchmark(kernel, niter);
	  error = cudaGetLastError();
	  double flops = blas_quda_flops;
	  double bytes = blas_quda_bytes;
	  
	  double gflops = (flops*1e-9)/(secs);
	  double gbytes = bytes/(secs*(1<<30));

	  // prevents selection of failed parameters
	  if (gbytes > gbytes_max && error == cudaSuccess) { 
	    gflops_max = gflops;
	    gbytes_max = gbytes;
	    threads_max = thread;
	    blocks_max = grid;
	  }
	  //printf("%d %d %-35s %f s, flops = %e, Gflop/s = %f, GiB/s = %f\n", 
	  // thread, grid, names[kernel], secs, flops, gflops, gbytes);
	}
      }

      if (threads_max == 0) {
	errorQuda("Autotuning failed for %s kernel: %s", names[kernel], cudaGetErrorString(error));
      } else {
	// now rerun with more iterations to get accurate speed measurements
	setBlasParam(kernel, prec, threads_max, blocks_max);
	benchmark(kernel, 1);
	blas_quda_flops = 0;
	blas_quda_bytes = 0;
	
	double secs = benchmark(kernel, 100*niter);
	
	gflops_max = (blas_quda_flops*1e-9)/(secs);
	gbytes_max = blas_quda_bytes/(secs*(1<<30));
      }

      printf("%-35s: %4d threads per block, %5d blocks per grid, Gflop/s = %8.4f, GiB/s = %8.4f\n", 
	     names[kernel], threads_max, blocks_max, gflops_max, gbytes_max);

      blas_threads[kernel][prec] = threads_max;
      blas_blocks[kernel][prec] = blocks_max;
    }
    freeFields();

    // halve the number of iterations for the next precision
    niter /= 2; 
    if (niter==0) niter = 1;
  }
  write(names, blas_threads, blas_blocks);
  endQuda();
}
