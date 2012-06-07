#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <test_util.h>
#include <face_quda.h>

// include because of nasty globals used in the tests
#include <dslash_util.h>

// Wilson, clover-improved Wilson, and twisted mass are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern int niter;

extern void usage(char** );

const int Nkernels = 31;

cpuColorSpinorField *xH, *yH, *zH, *wH, *vH, *hH, *lH;
cudaColorSpinorField *xD, *yD, *zD, *wD, *vD, *hD, *lD;
int Nspin;

void setPrec(ColorSpinorParam &param, const QudaPrecision precision)
{
  param.precision = precision;
  if (Nspin == 1 || precision == QUDA_DOUBLE_PRECISION) {
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  } else {
    param.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
  }
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("S_dimension T_dimension Nspin\n");
  printfQuda("%d/%d/%d        %d      %d\n", xdim, ydim, zdim, tdim, Nspin);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3)); 
  
  return;  
}

void initFields(int prec)
{
  // precisions used for the source field in the copyCuda() benchmark
  QudaPrecision high_aux_prec;
  QudaPrecision low_aux_prec;

  ColorSpinorParam param;
  param.nColor = 3;
  // set spin according to the type of dslash
  Nspin = (dslash_type == QUDA_ASQTAD_DSLASH) ? 1 : 4;
  param.nSpin = Nspin;
  param.nDim = 4; // number of spacetime dimensions

  param.pad = 0; // padding must be zero for cpu fields
  param.siteSubset = QUDA_PARITY_SITE_SUBSET;
  if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) param.x[0] = xdim/2;
  else param.x[0] = xdim;
  param.x[1] = ydim;
  param.x[2] = zdim;
  param.x[3] = tdim;

  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.precision = QUDA_DOUBLE_PRECISION;
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

  param.create = QUDA_ZERO_FIELD_CREATE;

  vH = new cpuColorSpinorField(param);
  wH = new cpuColorSpinorField(param);
  xH = new cpuColorSpinorField(param);
  yH = new cpuColorSpinorField(param);
  zH = new cpuColorSpinorField(param);
  hH = new cpuColorSpinorField(param);
  lH = new cpuColorSpinorField(param);

  vH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  wH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  xH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  yH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  zH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  hH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  lH->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);

  // Now set the parameters for the cuda fields
  param.pad = 0; //LX*LY*LZ/2;
  
  if (param.nSpin == 4) param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  param.create = QUDA_ZERO_FIELD_CREATE;

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

  checkCudaError();

  vD = new cudaColorSpinorField(param);
  wD = new cudaColorSpinorField(param);
  xD = new cudaColorSpinorField(param);
  yD = new cudaColorSpinorField(param);
  zD = new cudaColorSpinorField(param);

  setPrec(param, high_aux_prec);
  hD = new cudaColorSpinorField(param);

  setPrec(param, low_aux_prec);
  lD = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  *vD = *vH;
  *wD = *wH;
  *xD = *xH;
  *yD = *yH;
  *zD = *zH;
  *hD = *hH;
  *lD = *lH;
}


void freeFields()
{

  // release memory
  delete vD;
  delete wD;
  delete xD;
  delete yD;
  delete zD;
  delete hD;
  delete lD;

  // release memory
  delete vH;
  delete wH;
  delete xH;
  delete yH;
  delete zH;
  delete hH;
  delete lH;
}


double benchmark(int kernel, const int niter) {

  double a, b, c;
  quda::Complex a2, b2, c2;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  for (int i=0; i < niter; ++i) {

    switch (kernel) {

    case 0:
      copyCuda(*yD, *hD);
      break;

    case 1:
      copyCuda(*yD, *lD);
      break;
      
    case 2:
      axpbyCuda(a, *xD, b, *yD);
      break;

    case 3:
      xpyCuda(*xD, *yD);
      break;

    case 4:
      axpyCuda(a, *xD, *yD);
      break;

    case 5:
      xpayCuda(*xD, a, *yD);
      break;

    case 6:
      mxpyCuda(*xD, *yD);
      break;

    case 7:
      axCuda(a, *xD);
      break;

    case 8:
      caxpyCuda(a2, *xD, *yD);
      break;

    case 9:
      caxpbyCuda(a2, *xD, b2, *yD);
      break;

    case 10:
      cxpaypbzCuda(*xD, a2, *yD, b2, *zD);
      break;

    case 11:
      axpyBzpcxCuda(a, *xD, *yD, b, *zD, c);
      break;

    case 12:
      axpyZpbxCuda(a, *xD, *yD, *zD, b);
      break;

    case 13:
      caxpbypzYmbwCuda(a2, *xD, b2, *yD, *zD, *wD);
      break;
      
    case 14:
      cabxpyAxCuda(a, b2, *xD, *yD);
      break;

    case 15:
      caxpbypzCuda(a2, *xD, b2, *yD, *zD);
      break;

    case 16:
      caxpbypczpwCuda(a2, *xD, b2, *yD, c2, *zD, *wD);
      break;

    case 17:
      caxpyXmazCuda(a2, *xD, *yD, *zD);
      break;

      // double
    case 18:
      normCuda(*xD);
      break;

    case 19:
      reDotProductCuda(*xD, *yD);
      break;

    case 20:
      axpyNormCuda(a, *xD, *yD);
      break;

    case 21:
      xmyNormCuda(*xD, *yD);
      break;
      
    case 22:
      caxpyNormCuda(a2, *xD, *yD);
      break;

    case 23:
      caxpyXmazNormXCuda(a2, *xD, *yD, *zD);
      break;

    case 24:
      cabxpyAxNormCuda(a, b2, *xD, *yD);
      break;

    // double2
    case 25:
      cDotProductCuda(*xD, *yD);
      break;

    case 26:
      xpaycDotzyCuda(*xD, a, *yD, *zD);
      break;
      
    case 27:
      caxpyDotzyCuda(a2, *xD, *yD, *zD);
      break;

    // double3
    case 28:
      cDotProductNormACuda(*xD, *yD);
      break;

    case 29:
      cDotProductNormBCuda(*xD, *yD);
      break;

    case 30:
      caxpbypzYmbwcDotProductUYNormYCuda(a2, *xD, b2, *yD, *zD, *wD, *vD);
      break;

    default:
      errorQuda("Undefined blas kernel %d\n", kernel);
    }
  }
  
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000;
  return secs;
}

#define ERROR(a) fabs(norm2(*a##D) - norm2(*a##H)) / norm2(*a##H)

double test(int kernel) {

  double a = 1.5, b = 2.5, c = 3.5;
  quda::Complex a2(a, b), b2(b, -c), c2(a+b, c*a);
  double error = 0;

  switch (kernel) {

  case 0:
    *hD = *hH;
    copyCuda(*yD, *hD);
    yH->copy(*hH);
    error = ERROR(y);
    break;

  case 1:
    *lD = *lH;
    copyCuda(*yD, *lD);
    yH->copy(*lH);
    error = ERROR(y);
    break;
      
  case 2:
    *xD = *xH;
    *yD = *yH;
    axpbyCuda(a, *xD, b, *yD);
    axpbyCpu(a, *xH, b, *yH);
    error = ERROR(y);
    break;

  case 3:
    *xD = *xH;
    *yD = *yH;
    xpyCuda(*xD, *yD);
    xpyCpu(*xH, *yH);
    error = ERROR(y);
    break;

  case 4:
    *xD = *xH;
    *yD = *yH;
    axpyCuda(a, *xD, *yD);
    axpyCpu(a, *xH, *yH);
    error = ERROR(y);
    break;

  case 5:
    *xD = *xH;
    *yD = *yH;
    xpayCuda(*xD, a, *yD);
    xpayCpu(*xH, a, *yH);
    error = ERROR(y);
    break;

  case 6:
    *xD = *xH;
    *yD = *yH;
    mxpyCuda(*xD, *yD);
    mxpyCpu(*xH, *yH);
    error = ERROR(y);
    break;

  case 7:
    *xD = *xH;
    axCuda(a, *xD);
    axCpu(a, *xH);
    error = ERROR(x);
    break;

  case 8:
    *xD = *xH;
    *yD = *yH;
    caxpyCuda(a2, *xD, *yD);
    caxpyCpu(a2, *xH, *yH);
    error = ERROR(y);
    break;

  case 9:
    *xD = *xH;
    *yD = *yH;
    caxpbyCuda(a2, *xD, b2, *yD);
    caxpbyCpu(a2, *xH, b2, *yH);
    error = ERROR(y);
    break;

  case 10:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    cxpaypbzCuda(*xD, a2, *yD, b2, *zD);
    cxpaypbzCpu(*xH, a2, *yH, b2, *zH);
    error = ERROR(z);
    break;

  case 11:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    axpyBzpcxCuda(a, *xD, *yD, b, *zD, c);
    axpyBzpcxCpu(a, *xH, *yH, b, *zH, c);
    error = ERROR(x) + ERROR(y);
    break;

  case 12:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    axpyZpbxCuda(a, *xD, *yD, *zD, b);
    axpyZpbxCpu(a, *xH, *yH, *zH, b);
    error = ERROR(x) + ERROR(y);
    break;

  case 13:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    caxpbypzYmbwCuda(a2, *xD, b2, *yD, *zD, *wD);
    caxpbypzYmbwCpu(a2, *xH, b2, *yH, *zH, *wH);
    error = ERROR(z) + ERROR(y);
    break;
      
  case 14:
    *xD = *xH;
    *yD = *yH;
    cabxpyAxCuda(a, b2, *xD, *yD);
    cabxpyAxCpu(a, b2, *xH, *yH);
    error = ERROR(y) + ERROR(x);
    break;

  case 15:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {caxpbypzCuda(a2, *xD, b2, *yD, *zD);
      caxpbypzCpu(a2, *xH, b2, *yH, *zH);
      error = ERROR(z); }
    break;
    
  case 16:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    {caxpbypczpwCuda(a2, *xD, b2, *yD, c2, *zD, *wD);
      caxpbypczpwCpu(a2, *xH, b2, *yH, c2, *zH, *wH);
      error = ERROR(w); }
    break;

  case 17:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {caxpyXmazCuda(a, *xD, *yD, *zD);
     caxpyXmazCpu(a, *xH, *yH, *zH);
     error = ERROR(y) + ERROR(x);}
    break;

    // double
  case 18:
    *xD = *xH;
    error = fabs(normCuda(*xD) - normCpu(*xH)) / normCpu(*xH);
    break;
    
  case 19:
    *xD = *xH;
    *yD = *yH;
    error = fabs(reDotProductCuda(*xD, *yD) - reDotProductCpu(*xH, *yH)) / fabs(reDotProductCpu(*xH, *yH));
    break;

  case 20:
    *xD = *xH;
    *yD = *yH;
    {double d = axpyNormCuda(a, *xD, *yD);
    double h = axpyNormCpu(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 21:
    *xD = *xH;
    *yD = *yH;
    {double d = xmyNormCuda(*xD, *yD);
    double h = xmyNormCpu(*xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;
    
  case 22:
    *xD = *xH;
    *yD = *yH;
    {double d = caxpyNormCuda(a, *xD, *yD);
    double h = caxpyNormCpu(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 23:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {double d = caxpyXmazNormXCuda(a, *xD, *yD, *zD);
      double h = caxpyXmazNormXCpu(a, *xH, *yH, *zH);
      error = ERROR(y) + ERROR(x) + fabs(d-h)/fabs(h);}
    break;

  case 24:
    *xD = *xH;
    *yD = *yH;
    {double d = cabxpyAxNormCuda(a, b2, *xD, *yD);
      double h = cabxpyAxNormCpu(a, b2, *xH, *yH);
      error = ERROR(x) + ERROR(y) + fabs(d-h)/fabs(h);}
    break;

    // double2
  case 25:
    *xD = *xH;
    *yD = *yH;
    error = abs(cDotProductCuda(*xD, *yD) - cDotProductCpu(*xH, *yH)) / abs(cDotProductCpu(*xH, *yH));
    break;
    
  case 26:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    { quda::Complex d = xpaycDotzyCuda(*xD, a, *yD, *zD);
      quda::Complex h = xpaycDotzyCpu(*xH, a, *yH, *zH);
      error =  fabs(norm2(*yD) - norm2(*yH)) / norm2(*yH) + abs(d-h)/abs(h);
    }
    break;
    
  case 27:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {quda::Complex d = caxpyDotzyCuda(a, *xD, *yD, *zD);
      quda::Complex h = caxpyDotzyCpu(a, *xH, *yH, *zH);
    error = ERROR(y) + abs(d-h)/abs(h);}
    break;

    // double3
  case 28:
    *xD = *xH;
    *yD = *yH;
    { double3 d = cDotProductNormACuda(*xD, *yD);
      double3 h = cDotProductNormACpu(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;
    
  case 29:
    *xD = *xH;
    *yD = *yH;
    { double3 d = cDotProductNormBCuda(*xD, *yD);
      double3 h = cDotProductNormBCpu(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;
    
  case 30:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    *vD = *vH;
    { double3 d = caxpbypzYmbwcDotProductUYNormYCuda(a2, *xD, b2, *yD, *zD, *wD, *vD);
      double3 h = caxpbypzYmbwcDotProductUYNormYCpu(a2, *xH, b2, *yH, *zH, *wH, *vH);
      error = ERROR(z) + ERROR(y) + fabs(d.x - h.x) / fabs(h.x) + 
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  default:
    errorQuda("Undefined blas kernel %d\n", kernel);
  }

  return error;
}

int main(int argc, char** argv)
{
  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  setSpinorSiteSize(24);
  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
  display_test_info();
  initQuda(device);

  char *names[] = {
    "copyHS",
    "copyLS",
    "axpby",
    "xpy",
    "axpy",
    "xpay",
    "mxpy",
    "ax",
    "caxpy",
    "caxpby",
    "cxpaypbz",
    "axpyBzpcx",
    "axpyZpbx",
    "caxpbypzYmbw",
    "cabxpyAx",
    "caxpbypz",
    "caxpbypczpw",
    "caxpyXmaz",
    "norm",
    "reDotProduct",
    "axpyNorm",
    "xmyNorm",
    "caxpyNorm",
    "caxpyXmazNormX",
    "cabxpyAxNorm",
    "cDotProduct",
    "xpaycDotzy",
    "caxpyDotzy",
    "cDotProductNormA",
    "cDotProductNormB",
    "caxpbypzYmbwcDotProductWYNormY"
  };

  char *prec_str[] = {"half", "single", "double"};
  
  // Only benchmark double precision if supported
#if (__COMPUTE_CAPABILITY__ >= 130)
  int Nprec = 3;
#else
  int Nprec = 2;
#endif

  // enable the tuning
  quda::setBlasTuning(QUDA_TUNE_YES, QUDA_SILENT);

  for (int prec = 0; prec < Nprec; prec++) {

    printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", prec_str[prec], niter);
    initFields(prec);

    for (int kernel = 0; kernel < Nkernels; kernel++) {
      // only benchmark "high precision" copyCuda() if double is supported
      if ((Nprec < 3) && (kernel == 0)) continue;

      // do the initial tune
      benchmark(kernel, 1);
    
      // now rerun with more iterations to get accurate speed measurements
      quda::blas_flops = 0;
      quda::blas_bytes = 0;
      
      double secs = benchmark(kernel, niter);
      
      double gflops = (quda::blas_flops*1e-9)/(secs);
      double gbytes = quda::blas_bytes/(secs*1e9);
    
      printfQuda("%-31s: Gflop/s = %6.1f, GB/s = %6.1f\n", names[kernel], gflops, gbytes);
    }
    freeFields();
  }

  // clear the error state
  cudaGetLastError();

  // lastly check for correctness
  for (int prec = 0; prec < Nprec; prec++) {
    printfQuda("\nTesting %s precision...\n\n", prec_str[prec]);
    initFields(prec);
    
    for (int kernel = 0; kernel < Nkernels; kernel++) {
      // only benchmark "high precision" copyCuda() if double is supported
      if ((Nprec < 3) && (kernel == 0)) continue;
      double error = test(kernel);
      printfQuda("%-35s error = %e, \n", names[kernel], error);
    }
    freeFields();
  }

  endQuda();

  endCommsQuda();
}
