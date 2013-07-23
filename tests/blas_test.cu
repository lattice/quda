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

extern bool tune;

extern void usage(char** );

#if (__COMPUTE_CAPABILITY__ >= 200)
const int Nkernels = 32;
#else // exclude Heavy Quark Norm if on Tesla architecture
const int Nkernels = 31;
#endif

using namespace quda;

ColorSpinorField *xH, *yH, *zH, *wH, *vH, *hH, *lH;
ColorSpinorField *xD, *yD, *zD, *wD, *vD, *hD, *lD;
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
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
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

  static_cast<cpuColorSpinorField*>(vH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(wH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(xH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(yH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(zH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(hH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(lH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);

  // Now set the parameters for the cuda fields
  //param.pad = xdim*ydim*zdim/2;
  
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
      blas::copy(*yD, *hD);
      break;

    case 1:
      blas::copy(*yD, *lD);
      break;
      
    case 2:
      blas::axpby(a, *xD, b, *yD);
      break;

    case 3:
      blas::xpy(*xD, *yD);
      break;

    case 4:
      blas::axpy(a, *xD, *yD);
      break;

    case 5:
      blas::xpay(*xD, a, *yD);
      break;

    case 6:
      blas::mxpy(*xD, *yD);
      break;

    case 7:
      blas::ax(a, *xD);
      break;

    case 8:
      blas::caxpy(a2, *xD, *yD);
      break;

    case 9:
      blas::caxpby(a2, *xD, b2, *yD);
      break;

    case 10:
      blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
      break;

    case 11:
      blas::axpyBzpcx(a, *xD, *yD, b, *zD, c);
      break;

    case 12:
      blas::axpyZpbx(a, *xD, *yD, *zD, b);
      break;

    case 13:
      blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
      break;
      
    case 14:
      blas::cabxpyAx(a, b2, *xD, *yD);
      break;

    case 15:
      blas::caxpbypz(a2, *xD, b2, *yD, *zD);
      break;

    case 16:
      blas::caxpbypczpw(a2, *xD, b2, *yD, c2, *zD, *wD);
      break;

    case 17:
      blas::caxpyXmaz(a2, *xD, *yD, *zD);
      break;

      // double
    case 18:
      blas::norm2(*xD);
      break;

    case 19:
      blas::reDotProduct(*xD, *yD);
      break;

    case 20:
      blas::axpyNorm(a, *xD, *yD);
      break;

    case 21:
      blas::xmyNorm(*xD, *yD);
      break;
      
    case 22:
      blas::caxpyNorm(a2, *xD, *yD);
      break;

    case 23:
      blas::caxpyXmazNormX(a2, *xD, *yD, *zD);
      break;

    case 24:
      blas::cabxpyAxNorm(a, b2, *xD, *yD);
      break;

    // double2
    case 25:
      blas::cDotProduct(*xD, *yD);
      break;

    case 26:
      blas::xpaycDotzy(*xD, a, *yD, *zD);
      break;
      
    case 27:
      blas::caxpyDotzy(a2, *xD, *yD, *zD);
      break;

    // double3
    case 28:
      blas::cDotProductNormA(*xD, *yD);
      break;

    case 29:
      blas::cDotProductNormB(*xD, *yD);
      break;

    case 30:
      blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zD, *wD, *vD);
      break;

    case 31:
      blas::HeavyQuarkResidualNorm(*xD, *yD);
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

#define ERROR(a) fabs(blas::norm2(*a##D) - blas::norm2(*a##H)) / blas::norm2(*a##H)

double test(int kernel) {

  double a = 1.5, b = 2.5, c = 3.5;
  quda::Complex a2(a, b), b2(b, -c), c2(a+b, c*a);
  double error = 0;

  switch (kernel) {

  case 0:
    *hD = *hH;
    blas::copy(*yD, *hD);
    blas::copy(*yH, *hH);
    error = ERROR(y);
    break;

  case 1:
    *lD = *lH;
    blas::copy(*yD, *lD);
    blas::copy(*yH, *lH);
    error = ERROR(y);
    break;
      
  case 2:
    *xD = *xH;
    *yD = *yH;
    blas::axpby(a, *xD, b, *yD);
    blas::axpby(a, *xH, b, *yH);
    error = ERROR(y);
    break;

  case 3:
    *xD = *xH;
    *yD = *yH;
    blas::xpy(*xD, *yD);
    blas::xpy(*xH, *yH);
    error = ERROR(y);
    break;

  case 4:
    *xD = *xH;
    *yD = *yH;
    blas::axpy(a, *xD, *yD);
    blas::axpy(a, *xH, *yH);
    error = ERROR(y);
    break;

  case 5:
    *xD = *xH;
    *yD = *yH;
    blas::xpay(*xD, a, *yD);
    blas::xpay(*xH, a, *yH);
    error = ERROR(y);
    break;

  case 6:
    *xD = *xH;
    *yD = *yH;
    blas::mxpy(*xD, *yD);
    blas::mxpy(*xH, *yH);
    error = ERROR(y);
    break;

  case 7:
    *xD = *xH;
    blas::ax(a, *xD);
    blas::ax(a, *xH);
    error = ERROR(x);
    break;

  case 8:
    *xD = *xH;
    *yD = *yH;
    blas::caxpy(a2, *xD, *yD);
    blas::caxpy(a2, *xH, *yH);
    error = ERROR(y);
    break;

  case 9:
    *xD = *xH;
    *yD = *yH;
    blas::caxpby(a2, *xD, b2, *yD);
    blas::caxpby(a2, *xH, b2, *yH);
    error = ERROR(y);
    break;

  case 10:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
    blas::cxpaypbz(*xH, a2, *yH, b2, *zH);
    error = ERROR(z);
    break;

  case 11:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::axpyBzpcx(a, *xD, *yD, b, *zD, c);
    blas::axpyBzpcx(a, *xH, *yH, b, *zH, c);
    error = ERROR(x) + ERROR(y);
    break;

  case 12:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::axpyZpbx(a, *xD, *yD, *zD, b);
    blas::axpyZpbx(a, *xH, *yH, *zH, b);
    error = ERROR(x) + ERROR(y);
    break;

  case 13:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
    blas::caxpbypzYmbw(a2, *xH, b2, *yH, *zH, *wH);
    error = ERROR(z) + ERROR(y);
    break;
      
  case 14:
    *xD = *xH;
    *yD = *yH;
    blas::cabxpyAx(a, b2, *xD, *yD);
    blas::cabxpyAx(a, b2, *xH, *yH);
    error = ERROR(y) + ERROR(x);
    break;

  case 15:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {blas::caxpbypz(a2, *xD, b2, *yD, *zD);
      blas::caxpbypz(a2, *xH, b2, *yH, *zH);
      error = ERROR(z); }
    break;
    
  case 16:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    {blas::caxpbypczpw(a2, *xD, b2, *yD, c2, *zD, *wD);
      blas::caxpbypczpw(a2, *xH, b2, *yH, c2, *zH, *wH);
      error = ERROR(w); }
    break;

  case 17:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {blas::caxpyXmaz(a, *xD, *yD, *zD);
     blas::caxpyXmaz(a, *xH, *yH, *zH);
     error = ERROR(y) + ERROR(x);}
    break;

    // double
  case 18:
    *xD = *xH;
    error = fabs(blas::norm2(*xD) - blas::norm2(*xH)) / blas::norm2(*xH);
    break;
    
  case 19:
    *xD = *xH;
    *yD = *yH;
    error = fabs(blas::reDotProduct(*xD, *yD) - blas::reDotProduct(*xH, *yH)) / fabs(blas::reDotProduct(*xH, *yH));
    break;

  case 20:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::axpyNorm(a, *xD, *yD);
    double h = blas::axpyNorm(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 21:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::xmyNorm(*xD, *yD);
    double h = blas::xmyNorm(*xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;
    
  case 22:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::caxpyNorm(a, *xD, *yD);
    double h = blas::caxpyNorm(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 23:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {double d = blas::caxpyXmazNormX(a, *xD, *yD, *zD);
      double h = blas::caxpyXmazNormX(a, *xH, *yH, *zH);
      error = ERROR(y) + ERROR(x) + fabs(d-h)/fabs(h);}
    break;

  case 24:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::cabxpyAxNorm(a, b2, *xD, *yD);
      double h = blas::cabxpyAxNorm(a, b2, *xH, *yH);
      error = ERROR(x) + ERROR(y) + fabs(d-h)/fabs(h);}
    break;

    // double2
  case 25:
    *xD = *xH;
    *yD = *yH;
    error = abs(blas::cDotProduct(*xD, *yD) - blas::cDotProduct(*xH, *yH)) / abs(blas::cDotProduct(*xH, *yH));
    break;
    
  case 26:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    { quda::Complex d = blas::xpaycDotzy(*xD, a, *yD, *zD);
      quda::Complex h = blas::xpaycDotzy(*xH, a, *yH, *zH);
      error =  fabs(blas::norm2(*yD) - blas::norm2(*yH)) / blas::norm2(*yH) + abs(d-h)/abs(h);
    }
    break;
    
  case 27:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {quda::Complex d = blas::caxpyDotzy(a, *xD, *yD, *zD);
      quda::Complex h = blas::caxpyDotzy(a, *xH, *yH, *zH);
    error = ERROR(y) + abs(d-h)/abs(h);}
    break;

    // double3
  case 28:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::cDotProductNormA(*xD, *yD);
      double3 h = blas::cDotProductNormA(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;
    
  case 29:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::cDotProductNormB(*xD, *yD);
      double3 h = blas::cDotProductNormB(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;
    
  case 30:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    *vD = *vH;
    { double3 d = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zD, *wD, *vD);
      double3 h = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xH, b2, *yH, *zH, *wH, *vH);
      error = ERROR(z) + ERROR(y) + fabs(d.x - h.x) / fabs(h.x) + 
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  case 31:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::HeavyQuarkResidualNorm(*xD, *yD);
      double3 h = blas::HeavyQuarkResidualNorm(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + 
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
  initComms(argc, argv, gridsize_from_cmdline);
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
    "caxpbypzYmbwcDotProductWYNormY",
    "HeavyQuarkResidualNorm"
  };

  char *prec_str[] = {"half", "single", "double"};
  
  // Only benchmark double precision if supported
#if (__COMPUTE_CAPABILITY__ >= 130)
  int Nprec = 3;
#else
  int Nprec = 2;
#endif

  // enable the tuning
  quda::blas::setTuning(tune ? QUDA_TUNE_YES : QUDA_TUNE_NO, QUDA_SILENT);

  for (int prec = 0; prec < Nprec; prec++) {

    printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", prec_str[prec], niter);
    initFields(prec);

    for (int kernel = 0; kernel < Nkernels; kernel++) {
      // only benchmark "high precision" copyCuda() if double is supported
      if ((Nprec < 3) && (kernel == 0)) continue;

      // do the initial tune
      benchmark(kernel, 1);
    
      // now rerun with more iterations to get accurate speed measurements
      quda::blas::flops = 0;
      quda::blas::bytes = 0;
      
      double secs = benchmark(kernel, niter);
      
      double gflops = (quda::blas::flops*1e-9)/(secs);
      double gbytes = quda::blas::bytes/(secs*1e9);
    
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

  finalizeComms();
}
