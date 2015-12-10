#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <test_util.h>
#include <face_quda.h>

// include because of nasty globals used in the tests
#include <dslash_util.h>

// google test
#include <gtest.h>

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
extern bool verify_results;

extern void usage(char** );

const int Nkernels = 32;

using namespace quda;

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
  Nspin = (dslash_type == QUDA_ASQTAD_DSLASH || 
	   dslash_type == QUDA_STAGGERED_DSLASH) ? 1 : 4;
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

  {
    switch (kernel) {

    case 0:
      for (int i=0; i < niter; ++i) copyCuda(*yD, *hD);
      break;

    case 1:
      for (int i=0; i < niter; ++i) copyCuda(*yD, *lD);
      break;
      
    case 2:
      for (int i=0; i < niter; ++i) axpbyCuda(a, *xD, b, *yD);
      break;

    case 3:
      for (int i=0; i < niter; ++i) xpyCuda(*xD, *yD);
      break;

    case 4:
      for (int i=0; i < niter; ++i) axpyCuda(a, *xD, *yD);
      break;

    case 5:
      for (int i=0; i < niter; ++i) xpayCuda(*xD, a, *yD);
      break;

    case 6:
      for (int i=0; i < niter; ++i) mxpyCuda(*xD, *yD);
      break;

    case 7:
      for (int i=0; i < niter; ++i) axCuda(a, *xD);
      break;

    case 8:
      for (int i=0; i < niter; ++i) caxpyCuda(a2, *xD, *yD);
      break;

    case 9:
      for (int i=0; i < niter; ++i) caxpbyCuda(a2, *xD, b2, *yD);
      break;

    case 10:
      for (int i=0; i < niter; ++i) cxpaypbzCuda(*xD, a2, *yD, b2, *zD);
      break;

    case 11:
      for (int i=0; i < niter; ++i) axpyBzpcxCuda(a, *xD, *yD, b, *zD, c);
      break;

    case 12:
      for (int i=0; i < niter; ++i) axpyZpbxCuda(a, *xD, *yD, *zD, b);
      break;

    case 13:
      for (int i=0; i < niter; ++i) caxpbypzYmbwCuda(a2, *xD, b2, *yD, *zD, *wD);
      break;
      
    case 14:
      for (int i=0; i < niter; ++i) cabxpyAxCuda(a, b2, *xD, *yD);
      break;

    case 15:
      for (int i=0; i < niter; ++i) caxpbypzCuda(a2, *xD, b2, *yD, *zD);
      break;

    case 16:
      for (int i=0; i < niter; ++i) caxpbypczpwCuda(a2, *xD, b2, *yD, c2, *zD, *wD);
      break;

    case 17:
      for (int i=0; i < niter; ++i) caxpyXmazCuda(a2, *xD, *yD, *zD);
      break;

      // double
    case 18:
      for (int i=0; i < niter; ++i) normCuda(*xD);
      break;

    case 19:
      for (int i=0; i < niter; ++i) reDotProductCuda(*xD, *yD);
      break;

    case 20:
      for (int i=0; i < niter; ++i) axpyNormCuda(a, *xD, *yD);
      break;

    case 21:
      for (int i=0; i < niter; ++i) xmyNormCuda(*xD, *yD);
      break;
      
    case 22:
      for (int i=0; i < niter; ++i) caxpyNormCuda(a2, *xD, *yD);
      break;

    case 23:
      for (int i=0; i < niter; ++i) caxpyXmazNormXCuda(a2, *xD, *yD, *zD);
      break;

    case 24:
      for (int i=0; i < niter; ++i) cabxpyAxNormCuda(a, b2, *xD, *yD);
      break;

    // double2
    case 25:
      for (int i=0; i < niter; ++i) cDotProductCuda(*xD, *yD);
      break;

    case 26:
      for (int i=0; i < niter; ++i) xpaycDotzyCuda(*xD, a, *yD, *zD);
      break;
      
    case 27:
      for (int i=0; i < niter; ++i) caxpyDotzyCuda(a2, *xD, *yD, *zD);
      break;

    // double3
    case 28:
      for (int i=0; i < niter; ++i) cDotProductNormACuda(*xD, *yD);
      break;

    case 29:
      for (int i=0; i < niter; ++i) cDotProductNormBCuda(*xD, *yD);
      break;

    case 30:
      for (int i=0; i < niter; ++i) caxpbypzYmbwcDotProductUYNormYCuda(a2, *xD, b2, *yD, *zD, *wD, *vD);
      break;

    case 31:
      for (int i=0; i < niter; ++i) HeavyQuarkResidualNormCuda(*xD, *yD);
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

  double a = M_PI, b = M_PI*exp(1.0), c = sqrt(M_PI);
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

  case 31:
    *xD = *xH;
    *yD = *yH;
    { double3 d = HeavyQuarkResidualNormCuda(*xD, *yD);
      double3 h = HeavyQuarkResidualNormCpu(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + 
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  default:
    errorQuda("Undefined blas kernel %d\n", kernel);
  }

  return error;
}

int Nprec = 3;

const char *prec_str[] = {"half", "single", "double"};

const char *names[] = {
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

  // enable the tuning
  setTuning(tune ? QUDA_TUNE_YES : QUDA_TUNE_NO);
  setVerbosity(QUDA_SILENT);

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
  if (verify_results) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS() != 0) warningQuda("Tests failed");
  }

  endQuda();

  finalizeComms();
}

// The following tests each kernel at each precision using the google testing framework

class BlasTest : public ::testing::TestWithParam<int2> {
protected:
  int2 param;

public:
  virtual ~BlasTest() { }
  virtual void SetUp() { 
    param = GetParam();
    initFields(param.x); 
  }
  virtual void TearDown() { freeFields(); }

  virtual void NormalExit() { printf("monkey\n"); }

};

TEST_P(BlasTest, verify) {
  int prec = param.x;
  int kernel = param.y;
  double deviation = test(kernel);
  printfQuda("%-35s error = %e\n", names[kernel], deviation);
  double tol = (prec == 2 ? 1e-12 : (prec == 1 ? 1e-5 : 1e-3));
  tol = (kernel < 2) ? 1e-4 : tol; // use different tolerance for copy
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

// half precision
INSTANTIATE_TEST_CASE_P(copyHS_half, BlasTest, ::testing::Values( make_int2(0,0) ));
INSTANTIATE_TEST_CASE_P(copyLS_half, BlasTest, ::testing::Values( make_int2(0,1) ));
INSTANTIATE_TEST_CASE_P(axpby_half, BlasTest, ::testing::Values( make_int2(0,2) ));
INSTANTIATE_TEST_CASE_P(xpy_half, BlasTest, ::testing::Values( make_int2(0,3) ));
INSTANTIATE_TEST_CASE_P(axpy_half, BlasTest, ::testing::Values( make_int2(0,4) ));
INSTANTIATE_TEST_CASE_P(xpay_half, BlasTest, ::testing::Values( make_int2(0,5) ));
INSTANTIATE_TEST_CASE_P(mxpy_half, BlasTest, ::testing::Values( make_int2(0,6) ));
INSTANTIATE_TEST_CASE_P(ax_half, BlasTest, ::testing::Values( make_int2(0,7) ));
INSTANTIATE_TEST_CASE_P(caxpy_half, BlasTest, ::testing::Values( make_int2(0,8) ));
INSTANTIATE_TEST_CASE_P(caxpby_half, BlasTest, ::testing::Values( make_int2(0,9) ));
INSTANTIATE_TEST_CASE_P(cxpaypbz_half, BlasTest, ::testing::Values( make_int2(0,10) ));
INSTANTIATE_TEST_CASE_P(axpyBzpcx_half, BlasTest, ::testing::Values( make_int2(0,11) ));
INSTANTIATE_TEST_CASE_P(axpyZpbx_half, BlasTest, ::testing::Values( make_int2(0,12) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbw_half, BlasTest, ::testing::Values( make_int2(0,13) ));
INSTANTIATE_TEST_CASE_P(cabxpyAx_half, BlasTest, ::testing::Values( make_int2(0,14) ));
INSTANTIATE_TEST_CASE_P(caxpbypz_half, BlasTest, ::testing::Values( make_int2(0,15) ));
INSTANTIATE_TEST_CASE_P(caxpbypczpw_half, BlasTest, ::testing::Values( make_int2(0,16) ));
INSTANTIATE_TEST_CASE_P(caxpyXmaz_half, BlasTest, ::testing::Values( make_int2(0,17) ));
INSTANTIATE_TEST_CASE_P(norm2_half, BlasTest, ::testing::Values( make_int2(0,18) ));
INSTANTIATE_TEST_CASE_P(reDotProduct_half, BlasTest, ::testing::Values( make_int2(0,19) ));
INSTANTIATE_TEST_CASE_P(axpyNorm_half, BlasTest, ::testing::Values( make_int2(0,20) ));
INSTANTIATE_TEST_CASE_P(xmyNorm_half, BlasTest, ::testing::Values( make_int2(0,21) ));
INSTANTIATE_TEST_CASE_P(caxpyNorm_half, BlasTest, ::testing::Values( make_int2(0,22) ));
INSTANTIATE_TEST_CASE_P(caxpyXmazNormX_half, BlasTest, ::testing::Values( make_int2(0,23) ));
INSTANTIATE_TEST_CASE_P(cabxpyAxNorm_half, BlasTest, ::testing::Values( make_int2(0,24) ));
INSTANTIATE_TEST_CASE_P(cDotProduct_half, BlasTest, ::testing::Values( make_int2(0,25) ));
INSTANTIATE_TEST_CASE_P(xpaycDotzy_half, BlasTest, ::testing::Values( make_int2(0,26) ));
INSTANTIATE_TEST_CASE_P(caxpyDotzy_half, BlasTest, ::testing::Values( make_int2(0,27) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormA_half, BlasTest, ::testing::Values( make_int2(0,28) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormB_half, BlasTest, ::testing::Values( make_int2(0,29) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbwcDotProductWYNormY_half, BlasTest, ::testing::Values( make_int2(0,30) ));
INSTANTIATE_TEST_CASE_P(HeavyQuarkResidualNorm_half, BlasTest, ::testing::Values( make_int2(0,31) ));

// single precision
INSTANTIATE_TEST_CASE_P(copyHS_single, BlasTest, ::testing::Values( make_int2(1,0) ));
INSTANTIATE_TEST_CASE_P(copyLS_single, BlasTest, ::testing::Values( make_int2(1,1) ));
INSTANTIATE_TEST_CASE_P(axpby_single, BlasTest, ::testing::Values( make_int2(1,2) ));
INSTANTIATE_TEST_CASE_P(xpy_single, BlasTest, ::testing::Values( make_int2(1,3) ));
INSTANTIATE_TEST_CASE_P(axpy_single, BlasTest, ::testing::Values( make_int2(1,4) ));
INSTANTIATE_TEST_CASE_P(xpay_single, BlasTest, ::testing::Values( make_int2(1,5) ));
INSTANTIATE_TEST_CASE_P(mxpy_single, BlasTest, ::testing::Values( make_int2(1,6) ));
INSTANTIATE_TEST_CASE_P(ax_single, BlasTest, ::testing::Values( make_int2(1,7) ));
INSTANTIATE_TEST_CASE_P(caxpy_single, BlasTest, ::testing::Values( make_int2(1,8) ));
INSTANTIATE_TEST_CASE_P(caxpby_single, BlasTest, ::testing::Values( make_int2(1,9) ));
INSTANTIATE_TEST_CASE_P(cxpaypbz_single, BlasTest, ::testing::Values( make_int2(1,10) ));
INSTANTIATE_TEST_CASE_P(axpyBzpcx_single, BlasTest, ::testing::Values( make_int2(1,11) ));
INSTANTIATE_TEST_CASE_P(axpyZpbx_single, BlasTest, ::testing::Values( make_int2(1,12) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbw_single, BlasTest, ::testing::Values( make_int2(1,13) ));
INSTANTIATE_TEST_CASE_P(cabxpyAx_single, BlasTest, ::testing::Values( make_int2(1,14) ));
INSTANTIATE_TEST_CASE_P(caxpbypz_single, BlasTest, ::testing::Values( make_int2(1,15) ));
INSTANTIATE_TEST_CASE_P(caxpbypczpw_single, BlasTest, ::testing::Values( make_int2(1,16) ));
INSTANTIATE_TEST_CASE_P(caxpyXmaz_single, BlasTest, ::testing::Values( make_int2(1,17) ));
INSTANTIATE_TEST_CASE_P(norm2_single, BlasTest, ::testing::Values( make_int2(1,18) ));
INSTANTIATE_TEST_CASE_P(reDotProduct_single, BlasTest, ::testing::Values( make_int2(1,19) ));
INSTANTIATE_TEST_CASE_P(axpyNorm_single, BlasTest, ::testing::Values( make_int2(1,20) ));
INSTANTIATE_TEST_CASE_P(xmyNorm_single, BlasTest, ::testing::Values( make_int2(1,21) ));
INSTANTIATE_TEST_CASE_P(caxpyNorm_single, BlasTest, ::testing::Values( make_int2(1,22) ));
INSTANTIATE_TEST_CASE_P(caxpyXmazNormX_single, BlasTest, ::testing::Values( make_int2(1,23) ));
INSTANTIATE_TEST_CASE_P(cabxpyAxNorm_single, BlasTest, ::testing::Values( make_int2(1,24) ));
INSTANTIATE_TEST_CASE_P(cDotProduct_single, BlasTest, ::testing::Values( make_int2(1,25) ));
INSTANTIATE_TEST_CASE_P(xpaycDotzy_single, BlasTest, ::testing::Values( make_int2(1,26) ));
INSTANTIATE_TEST_CASE_P(caxpyDotzy_single, BlasTest, ::testing::Values( make_int2(1,27) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormA_single, BlasTest, ::testing::Values( make_int2(1,28) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormB_single, BlasTest, ::testing::Values( make_int2(1,29) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbwcDotProductWYNormY_single, BlasTest, ::testing::Values( make_int2(1,30) ));
INSTANTIATE_TEST_CASE_P(HeavyQuarkResidualNorm_single, BlasTest, ::testing::Values( make_int2(1,31) ));

// double precision
INSTANTIATE_TEST_CASE_P(copyHS_double, BlasTest, ::testing::Values( make_int2(2,0) ));
INSTANTIATE_TEST_CASE_P(copyLS_double, BlasTest, ::testing::Values( make_int2(2,1) ));
INSTANTIATE_TEST_CASE_P(axpby_double, BlasTest, ::testing::Values( make_int2(2,2) ));
INSTANTIATE_TEST_CASE_P(xpy_double, BlasTest, ::testing::Values( make_int2(2,3) ));
INSTANTIATE_TEST_CASE_P(axpy_double, BlasTest, ::testing::Values( make_int2(2,4) ));
INSTANTIATE_TEST_CASE_P(xpay_double, BlasTest, ::testing::Values( make_int2(2,5) ));
INSTANTIATE_TEST_CASE_P(mxpy_double, BlasTest, ::testing::Values( make_int2(2,6) ));
INSTANTIATE_TEST_CASE_P(ax_double, BlasTest, ::testing::Values( make_int2(2,7) ));
INSTANTIATE_TEST_CASE_P(caxpy_double, BlasTest, ::testing::Values( make_int2(2,8) ));
INSTANTIATE_TEST_CASE_P(caxpby_double, BlasTest, ::testing::Values( make_int2(2,9) ));
INSTANTIATE_TEST_CASE_P(cxpaypbz_double, BlasTest, ::testing::Values( make_int2(2,10) ));
INSTANTIATE_TEST_CASE_P(axpyBzpcx_double, BlasTest, ::testing::Values( make_int2(2,11) ));
INSTANTIATE_TEST_CASE_P(axpyZpbx_double, BlasTest, ::testing::Values( make_int2(2,12) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbw_double, BlasTest, ::testing::Values( make_int2(2,13) ));
INSTANTIATE_TEST_CASE_P(cabxpyAx_double, BlasTest, ::testing::Values( make_int2(2,14) ));
INSTANTIATE_TEST_CASE_P(caxpbypz_double, BlasTest, ::testing::Values( make_int2(2,15) ));
INSTANTIATE_TEST_CASE_P(caxpbypczpw_double, BlasTest, ::testing::Values( make_int2(2,16) ));
INSTANTIATE_TEST_CASE_P(caxpyXmaz_double, BlasTest, ::testing::Values( make_int2(2,17) ));
INSTANTIATE_TEST_CASE_P(norm2_double, BlasTest, ::testing::Values( make_int2(2,18) ));
INSTANTIATE_TEST_CASE_P(reDotProduct_double, BlasTest, ::testing::Values( make_int2(2,19) ));
INSTANTIATE_TEST_CASE_P(axpyNorm_double, BlasTest, ::testing::Values( make_int2(2,20) ));
INSTANTIATE_TEST_CASE_P(xmyNorm_double, BlasTest, ::testing::Values( make_int2(2,21) ));
INSTANTIATE_TEST_CASE_P(caxpyNorm_double, BlasTest, ::testing::Values( make_int2(2,22) ));
INSTANTIATE_TEST_CASE_P(caxpyXmazNormX_double, BlasTest, ::testing::Values( make_int2(2,23) ));
INSTANTIATE_TEST_CASE_P(cabxpyAxNorm_double, BlasTest, ::testing::Values( make_int2(2,24) ));
INSTANTIATE_TEST_CASE_P(cDotProduct_double, BlasTest, ::testing::Values( make_int2(2,25) ));
INSTANTIATE_TEST_CASE_P(xpaycDotzy_double, BlasTest, ::testing::Values( make_int2(2,26) ));
INSTANTIATE_TEST_CASE_P(caxpyDotzy_double, BlasTest, ::testing::Values( make_int2(2,27) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormA_double, BlasTest, ::testing::Values( make_int2(2,28) ));
INSTANTIATE_TEST_CASE_P(cDotProductNormB_double, BlasTest, ::testing::Values( make_int2(2,29) ));
INSTANTIATE_TEST_CASE_P(caxpbypzYmbwcDotProductWYNormY_double, BlasTest, ::testing::Values( make_int2(2,30) ));
INSTANTIATE_TEST_CASE_P(HeavyQuarkResidualNorm_double, BlasTest, ::testing::Values( make_int2(2,31) ));

