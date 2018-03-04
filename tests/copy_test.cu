#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <test_util.h>

// include because of nasty globals used in the tests
#include <dslash_util.h>

// google test
#include <gtest.h>

extern int test_type;
extern QudaPrecision prec;
extern QudaDslashType dslash_type;
extern QudaInverterType inv_type;
extern int nvec[QUDA_MAX_MG_LEVEL];
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern int niter;

extern bool verify_results;
extern int Nsrc;
extern int Msrc;

extern void usage(char** );

const int Nkernels = 3;

using namespace quda;

ColorSpinorField *xH, *yH, *zH, *wH, *vH, *hH, *mH, *lH;
ColorSpinorField *xD, *yD, *zD, *wD, *vD, *hD, *mD, *lD, *xmD, *ymD, *zmD;
int Nspin;
int Ncolor;

void setPrec(ColorSpinorParam &param, const QudaPrecision precision)
{
  param.setPrecision(precision);
  if (Nspin == 1 || Nspin == 2 || precision == QUDA_DOUBLE_PRECISION) {
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  } else {
    param.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
  }
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("S_dimension T_dimension Nspin Ncolor\n");
  printfQuda("%3d /%3d / %3d   %3d      %d     %d\n", xdim, ydim, zdim, tdim, Nspin, Ncolor);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));
  return;
}

int Nprec = 4;

bool skip_kernel(int precision, int kernel) {
  // if we've selected a given kernel then make sure we only run that
  if (test_type != -1 && kernel != test_type) return true;

  // if we've selected a given precision then make sure we only run that
  QudaPrecision this_prec = precision == 3 ? QUDA_DOUBLE_PRECISION : precision  == 2 ? QUDA_SINGLE_PRECISION : precision == 1 ? QUDA_HALF_PRECISION : QUDA_QUARTER_PRECISION;
  if (prec != QUDA_INVALID_PRECISION && this_prec != prec) return true;

  // mg copy isn't initialized for quarter or half
  if (precision == 0 || precision == 1) return true;

  // if we're doing an MG test, skip copying from single or double to fixed.
  if ((precision == 3 || precision == 2) && (kernel == 1 || kernel == 2)) return true;

  return false;
}

void printFieldInfo(ColorSpinorField& field, const char* name)
{
  printf("\nField %s lives on the ");
  switch (field.Location())
  {
    case QUDA_CUDA_FIELD_LOCATION:
      printf("GPU");
      break;
    case QUDA_CPU_FIELD_LOCATION:
      printf("CPU");
      break;
  }
  printf(", is ");
  switch (field.Precision())
  {
    case QUDA_QUARTER_PRECISION:
      printf("quarter");
      break;
    case QUDA_HALF_PRECISION:
      printf("half");
      break;
    case QUDA_SINGLE_PRECISION:
      printf("single");
      break;
    case QUDA_DOUBLE_PRECISION:
      printf("double");
      break;
  }
  printf(" precision. Its layout is ");
  switch (field.FieldOrder())
  {
    case QUDA_FLOAT_FIELD_ORDER:
      printf("QUDA_FLOAT_FIELD_ORDER");
      break;
    case QUDA_FLOAT2_FIELD_ORDER:
      printf("QUDA_FLOAT2_FIELD_ORDER");
      break;
    case QUDA_FLOAT4_FIELD_ORDER:
      printf("QUDA_FLOAT4_FIELD_ORDER");
      break;
    case QUDA_SPACE_SPIN_COLOR_FIELD_ORDER:
      printf("QUDA_SPACE_SPIN_COLOR_FIELD_ORDER");
      break;
    case QUDA_SPACE_COLOR_SPIN_FIELD_ORDER:
      printf("QUDA_SPACE_COLOR_SPIN_FIELD_ORDER");
      break;
  }
  printf(", which is %snative.\n", field.isNative() ? "" : "not ");
}


void initFields(int prec)
{
  // precisions used for the source field in the copyCuda() benchmark
  QudaPrecision high_aux_prec = QUDA_INVALID_PRECISION;
  QudaPrecision mid_aux_prec = QUDA_INVALID_PRECISION;
  QudaPrecision low_aux_prec = QUDA_INVALID_PRECISION;

  ColorSpinorParam param;
  param.nColor = Ncolor;
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
  param.setPrecision(QUDA_DOUBLE_PRECISION);
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

  param.create = QUDA_ZERO_FIELD_CREATE;

  vH = new cpuColorSpinorField(param);
  wH = new cpuColorSpinorField(param);
  xH = new cpuColorSpinorField(param);
  yH = new cpuColorSpinorField(param);
  zH = new cpuColorSpinorField(param);
  hH = new cpuColorSpinorField(param);
  mH = new cpuColorSpinorField(param);
  lH = new cpuColorSpinorField(param);

  static_cast<cpuColorSpinorField*>(vH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(wH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(xH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(yH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(zH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(hH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(mH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(lH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  // Now set the parameters for the cuda fields
  //param.pad = xdim*ydim*zdim/2;

  if (param.nSpin == 4) param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  param.create = QUDA_ZERO_FIELD_CREATE;

  switch(prec) {
  case 0:
    setPrec(param, QUDA_QUARTER_PRECISION);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_SINGLE_PRECISION;
    low_aux_prec = QUDA_HALF_PRECISION;
    break;
  case 1:
    setPrec(param, QUDA_HALF_PRECISION);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_SINGLE_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  case 2:
    setPrec(param, QUDA_SINGLE_PRECISION);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_HALF_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  case 3:
    setPrec(param, QUDA_DOUBLE_PRECISION);
    high_aux_prec = QUDA_SINGLE_PRECISION;
    mid_aux_prec = QUDA_HALF_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  default:
    errorQuda("Precision option not defined");
  }

  checkCudaError();

  vD = new cudaColorSpinorField(param);
  wD = new cudaColorSpinorField(param);
  xD = new cudaColorSpinorField(param);
  yD = new cudaColorSpinorField(param);
  zD = new cudaColorSpinorField(param);

  param.is_composite = false;
  param.is_component = false;
  param.composite_dim = 1;

  setPrec(param, high_aux_prec);
  hD = new cudaColorSpinorField(param);

  setPrec(param, mid_aux_prec);
  mD = new cudaColorSpinorField(param);

  setPrec(param, low_aux_prec);
  lD = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  // only do copy if not doing half or quarter precision with mg
  bool flag = !(param.nSpin == 2 &&
		(prec == 0 || prec == 1 || low_aux_prec == QUDA_HALF_PRECISION ||
        low_aux_prec == QUDA_QUARTER_PRECISION || mid_aux_prec == QUDA_HALF_PRECISION ||
        mid_aux_prec == QUDA_QUARTER_PRECISION) );

  if ( flag ) {
    *vD = *vH;
    *wD = *wH;
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *hD = *hH;
    *mD = *mH;
    *lD = *lH;
  }
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
  delete mD;
  delete lD;

  // release memory
  delete vH;
  delete wH;
  delete xH;
  delete yH;
  delete zH;
  delete hH;
  delete mH;
  delete lH;
}


double benchmark(int kernel, const int niter) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  {
    switch (kernel) {

    case 0:
      for (int i=0; i < niter; ++i) blas::copy(*yD, *hD);
      break;

    case 1:
      for (int i=0; i < niter; ++i) blas::copy(*yD, *mD);
      break;

    case 2:
      for (int i=0; i < niter; ++i) blas::copy(*yD, *lD);
      break;

    default:
      errorQuda("Undefined copy kernel %d\n", kernel);
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

  double error = 0;

  switch (kernel) {

  case 0:
    //printFieldInfo(*hD, "hD");
    //hD->PrintVector(0);
    *hD = *hH;
    blas::copy(*yD, *hD);
    blas::copy(*yH, *hH);
    error = ERROR(y);
    break;    

  case 1:
    *mD = *mH;
    blas::copy(*yD, *mD);
    blas::copy(*yH, *mH);
    error = ERROR(y);
    break;

  case 2:
    *lD = *lH;
    blas::copy(*yD, *lD);
    blas::copy(*yH, *lH);
    error = ERROR(y);
    break;

  default:
    errorQuda("Undefined blas kernel %d\n", kernel);
  }

  return error;
}

const char *prec_str[] = {"quarter", "half", "single", "double"};

const char *names[] = {
  "copyHS",
  "copyMS",
  "copyLS"
};

int main(int argc, char** argv)
{
  prec = QUDA_INVALID_PRECISION;
  test_type = -1;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  // override spin setting if mg solver is set to test coarse grids
  if (inv_type == QUDA_MG_INVERTER) {
    Nspin = 2;
    Ncolor = nvec[0];
    if (Ncolor == 0) Ncolor = 24;
  } else {
    // set spin according to the type of dslash
    Nspin = (dslash_type == QUDA_ASQTAD_DSLASH ||
	     dslash_type == QUDA_STAGGERED_DSLASH) ? 1 : 4;
    Ncolor = 3;
  }

  setSpinorSiteSize(24);
  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device);

  setVerbosity(QUDA_SILENT);

  for (int prec = 0; prec < Nprec; prec++) {
    if (Nspin == 2 && (prec == 0 || prec == 1)) continue;

    printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", prec_str[prec], niter);
    initFields(prec);

    for (int kernel = 0; kernel < Nkernels; kernel++) {
      if (skip_kernel(prec, kernel)) continue;

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
  if (verify_results) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS() != 0) warningQuda("Tests failed");
  }

  endQuda();

  finalizeComms();
}

// The following tests each kernel at each precision using the google testing framework

class CopyTest : public ::testing::TestWithParam<int2> {
protected:
  int2 param;

public:
  virtual ~CopyTest() { }
  virtual void SetUp() {
    param = GetParam();
    initFields(param.x);
  }
  virtual void TearDown() { freeFields(); }

  virtual void NormalExit() { printf("monkey\n"); }

};

TEST_P(CopyTest, verify) {
  int prec = param.x;
  int kernel = param.y;

  // certain tests will fail to run for coarse grids so mark these as
  // failed without running
  double deviation =  skip_kernel(prec,kernel) ? 1.0 : test(kernel);
  printfQuda("%-35s error = %e\n", names[kernel], deviation);
  //double tol = (prec == 3 ? 1e-10 : (prec == 2 ? 1e-5 : (prec == 1 ? 1e-3 : 1e-1)));
  double tol = 5e-2;
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

// quarter precision
INSTANTIATE_TEST_CASE_P(copyHS_quarter, CopyTest, ::testing::Values( make_int2(0,0) ));
INSTANTIATE_TEST_CASE_P(copyMS_quarter, CopyTest, ::testing::Values( make_int2(0,1) ));
INSTANTIATE_TEST_CASE_P(copyLS_quarter, CopyTest, ::testing::Values( make_int2(0,2) ));

// half precision
INSTANTIATE_TEST_CASE_P(copyHS_half, CopyTest, ::testing::Values( make_int2(1,0) ));
INSTANTIATE_TEST_CASE_P(copyMS_half, CopyTest, ::testing::Values( make_int2(1,1) ));
INSTANTIATE_TEST_CASE_P(copyLS_half, CopyTest, ::testing::Values( make_int2(1,2) ));

// single precision
INSTANTIATE_TEST_CASE_P(copyHS_single, CopyTest, ::testing::Values( make_int2(2,0) ));
INSTANTIATE_TEST_CASE_P(copyMS_single, CopyTest, ::testing::Values( make_int2(2,1) ));
INSTANTIATE_TEST_CASE_P(copyLS_single, CopyTest, ::testing::Values( make_int2(2,2) ));

// double precision
INSTANTIATE_TEST_CASE_P(copyHS_double, CopyTest, ::testing::Values( make_int2(3,0) ));
INSTANTIATE_TEST_CASE_P(copyMS_double, CopyTest, ::testing::Values( make_int2(3,1) ));
INSTANTIATE_TEST_CASE_P(copyLS_double, CopyTest, ::testing::Values( make_int2(3,2) ));
