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

// ESW: multireduce-specific hacks.
// Sort of self explanatory, set this equal to
// 0 for r^2 tests (x = y), 1 for pAp tests (x != y)
#define ZERO_FOR_R2_ONE_FOR_PAP 0

extern QudaDslashType dslash_type;
extern QudaInverterType inv_type;
extern int nvec;
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

const int Nkernels = 1;

using namespace quda;

ColorSpinorField *xH, *yH, *zH, *wH, *vH, *hH, *lH;
ColorSpinorField *xD, *yD, *zD, *wD, *vD, *hD, *lD, *xmD, *ymD, *zmD;
std::vector<cpuColorSpinorField*> xmH;
std::vector<cpuColorSpinorField*> ymH;
std::vector<cpuColorSpinorField*> zmH;
int Nspin;
int Ncolor;

void setPrec(ColorSpinorParam &param, const QudaPrecision precision)
{
  param.precision = precision;
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

int Nprec = 3;

bool skip_kernel(int precision, int kernel) {
  if ( Nspin == 2 && precision == 0) {
    // avoid half precision tests if doing coarse fields
    return true;
  } else if (Nspin == 2 && kernel == 1) {
    // avoid low-precision copy if doing coarse fields
    return true;
  } else if (Ncolor != 3 && (kernel == 31 || kernel == 32)) {
    // only benchmark heavy-quark norm if doing 3 colors
    return true;
  } else if ((Nprec < 3) && (kernel == 0)) {
    // only benchmark high-precision copy() if double is supported
    return true;
  }

  return false;
}

void initFields(int prec)
{
  // precisions used for the source field in the copyCuda() benchmark
  QudaPrecision high_aux_prec = QUDA_INVALID_PRECISION;
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

// create composite fields

  // xmH = new cpuColorSpinorField(param);
  // ymH = new cpuColorSpinorField(param);



  xmH.reserve(Nsrc);
  for (int cid = 0; cid < Nsrc; cid++) xmH.push_back(new cpuColorSpinorField(param));
  ymH.reserve(Msrc);
  for (int cid = 0; cid < Msrc; cid++) ymH.push_back(new cpuColorSpinorField(param));
  zmH.reserve(Nsrc);
  for (int cid = 0; cid < Nsrc; cid++) zmH.push_back(new cpuColorSpinorField(param));


  static_cast<cpuColorSpinorField*>(vH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(wH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(xH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(yH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(zH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(hH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField*>(lH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  for(int i=0; i<Nsrc; i++){
    static_cast<cpuColorSpinorField*>(xmH[i])->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  }
  for(int i=0; i<Msrc; i++){
    static_cast<cpuColorSpinorField*>(ymH[i])->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  }
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
  default:
    errorQuda("Precision option not defined");
  }

  checkCudaError();

  vD = new cudaColorSpinorField(param);
  wD = new cudaColorSpinorField(param);
  xD = new cudaColorSpinorField(param);
  yD = new cudaColorSpinorField(param);
  zD = new cudaColorSpinorField(param);

  param.is_composite = true;
  param.is_component = false;

// create composite fields
  param.composite_dim = Nsrc;
  xmD = new cudaColorSpinorField(param);

  param.composite_dim = Msrc;
  ymD = new cudaColorSpinorField(param);

  param.composite_dim = Nsrc;
  zmD = new cudaColorSpinorField(param);

  param.is_composite = false;
  param.is_component = false;
  param.composite_dim = 1;

  setPrec(param, high_aux_prec);
  hD = new cudaColorSpinorField(param);

  setPrec(param, low_aux_prec);
  lD = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  // only do copy if not doing half precision with mg
  bool flag = !(param.nSpin == 2 &&
		(prec == 0 || low_aux_prec == QUDA_HALF_PRECISION) );

  if ( flag ) {
    *vD = *vH;
    *wD = *wH;
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *hD = *hH;
    *lD = *lH;
    // for (int i=0; i < Nsrc; i++){
    //   xmD->Component(i) = *(xmH[i]);
    //   ymD->Component(i) = *(ymH[i]);
    // }
    // *ymD = *ymH;
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
  delete lD;
  delete xmD;
  delete ymD;
  delete zmD;

  // release memory
  delete vH;
  delete wH;
  delete xH;
  delete yH;
  delete zH;
  delete hH;
  delete lH;
  for (int i=0; i < Nsrc; i++) delete xmH[i];
  for (int i=0; i < Msrc; i++) delete ymH[i];
  for (int i=0; i < Nsrc; i++) delete zmH[i];
  xmH.clear();
  ymH.clear();
  zmH.clear();
}


double benchmark(int kernel, const int niter) {

  double a, b, c;
  quda::Complex a2, b2, c2;
  quda::Complex * A = new quda::Complex[Nsrc*Msrc];
  quda::Complex * B = new quda::Complex[Nsrc*Msrc];
  quda::Complex * C = new quda::Complex[Nsrc*Msrc];

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  {
    switch (kernel) {

    case 0:
#if ZERO_FOR_R2_ONE_FOR_PAP == 0
      for (int i=0; i < niter; ++i) blas::cDotProduct(A, xmD->Components(), xmD->Components());
#elif ZERO_FOR_R2_ONE_FOR_PAP == 1
      for (int i=0; i < niter; ++i) blas::cDotProduct(A, xmD->Components(), ymD->Components());
#endif
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
  delete[] A;
  delete[] B;
  delete[] C;
  double secs = runTime / 1000;
  return secs;
}

#define ERROR(a) fabs(blas::norm2(*a##D) - blas::norm2(*a##H)) / blas::norm2(*a##H)

double test(int kernel) {

  double a = M_PI, b = M_PI*exp(1.0), c = sqrt(M_PI);
  quda::Complex a2(a, b), b2(b, -c), c2(a+b, c*a);
  double error = 0;
  quda::Complex * A = new quda::Complex[Nsrc*Msrc];
  quda::Complex * B = new quda::Complex[Nsrc*Msrc];
  quda::Complex * C = new quda::Complex[Nsrc*Msrc];
  for(int i=0; i < Nsrc*Msrc; i++){
    A[i] = a2*  (1.0*((i/Nsrc) + i)) + b2 * (1.0*i) + c2 *(1.0*(Nsrc*Msrc/2-i));
    B[i] = a2*  (1.0*((i/Nsrc) + i)) - b2 * (M_PI*i) + c2 *(1.0*(Nsrc*Msrc/2-i));
    C[i] = a2*  (1.0*((M_PI/Nsrc) + i)) + b2 * (1.0*i) + c2 *(1.0*(Nsrc*Msrc/2-i));
  }
  // A[0] = a2;
  // A[1] = 0.;
  // A[2] = 0.;
  // A[3] = 0.;

  switch (kernel) {

  case 0:
    for (int i=0; i < Nsrc; i++) {
      xmD->Component(i) = *(xmH[i]);
    }
    for (int i=0; i < Msrc; i++) {
#if ZERO_FOR_R2_ONE_FOR_PAP == 0
      ymD->Component(i) = *(xmH[i]);
#elif ZERO_FOR_R2_ONE_FOR_PAP == 1
      ymD->Component(i) = *(ymH[i]);
#endif
    }

    blas::cDotProduct(A, xmD->Components(), ymD->Components());

    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) {
        B[j*Nsrc+i] = blas::cDotProduct(xmD->Component(i), ymD->Component(j));
      }
    }
    error = 0;
    for (int i = 0; i < Nsrc*Msrc; i++)
    {
      error+= fabs(A[i] - B[i])/fabs(B[i]);
    }
    error /= (Nsrc*Msrc);
    break; 

  default:
    errorQuda("Undefined blas kernel %d\n", kernel);
  }
  delete[] A;
  delete[] B;
  delete[] C;
  return error;
}

const char *prec_str[] = {"half", "single", "double"};

const char *names[] = {
  "cDotProduct (block)"
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

  // override spin setting if mg solver is set to test coarse grids
  if (inv_type == QUDA_MG_INVERTER) {
    Nspin = 2;
    Ncolor = nvec;
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
    if (Nspin == 2 && prec == 0) continue;

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

  // certain tests will fail to run for coarse grids so mark these as
  // failed without running
  double deviation =  skip_kernel(prec,kernel) ? 1.0 : test(kernel);
  printfQuda("%-35s error = %e\n", names[kernel], deviation);
  double tol = (prec == 2 ? 1e-11 : (prec == 1 ? 1e-5 : 1e-3));
  tol = (kernel < 2) ? 1e-4 : tol; // use different tolerance for copy
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

// half precision
INSTANTIATE_TEST_CASE_P(multicDotProduct_half, BlasTest, ::testing::Values( make_int2(0,0) ));

// single precision
INSTANTIATE_TEST_CASE_P(multicDotProduct_single, BlasTest, ::testing::Values( make_int2(1,0) ));

// double precision
INSTANTIATE_TEST_CASE_P(multicDotProduct_double, BlasTest, ::testing::Values( make_int2(2,0) ));


