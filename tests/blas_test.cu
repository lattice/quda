#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>

// include because of nasty globals used in the tests
#include <dslash_reference.h>

// google test
#include <gtest/gtest.h>

constexpr int Nkernels = 43;

using namespace quda;

ColorSpinorField *xH, *yH, *zH, *wH, *vH, *hH, *mH, *lH;
ColorSpinorField *xD, *yD, *zD, *wD, *vD, *hD, *mD, *lD, *xmD, *ymD, *zmD;
std::vector<cpuColorSpinorField*> xmH;
std::vector<cpuColorSpinorField*> ymH;
std::vector<cpuColorSpinorField*> zmH;
int Nspin;
int Ncolor;

void setPrec(ColorSpinorParam &param, QudaPrecision precision, int order = 0)
{
  param.setPrecision(precision);
  if (order == 2) {
    param.fieldOrder = QUDA_FLOAT8_FIELD_ORDER;
  } else if (order == 1) {
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  } else if (Nspin == 1 || Nspin == 2 || precision == QUDA_DOUBLE_PRECISION) {
    param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  } else {
    param.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
  }
}

void display_test_info()
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

const char *prec_str[] = {"quarter", "half", "single", "double"};
const char *order_str[] = {"default", "float2", "float8"};

// For googletest names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore
const char *names[] = {"copyHS",
                       "copyMS",
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
                       "caxpyXmaz",
                       "norm",
                       "reDotProduct",
                       "axpyNorm",
                       "xmyNorm",
                       "caxpyNorm",
                       "caxpyXmazNormX",
                       "cabxpyzAxNorm",
                       "cDotProduct",
                       "caxpyDotzy",
                       "cDotProductNormA",
                       "cDotProductNormB",
                       "caxpbypzYmbwcDotProductUYNormY",
                       "HeavyQuarkResidualNorm",
                       "xpyHeavyQuarkResidualNorm",
                       "tripleCGReduction",
                       "tripleCGUpdate",
                       "axpyReDot",
                       "caxpy_block",
                       "axpyBzpcx_block",
                       "caxpyBxpz",
                       "caxpyBzpx",
                       "cDotProductNorm_block",
                       "cDotProduct_block",
                       "reDotProductNorm_block",
                       "reDotProduct_block",
                       "axpy_block"};

// kernels that utilize multi-blas
bool is_multi(int kernel) { return std::string(names[kernel]).find("_block") != std::string::npos ? true : false; }

// kernels that require site unrolling
bool is_site_unroll(int kernel) { return std::string(names[kernel]).find("HeavyQuark") != std::string::npos ? true : false; }

bool skip_kernel(int precision, int kernel, int order)
{
  if ((QUDA_PRECISION & getPrecision(precision)) == 0) return true;

  // if we've selected a given kernel then make sure we only run that
  if (test_type != -1 && kernel != test_type) return true;

  // if we've selected a given precision then make sure we only run that
  auto this_prec = getPrecision(precision);
  if (prec != QUDA_INVALID_PRECISION && this_prec != prec) return true;

  if ( Nspin == 2 && ( precision == 0 || precision ==1 ) ) {
    // avoid quarter, half precision tests if doing coarse fields
    return true;
  } else if (Nspin == 2 && (kernel == 1 || kernel == 2)) {
    // avoid low-precision copy if doing coarse fields
    return true;
  } else if (Ncolor != 3 && is_site_unroll(kernel)) {
    // only benchmark heavy-quark norm if doing 3 colors
    return true;
  } else if ((Nprec < 4) && (kernel == 0)) {
    // only benchmark high-precision copy() if double is supported
    return true;
  }

  if (order == 1) {
#ifdef GPU_MULTIGRID
    // order == 1 represents the case of multigrid testing for float-2
    // ordered nspin-4 fields in single precision and less, skip all other cases
    if (Nspin == 1 || Nspin == 2 || this_prec == QUDA_DOUBLE_PRECISION) {
      return true;
    } else if (Nspin == 4 && (this_prec != QUDA_DOUBLE_PRECISION && is_multi(kernel) ||
                              this_prec == QUDA_SINGLE_PRECISION && is_site_unroll(kernel))) {
      // we don't instantiate multi-blas kernels for float-2 nspin-4
      // fields, so skip these
      return true;
    }
#else
    return true;
#endif
  }

  // this is for float-8 testing
  if (order == 2) {
#ifdef FLOAT8
    // order == 2 represents the case of float-8 nspin-4 fields
    // only run fixed-precision fields, skip all other cases
    if (Nspin == 1 || Nspin == 2 || this_prec >= QUDA_HALF_PRECISION) {
      return true;
    } else if (Nspin == 4 && is_multi(kernel)) {
      // we currently don't instantiate multi-blas kernels for float-8
      // fields, so skip these
      return true;
    }
#else
    return true;
#endif
  }

  return false;
}

void initFields(int prec, int order)
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

  switch (solve_type) {
  case QUDA_DIRECT_PC_SOLVE:
  case QUDA_NORMOP_PC_SOLVE: param.siteSubset = QUDA_PARITY_SITE_SUBSET; break;
  case QUDA_DIRECT_SOLVE:
  case QUDA_NORMOP_SOLVE: param.siteSubset = QUDA_FULL_SITE_SUBSET; break;
  default: errorQuda("Unexpected solve_type=%d\n", solve_type);
  }

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
  static_cast<cpuColorSpinorField*>(mH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
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
    setPrec(param, QUDA_QUARTER_PRECISION, order);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_SINGLE_PRECISION;
    low_aux_prec = QUDA_HALF_PRECISION;
    break;
  case 1:
    setPrec(param, QUDA_HALF_PRECISION, order);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_SINGLE_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  case 2:
    setPrec(param, QUDA_SINGLE_PRECISION, order);
    high_aux_prec = QUDA_DOUBLE_PRECISION;
    mid_aux_prec = QUDA_HALF_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  case 3:
    setPrec(param, QUDA_DOUBLE_PRECISION, order);
    high_aux_prec = QUDA_SINGLE_PRECISION;
    mid_aux_prec = QUDA_HALF_PRECISION;
    low_aux_prec = QUDA_QUARTER_PRECISION;
    break;
  default:
    errorQuda("Precision option not defined");
  }

  // ensure we don't enable copying between precisions that are not compiled
  if ( (high_aux_prec != QUDA_DOUBLE_PRECISION) && !(high_aux_prec & QUDA_PRECISION) ) high_aux_prec = getPrecision(prec);
  if ( (mid_aux_prec != QUDA_DOUBLE_PRECISION) && !(mid_aux_prec & QUDA_PRECISION) ) mid_aux_prec = getPrecision(prec);
  if ( (low_aux_prec != QUDA_DOUBLE_PRECISION) && !(low_aux_prec & QUDA_PRECISION) ) low_aux_prec = getPrecision(prec);

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

  setPrec(param, mid_aux_prec);
  mD = new cudaColorSpinorField(param);

  setPrec(param, low_aux_prec);
  lD = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  // only do copy if not doing half precision with mg
  bool flag = !(param.nSpin == 2 &&
		(prec == 0 || prec == 1 || low_aux_prec == QUDA_HALF_PRECISION || mid_aux_prec == QUDA_HALF_PRECISION ||
                                low_aux_prec == QUDA_QUARTER_PRECISION || mid_aux_prec == QUDA_QUARTER_PRECISION) );

  if ( flag ) {

    *vD = *vH;
    *wD = *wH;
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *hD = *hH;
    *mD = *mH;
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
  delete mD;
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
  delete mH;
  delete lH;
  for (int i=0; i < Nsrc; i++) delete xmH[i];
  for (int i=0; i < Msrc; i++) delete ymH[i];
  for (int i=0; i < Nsrc; i++) delete zmH[i];
  xmH.clear();
  ymH.clear();
  zmH.clear();
}


double benchmark(int kernel, const int niter) {

  double a = 1.0, b = 2.0, c = 3.0;
  quda::Complex a2, b2;
  quda::Complex * A = new quda::Complex[Nsrc*Msrc];
  quda::Complex * B = new quda::Complex[Nsrc*Msrc];
  quda::Complex * C = new quda::Complex[Nsrc*Msrc];
  quda::Complex * A2 = new quda::Complex[Nsrc*Nsrc]; // for the block cDotProductNorm test
  double *Ar = new double[Nsrc * Msrc];

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

    case 3:
      for (int i=0; i < niter; ++i) blas::axpby(a, *xD, b, *yD);
      break;

    case 4:
      for (int i=0; i < niter; ++i) blas::xpy(*xD, *yD);
      break;

    case 5:
      for (int i=0; i < niter; ++i) blas::axpy(a, *xD, *yD);
      break;

    case 6:
      for (int i=0; i < niter; ++i) blas::xpay(*xD, a, *yD);
      break;

    case 7:
      for (int i=0; i < niter; ++i) blas::mxpy(*xD, *yD);
      break;

    case 8:
      for (int i=0; i < niter; ++i) blas::ax(a, *xD);
      break;

    case 9:
      for (int i=0; i < niter; ++i) blas::caxpy(a2, *xD, *yD);
      break;

    case 10:
      for (int i=0; i < niter; ++i) blas::caxpby(a2, *xD, b2, *yD);
      break;

    case 11:
      for (int i=0; i < niter; ++i) blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
      break;

    case 12:
      for (int i=0; i < niter; ++i) blas::axpyBzpcx(a, *xD, *yD, b, *zD, c);
      break;

    case 13:
      for (int i=0; i < niter; ++i) blas::axpyZpbx(a, *xD, *yD, *zD, b);
      break;

    case 14:
      for (int i=0; i < niter; ++i) blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
      break;

    case 15:
      for (int i=0; i < niter; ++i) blas::cabxpyAx(a, b2, *xD, *yD);
      break;

    case 16:
      for (int i=0; i < niter; ++i) blas::caxpyXmaz(a2, *xD, *yD, *zD);
      break;

      // double
    case 17:
      for (int i=0; i < niter; ++i) blas::norm2(*xD);
      break;

    case 18:
      for (int i=0; i < niter; ++i) blas::reDotProduct(*xD, *yD);
      break;

    case 19:
      for (int i=0; i < niter; ++i) blas::axpyNorm(a, *xD, *yD);
      break;

    case 20:
      for (int i=0; i < niter; ++i) blas::xmyNorm(*xD, *yD);
      break;

    case 21:
      for (int i=0; i < niter; ++i) blas::caxpyNorm(a2, *xD, *yD);
      break;

    case 22:
      for (int i=0; i < niter; ++i) blas::caxpyXmazNormX(a2, *xD, *yD, *zD);
      break;

    case 23:
      for (int i=0; i < niter; ++i) blas::cabxpyzAxNorm(a, b2, *xD, *yD, *yD);
      break;

    // double2
    case 24:
      for (int i=0; i < niter; ++i) blas::cDotProduct(*xD, *yD);
      break;

    case 25:
      for (int i=0; i < niter; ++i) blas::caxpyDotzy(a2, *xD, *yD, *zD);
      break;

    // double3
    case 26:
      for (int i=0; i < niter; ++i) blas::cDotProductNormA(*xD, *yD);
      break;

    case 27:
      for (int i=0; i < niter; ++i) blas::cDotProductNormB(*xD, *yD);
      break;

    case 28:
      for (int i=0; i < niter; ++i) blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zD, *wD, *vD);
      break;

    case 29:
      for (int i=0; i < niter; ++i) blas::HeavyQuarkResidualNorm(*xD, *yD);
      break;

    case 30:
      for (int i=0; i < niter; ++i) blas::xpyHeavyQuarkResidualNorm(*xD, *yD, *zD);
      break;

    case 31:
      for (int i=0; i < niter; ++i) blas::tripleCGReduction(*xD, *yD, *zD);
      break;

    case 32:
      for (int i=0; i < niter; ++i) blas::tripleCGUpdate(a, b, *xD, *yD, *zD, *wD);
      break;

    case 33:
      for (int i=0; i < niter; ++i) blas::axpyReDot(a, *xD, *yD);
      break;

    case 34:
      for (int i=0; i < niter; ++i) blas::caxpy(A, *xmD,* ymD);
      break;

    case 35:
      for (int i=0; i < niter; ++i) blas::axpyBzpcx((double*)A, xmD->Components(), zmD->Components(), (double*)B, *yD, (double*)C);
      break;

    case 36:
      for (int i=0; i < niter; ++i) blas::caxpyBxpz(a2, *xD, *yD, b2, *zD);
      break;

    case 37:
      for (int i=0; i < niter; ++i) blas::caxpyBzpx(a2, *xD, *yD, b2, *zD);
      break;

    case 38:
      for (int i=0; i < niter; ++i) blas::cDotProduct(A2, xmD->Components(), xmD->Components());
      break;

    case 39:
      for (int i=0; i < niter; ++i) blas::cDotProduct(A, xmD->Components(), ymD->Components());
      break;

    case 40:
      for (int i=0; i < niter; ++i) blas::reDotProduct((double*)A2, xmD->Components(), xmD->Components());
      break;

    case 41:
      for (int i=0; i < niter; ++i) blas::reDotProduct((double*)A, xmD->Components(), ymD->Components());
      break;

    case 42:
      for (int i = 0; i < niter; ++i) blas::axpy(Ar, xmD->Components(), ymD->Components());
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
  delete[] A2;
  delete[] Ar;
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
  quda::Complex * A2 = new quda::Complex[Nsrc*Nsrc]; // for the block cDotProductNorm test
  quda::Complex * B2 = new quda::Complex[Nsrc*Nsrc]; // for the block cDotProductNorm test
  double *Ar = new double[Nsrc * Msrc];

  for (int i = 0; i < Nsrc * Msrc; i++) {
    A[i] = a2 * (1.0 * ((i / (double)Nsrc) + i)) + b2 * (1.0 * i) + c2 * (1.0 * (0.5 * Nsrc * Msrc - i));
    B[i] = a2 * (1.0 * ((i / (double)Nsrc) + i)) - b2 * (M_PI * i) + c2 * (1.0 * (0.5 * Nsrc * Msrc - i));
    C[i] = a2 * (1.0 * ((M_PI / (double)Nsrc) + i)) + b2 * (1.0 * i) + c2 * (1.0 * (0.5 * Nsrc * Msrc - i));
    Ar[i] = A[i].real();
  }
  for (int i = 0; i < Nsrc * Nsrc; i++) {
    A2[i] = a2 * (1.0 * ((i / (double)Nsrc) + i)) + b2 * (1.0 * i) + c2 * (1.0 * (0.5 * Nsrc * Nsrc - i));
    B2[i] = a2 * (1.0 * ((i / (double)Nsrc) + i)) - b2 * (M_PI * i) + c2 * (1.0 * (0.5 * Nsrc * Nsrc - i));
  }
  // A[0] = a2;
  // A[1] = 0.;
  // A[2] = 0.;
  // A[3] = 0.;

  switch (kernel) {

  case 0:
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

  case 3:
    *xD = *xH;
    *yD = *yH;
    blas::axpby(a, *xD, b, *yD);
    blas::axpby(a, *xH, b, *yH);
    error = ERROR(y);
    break;

  case 4:
    *xD = *xH;
    *yD = *yH;
    blas::xpy(*xD, *yD);
    blas::xpy(*xH, *yH);
    error = ERROR(y);
    break;

  case 5:
    *xD = *xH;
    *yD = *yH;
    blas::axpy(a, *xD, *yD);
    blas::axpy(a, *xH, *yH);
    *zH = *yD;
    error = ERROR(y);
    break;

  case 6:
    *xD = *xH;
    *yD = *yH;
    blas::xpay(*xD, a, *yD);
    blas::xpay(*xH, a, *yH);
    error = ERROR(y);
    break;

  case 7:
    *xD = *xH;
    *yD = *yH;
    blas::mxpy(*xD, *yD);
    blas::mxpy(*xH, *yH);
    error = ERROR(y);
    break;

  case 8:
    *xD = *xH;
    blas::ax(a, *xD);
    blas::ax(a, *xH);
    error = ERROR(x);
    break;

  case 9:
    *xD = *xH;
    *yD = *yH;
    blas::caxpy(a2, *xD, *yD);
    blas::caxpy(a2, *xH, *yH);
    error = ERROR(y);
    break;

  case 10:
    *xD = *xH;
    *yD = *yH;
    blas::caxpby(a2, *xD, b2, *yD);
    blas::caxpby(a2, *xH, b2, *yH);
    error = ERROR(y);
    break;

  case 11:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
    blas::cxpaypbz(*xH, a2, *yH, b2, *zH);
    error = ERROR(z);
    break;

  case 12:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::axpyBzpcx(a, *xD, *yD, b, *zD, c);
    blas::axpyBzpcx(a, *xH, *yH, b, *zH, c);
    error = ERROR(x) + ERROR(y);
    break;

  case 13:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::axpyZpbx(a, *xD, *yD, *zD, b);
    blas::axpyZpbx(a, *xH, *yH, *zH, b);
    error = ERROR(x) + ERROR(y);
    break;

  case 14:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
    blas::caxpbypzYmbw(a2, *xH, b2, *yH, *zH, *wH);
    error = ERROR(z) + ERROR(y);
    break;

  case 15:
    *xD = *xH;
    *yD = *yH;
    blas::cabxpyAx(a, b2, *xD, *yD);
    blas::cabxpyAx(a, b2, *xH, *yH);
    error = ERROR(y) + ERROR(x);
    break;

  case 16:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {blas::caxpyXmaz(a, *xD, *yD, *zD);
     blas::caxpyXmaz(a, *xH, *yH, *zH);
     error = ERROR(y) + ERROR(x);}
    break;

  case 17:
    *xD = *xH;
    *yH = *xD;
    error = fabs(blas::norm2(*xD) - blas::norm2(*xH)) / blas::norm2(*xH);
    break;

  case 18:
    *xD = *xH;
    *yD = *yH;
    error = fabs(blas::reDotProduct(*xD, *yD) - blas::reDotProduct(*xH, *yH)) / fabs(blas::reDotProduct(*xH, *yH));
    break;

  case 19:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::axpyNorm(a, *xD, *yD);
    double h = blas::axpyNorm(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 20:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::xmyNorm(*xD, *yD);
    double h = blas::xmyNorm(*xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 21:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::caxpyNorm(a, *xD, *yD);
    double h = blas::caxpyNorm(a, *xH, *yH);
    error = ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 22:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {double d = blas::caxpyXmazNormX(a, *xD, *yD, *zD);
      double h = blas::caxpyXmazNormX(a, *xH, *yH, *zH);
      error = ERROR(y) + ERROR(x) + fabs(d-h)/fabs(h);}
    break;

  case 23:
    *xD = *xH;
    *yD = *yH;
    {double d = blas::cabxpyzAxNorm(a, b2, *xD, *yD, *yD);
      double h = blas::cabxpyzAxNorm(a, b2, *xH, *yH, *yH);
      error = ERROR(x) + ERROR(y) + fabs(d-h)/fabs(h);}
    break;

  case 24:
    *xD = *xH;
    *yD = *yH;
    error = abs(blas::cDotProduct(*xD, *yD) - blas::cDotProduct(*xH, *yH)) / abs(blas::cDotProduct(*xH, *yH));
    break;

  case 25:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {quda::Complex d = blas::caxpyDotzy(a, *xD, *yD, *zD);
      quda::Complex h = blas::caxpyDotzy(a, *xH, *yH, *zH);
    error = ERROR(y) + abs(d-h)/abs(h);}
    break;

  case 26:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::cDotProductNormA(*xD, *yD);
      double3 h = blas::cDotProductNormA(*xH, *yH);
      error = abs(Complex(d.x - h.x, d.y - h.y)) / abs(Complex(h.x, h.y)) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case 27:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::cDotProductNormB(*xD, *yD);
      double3 h = blas::cDotProductNormB(*xH, *yH);
      error = abs(Complex(d.x - h.x, d.y - h.y)) / abs(Complex(h.x, h.y)) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case 28:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    *vD = *vH;
    { double3 d = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zD, *wD, *vD);
      double3 h = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xH, b2, *yH, *zH, *wH, *vH);
      error = ERROR(z) + ERROR(y) + abs(Complex(d.x - h.x, d.y - h.y)) / abs(Complex(h.x, h.y))
          + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case 29:
    *xD = *xH;
    *yD = *yH;
    { double3 d = blas::HeavyQuarkResidualNorm(*xD, *yD);
      double3 h = blas::HeavyQuarkResidualNorm(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) +
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  case 30:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    { double3 d = blas::xpyHeavyQuarkResidualNorm(*xD, *yD, *zD);
      double3 h = blas::xpyHeavyQuarkResidualNorm(*xH, *yH, *zH);
      error = ERROR(y) + fabs(d.x - h.x) / fabs(h.x) +
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  case 31:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    { double3 d = blas::tripleCGReduction(*xD, *yD, *zD);
      double3 h = make_double3(blas::norm2(*xH), blas::norm2(*yH), blas::reDotProduct(*yH, *zH));
      error = fabs(d.x - h.x) / fabs(h.x) +
	fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z); }
    break;

  case 32:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    { blas::tripleCGUpdate(a, b, *xD, *yD, *zD, *wD);
      blas::tripleCGUpdate(a, b, *xH, *yH, *zH, *wH);
      error = ERROR(y) + ERROR(z) + ERROR(w); }
    break;

  case 33:
    *xD = *xH;
    *yD = *yH;
    { double d = blas::axpyReDot(a, *xD, *yD);
      double h = blas::axpyReDot(a, *xH, *yH);
      error = ERROR(y) + fabs(d-h)/fabs(h); }
    break;

  case 34:
    for (int i=0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i=0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);

    blas::caxpy(A, *xmD, *ymD);
    for (int i=0; i < Nsrc; i++){
      for(int j=0; j < Msrc; j++){
	blas::caxpy(A[Msrc*i+j], *(xmH[i]), *(ymH[j]));
      }
    }
    error = 0;
    for (int i=0; i < Msrc; i++){
      error+= fabs(blas::norm2((ymD->Component(i))) - blas::norm2(*(ymH[i]))) / blas::norm2(*(ymH[i]));
    }
    error/= Msrc;
    break;

  case 35:
    for (int i=0; i < Nsrc; i++) {
      xmD->Component(i) = *(xmH[i]);
      zmD->Component(i) = *(zmH[i]);
    }
    *yD = *yH;

    blas::axpyBzpcx((double*)A, xmD->Components(), zmD->Components(), (double*)B, *yD, (const double*)C);

    for (int i=0; i<Nsrc; i++) {
      blas::axpyBzpcx(((double*)A)[i], *xmH[i], *zmH[i], ((double*)B)[i], *yH, ((double*)C)[i]);
    }

    error = 0;
    for (int i=0; i < Nsrc; i++){
      error+= fabs(blas::norm2((xmD->Component(i))) - blas::norm2(*(xmH[i]))) / blas::norm2(*(xmH[i]));
      //error+= fabs(blas::norm2((zmD->Component(i))) - blas::norm2(*(zmH[i]))) / blas::norm2(*(zmH[i]));
    }
    error/= Nsrc;
    break;

  case 36:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {blas::caxpyBxpz(a, *xD, *yD, b2, *zD);
     blas::caxpyBxpz(a, *xH, *yH, b2, *zH);
     error = ERROR(x) + ERROR(z);}
    break;

  case 37:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {blas::caxpyBzpx(a, *xD, *yD, b2, *zD);
     blas::caxpyBzpx(a, *xH, *yH, b2, *zH);
     error = ERROR(x) + ERROR(z);}
    break;

  case 38:
    for (int i=0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    blas::cDotProduct(A2, xmD->Components(), xmD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Nsrc; j++) {
	B2[i*Nsrc+j] = blas::cDotProduct(xmD->Component(i), xmD->Component(j));
	error += std::abs(A2[i*Nsrc+j] - B2[i*Nsrc+j])/std::abs(B2[i*Nsrc+j]);
      }
    }
    error /= Nsrc*Nsrc;
    break;

  case 39:
    for (int i=0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i=0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);
    blas::cDotProduct(A, xmD->Components(), ymD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) {
	B[i*Msrc+j] = blas::cDotProduct(xmD->Component(i), ymD->Component(j));
	error += std::abs(A[i*Msrc+j] - B[i*Msrc+j])/std::abs(B[i*Msrc+j]);
      }
    }
    error /= Nsrc*Msrc;
    break;

  case 40:
    for (int i=0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    blas::reDotProduct((double*)A2, xmD->Components(), xmD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Nsrc; j++) {
        ((double*)B2)[i*Nsrc+j] = blas::reDotProduct(xmD->Component(i), xmD->Component(j));
        error += std::abs(((double*)A2)[i*Nsrc+j] - ((double*)B2)[i*Nsrc+j])/std::abs(((double*)B2)[i*Nsrc+j]);
      }
    }
    error /= Nsrc*Nsrc;
    break;

  case 41:
    for (int i=0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i=0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);
    blas::reDotProduct((double*)A, xmD->Components(), ymD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) {
        ((double*)B)[i*Msrc+j] = blas::reDotProduct(xmD->Component(i), ymD->Component(j));
        error += std::abs(((double*)A)[i*Msrc+j] - ((double*)B)[i*Msrc+j])/std::abs(((double*)B)[i*Msrc+j]);
      }
    }
    error /= Nsrc*Msrc;
    break;

  case 42:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i = 0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);

    blas::axpy(Ar, *xmD, *ymD);
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) { blas::axpy(Ar[Msrc * i + j], *(xmH[i]), *(ymH[j])); }
    }

    error = 0;
    for (int i = 0; i < Msrc; i++) {
      error += fabs(blas::norm2((ymD->Component(i))) - blas::norm2(*(ymH[i]))) / blas::norm2(*(ymH[i]));
    }
    error /= Msrc;
    break;

  default:
    errorQuda("Undefined blas kernel %d\n", kernel);
  }
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] A2;
  delete[] B2;
  delete[] Ar;
  return error;
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int result = 0;

  prec = QUDA_INVALID_PRECISION;
  test_type = -1;

  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);

  app->add_option("--test", test_type, "Kernel to test (-1: -> all kernels)")->check(CLI::Range(0, Nkernels - 1));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
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

  setVerbosity(verbosity);

  // clear the error state
  cudaGetLastError();

  // lastly check for correctness
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
  result = RUN_ALL_TESTS();

  endQuda();

  finalizeComms();
  return result;
}

// The following tests each kernel at each precision using the google testing framework

using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;

class BlasTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>>
{
protected:
  ::testing::tuple<int, int, int> param;
  const int &prec;
  const int &kernel;
  const int &order;

public:
  BlasTest() :
    param(GetParam()),
    prec(::testing::get<0>(param)),
    kernel(::testing::get<1>(param)),
    order(::testing::get<2>(param))
  {
  }
  virtual void SetUp() {
    if (!skip_kernel(prec, kernel, order)) initFields(prec, order);
  }
  virtual void TearDown()
  {
    if (!skip_kernel(prec, kernel, order)) {
      freeFields();
    }
  }
};

TEST_P(BlasTest, verify) {
  int prec = ::testing::get<0>(GetParam());
  int kernel = ::testing::get<1>(GetParam());
  int order = ::testing::get<2>(GetParam());
  if (skip_kernel(prec, kernel, order)) GTEST_SKIP();

  // certain tests will fail to run for coarse grids so mark these as
  // failed without running
  double deviation = test(kernel);
  // printfQuda("%-35s error = %e\n", names[kernel], deviation);
  double tol = (prec == 3 ? 1e-12 : (prec == 2 ? 1e-6 : (prec == 1 ? 1e-4 : 1e-2)));
  tol = (kernel < 4) ? 5e-2 : tol; // use different tolerance for copy
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

TEST_P(BlasTest, benchmark) {
  int prec = ::testing::get<0>(GetParam());
  int kernel = ::testing::get<1>(GetParam());
  int order = ::testing::get<2>(GetParam());
  if (skip_kernel(prec, kernel, order)) GTEST_SKIP();

  // do the initial tune
  benchmark(kernel, 1);

  // now rerun with more iterations to get accurate speed measurements
  quda::blas::flops = 0;
  quda::blas::bytes = 0;

  double secs = benchmark(kernel, niter);

  double gflops = (quda::blas::flops*1e-9)/(secs);
  double gbytes = quda::blas::bytes/(secs*1e9);
  RecordProperty("Gflops", std::to_string(gflops));
  RecordProperty("GBs", std::to_string(gbytes));
  printfQuda("%-31s: Gflop/s = %6.1f, GB/s = %6.1f\n", names[kernel], gflops, gbytes);
}

std::string getblasname(testing::TestParamInfo<::testing::tuple<int, int, int>> param)
{
  int prec = ::testing::get<0>(param.param);
  int kernel = ::testing::get<1>(param.param);
  int order = ::testing::get<2>(param.param);
  std::string str(names[kernel]);
  str += std::string("_") + std::string(prec_str[prec]);
  str += std::string("_") + std::string(order_str[order]);
  return str;
}

// instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, BlasTest, Combine(Range(0, Nprec), Range(0, Nkernels), Range(0, 3)), getblasname);
