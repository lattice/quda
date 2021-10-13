#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <timer.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>

// include because of nasty globals used in the tests
#include <dslash_reference.h>

// google test
#include <gtest/gtest.h>

using namespace quda;

/**
   This is the blas_test for checking correctness of the blas and
   reduction functions used by the lienar solvers.  Any blas kernels
   that are added to QUDA should have a test added here.

   For kernels that have mixed-precision support, we can test this
   functionality as well.  The number of precision cominations we test
   are Nprec * (Nprec + 1)/2, which corresponds to all uni-precision
   combinations ("this precision") and all combinations with the
   "other precision" > "this precision".
*/

// these are pointers to the host fields
ColorSpinorField *xH, *yH, *zH, *wH, *vH, *xoH, *yoH, *zoH;

// these are pointers to the device fields that have "this precision"
ColorSpinorField *xD, *yD, *zD, *wD, *vD;

// these are pointers to the device multi-fields that have "this precision"
ColorSpinorField *xmD, *ymD, *zmD;

// these are pointers to the device fields that have "this precision"
ColorSpinorField *xoD, *yoD, *zoD, *woD, *voD;

// these are pointers to the device multi-fields that have "other precision"
ColorSpinorField *xmoD, *ymoD, *zmoD;

// these are pointers to the host multi-fields that have "this precision"
std::vector<cpuColorSpinorField *> xmH;
std::vector<cpuColorSpinorField *> ymH;
std::vector<cpuColorSpinorField *> zmH;
int Nspin;
int Ncolor;

void setPrec(ColorSpinorParam &param, QudaPrecision precision) { param.setPrecision(precision, precision, true); }

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("S_dimension T_dimension Nspin Ncolor Msrc Nsrc\n");
  printfQuda("%3d /%3d / %3d   %3d      %d     %d     %3d  %3d\n", xdim, ydim, zdim, tdim, Nspin, Ncolor, Msrc, Nsrc);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

using prec_pair_t = std::pair<QudaPrecision, QudaPrecision>;

const std::map<QudaPrecision, std::string> prec_map = {{QUDA_QUARTER_PRECISION, "quarter"},
                                                       {QUDA_HALF_PRECISION, "half"},
                                                       {QUDA_SINGLE_PRECISION, "single"},
                                                       {QUDA_DOUBLE_PRECISION, "double"}};

const int Nprec = prec_map.size();

enum class Kernel {
  copyHS,
  copyLS,
  axpbyz,
  ax,
  caxpy,
  caxpby,
  cxpaypbz,
  axpyBzpcx,
  axpyZpbx,
  caxpbypzYmbw,
  cabxpyAx,
  caxpyXmaz,
  norm2,
  reDotProduct,
  axpbyzNorm,
  axpyCGNorm,
  caxpyNorm,
  caxpyXmazNormX,
  cabxpyzAxNorm,
  cDotProduct,
  caxpyDotzy,
  cDotProductNormA,
  caxpbypzYmbwcDotProductUYNormY,
  HeavyQuarkResidualNorm,
  xpyHeavyQuarkResidualNorm,
  tripleCGReduction,
  tripleCGUpdate,
  axpyReDot,
  caxpyBxpz,
  caxpyBzpx,
  axpy_block,
  caxpy_block,
  axpyBzpcx_block,
  reDotProductNorm_block,
  reDotProduct_block,
  cDotProductNorm_block,
  cDotProduct_block,
  caxpyXmazMR
};

// For googletest names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore
const std::map<Kernel, std::string> kernel_map
  = {{Kernel::copyHS, "copyHS"},
     {Kernel::copyLS, "copyLS"},
     {Kernel::axpbyz, "axpbyz"},
     {Kernel::ax, "ax"},
     {Kernel::caxpy, "caxpy"},
     {Kernel::caxpby, "caxpby"},
     {Kernel::cxpaypbz, "cxpaypbz"},
     {Kernel::axpyBzpcx, "axpyBzpcx"},
     {Kernel::axpyZpbx, "axpyZpbx"},
     {Kernel::caxpbypzYmbw, "caxpbypzYmbw"},
     {Kernel::cabxpyAx, "cabxpyAx"},
     {Kernel::caxpyXmaz, "caxpyXmaz"},
     {Kernel::norm2, "norm2"},
     {Kernel::reDotProduct, "reDotProduct"},
     {Kernel::axpbyzNorm, "axpbyzNorm"},
     {Kernel::axpyCGNorm, "axpyCGNorm"},
     {Kernel::caxpyNorm, "caxpyNorm"},
     {Kernel::caxpyXmazNormX, "caxpyXmazNormX"},
     {Kernel::cabxpyzAxNorm, "cabxpyzAxNorm"},
     {Kernel::cDotProduct, "cDotProduct"},
     {Kernel::caxpyDotzy, "caxpyDotzy"},
     {Kernel::cDotProductNormA, "cDotProductNormA"},
     {Kernel::caxpbypzYmbwcDotProductUYNormY, "caxpbypzYmbwcDotProductUYNormY"},
     {Kernel::HeavyQuarkResidualNorm, "HeavyQuarkResidualNorm"},
     {Kernel::xpyHeavyQuarkResidualNorm, "xpyHeavyQuarkResidualNorm"},
     {Kernel::tripleCGReduction, "tripleCGReduction"},
     {Kernel::tripleCGUpdate, "tripleCGUpdate"},
     {Kernel::axpyReDot, "axpyReDot"},
     {Kernel::caxpyBxpz, "caxpyBxpz"},
     {Kernel::caxpyBzpx, "caxpyBzpx"},
     {Kernel::axpy_block, "axpy_block"},
     {Kernel::caxpy_block, "caxpy_block"},
     {Kernel::axpyBzpcx_block, "axpyBzpcx_block"},
     {Kernel::reDotProductNorm_block, "reDotProductNorm_block"},
     {Kernel::reDotProduct_block, "reDotProduct_block"},
     {Kernel::cDotProductNorm_block, "cDotProductNorm_block"},
     {Kernel::cDotProduct_block, "cDotProduct_block"},
     {Kernel::caxpyXmazMR, "caxpyXmazMR"}};

const int Nkernels = kernel_map.size();

// kernels that utilize multi-blas
bool is_multi(Kernel kernel)
{
  return std::string(kernel_map.at(kernel)).find("_block") != std::string::npos ? true : false;
}

bool is_copy(Kernel kernel) { return (kernel == Kernel::copyHS || kernel == Kernel::copyLS); }

// kernels that require site unrolling
bool is_site_unroll(Kernel kernel)
{
  return (kernel == Kernel::HeavyQuarkResidualNorm || kernel == Kernel::xpyHeavyQuarkResidualNorm);
}

bool skip_kernel(prec_pair_t pair, Kernel kernel)
{
  auto &this_prec = pair.first;
  auto &other_prec = pair.second;

  if ((QUDA_PRECISION & this_prec) == 0) return true;
  if ((QUDA_PRECISION & other_prec) == 0) return true;

  // if we've selected a given kernel then make sure we only run that
  if (test_type != -1 && (int)kernel != test_type) return true;

  // if we've selected a given precision then make sure we only run that
  if (prec != QUDA_INVALID_PRECISION && this_prec != prec) return true;

  // if we've selected a given precision then make sure we only run that
  if (prec_sloppy != QUDA_INVALID_PRECISION && other_prec != prec_sloppy) return true;

  if (Nspin == 2 && this_prec < QUDA_SINGLE_PRECISION) {
    // avoid quarter, half precision tests if doing coarse fields
    return true;
  } else if (Ncolor != 3 && is_site_unroll(kernel)) {
    // only benchmark heavy-quark norm if doing 3 colors
    return true;
  }

  return false;
}

void initFields(prec_pair_t prec_pair)
{
  ColorSpinorParam param;
  param.nColor = Ncolor;
  param.nSpin = Nspin;
  param.nDim = 4; // number of spacetime dimensions
  param.pad = 0;  // padding must be zero for cpu fields

  switch (solve_type) {
  case QUDA_DIRECT_PC_SOLVE:
  case QUDA_NORMOP_PC_SOLVE: param.siteSubset = QUDA_PARITY_SITE_SUBSET; break;
  case QUDA_DIRECT_SOLVE:
  case QUDA_NORMOP_SOLVE: param.siteSubset = QUDA_FULL_SITE_SUBSET; break;
  default: errorQuda("Unexpected solve_type=%d\n", solve_type);
  }

  if (param.siteSubset == QUDA_PARITY_SITE_SUBSET)
    param.x[0] = xdim / 2;
  else
    param.x[0] = xdim;
  param.x[1] = ydim;
  param.x[2] = zdim;
  param.x[3] = tdim;

  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.setPrecision(QUDA_DOUBLE_PRECISION);
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.pc_type = QUDA_4D_PC;

  vH = new cpuColorSpinorField(param);
  wH = new cpuColorSpinorField(param);
  xH = new cpuColorSpinorField(param);
  yH = new cpuColorSpinorField(param);
  zH = new cpuColorSpinorField(param);

  // all host fields are double precision, so the "other" fields just alias the regular fields
  xoH = xH;
  yoH = yH;
  zoH = zH;

  xmH.reserve(Nsrc);
  for (int cid = 0; cid < Nsrc; cid++) xmH.push_back(new cpuColorSpinorField(param));
  ymH.reserve(Msrc);
  for (int cid = 0; cid < Msrc; cid++) ymH.push_back(new cpuColorSpinorField(param));
  zmH.reserve(Nsrc);
  for (int cid = 0; cid < Nsrc; cid++) zmH.push_back(new cpuColorSpinorField(param));

  static_cast<cpuColorSpinorField *>(vH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField *>(wH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField *>(xH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField *>(yH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  static_cast<cpuColorSpinorField *>(zH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  for (int i = 0; i < Nsrc; i++) { static_cast<cpuColorSpinorField *>(xmH[i])->Source(QUDA_RANDOM_SOURCE, 0, 0, 0); }
  for (int i = 0; i < Msrc; i++) { static_cast<cpuColorSpinorField *>(ymH[i])->Source(QUDA_RANDOM_SOURCE, 0, 0, 0); }
  // Now set the parameters for the cuda fields
  // param.pad = xdim*ydim*zdim/2;

  if (param.nSpin == 4) param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  param.create = QUDA_ZERO_FIELD_CREATE;

  QudaPrecision prec = prec_pair.first;
  QudaPrecision prec_other = prec_pair.second;

  param.setPrecision(prec, prec, true);
  vD = new cudaColorSpinorField(param);
  wD = new cudaColorSpinorField(param);
  xD = new cudaColorSpinorField(param);
  yD = new cudaColorSpinorField(param);
  zD = new cudaColorSpinorField(param);

  param.setPrecision(prec_other, prec_other, true);
  voD = new cudaColorSpinorField(param);
  woD = new cudaColorSpinorField(param);
  xoD = new cudaColorSpinorField(param);
  yoD = new cudaColorSpinorField(param);
  zoD = new cudaColorSpinorField(param);

  // create composite fields
  param.is_composite = true;
  param.is_component = false;

  param.setPrecision(prec, prec, true);
  param.composite_dim = Nsrc;
  xmD = new cudaColorSpinorField(param);

  param.composite_dim = Msrc;
  ymD = new cudaColorSpinorField(param);

  param.composite_dim = Nsrc;
  zmD = new cudaColorSpinorField(param);

  param.setPrecision(prec_other, prec_other, true);
  param.composite_dim = Nsrc;
  xmoD = new cudaColorSpinorField(param);

  param.composite_dim = Msrc;
  ymoD = new cudaColorSpinorField(param);

  param.composite_dim = Nsrc;
  zmoD = new cudaColorSpinorField(param);

  // only do copy if not doing half precision with mg
  bool flag = !(param.nSpin == 2 && (prec < QUDA_SINGLE_PRECISION || prec_other < QUDA_HALF_PRECISION));

  if (flag) {
    *vD = *vH;
    *wD = *wH;
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
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
  delete voD;
  delete woD;
  delete xoD;
  delete yoD;
  delete zoD;
  delete xmD;
  delete ymD;
  delete zmD;
  delete xmoD;
  delete ymoD;
  delete zmoD;

  // release memory
  delete vH;
  delete wH;
  delete xH;
  delete yH;
  delete zH;
  for (int i = 0; i < Nsrc; i++) delete xmH[i];
  for (int i = 0; i < Msrc; i++) delete ymH[i];
  for (int i = 0; i < Nsrc; i++) delete zmH[i];
  xmH.clear();
  ymH.clear();
  zmH.clear();
}

double benchmark(Kernel kernel, const int niter)
{
  double a = 1.0, b = 2.0, c = 3.0;
  quda::Complex a2, b2;
  quda::Complex *A = new quda::Complex[Nsrc * Msrc];
  quda::Complex *B = new quda::Complex[Nsrc * Msrc];
  quda::Complex *C = new quda::Complex[Nsrc * Msrc];
  quda::Complex *A2 = new quda::Complex[Nsrc * Nsrc]; // for the block cDotProductNorm test
  double *Ar = new double[Nsrc * Msrc];

  device_timer_t timer;
  timer.start();

  {
    switch (kernel) {

    case Kernel::copyHS:
      for (int i = 0; i < niter; ++i) blas::copy(*yD, *xoD);
      break;

    case Kernel::copyLS:
      for (int i = 0; i < niter; ++i) blas::copy(*yoD, *xD);
      break;

    case Kernel::axpbyz:
      for (int i = 0; i < niter; ++i) blas::axpbyz(a, *xD, b, *yoD, *zoD);
      break;

    case Kernel::ax:
      for (int i = 0; i < niter; ++i) blas::ax(a, *xD);
      break;

    case Kernel::caxpy:
      for (int i = 0; i < niter; ++i) blas::caxpy(a2, *xD, *yoD);
      break;

    case Kernel::caxpby:
      for (int i = 0; i < niter; ++i) blas::caxpby(a2, *xD, b2, *yD);
      break;

    case Kernel::cxpaypbz:
      for (int i = 0; i < niter; ++i) blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
      break;

    case Kernel::axpyBzpcx:
      for (int i = 0; i < niter; ++i) blas::axpyBzpcx(a, *xD, *yoD, b, *zD, c);
      break;

    case Kernel::axpyZpbx:
      for (int i = 0; i < niter; ++i) blas::axpyZpbx(a, *xD, *yoD, *zD, b);
      break;

    case Kernel::caxpbypzYmbw:
      for (int i = 0; i < niter; ++i) blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
      break;

    case Kernel::cabxpyAx:
      for (int i = 0; i < niter; ++i) blas::cabxpyAx(a, b2, *xD, *yD);
      break;

    case Kernel::caxpyXmaz:
      for (int i = 0; i < niter; ++i) blas::caxpyXmaz(a2, *xD, *yD, *zD);
      break;

    case Kernel::norm2:
      for (int i = 0; i < niter; ++i) blas::norm2(*xD);
      break;

    case Kernel::reDotProduct:
      for (int i = 0; i < niter; ++i) blas::reDotProduct(*xD, *yD);
      break;

    case Kernel::axpbyzNorm:
      for (int i = 0; i < niter; ++i) blas::axpbyzNorm(a, *xD, b, *yD, *zD);
      break;

    case Kernel::axpyCGNorm:
      for (int i = 0; i < niter; ++i) blas::axpyCGNorm(a, *xD, *yoD);
      break;

    case Kernel::caxpyNorm:
      for (int i = 0; i < niter; ++i) blas::caxpyNorm(a2, *xD, *yD);
      break;

    case Kernel::caxpyXmazNormX:
      for (int i = 0; i < niter; ++i) blas::caxpyXmazNormX(a2, *xD, *yD, *zD);
      break;

    case Kernel::cabxpyzAxNorm:
      for (int i = 0; i < niter; ++i) blas::cabxpyzAxNorm(a, b2, *xD, *yD, *yD);
      break;

    case Kernel::cDotProduct:
      for (int i = 0; i < niter; ++i) blas::cDotProduct(*xD, *yD);
      break;

    case Kernel::caxpyDotzy:
      for (int i = 0; i < niter; ++i) blas::caxpyDotzy(a2, *xD, *yD, *zD);
      break;

    case Kernel::cDotProductNormA:
      for (int i = 0; i < niter; ++i) blas::cDotProductNormA(*xD, *yD);
      break;

    case Kernel::caxpbypzYmbwcDotProductUYNormY:
      for (int i = 0; i < niter; ++i) blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zoD, *wD, *vD);
      break;

    case Kernel::HeavyQuarkResidualNorm:
      for (int i = 0; i < niter; ++i) blas::HeavyQuarkResidualNorm(*xD, *yD);
      break;

    case Kernel::xpyHeavyQuarkResidualNorm:
      for (int i = 0; i < niter; ++i) blas::xpyHeavyQuarkResidualNorm(*xD, *yD, *zD);
      break;

    case Kernel::tripleCGReduction:
      for (int i = 0; i < niter; ++i) blas::tripleCGReduction(*xD, *yD, *zD);
      break;

    case Kernel::tripleCGUpdate:
      for (int i = 0; i < niter; ++i) blas::tripleCGUpdate(a, b, *xD, *yD, *zD, *wD);
      break;

    case Kernel::axpyReDot:
      for (int i = 0; i < niter; ++i) blas::axpyReDot(a, *xD, *yD);
      break;

    case Kernel::caxpyBxpz:
      for (int i = 0; i < niter; ++i) blas::caxpyBxpz(a2, *xD, *yD, b2, *zD);
      break;

    case Kernel::caxpyBzpx:
      for (int i = 0; i < niter; ++i) blas::caxpyBzpx(a2, *xD, *yD, b2, *zD);
      break;

    case Kernel::axpy_block:
      for (int i = 0; i < niter; ++i) blas::axpy(Ar, xmD->Components(), ymoD->Components());
      break;

    case Kernel::caxpy_block:
      for (int i = 0; i < niter; ++i) blas::caxpy(A, *xmD, *ymoD);
      break;

    case Kernel::axpyBzpcx_block:
      for (int i = 0; i < niter; ++i)
        blas::axpyBzpcx((double *)A, xmD->Components(), zmoD->Components(), (double *)B, *yD, (double *)C);
      break;

    case Kernel::reDotProductNorm_block:
      for (int i = 0; i < niter; ++i) blas::reDotProduct((double *)A2, xmD->Components(), xmD->Components());
      break;

    case Kernel::reDotProduct_block:
      for (int i = 0; i < niter; ++i) blas::reDotProduct((double *)A, xmD->Components(), ymoD->Components());
      break;

    case Kernel::cDotProductNorm_block:
      for (int i = 0; i < niter; ++i) blas::cDotProduct(A2, xmD->Components(), xmD->Components());
      break;

    case Kernel::cDotProduct_block:
      for (int i = 0; i < niter; ++i) blas::cDotProduct(A, xmD->Components(), ymoD->Components());
      break;

    case Kernel::caxpyXmazMR:
      commAsyncReductionSet(true);
      for (int i = 0; i < niter; ++i) blas::caxpyXmazMR(a, *xD, *yD, *zD);
      commAsyncReductionSet(false);
      break;

    default: errorQuda("Undefined blas kernel %s\n", kernel_map.at(kernel).c_str());
    }
  }

  timer.stop();

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] A2;
  delete[] Ar;

  return timer.last();
}

#define ERROR(a) fabs(blas::norm2(*a##D) - blas::norm2(*a##H)) / blas::norm2(*a##H)

double test(Kernel kernel)
{
  double a = M_PI, b = M_PI * exp(1.0), c = sqrt(M_PI);
  quda::Complex a2(a, b), b2(b, -c), c2(a + b, c * a);
  double error = 0;
  quda::Complex *A = new quda::Complex[Nsrc * Msrc];
  quda::Complex *B = new quda::Complex[Nsrc * Msrc];
  quda::Complex *C = new quda::Complex[Nsrc * Msrc];
  quda::Complex *A2 = new quda::Complex[Nsrc * Nsrc]; // for the block cDotProductNorm test
  quda::Complex *B2 = new quda::Complex[Nsrc * Nsrc]; // for the block cDotProductNorm test
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

  case Kernel::copyHS:
    *xoD = *xH;
    blas::copy(*yD, *xoD);
    blas::copy(*yH, *xH);
    error = ERROR(y);
    break;

  case Kernel::copyLS:
    *xD = *xH;
    blas::copy(*yoD, *xD);
    blas::copy(*yH, *xH);
    error = ERROR(yo);
    break;

  case Kernel::axpbyz:
    *xD = *xH;
    *yoD = *yH;
    blas::axpbyz(a, *xD, b, *yoD, *zoD);
    blas::axpbyz(a, *xH, b, *yH, *zH);
    error = ERROR(zo);
    break;

  case Kernel::ax:
    *xD = *xH;
    blas::ax(a, *xD);
    blas::ax(a, *xH);
    error = ERROR(x);
    break;

  case Kernel::caxpy:
    *xD = *xH;
    *yoD = *yH;
    blas::caxpy(a2, *xD, *yoD);
    blas::caxpy(a2, *xH, *yH);
    error = ERROR(yo);
    break;

  case Kernel::caxpby:
    *xD = *xH;
    *yD = *yH;
    blas::caxpby(a2, *xD, b2, *yD);
    blas::caxpby(a2, *xH, b2, *yH);
    error = ERROR(y);
    break;

  case Kernel::cxpaypbz:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    blas::cxpaypbz(*xD, a2, *yD, b2, *zD);
    blas::cxpaypbz(*xH, a2, *yH, b2, *zH);
    error = ERROR(z);
    break;

  case Kernel::axpyBzpcx:
    *xD = *xH;
    *yoD = *yH;
    *zD = *zH;
    blas::axpyBzpcx(a, *xD, *yoD, b, *zD, c);
    blas::axpyBzpcx(a, *xH, *yH, b, *zH, c);
    error = ERROR(x) + ERROR(yo);
    break;

  case Kernel::axpyZpbx:
    *xD = *xH;
    *yoD = *yH;
    *zD = *zH;
    blas::axpyZpbx(a, *xD, *yoD, *zD, b);
    blas::axpyZpbx(a, *xH, *yH, *zH, b);
    error = ERROR(x) + ERROR(yo);
    break;

  case Kernel::caxpbypzYmbw:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    blas::caxpbypzYmbw(a2, *xD, b2, *yD, *zD, *wD);
    blas::caxpbypzYmbw(a2, *xH, b2, *yH, *zH, *wH);
    error = ERROR(z) + ERROR(y);
    break;

  case Kernel::cabxpyAx:
    *xD = *xH;
    *yD = *yH;
    blas::cabxpyAx(a, b2, *xD, *yD);
    blas::cabxpyAx(a, b2, *xH, *yH);
    error = ERROR(y) + ERROR(x);
    break;

  case Kernel::caxpyXmaz:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      blas::caxpyXmaz(a, *xD, *yD, *zD);
      blas::caxpyXmaz(a, *xH, *yH, *zH);
      error = ERROR(y) + ERROR(x);
    }
    break;

  case Kernel::norm2:
    *xD = *xH;
    error = fabs(blas::norm2(*xD) - blas::norm2(*xH)) / blas::norm2(*xH);
    break;

  case Kernel::reDotProduct:
    *xD = *xH;
    *yD = *yH;
    error = fabs(blas::reDotProduct(*xD, *yD) - blas::reDotProduct(*xH, *yH)) / fabs(blas::reDotProduct(*xH, *yH));
    break;

  case Kernel::axpbyzNorm:
    *xD = *xH;
    *yD = *yH;
    {
      double d = blas::axpbyzNorm(a, *xD, b, *yD, *zD);
      double h = blas::axpbyzNorm(a, *xH, b, *yH, *zH);
      error = ERROR(z) + fabs(d - h) / fabs(h);
    }
    break;

  case Kernel::axpyCGNorm:
    *xD = *xH;
    *yoD = *yH;
    {
      quda::Complex d = blas::axpyCGNorm(a, *xD, *yoD);
      quda::Complex h = blas::axpyCGNorm(a, *xH, *yH);
      error = ERROR(yo) + fabs(d.real() - h.real()) / fabs(h.real()) + fabs(d.imag() - h.imag()) / fabs(h.imag());
    }
    break;

  case Kernel::caxpyNorm:
    *xD = *xH;
    *yD = *yH;
    {
      double d = blas::caxpyNorm(a, *xD, *yD);
      double h = blas::caxpyNorm(a, *xH, *yH);
      error = ERROR(y) + fabs(d - h) / fabs(h);
    }
    break;

  case Kernel::caxpyXmazNormX:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      double d = blas::caxpyXmazNormX(a, *xD, *yD, *zD);
      double h = blas::caxpyXmazNormX(a, *xH, *yH, *zH);
      error = ERROR(y) + ERROR(x) + fabs(d - h) / fabs(h);
    }
    break;

  case Kernel::cabxpyzAxNorm:
    *xD = *xH;
    *yD = *yH;
    {
      double d = blas::cabxpyzAxNorm(a, b2, *xD, *yD, *yD);
      double h = blas::cabxpyzAxNorm(a, b2, *xH, *yH, *yH);
      error = ERROR(x) + ERROR(y) + fabs(d - h) / fabs(h);
    }
    break;

  case Kernel::cDotProduct:
    *xD = *xH;
    *yD = *yH;
    error = abs(blas::cDotProduct(*xD, *yD) - blas::cDotProduct(*xH, *yH)) / abs(blas::cDotProduct(*xH, *yH));
    break;

  case Kernel::caxpyDotzy:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      quda::Complex d = blas::caxpyDotzy(a, *xD, *yD, *zD);
      quda::Complex h = blas::caxpyDotzy(a, *xH, *yH, *zH);
      error = ERROR(y) + abs(d - h) / abs(h);
    }
    break;

  case Kernel::cDotProductNormA:
    *xD = *xH;
    *yD = *yH;
    {
      double3 d = blas::cDotProductNormA(*xD, *yD);
      double3 h = blas::cDotProductNormA(*xH, *yH);
      error = abs(Complex(d.x - h.x, d.y - h.y)) / abs(Complex(h.x, h.y)) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case Kernel::caxpbypzYmbwcDotProductUYNormY:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    *vD = *vH;
    {
      double3 d = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xD, b2, *yD, *zD, *wD, *vD);
      double3 h = blas::caxpbypzYmbwcDotProductUYNormY(a2, *xH, b2, *yH, *zH, *wH, *vH);
      error = ERROR(z) + ERROR(y) + abs(Complex(d.x - h.x, d.y - h.y)) / abs(Complex(h.x, h.y))
        + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case Kernel::HeavyQuarkResidualNorm:
    *xD = *xH;
    *yD = *yH;
    {
      double3 d = blas::HeavyQuarkResidualNorm(*xD, *yD);
      double3 h = blas::HeavyQuarkResidualNorm(*xH, *yH);
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case Kernel::xpyHeavyQuarkResidualNorm:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      double3 d = blas::xpyHeavyQuarkResidualNorm(*xD, *yD, *zD);
      double3 h = blas::xpyHeavyQuarkResidualNorm(*xH, *yH, *zH);
      error = ERROR(y) + fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case Kernel::tripleCGReduction:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      double3 d = blas::tripleCGReduction(*xD, *yD, *zD);
      double3 h = make_double3(blas::norm2(*xH), blas::norm2(*yH), blas::reDotProduct(*yH, *zH));
      error = fabs(d.x - h.x) / fabs(h.x) + fabs(d.y - h.y) / fabs(h.y) + fabs(d.z - h.z) / fabs(h.z);
    }
    break;

  case Kernel::tripleCGUpdate:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    *wD = *wH;
    {
      blas::tripleCGUpdate(a, b, *xD, *yD, *zD, *wD);
      blas::tripleCGUpdate(a, b, *xH, *yH, *zH, *wH);
      error = ERROR(y) + ERROR(z) + ERROR(w);
    }
    break;

  case Kernel::axpyReDot:
    *xD = *xH;
    *yD = *yH;
    {
      double d = blas::axpyReDot(a, *xD, *yD);
      double h = blas::axpyReDot(a, *xH, *yH);
      error = ERROR(y) + fabs(d - h) / fabs(h);
    }
    break;

  case Kernel::caxpyBxpz:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      blas::caxpyBxpz(a, *xD, *yD, b2, *zD);
      blas::caxpyBxpz(a, *xH, *yH, b2, *zH);
      error = ERROR(x) + ERROR(z);
    }
    break;

  case Kernel::caxpyBzpx:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;
    {
      blas::caxpyBzpx(a, *xD, *yD, b2, *zD);
      blas::caxpyBzpx(a, *xH, *yH, b2, *zH);
      error = ERROR(x) + ERROR(z);
    }
    break;

  case Kernel::axpy_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i = 0; i < Msrc; i++) ymoD->Component(i) = *(ymH[i]);

    blas::axpy(Ar, *xmD, *ymoD);
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) { blas::axpy(Ar[Msrc * i + j], *(xmH[i]), *(ymH[j])); }
    }

    error = 0;
    for (int i = 0; i < Msrc; i++) {
      error += fabs(blas::norm2((ymoD->Component(i))) - blas::norm2(*(ymH[i]))) / blas::norm2(*(ymH[i]));
    }
    error /= Msrc;
    break;

  case Kernel::caxpy_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i = 0; i < Msrc; i++) ymoD->Component(i) = *(ymH[i]);

    blas::caxpy(A, *xmD, *ymoD);
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) { blas::caxpy(A[Msrc * i + j], *(xmH[i]), *(ymH[j])); }
    }
    error = 0;
    for (int i = 0; i < Msrc; i++) {
      error += fabs(blas::norm2((ymoD->Component(i))) - blas::norm2(*(ymH[i]))) / blas::norm2(*(ymH[i]));
    }
    error /= Msrc;
    break;

  case Kernel::axpyBzpcx_block:
    for (int i = 0; i < Nsrc; i++) {
      xmD->Component(i) = *(xmH[i]);
      zmoD->Component(i) = *(zmH[i]);
    }
    *yD = *yH;

    blas::axpyBzpcx((double *)A, xmD->Components(), zmoD->Components(), (double *)B, *yD, (const double *)C);

    for (int i = 0; i < Nsrc; i++) {
      blas::axpyBzpcx(((double *)A)[i], *xmH[i], *zmH[i], ((double *)B)[i], *yH, ((double *)C)[i]);
    }

    error = 0;
    for (int i = 0; i < Nsrc; i++) {
      error += fabs(blas::norm2((xmD->Component(i))) - blas::norm2(*(xmH[i]))) / blas::norm2(*(xmH[i]));
      error += fabs(blas::norm2((zmoD->Component(i))) - blas::norm2(*(zmH[i]))) / blas::norm2(*(zmH[i]));
    }
    error /= Nsrc;
    break;

  case Kernel::reDotProductNorm_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    blas::reDotProduct((double *)A2, xmD->Components(), xmD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Nsrc; j++) {
        ((double *)B2)[i * Nsrc + j] = blas::reDotProduct(xmD->Component(i), xmD->Component(j));
        error += std::abs(((double *)A2)[i * Nsrc + j] - ((double *)B2)[i * Nsrc + j])
          / std::abs(((double *)B2)[i * Nsrc + j]);
      }
    }
    error /= Nsrc * Nsrc;
    break;

  case Kernel::reDotProduct_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i = 0; i < Msrc; i++) ymoD->Component(i) = *(ymH[i]);
    for (int i = 0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);
    blas::reDotProduct((double *)A, xmD->Components(), ymoD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) {
        ((double *)B)[i * Msrc + j] = blas::reDotProduct(xmD->Component(i), ymD->Component(j));
        error
          += std::abs(((double *)A)[i * Msrc + j] - ((double *)B)[i * Msrc + j]) / std::abs(((double *)B)[i * Msrc + j]);
      }
    }
    error /= Nsrc * Msrc;
    break;

  case Kernel::cDotProductNorm_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    blas::cDotProduct(A2, xmD->Components(), xmD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Nsrc; j++) {
        B2[i * Nsrc + j] = blas::cDotProduct(xmD->Component(i), xmD->Component(j));
        error += std::abs(A2[i * Nsrc + j] - B2[i * Nsrc + j]) / std::abs(B2[i * Nsrc + j]);
      }
    }
    error /= Nsrc * Nsrc;
    break;

  case Kernel::cDotProduct_block:
    for (int i = 0; i < Nsrc; i++) xmD->Component(i) = *(xmH[i]);
    for (int i = 0; i < Msrc; i++) ymoD->Component(i) = *(ymH[i]);
    for (int i = 0; i < Msrc; i++) ymD->Component(i) = *(ymH[i]);
    blas::cDotProduct(A, xmD->Components(), ymoD->Components());
    error = 0.0;
    for (int i = 0; i < Nsrc; i++) {
      for (int j = 0; j < Msrc; j++) {
        B[i * Msrc + j] = blas::cDotProduct(xmD->Component(i), ymD->Component(j));
        error += std::abs(A[i * Msrc + j] - B[i * Msrc + j]) / std::abs(B[i * Msrc + j]);
      }
    }
    error /= Nsrc * Msrc;
    break;

  case Kernel::caxpyXmazMR:
    *xD = *xH;
    *yD = *yH;
    *zD = *zH;

    commGlobalReductionPush(false); // switch off global reductions for this test

    commAsyncReductionSet(true);
    blas::cDotProductNormA(*zD, *xD);
    blas::caxpyXmazMR(a, *xD, *yD, *zD);
    commAsyncReductionSet(false);

    *vD = *xH;
    *wD = *yH;
    *zD = *zH;
    {
      double3 Ar3 = blas::cDotProductNormA(*zD, *vD);
      auto alpha = Complex(Ar3.x, Ar3.y) / Ar3.z;
      blas::caxpyXmaz(a * alpha, *vD, *wD, *zD);
    }
    *xH = *vD;
    *yH = *wD;

    commGlobalReductionPop(); // restore global reductions

    error = ERROR(x) + ERROR(y);
    break;

  default: errorQuda("Undefined blas kernel %s\n", kernel_map.at(kernel).c_str());
  }
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] A2;
  delete[] B2;
  delete[] Ar;
  return error;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int result = 0;

  prec = QUDA_INVALID_PRECISION;
  prec_sloppy = QUDA_INVALID_PRECISION;
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
    Nspin = (dslash_type == QUDA_ASQTAD_DSLASH || dslash_type == QUDA_STAGGERED_DSLASH) ? 1 : 4;
    Ncolor = 3;
  }

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // lastly check for correctness
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
  result = RUN_ALL_TESTS();

  endQuda();

  finalizeComms();
  return result;
}

// The following tests each kernel at each precision using the google testing framework

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

// map the 1-d precision test index into 2-d mixed prec
prec_pair_t prec_idx_map(int idx)
{
  switch (idx) {
  case 0: return std::make_pair(QUDA_QUARTER_PRECISION, QUDA_QUARTER_PRECISION);
  case 1: return std::make_pair(QUDA_QUARTER_PRECISION, QUDA_HALF_PRECISION);
  case 2: return std::make_pair(QUDA_QUARTER_PRECISION, QUDA_SINGLE_PRECISION);
  case 3: return std::make_pair(QUDA_QUARTER_PRECISION, QUDA_DOUBLE_PRECISION);
  case 4: return std::make_pair(QUDA_HALF_PRECISION, QUDA_HALF_PRECISION);
  case 5: return std::make_pair(QUDA_HALF_PRECISION, QUDA_SINGLE_PRECISION);
  case 6: return std::make_pair(QUDA_HALF_PRECISION, QUDA_DOUBLE_PRECISION);
  case 7: return std::make_pair(QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION);
  case 8: return std::make_pair(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION);
  case 9: return std::make_pair(QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION);
  default: errorQuda("Unexpect precision index %d", idx);
  }
  return std::make_pair(QUDA_INVALID_PRECISION, QUDA_INVALID_PRECISION);
}

class BlasTest : public ::testing::TestWithParam<::testing::tuple<int, int>>
{
protected:
  ::testing::tuple<int, int> param;
  const prec_pair_t prec_pair;
  const int &kernel;

public:
  BlasTest() : param(GetParam()), prec_pair(prec_idx_map(::testing::get<0>(param))), kernel(::testing::get<1>(param)) {}
  virtual void SetUp()
  {
    if (!skip_kernel(prec_pair, (Kernel)kernel)) initFields(prec_pair);
  }
  virtual void TearDown()
  {
    if (!skip_kernel(prec_pair, (Kernel)kernel)) { freeFields(); }
  }
};

TEST_P(BlasTest, verify)
{
  prec_pair_t prec_pair = ::prec_idx_map(testing::get<0>(GetParam()));
  Kernel kernel = (Kernel)::testing::get<1>(GetParam());
  if (skip_kernel(prec_pair, kernel)) GTEST_SKIP();

  // certain tests will fail to run for coarse grids so mark these as
  // failed without running
  double deviation = test(kernel);
  // printfQuda("%-35s error = %e\n", names[kernel], deviation);
  double tol_x
    = (prec_pair.first == QUDA_DOUBLE_PRECISION ?
         1e-12 :
         (prec_pair.first == QUDA_SINGLE_PRECISION ? 1e-6 : (prec_pair.first == QUDA_HALF_PRECISION ? 1e-4 : 1e-2)));
  double tol_y
    = (prec_pair.second == QUDA_DOUBLE_PRECISION ?
         1e-12 :
         (prec_pair.second == QUDA_SINGLE_PRECISION ? 1e-6 : (prec_pair.second == QUDA_HALF_PRECISION ? 1e-4 : 1e-2)));
  double tol = std::max(tol_x, tol_y);
  tol = is_copy(kernel) ? 5e-2 : tol; // use different tolerance for copy
  EXPECT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
  EXPECT_EQ(false, std::isnan(deviation)) << "Nan has propagated into the result";
}

TEST_P(BlasTest, benchmark)
{
  prec_pair_t prec_pair = prec_idx_map(::testing::get<0>(GetParam()));
  Kernel kernel = (Kernel)::testing::get<1>(GetParam());

  if (skip_kernel(prec_pair, kernel)) GTEST_SKIP();

  // do the initial tune
  benchmark(kernel, 1);

  // now rerun with more iterations to get accurate speed measurements
  quda::blas::flops = 0;
  quda::blas::bytes = 0;

  double secs = benchmark(kernel, niter);

  double gflops = (quda::blas::flops * 1e-9) / (secs);
  double gbytes = quda::blas::bytes / (secs * 1e9);
  RecordProperty("Gflops", std::to_string(gflops));
  RecordProperty("GBs", std::to_string(gbytes));
  printfQuda("%-31s: Gflop/s = %6.1f, GB/s = %6.1f\n", kernel_map.at(kernel).c_str(), gflops, gbytes);
}

std::string getblasname(testing::TestParamInfo<::testing::tuple<int, int>> param)
{
  prec_pair_t prec_pair = prec_idx_map(::testing::get<0>(param.param));
  int kernel = ::testing::get<1>(param.param);
  std::string str(kernel_map.at((Kernel)kernel));
  str += std::string("_") + prec_map.at(prec_pair.first) + std::string("_") + prec_map.at(prec_pair.second);
  return str;
}

// instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, BlasTest, Combine(Range(0, (Nprec * (Nprec + 1)) / 2), Range(0, Nkernels)), getblasname);
