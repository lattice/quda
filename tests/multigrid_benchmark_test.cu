#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <misc.h>

// include because of nasty globals used in the tests
#include <dslash_reference.h>
#include <dirac_quda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))


extern void usage(char** );

using namespace quda;

ColorSpinorField *xH, *yH;
ColorSpinorField *xD, *yD;

cpuGaugeField *Y_h, *X_h, *Xinv_h, *Yhat_h;
cudaGaugeField *Y_d, *X_d, *Xinv_d, *Yhat_d;

int Nspin;
int Ncolor;

#define MAX(a,b) ((a)>(b)?(a):(b))

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
}

void initFields(QudaPrecision prec)
{
  ColorSpinorParam param;
  param.nColor = Ncolor;
  param.nSpin = Nspin;
  param.nDim = 5; // number of spacetime dimensions

  param.pad = 0; // padding must be zero for cpu fields
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.x[0] = xdim;
  param.x[1] = ydim;
  param.x[2] = zdim;
  param.x[3] = tdim;
  param.x[4] = Nsrc;
  param.pc_type = QUDA_4D_PC;

  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.setPrecision(QUDA_DOUBLE_PRECISION);
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

  param.create = QUDA_ZERO_FIELD_CREATE;

  xH = new cpuColorSpinorField(param);
  yH = new cpuColorSpinorField(param);

  //static_cast<cpuColorSpinorField*>(xH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);
  //static_cast<cpuColorSpinorField*>(yH)->Source(QUDA_RANDOM_SOURCE, 0, 0, 0);

  // Now set the parameters for the cuda fields
  //param.pad = xdim*ydim*zdim/2;

  if (param.nSpin == 4) param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.setPrecision(prec);
  param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;

  xD = new cudaColorSpinorField(param);
  yD = new cudaColorSpinorField(param);

  // check for successful allocation
  checkCudaError();

  //*xD = *xH;
  //*yD = *yH;

  GaugeFieldParam gParam;
  gParam.x[0] = xdim;
  gParam.x[1] = ydim;
  gParam.x[2] = zdim;
  gParam.x[3] = tdim;
  gParam.nColor = param.nColor*param.nSpin;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.order = QUDA_QDP_GAUGE_ORDER;
  gParam.link_type = QUDA_COARSE_LINKS;
  gParam.t_boundary = QUDA_PERIODIC_T;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.setPrecision(param.Precision());
  gParam.nDim = 4;
  gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  gParam.nFace = 1;

  gParam.geometry = QUDA_COARSE_GEOMETRY;
  Y_h = new cpuGaugeField(gParam);
  Yhat_h = new cpuGaugeField(gParam);

  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.nFace = 0;
  X_h = new cpuGaugeField(gParam);
  Xinv_h = new cpuGaugeField(gParam);

  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.geometry = QUDA_COARSE_GEOMETRY;
  gParam.nFace = 1;

  int x_face_size = gParam.x[1]*gParam.x[2]*gParam.x[3]/2;
  int y_face_size = gParam.x[0]*gParam.x[2]*gParam.x[3]/2;
  int z_face_size = gParam.x[0]*gParam.x[1]*gParam.x[3]/2;
  int t_face_size = gParam.x[0]*gParam.x[1]*gParam.x[2]/2;
  int pad = MAX(x_face_size, y_face_size);
  pad = MAX(pad, z_face_size);
  pad = MAX(pad, t_face_size);
  gParam.pad = gParam.nFace * pad * 2;

  gParam.setPrecision(prec_sloppy);

  Y_d = new cudaGaugeField(gParam);
  Yhat_d = new cudaGaugeField(gParam);
  Y_d->copy(*Y_h);
  Yhat_d->copy(*Yhat_h);

  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.nFace = 0;
  X_d = new cudaGaugeField(gParam);
  Xinv_d = new cudaGaugeField(gParam);
  X_d->copy(*X_h);
  Xinv_d->copy(*Xinv_h);
}


void freeFields()
{
  delete xD;
  delete yD;

  delete xH;
  delete yH;

  delete Y_h;
  delete X_h;
  delete Xinv_h;
  delete Yhat_h;

  delete Y_d;
  delete X_d;
  delete Xinv_d;
  delete Yhat_d;
}

DiracCoarse *dirac;

double benchmark(int test, const int niter) {

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  switch(test) {
  case 0:
    for (int i=0; i < niter; ++i) dirac->Dslash(xD->Even(), yD->Odd(), QUDA_EVEN_PARITY);
    break;
  case 1:
    for (int i=0; i < niter; ++i) dirac->M(*xD, *yD);
    break;
  case 2:
    for (int i=0; i < niter; ++i) dirac->Clover(xD->Even(), yD->Even(), QUDA_EVEN_PARITY);
    break;
  default:
    errorQuda("Undefined test %d", test);
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


const char *names[] = {
  "Dslash",
  "Mat",
  "Clover"
};

int main(int argc, char** argv)
{
  // Set some defaults that lets the benchmark fit in memory if you run it
  // with default parameters.
  xdim = ydim = zdim = tdim = 8;

  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  add_multigrid_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"Dslash", 0}, {"Mat", 1}, {"Clover", 2}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device);

  setVerbosity(verbosity);

  Nspin = 2;

  printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", get_prec_str(prec), niter);
  for (int c=24; c<=32; c+=8) {
    Ncolor = c;

    initFields(prec);

    DiracParam param;
    param.halo_precision = smoother_halo_prec;
    dirac = new DiracCoarse(param, Y_h, X_h, Xinv_h, Yhat_h, Y_d, X_d, Xinv_d, Yhat_d);

    // do the initial tune
    benchmark(test_type, 1);

    // now rerun with more iterations to get accurate speed measurements
    dirac->Flops(); // reset flops counter

    double secs = benchmark(test_type, niter);
    double gflops = (dirac->Flops()*1e-9)/(secs);

    printfQuda("Ncolor = %2d, %-31s: Gflop/s = %6.1f\n", Ncolor, names[test_type], gflops);

    delete dirac;
    freeFields();
  }

  // clear the error state
  cudaGetLastError();

  endQuda();

  finalizeComms();
}
