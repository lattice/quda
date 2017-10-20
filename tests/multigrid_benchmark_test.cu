#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <test_util.h>
#include <misc.h>

// include because of nasty globals used in the tests
#include <dslash_util.h>
#include <dirac_quda.h>
#include <algorithm>

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

extern int Nsrc; // number of spinors to apply to simultaneously

extern bool verify_results;

extern int test_type;

extern QudaPrecision prec;

extern void usage(char** );

using namespace quda;

ColorSpinorField *xH, *yH;
ColorSpinorField *xD, *yD;

cpuGaugeField *Y_h, *X_h, *Xinv_h, *Yhat_h;
cudaGaugeField *Y_d, *X_d, *Xinv_d, *Yhat_d;

int Nspin;
int Ncolor;

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
  param.PCtype = QUDA_4D_PC;

  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.precision = QUDA_DOUBLE_PRECISION;
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
  param.precision = prec;
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
  gParam.precision = param.precision;
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
  int pad = std::max( { (gParam.x[0]*gParam.x[1]*gParam.x[2])/2,
	(gParam.x[1]*gParam.x[2]*gParam.x[3])/2,
	(gParam.x[0]*gParam.x[2]*gParam.x[3])/2,
	(gParam.x[0]*gParam.x[1]*gParam.x[3])/2 } );
  gParam.pad = gParam.nFace * pad * 2;
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
  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device);

  // enable the tuning
  setVerbosity(QUDA_SUMMARIZE);

  Nspin = 2;

  printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", get_prec_str(prec), niter);
  for (int c=24; c<=32; c+=8) {
    Ncolor = c;

    initFields(prec);

    DiracParam param;
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
