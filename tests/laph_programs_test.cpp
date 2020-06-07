#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <contract_reference.h>
#include "misc.h"

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

namespace quda
{
  extern void setTransferGPU(bool);
}

void laphColorCrossCheck(void *spinorY, void *spinorZ, void *spinorT) {

  double total_diff = 0.0;
  void *spinorC = pinned_malloc(3 * sizeof(double _Complex));
  for(int offset = 0; offset < V * 3; offset += 3) {

    ((complex<double>*)spinorC)[0] = ((complex<double>*)spinorY)[offset+1] * ((complex<double>*)spinorZ)[offset+2] - ((complex<double>*)spinorY)[offset+2] * ((complex<double>*)spinorZ)[offset+1];
    ((complex<double>*)spinorC)[1] = ((complex<double>*)spinorY)[offset+2] * ((complex<double>*)spinorZ)[offset+0] - ((complex<double>*)spinorY)[offset+0] * ((complex<double>*)spinorZ)[offset+2];
    ((complex<double>*)spinorC)[2] = ((complex<double>*)spinorY)[offset+0] * ((complex<double>*)spinorZ)[offset+1] - ((complex<double>*)spinorY)[offset+1] * ((complex<double>*)spinorZ)[offset+0];
    
    //printfQuda("color cross elem[%d] CPU - GPU ", offset/3);
    double diff = 0.0;
    for(int i=0; i<6; i++) {
      diff += ((double*)spinorC)[i] - ((double*)spinorT)[i + offset*2];
    }
    total_diff += diff;
    //printfQuda(" = %.16e\n", diff);
  }
  printfQuda("color cross CPU - GPU L2 norm = %.16e\n", total_diff/(xdim * ydim * zdim * 3));
  host_free(spinorC);
}

void laphColorContractCheck(void *spinorY, void *spinorZ, void *gpu_res) {

  double total_diff = 0.0;
  for(int offset = 0; offset < V * 3; offset+=3) {
    complex<double> res = 
      ((complex<double>*)spinorY)[offset+0] * ((complex<double>*)spinorZ)[offset+0] + 
      ((complex<double>*)spinorY)[offset+1] * ((complex<double>*)spinorZ)[offset+1] + 
      ((complex<double>*)spinorY)[offset+2] * ((complex<double>*)spinorZ)[offset+2];
    
    //printfQuda("color contract elem [%d] CPU - GPU L2 norm:  ", offset/3);
    double diff = 0.0;
    diff += res.real() - ((complex<double>*)gpu_res)[offset/3].real();
    diff += res.imag() - ((complex<double>*)gpu_res)[offset/3].imag();
    total_diff += diff;
    //printfQuda(" = %.16e\n", diff);
  }

  printfQuda("color contract CPU - GPU L2 norm = %.16e\n", total_diff/(xdim * ydim * zdim * 3));
}
  


void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %d/%d/%d          %d         %d\n", get_prec_str(prec), get_prec_str(prec_sloppy),
             xdim, ydim, zdim, tdim, Lsdim);

  printfQuda("Evec project test");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{

  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();
  display_test_info();

  // initialize the QUDA library
  initQuda(device);
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);
  setSpinorSiteSize(24);
  //-----------------------------------------------------------------------------

  QudaInvertParam inv_param = newQudaInvertParam();
  setContractInvertParam(inv_param);
  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cpu_prec;
  inv_param.cuda_prec_sloppy = prec;
  inv_param.cuda_prec_precondition = prec;

  size_t data_size = cpu_prec;
  void *spinorX = pinned_malloc(V * 2 * 4 * 3 * data_size);
  void *spinorY = pinned_malloc(V * 2 * 1 * 3 * data_size);
  void *spinorZ = pinned_malloc(V * 2 * 1 * 3 * data_size);
  void *spinorT = pinned_malloc(V * 2 * 1 * 3 * data_size);

  double _Complex *dsp_result = (double _Complex*)malloc(4 * X[3] * sizeof(double _Complex));
  
  {
    quda::RNG rng(X, 1234);
    rng.Init();

    constructRandomSpinorSource(spinorX, 4, 3, inv_param.cpu_prec, X, rng);
    constructRandomSpinorSource(spinorY, 1, 3, inv_param.cpu_prec, X, rng);
    constructRandomSpinorSource(spinorZ, 1, 3, inv_param.cpu_prec, X, rng);
    constructRandomSpinorSource(spinorT, 1, 3, inv_param.cpu_prec, X, rng);

    rng.Release();
  }

  // Perform GPU evec projection
  for (int i=0; i<niter; i++) {
    laphSinkProject(spinorX, &spinorY, dsp_result, inv_param, 1, X);
    if (i==niter-1) {
      // Eyeball the data.
      for(int t=0; t<X[3]; t++) {
	for(int s=0; s<4; s++) {
	  //printfQuda("elem (%d,%d) = (%.16e,%.16e)\n", X[3] * comm_coord(3) + t,
	  //s, ((complex<double>*)dsp_result)[t*4 + s].real(), ((complex<double>*)dsp_result)[t*4 + s].imag());
	}
      }
    }
  }

  for (int n=0; n<niter; n++) {
    laphColorCross(spinorY, spinorZ, spinorT, inv_param, X);
    if (n == niter-1) laphColorCrossCheck(spinorY, spinorZ, spinorT);
  }
  
  double _Complex *dcc_result = (double _Complex*)malloc(V * sizeof(double _Complex));
  for (int n=0; n<niter; n++) {
    laphColorContract(spinorY, spinorZ, (void*)dcc_result, inv_param, X);
    if (n == niter-1) laphColorContractCheck(spinorY, spinorZ, dcc_result);
  }

  double _Complex *c1 = (double _Complex*)malloc(48 * 384 * sizeof(double _Complex));
  double _Complex *c2 = (double _Complex*)malloc(48 * 384 * sizeof(double _Complex));
  double _Complex *c3 = (double _Complex*)malloc(48 * 384 * sizeof(double _Complex));

  for (uint64_t i = 0; i < 2 * 48 * 384; i++) {
    ((double *)c1)[i] = rand() / (double)RAND_MAX;
    ((double *)c2)[i] = rand() / (double)RAND_MAX;
    ((double *)c3)[i] = rand() / (double)RAND_MAX;
  }

  void *evecs[384];
  for(int i=0; i<384; i++) {
    evecs[i] = malloc(48 * 48 * 48 * 3 * 2 * data_size);
    for (uint64_t j = 0; j < 2 * 3 * 48 * 48 * 48; j++) {
      ((double *)evecs[i])[j] = rand() / (double)RAND_MAX;
    }
  }

  double _Complex *momP = (double _Complex*)malloc(48 * 384 * sizeof(double _Complex));
  void *ret = malloc(data_size);
  laphBaryonKernel(48, 48, 48, 33, c1, c2, c3, momP, 384, evecs, ret, 10, inv_param, X);
  
  host_free(spinorX);
  host_free(spinorY);
  host_free(spinorZ);
  host_free(spinorT);
  free(dsp_result);
  free(dcc_result);
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}

