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

  void *spinorC = pinned_malloc(3 * sizeof(double _Complex));

  ((complex<double>*)spinorC)[0] = ((complex<double>*)spinorY)[1] * ((complex<double>*)spinorZ)[2] - ((complex<double>*)spinorY)[2] * ((complex<double>*)spinorZ)[1];
  ((complex<double>*)spinorC)[1] = ((complex<double>*)spinorY)[2] * ((complex<double>*)spinorZ)[0] - ((complex<double>*)spinorY)[0] * ((complex<double>*)spinorZ)[2];
  ((complex<double>*)spinorC)[2] = ((complex<double>*)spinorY)[0] * ((complex<double>*)spinorZ)[1] - ((complex<double>*)spinorY)[1] * ((complex<double>*)spinorZ)[0];
  
  printfQuda("color cross elem 0 CPU - GPU: ");
  for(int i=0; i<6; i++) printfQuda("%.16e ", ((double*)spinorC)[i] - ((double*)spinorT)[i]);
  printfQuda("\n");
  host_free(spinorC);
}

void laphColorContractCheck(void *spinorY, void *spinorZ, void *gpu_res) {

  complex<double> res = 
    ((complex<double>*)spinorY)[0] * ((complex<double>*)spinorZ)[0] + 
    ((complex<double>*)spinorY)[1] * ((complex<double>*)spinorZ)[1] + 
    ((complex<double>*)spinorY)[2] * ((complex<double>*)spinorZ)[2];

  printfQuda("color contract elem 0 CPU - GPU: ");
  printfQuda("(%.16e %.16e)", 
	     res.real() - ((complex<double>*)gpu_res)[0].real(), 
	     res.imag() - ((complex<double>*)gpu_res)[0].imag());
  printfQuda("\n");
  
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
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec;
  inv_param.cuda_prec_precondition = prec;

  size_t data_size = cpu_prec;
  void *spinorX = pinned_malloc(V * 2 * 4 * 3 * data_size);
  void *spinorY = pinned_malloc(V * 2 * 1 * 3 * data_size);
  void *spinorZ = pinned_malloc(V * 2 * 1 * 3 * data_size);
  void *spinorT = pinned_malloc(V * 2 * 1 * 3 * data_size);

  double _Complex *d_result = (double _Complex*)malloc(4 * X[3] * sizeof(double _Complex));
  
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
    laphSinkProject(spinorX, &spinorY, d_result, inv_param, 1, X);
    if (i==niter-1) {
      // Eyeball the data.
      for(int t=0; t<X[3]; t++) {
	for(int s=0; s<4; s++) {
	  //printfQuda("elem (%d,%d) = (%.16e,%.16e)\n", X[3] * comm_coord(3) + t,
	  //s, ((complex<double>*)d_result)[t*4 + s].real(), ((complex<double>*)d_result)[t*4 + s].imag());
	}
      }
    }
  }

  for (int n=0; n<niter; n++) {
    laphColorCross(spinorY, spinorZ, spinorT, inv_param, X);
        
    if (n == niter-1) {
      laphColorCrossCheck(spinorY, spinorZ, spinorT);
    }
  }

  double _Complex *cc_result = (double _Complex*)malloc(V * sizeof(double _Complex));

  for (int n=0; n<niter; n++) {

    laphColorContract(spinorY, spinorZ, (void*)cc_result, inv_param, X);
    
    if (n == niter-1) {
      laphColorContractCheck(spinorY, spinorZ, cc_result);
    }
  }



  host_free(spinorX);
  host_free(spinorY);
  host_free(spinorZ);
  host_free(spinorT);
  free(d_result);
  //free(cc_result);
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}

