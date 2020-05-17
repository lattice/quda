#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

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

  size_t data_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  void *spinorX = malloc(V * 2 * 4 * 3 * data_size);
  void *spinorY = malloc(V * 2 * 1 * 3 * data_size);
  void *d_result = malloc(2 * 4 * X[3] * data_size);

  if (prec == QUDA_SINGLE_PRECISION) {
    for (int i = 0; i < V * 2 * 4 * 3; i++) {
      ((float *)spinorX)[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < V * 2 * 1 * 3; i++) {
      ((float *)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (int i = 0; i < V * 2 * 4 * 3; i++) {
      ((double *)spinorX)[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < V * 2 * 1 * 3; i++) {
      ((double *)spinorY)[i] = rand() / (double)RAND_MAX;
    }
  }

  // Perform GPU evec projection
  laphSinkProject(spinorX, spinorY, d_result, inv_param, X);

  free(spinorX);
  free(spinorY);
  free(d_result);


  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}

