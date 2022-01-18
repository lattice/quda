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
  
int main(int argc, char **argv)
{
  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  setQudaDefaultMgTestParams();
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // initialize the QUDA library
  initQuda(device_ordinal);
  //-----------------------------------------------------------------------------
  
  
  QudaPrecision test_prec = QUDA_DOUBLE_PRECISION;  
  size_t data_size = (test_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  
  int Lx = 32;
  int Lt = 32;
  int Vol = Lx * Lx * Lx * Lt;
  int X[4] = {Lx, Lx, Lx, Lt};

  // spinor_site_size is a global variable, all pointers
  void *spinorX = safe_malloc(Vol * spinor_site_size * data_size);
  void *spinorY = safe_malloc(Vol * spinor_site_size * data_size);
  void *d_result = safe_malloc(2 * Vol * 16 * data_size);
 
  // fill fermion fields with random numbers 
  if (test_prec == QUDA_SINGLE_PRECISION) {
    for (int i = 0; i < Vol * spinor_site_size; i++) {
      ((float *)spinorX)[i] = rand() / (float)RAND_MAX;
      //((float *)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (int i = 0; i < Vol * spinor_site_size; i++) {
      ((double *)spinorX)[i] = rand() / (double)RAND_MAX;
      //((double *)spinorY)[i] = rand() / (double)RAND_MAX;
    }
  }
  // Is ^ the fastest way to fill with random numbers 
  //RNG *randstates = new RNG(spinorX, 1234);

  QudaContractType cType;
  //cType = QUDA_CONTRACT_TYPE_OPEN;
  cType = QUDA_CONTRACT_TYPE_DR;
  
  QudaInvertParam inv_param = newQudaInvertParam();
  setContractInvertParam(inv_param);
  inv_param.cpu_prec = test_prec;
  inv_param.cuda_prec = test_prec;
  inv_param.cuda_prec_sloppy = test_prec;
  inv_param.cuda_prec_precondition = test_prec;

  // Perform GPU contraction.
  
  if (comm_rank() == 0) printf("check test 1 \n");
  
  contractQuda(spinorX, spinorY, d_result, cType, &inv_param, X);
  
  // ---- START CLEAN UP ---- //
  
  free(spinorX);
  free(spinorY);
  free(d_result);
  
  endQuda();
  finalizeComms();
  
  // ---- FINISH CLEAN UP ---- //

  return 0;

}

