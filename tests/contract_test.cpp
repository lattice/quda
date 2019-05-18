#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <contract_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_null;

extern char vec_infile[];
extern char vec_outfile[];

extern QudaContractType contract_type;

extern void usage(char **);

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

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setInvertParam(QudaInvertParam &inv_param)
{

  inv_param.Ls = 1;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
}

int main(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    if (process_command_line_option(argc, argv, &i) == 0) { continue; }
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);
  setSpinorSiteSize(24);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  size_t sSize = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  void *spinorX = malloc(V * spinorSiteSize * sSize);
  void *spinorY = malloc(V * spinorSiteSize * sSize);
  void *d_result = malloc(2 * V * 16 * sSize);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  if (cpu_prec == QUDA_SINGLE_PRECISION) {
    for (int i = 0; i < V * spinorSiteSize; i++) {
      ((float *)spinorX)[i] = rand() / (float)RAND_MAX;
      ((float *)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (int i = 0; i < V * spinorSiteSize; i++) {
      ((double *)spinorX)[i] = rand() / (double)RAND_MAX;
      ((double *)spinorY)[i] = rand() / (double)RAND_MAX;
    }
  }

  // Host side spinor data and result passed to QUDA.
  // QUDA will allocate GPU memory, transfer the data,
  // perform the requested contraction, and return the
  // result in the array 'result'
  // We then compare the GPU result with a CPU refernce code

  QudaContractType cType = contract_type;
  contractQuda(spinorX, spinorY, d_result, cType, &inv_param, X);

  // This function will compare each color contraction from the host and device.
  // It returns the number of faults it detects.
  int faults = 0;
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    faults = contraction_reference((double *)spinorX, (double *)spinorY, (double *)d_result, cType, X);
  } else {
    faults = contraction_reference((float *)spinorX, (float *)spinorY, (float *)d_result, cType, X);
  }

  printfQuda("Contraction comparison for contraction type %s complete with %d/%d faults\n", get_contract_str(cType),
             faults, V * 16 * 2);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
