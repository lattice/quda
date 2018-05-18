#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include "misc.h"

#include <qio_field.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern double anisotropy;

extern char latfile[];

#define MAX(a,b) ((a)>(b)?(a):(b))

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;

void setGaugeParam(QudaGaugeParam &gauge_param) {

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}


extern void usage(char**);

void SU3test(int argc, char **argv) {

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) 
    prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) 
    link_recon_sloppy = link_recon;

  setGaugeParam(gauge_param);
  setDims(gauge_param.X);
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);

  void *gauge[4], *new_gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    new_gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  initQuda(device);

  // call srand() with a rank-dependent seed
  initRand();

  // load in the command line supplied gauge field
  if (strcmp(latfile,"")) {  
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { 
    // generate a random SU(3) field
    printf("Randomizing fields...");
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    printf("done.\n");
  }

  loadGaugeQuda(gauge, &gauge_param);
  saveGaugeQuda(new_gauge, &gauge_param);

#ifdef GPU_GAUGE_TOOLS
  double plaq[3];
  plaqQuda(plaq);
  printf("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Topological charge
  double qCharge;
  // start the timer
  double time0 = -((double)clock());
  qCharge = qChargeCuda();
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printf("Computed topological charge is %.16e Done in %g secs\n", qCharge, time0);

  // Stout smearing should be equivalent to APE smearing
  // on D dimensional lattices for rho = alpha/2*(D-1). 
  // Typical APE values are aplha=0.6, rho=0.1 for Stout.
  unsigned int nSteps = 50;
  double coeff_APE = 0.6;
  double coeff_STOUT = coeff_APE/(2*(4-1));
  QudaVerbosity verbosity = QUDA_VERBOSE;
  setVerbosity(verbosity);
  
  //STOUT
  // start the timer
  time0 = -((double)clock());
  performSTOUTnStep(nSteps, coeff_STOUT);
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for STOUT = %g secs\n", time0);
  qCharge = qChargeCuda();
  printf("Computed topological charge after is %.16e \n", qCharge);

  //APE
  // start the timer
  time0 = -((double)clock());
  performAPEnStep(nSteps, coeff_APE);  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for APE = %g secs\n", time0);
  qCharge = qChargeCuda();
  printf("Computed topological charge after is %.16e \n", qCharge);

  //Over Improved STOUT
  double epsilon = -0.25;
  coeff_STOUT = 0.06;
  nSteps = 200;
  // start the timer
  time0 = -((double)clock());
  performOvrImpSTOUTnStep(nSteps, coeff_STOUT, epsilon);  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for Over Improved STOUT = %g secs\n", time0);
  qCharge = qChargeCuda();
  printf("Computed topological charge after is %.16e \n", qCharge);

#else
  printf("Skipping plaquette tests since gauge tools have not been compiled\n");
#endif
  
  check_gauge(gauge, new_gauge, 1e-3, gauge_param.cpu_prec);
  freeGaugeQuda();
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(new_gauge[dir]);
  }

  finalizeComms();
}

int main(int argc, char **argv) {

  SU3test(argc, argv);

  return 0;
}
