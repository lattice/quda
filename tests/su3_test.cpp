#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_util.h>
#include <dslash_util.h>

#include <gauge_qio.h>
#include <util_quda.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

QudaGaugeParam param;
void *gauge[4], *new_gauge[4];

extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern char latfile[];

#define MAX(a,b) ((a)>(b)?(a):(b))

void init() {

  param = newQudaGaugeParam();

  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = prec;
  param.reconstruct = link_recon;
  param.cuda_prec_sloppy = prec;
  param.reconstruct_sloppy = link_recon;
  
  param.type = QUDA_WILSON_LINKS;
  param.tadpole_coeff = 0.8;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;

  param.X[0] = xdim;
  param.X[1] = ydim;
  param.X[2] = zdim;
  param.X[3] = tdim;
  setDims(param.X);

  param.anisotropy = 1.0;
  param.t_boundary = QUDA_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
#ifdef MULTI_GPU
  int x_face_size = param.X[1]*param.X[2]*param.X[3]/2;
  int y_face_size = param.X[0]*param.X[2]*param.X[3]/2;
  int z_face_size = param.X[0]*param.X[1]*param.X[3]/2;
  int t_face_size = param.X[0]*param.X[1]*param.X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  param.ga_pad = pad_size;    
#else
  param.ga_pad = 0;    
#endif

  // construct gauge fields
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
    new_gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }

  initQuda(device);
  if (tune) setTuning(QUDA_TUNE_YES);

}

void end() {
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(new_gauge[dir]);
  }

  finalizeComms();
}

extern void usage(char**);

void SU3Test(int argc, char **argv) {

  for (int i =1;i < argc; i++){    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  init();

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, param.cpu_prec, param.X, argc, argv);
    construct_gauge_field((void**)gauge, 2, param.cpu_prec, &param);
  } else { // generate a random SU(3) field
    printf("Randomizing fields...");
    construct_gauge_field((void**)gauge, 1, param.cpu_prec, &param);
    printf("done.\n");
  }

  loadGaugeQuda(gauge, &param);
  saveGaugeQuda(new_gauge, &param);

#ifdef GPU_GAUGE_TOOLS
  double plaq[3];
  plaqQuda(plaq);
  printf("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
#else
  printf("Skipping plaquette computation since gauge tools have not been compiled\n");
#endif

  check_gauge(gauge, new_gauge, 1e-3, param.cpu_prec);

  end();
}

int main(int argc, char **argv) {

  SU3Test(argc, argv);

  return 0;
}
