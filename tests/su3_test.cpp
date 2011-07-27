#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_util.h>
#include <wilson_dslash_reference.h>

#include <gauge_qio.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

QudaGaugeParam param;
void *gauge[4], *new_gauge[4];

#define MAX(a,b) ((a)>(b)?(a):(b))

void init() {

  param = newQudaGaugeParam();

  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = QUDA_HALF_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_8;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.type = QUDA_WILSON_LINKS;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;

  param.X[0] = 16;
  param.X[1] = 16;
  param.X[2] = 16;
  param.X[3] = 64;
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
#endif

  // construct gauge fields
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
    new_gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }

  int dev = 0;
  initQuda(dev);
}

void end() {
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(new_gauge[dir]);
  }
}

void SU3Test(int argc, char **argv) {

  init();

  char *latfile = "16_64.lat";
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

  check_gauge(gauge, new_gauge, 1e-3, param.cpu_prec);

  end();
}

int main(int argc, char **argv) {

#ifdef QMP_COMMS
  int ndim=4, dims[4];
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
  dims[0] = dims[1] = dims[2] = 1;
  dims[3] = QMP_get_number_of_nodes();
  QMP_declare_logical_topology(dims, ndim);
#endif  

  SU3Test(argc, argv);

#ifdef QMP_COMMS
  QMP_finalize_msg_passing();
#endif

  return 0;
}
