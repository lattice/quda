#include <stdio.h>
#include <stdlib.h>

#include <test_util.h>
#include <wilson_dslash_reference.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

QudaGaugeParam param;
void *gauge[4], *new_gauge[4];

void init() {

  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = QUDA_SINGLE_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_12;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.X[0] = 8;
  param.X[1] = 8;
  param.X[2] = 8;
  param.X[3] = 8;
  setDims(param.X);

  param.anisotropy = 1.0;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
#ifdef MULTI_GPU
  param.ga_pad = param.X[0]*param.X[1]*param.X[2]/2;
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

void SU3Test() {

  init();
    
  printf("Randomizing fields...");
  construct_gauge_field((void**)gauge, 1, param.cpu_prec, &param);
  printf("done.\n");

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

  SU3Test();

#ifdef QMP_COMMS
  QMP_finalize_msg_passing();
#endif

  return 0;
}
