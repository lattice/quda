#include <stdio.h>
#include <stdlib.h>

#include <test_util.h>
#include <dslash_reference.h>

QudaGaugeParam param;
void *gauge[4], *new_gauge[4];

void init() {

  param.blockDim = 64;

  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = QUDA_HALF_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_8;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.X[0] = 8;
  param.X[1] = 8;
  param.X[2] = 8;
  param.X[3] = 4;
  setDims(param.X);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  // construct gauge fields
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
    new_gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }

  int dev = 0;
  initQuda(dev);
}

void end() {
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
  SU3Test();
}
