#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>

#include <gauge_quda.h>
#include <spinor_quda.h>
#include <dslash_reference.h>

#define MAX_SHORT 32767
#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

inline short floatToShort(float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline short doubleToShort(double a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

QudaGaugeParam param;
void *gauge[4], *new_gauge[4];

void init() {

  param.blockDim = 64;

  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = QUDA_DOUBLE_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_12;
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
  gauge_param = &param;

  // construct gauge fields
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
    new_gauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }

  int dev = 0;
  cudaSetDevice(dev);
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
  construct_gauge_field((void**)gauge, 1, param.cpu_prec);
  printf("done.\n");

  loadGaugeQuda(gauge, &param);
  saveGaugeQuda(new_gauge);

  check_gauge(gauge, new_gauge, 1e-3, param.cpu_prec);

  end();
}

int main(int argc, char **argv) {
  SU3Test();
}
