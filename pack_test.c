#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>

#include <gauge_quda.h>
#include <spinor_quda.h>

#include <dslash_reference.h>

#define FULL_WILSON 1

QudaGaugeParam param;
FullSpinor cudaFullSpinor;
ParitySpinor cudaParitySpinor;

float *qdpGauge[4];
float *cpsGauge;
float *spinor;
    
float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
    
#define myalloc(type, n, m0) (type *) aligned_malloc(n*sizeof(type), m0)

#define ALIGN 16
void *
aligned_malloc(size_t n, void **m0)
{
  size_t m = (size_t) malloc(n+ALIGN);
  *m0 = (void*)m;
  size_t r = m % ALIGN;
  if(r) m += (ALIGN - r);
  return (void *)m;
}

void init() {

  Precision single = QUDA_SINGLE_PRECISION;

  param.blockDim = 64;

  param.cpu_prec = QUDA_SINGLE_PRECISION;
  param.cuda_prec = QUDA_SINGLE_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_12;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.X[0] = 4;
  param.X[1] = 4;
  param.X[2] = 4;
  param.X[3] = 4;
  setDims(param.X);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param = &param;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    qdpGauge[dir] = (float*)malloc(V*gaugeSiteSize*sizeof(float));
  }
  cpsGauge = (float*)malloc(4*V*gaugeSiteSize*sizeof(float));

  spinor = (float*)malloc(V*spinorSiteSize*sizeof(float));
  
  int dev = 0;
  cudaSetDevice(dev);

  param.X[0] /= 2;
  cudaFullSpinor = allocateSpinorField(param.X, single);
  cudaParitySpinor = allocateParitySpinor(param.X, single);
  param.X[0] *= 2;

}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(qdpGauge[dir]);
  free(cpsGauge);
  free(spinor);
  freeSpinorField(cudaFullSpinor);
  freeParitySpinor(cudaParitySpinor);
  freeSpinorBuffer();
}

void packTest() {

  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(float) / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", param.gaugeGiB);

  printf("Sending fields to GPU...\n"); fflush(stdout);
  
  stopwatchStart();
  param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;
  createGaugeField(&cudaGaugePrecise, cpsGauge, param.reconstruct, param.cuda_prec, param.X, 1.0, param.blockDim);
  double cpsGtime = stopwatchReadSeconds();
  printf("CPS Gauge send time = %e seconds\n", cpsGtime);

  stopwatchStart();
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  createGaugeField(&cudaGaugePrecise, qdpGauge, param.reconstruct, param.cuda_prec, param.X, 1.0, param.blockDim);
  double qdpGtime = stopwatchReadSeconds();
  printf("QDP Gauge send time = %e seconds\n", qdpGtime);

  stopwatchStart();
  loadSpinorField(cudaFullSpinor, (void*)spinor, QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
  double sSendTime = stopwatchReadSeconds();
  printf("Spinor send time = %e seconds\n", sSendTime);

  stopwatchStart();

  stopwatchStart();
  loadParitySpinor(cudaFullSpinor.even, (void*)spinor, QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
  double pSendTime = stopwatchReadSeconds();
  printf("Parity spinor send time = %e seconds\n", pSendTime);

  stopwatchStart();
  retrieveSpinorField(spinor, cudaFullSpinor, QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
  double sRecTime = stopwatchReadSeconds();
  printf("Spinor receive time = %e seconds\n", sRecTime);
  
  stopwatchStart();
  retrieveParitySpinor(spinor, cudaParitySpinor, QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
  double pRecTime = stopwatchReadSeconds();
  printf("Parity receive time = %e seconds\n", pRecTime);

  end();

}

int main(int argc, char **argv) {
  packTest();
}

