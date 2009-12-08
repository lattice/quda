#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <util_quda.h>
#include <gauge_quda.h>
#include <spinor_quda.h>

#include <test_util.h>
#include <dslash_reference.h>

QudaGaugeParam param;
FullSpinor cudaFullSpinor;
ParitySpinor cudaParitySpinor;

void *qdpGauge[4];
void *cpsGauge;
void *spinor;
void *spinor2;
    
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

  param.cpu_prec = QUDA_SINGLE_PRECISION;
  param.cuda_prec = QUDA_HALF_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_8;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.X[0] = 4;
  param.X[1] = 4;
  param.X[2] = 4;
  param.X[3] = 4;
  param.ga_pad = 0;
  setDims(param.X);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  int sp_pad = 256;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    qdpGauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }
  cpsGauge = malloc(4*V*gaugeSiteSize*param.cpu_prec);

  spinor = malloc(V*spinorSiteSize*param.cpu_prec);
  spinor2 = malloc(V*spinorSiteSize*param.cpu_prec);

  construct_spinor_field(spinor, 1, 0, 0, 0, param.cpu_prec);
  
  int dev = 0;
  cudaSetDevice(dev);

  param.X[0] /= 2;
  cudaFullSpinor = allocateSpinorField(param.X, param.cuda_prec, sp_pad);
  cudaParitySpinor = allocateParitySpinor(param.X, param.cuda_prec, sp_pad);
  param.X[0] *= 2;

}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(qdpGauge[dir]);
  free(cpsGauge);
  free(spinor);
  free(spinor2);
  freeSpinorField(cudaFullSpinor);
  freeParitySpinor(cudaParitySpinor);
  freeSpinorBuffer();
}

void packTest() {

  init();

  float spinorGiB = (float)Vh*spinorSiteSize*param.cuda_prec / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", param.gaugeGiB);

  printf("Sending fields to GPU...\n"); fflush(stdout);
  
  stopwatchStart();
  param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;
  createGaugeField(&cudaGaugePrecise, cpsGauge, param.cuda_prec, param.cpu_prec, param.gauge_order, param.reconstruct, 
		   param.gauge_fix, param.t_boundary, param.X, param.ga_pad, 1.0);
  double cpsGtime = stopwatchReadSeconds();
  printf("CPS Gauge send time = %e seconds\n", cpsGtime);

  stopwatchStart();
  restoreGaugeField(cpsGauge, &cudaGaugePrecise, param.cpu_prec, param.gauge_order);
  double cpsGRtime = stopwatchReadSeconds();
  printf("CPS Gauge restore time = %e seconds\n", cpsGRtime);

  stopwatchStart();
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  createGaugeField(&cudaGaugePrecise, qdpGauge, param.cuda_prec, param.cpu_prec, param.gauge_order, param.reconstruct, 
		   param.gauge_fix, param.t_boundary, param.X, param.ga_pad, 1.0);
  double qdpGtime = stopwatchReadSeconds();
  printf("QDP Gauge send time = %e seconds\n", qdpGtime);

  stopwatchStart();
  restoreGaugeField(qdpGauge, &cudaGaugePrecise, param.cpu_prec, param.gauge_order);
  double qdpGRtime = stopwatchReadSeconds();
  printf("QDP Gauge restore time = %e seconds\n", qdpGRtime);

  stopwatchStart();
  loadParitySpinor(cudaFullSpinor.even, (void*)spinor, param.cpu_prec, QUDA_DIRAC_ORDER);
  double pSendTime = stopwatchReadSeconds();
  printf("Parity spinor send time = %e seconds\n", pSendTime);

  stopwatchStart();
  retrieveParitySpinor(spinor2, cudaFullSpinor.even, param.cpu_prec, QUDA_DIRAC_ORDER);
  double pRecTime = stopwatchReadSeconds();
  printf("Parity receive time = %e seconds\n", pRecTime);

  stopwatchStart();
  loadSpinorField(cudaFullSpinor, (void*)spinor, param.cpu_prec, QUDA_DIRAC_ORDER);
  double sSendTime = stopwatchReadSeconds();
  printf("Spinor send time = %e seconds\n", sSendTime);

  stopwatchStart();
  retrieveSpinorField(spinor2, cudaFullSpinor, param.cpu_prec, QUDA_DIRAC_ORDER);
  double sRecTime = stopwatchReadSeconds();
  printf("Spinor receive time = %e seconds\n", sRecTime);
  
  compare_spinor(spinor, spinor2, V, param.cpu_prec);

  end();

}

int main(int argc, char **argv) {
  packTest();
}

