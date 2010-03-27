#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <quda_internal.h>
#include <gauge_quda.h>
#include <util_quda.h>

#include <test_util.h>
#include <dslash_reference.h>

#include <color_spinor_field.h>
#include <blas_quda.h>

QudaGaugeParam param;
FullGauge cudaGauge;
cudaColorSpinorField *cudaSpinor;

void *qdpGauge[4];
void *cpsGauge;
cpuColorSpinorField *spinor, *spinor2;

ColorSpinorParam csParam;

float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
    
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

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    qdpGauge[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }
  cpsGauge = malloc(4*V*gaugeSiteSize*param.cpu_prec);

  csParam.fieldType = QUDA_CPU_FIELD;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = param.X[d];
  csParam.precision = QUDA_SINGLE_PRECISION;
  csParam.pad = 0;
  csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  csParam.subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_ORDER;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;
  csParam.create = QUDA_NULL_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinor2 = new cpuColorSpinorField(csParam);

  spinor->Source(QUDA_RANDOM_SOURCE);

  initQuda(0);

  csParam.fieldType = QUDA_CUDA_FIELD;
  csParam.fieldOrder = QUDA_FLOAT4_ORDER;
  csParam.basis = QUDA_UKQCD_BASIS;
  csParam.pad = 0;
  csParam.precision = QUDA_HALF_PRECISION;

  cudaSpinor = new cudaColorSpinorField(csParam);
}

void end() {
  // release memory
  delete cudaSpinor;
  delete spinor2;
  delete spinor;

  for (int dir = 0; dir < 4; dir++) free(qdpGauge[dir]);
  free(cpsGauge);
  endQuda();
}

void packTest() {

  float spinorGiB = (float)Vh*spinorSiteSize*param.cuda_prec / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", param.gaugeGiB);

  printf("Sending fields to GPU...\n"); fflush(stdout);
  
  stopwatchStart();
  param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;
  createGaugeField(&cudaGauge, cpsGauge, param.cuda_prec, param.cpu_prec, param.gauge_order, param.reconstruct, 
		   param.gauge_fix, param.t_boundary, param.X, 1.0, param.ga_pad);
  double cpsGtime = stopwatchReadSeconds();
  printf("CPS Gauge send time = %e seconds\n", cpsGtime);

  stopwatchStart();
  restoreGaugeField(cpsGauge, &cudaGauge, param.cpu_prec, param.gauge_order);
  double cpsGRtime = stopwatchReadSeconds();
  printf("CPS Gauge restore time = %e seconds\n", cpsGRtime);

  stopwatchStart();
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  createGaugeField(&cudaGauge, qdpGauge, param.cuda_prec, param.cpu_prec, param.gauge_order, param.reconstruct, 
		   param.gauge_fix, param.t_boundary, param.X, 1.0, param.ga_pad);
  double qdpGtime = stopwatchReadSeconds();
  printf("QDP Gauge send time = %e seconds\n", qdpGtime);

  stopwatchStart();
  restoreGaugeField(qdpGauge, &cudaGauge, param.cpu_prec, param.gauge_order);
  double qdpGRtime = stopwatchReadSeconds();
  printf("QDP Gauge restore time = %e seconds\n", qdpGRtime);

  stopwatchStart();
  *cudaSpinor = *spinor;
  double sSendTime = stopwatchReadSeconds();
  printf("Spinor send time = %e seconds\n", sSendTime);

  stopwatchStart();
  *spinor2 = *cudaSpinor;
  double sRecTime = stopwatchReadSeconds();
  printf("Spinor receive time = %e seconds\n", sRecTime);
  
  std::cout << "Norm check: CPU = " << norm2(*spinor) << 
    ", CUDA = " << norm2(*cudaSpinor) << 
    ", CPU =  " << norm2(*spinor2) << std::endl;

  cpuColorSpinorField::Compare(*spinor, *spinor2, 1);

}

int main(int argc, char **argv) {
  init();
  packTest();
  end();
}

