#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <quda_internal.h>
#include <gauge_field.h>
#include <util_quda.h>

#include <test_util.h>
#include <wilson_dslash_reference.h>

#include <color_spinor_field.h>
#include <blas_quda.h>

QudaGaugeParam param;
cudaColorSpinorField *cudaSpinor;

void *qdpCpuGauge_p[4];
void *cpsCpuGauge_p;
cpuColorSpinorField *spinor, *spinor2;

ColorSpinorParam csParam;

float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
    
// where is the packing / unpacking taking place
//most orders are CPU only currently
const QudaFieldLocation location = QUDA_CPU_FIELD_LOCATION;

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
    qdpCpuGauge_p[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }
  cpsCpuGauge_p = malloc(4*V*gaugeSiteSize*param.cpu_prec);

  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = param.X[d];
  csParam.precision = QUDA_SINGLE_PRECISION;
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_NULL_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinor2 = new cpuColorSpinorField(csParam);

  spinor->Source(QUDA_RANDOM_SOURCE);

  initQuda(0);

  csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
  csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  csParam.pad = 0;
  csParam.precision = QUDA_HALF_PRECISION;

  cudaSpinor = new cudaColorSpinorField(csParam);
}

void end() {
  // release memory
  delete cudaSpinor;
  delete spinor2;
  delete spinor;

  for (int dir = 0; dir < 4; dir++) free(qdpCpuGauge_p[dir]);
  free(cpsCpuGauge_p);
  endQuda();
}

void packTest() {

  float spinorGiB = (float)Vh*spinorSiteSize*param.cuda_prec / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", param.gaugeGiB);

  printf("Sending fields to GPU...\n"); fflush(stdout);
  
  {
    param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;
    
    GaugeFieldParam cpsParam(cpsCpuGauge_p, param);
    cpuGaugeField cpsCpuGauge(cpsParam);
    cpsParam.create = QUDA_NULL_FIELD_CREATE;
    cpsParam.precision = param.cuda_prec;
    cpsParam.reconstruct = param.reconstruct;
    cpsParam.pad = param.ga_pad;
    cpsParam.order = (cpsParam.precision == QUDA_DOUBLE_PRECISION || 
		      cpsParam.reconstruct == QUDA_RECONSTRUCT_NO ) ?
      QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    cudaGaugeField cudaCpsGauge(cpsParam);

    stopwatchStart();
    cudaCpsGauge.loadCPUField(cpsCpuGauge, location);    
    double cpsGtime = stopwatchReadSeconds();
    printf("CPS Gauge send time = %e seconds\n", cpsGtime);

    stopwatchStart();
    cudaCpsGauge.saveCPUField(cpsCpuGauge, location);
    double cpsGRtime = stopwatchReadSeconds();
    printf("CPS Gauge restore time = %e seconds\n", cpsGRtime);
  }

  {
    param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    
    GaugeFieldParam qdpParam(qdpCpuGauge_p, param);
    cpuGaugeField qdpCpuGauge(qdpParam);
    qdpParam.create = QUDA_NULL_FIELD_CREATE;
    qdpParam.precision = param.cuda_prec;
    qdpParam.reconstruct = param.reconstruct;
    qdpParam.pad = param.ga_pad;
    qdpParam.order = (qdpParam.precision == QUDA_DOUBLE_PRECISION || 
		      qdpParam.reconstruct == QUDA_RECONSTRUCT_NO ) ?
      QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    cudaGaugeField cudaQdpGauge(qdpParam);

    stopwatchStart();
    cudaQdpGauge.loadCPUField(qdpCpuGauge, location);    
    double qdpGtime = stopwatchReadSeconds();
    printf("QDP Gauge send time = %e seconds\n", qdpGtime);

    stopwatchStart();
    cudaQdpGauge.saveCPUField(qdpCpuGauge, location);
    double qdpGRtime = stopwatchReadSeconds();
    printf("QDP Gauge restore time = %e seconds\n", qdpGRtime);
  }

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

