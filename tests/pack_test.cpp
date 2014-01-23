#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <quda_internal.h>
#include <gauge_field.h>
#include <util_quda.h>

#include <test_util.h>
#include <dslash_util.h>

#include <color_spinor_field.h>
#include <blas_quda.h>

using namespace quda;

QudaGaugeParam param;
cudaColorSpinorField *cudaSpinor;

void *qdpCpuGauge_p[4];
void *cpsCpuGauge_p;
cpuColorSpinorField *spinor, *spinor2;

ColorSpinorParam csParam;

float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern char latfile[];
extern int gridsize_from_cmdline[];
    
extern bool tune;

QudaPrecision prec_cpu = QUDA_DOUBLE_PRECISION;

// where is the packing / unpacking taking place
//most orders are CPU only currently
const QudaFieldLocation location = QUDA_CPU_FIELD_LOCATION;

void init() {

  param.cpu_prec = prec_cpu;
  param.cuda_prec = prec;
  param.reconstruct = link_recon;
  param.cuda_prec_sloppy = param.cuda_prec;
  param.reconstruct_sloppy = param.reconstruct;
  
  param.X[0] = xdim;
  param.X[1] = ydim;
  param.X[2] = zdim;
  param.X[3] = tdim;
#ifdef MULTI_GPU
  param.ga_pad = xdim*ydim*zdim/2;
#else
  param.ga_pad = 0;
#endif
  setDims(param.X);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    qdpCpuGauge_p[dir] = malloc(V*gaugeSiteSize*param.cpu_prec);
  }
  cpsCpuGauge_p = malloc(4*V*gaugeSiteSize*param.cpu_prec);

  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = param.X[d];
  csParam.precision = prec_cpu;
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_NULL_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinor2 = new cpuColorSpinorField(csParam);

  spinor->Source(QUDA_RANDOM_SOURCE);

  initQuda(device);

  setVerbosityQuda(QUDA_VERBOSE, "", stdout);
  setTuning(tune ? QUDA_TUNE_YES : QUDA_TUNE_NO);

  csParam.setPrecision(prec);
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.pad = param.X[0] * param.X[1] * param.X[2];

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
  
#ifdef BUILD_CPS_INTERFACE
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
#endif

#ifdef BUILD_QDP_INTERFACE
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
#endif

  stopwatchStart();

  *cudaSpinor = *spinor;
  double sSendTime = stopwatchReadSeconds();
  printf("Spinor send time = %e seconds\n", sSendTime); fflush(stdout);

  stopwatchStart();
  *spinor2 = *cudaSpinor;
  double sRecTime = stopwatchReadSeconds();
  printf("Spinor receive time = %e seconds\n", sRecTime); fflush(stdout);
  
  double spinor_norm = norm2(*spinor);
  double cuda_spinor_norm = norm2(*cudaSpinor);
  double spinor2_norm = norm2(*spinor2);

  printf("Norm check: CPU = %e, CUDA = %e, CPU = %e\n",
	 spinor_norm, cuda_spinor_norm, spinor2_norm);

  cpuColorSpinorField::Compare(*spinor, *spinor2, 1);

}

extern void usage(char**);

int main(int argc, char **argv) {
  for (int i=1; i<argc; i++){    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  init();
  packTest();
  end();

  finalizeComms();
}

