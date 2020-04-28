#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <quda_internal.h>
#include <gauge_field.h>
#include <util_quda.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#include <color_spinor_field.h>
#include <blas_quda.h>

using namespace quda;

QudaGaugeParam param;
cudaColorSpinorField *cudaSpinor;

void *qdpCpuGauge_p[4];
void *cpsCpuGauge_p;
cpuColorSpinorField *spinor, *spinor2;

ColorSpinorParam csParam;

int ODD_BIT = 0;
int DAGGER_BIT = 0;
    
QudaPrecision prec_cpu = QUDA_DOUBLE_PRECISION;

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
  for (int dir = 0; dir < 4; dir++) { qdpCpuGauge_p[dir] = malloc(V * gauge_site_size * param.cpu_prec); }
  cpsCpuGauge_p = malloc(4 * V * gauge_site_size * param.cpu_prec);

  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = param.X[d];
  csParam.setPrecision(prec_cpu);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_NULL_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinor2 = new cpuColorSpinorField(csParam);

  spinor->Source(QUDA_RANDOM_SOURCE);

  initQuda(device);

  setVerbosityQuda(QUDA_VERBOSE, "", stdout);

  csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  csParam.setPrecision(QUDA_DOUBLE_PRECISION);
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

  printfQuda("Sending fields to GPU...\n");

#ifdef BUILD_CPS_INTERFACE
  {
    param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;

    GaugeFieldParam cpsParam(cpsCpuGauge_p, param);
    cpuGaugeField cpsCpuGauge(cpsParam);
    cpsParam.create = QUDA_NULL_FIELD_CREATE;
    cpsParam.reconstruct = param.reconstruct;
    cpsParam.setPrecision(param.cuda_prec, true);
    cpsParam.pad = param.ga_pad;
    cudaGaugeField cudaCpsGauge(cpsParam);

    stopwatchStart();
    cudaCpsGauge.loadCPUField(cpsCpuGauge);
    double cpsGtime = stopwatchReadSeconds();
    printfQuda("CPS Gauge send time = %e seconds\n", cpsGtime);

    stopwatchStart();
    cudaCpsGauge.saveCPUField(cpsCpuGauge);
    double cpsGRtime = stopwatchReadSeconds();
    printfQuda("CPS Gauge restore time = %e seconds\n", cpsGRtime);
  }
#endif

#ifdef BUILD_QDP_INTERFACE
  {
    param.gauge_order = QUDA_QDP_GAUGE_ORDER;

    GaugeFieldParam qdpParam(qdpCpuGauge_p, param);
    cpuGaugeField qdpCpuGauge(qdpParam);
    qdpParam.create = QUDA_NULL_FIELD_CREATE;
    qdpParam.reconstruct = param.reconstruct;
    qdpParam.setPrecision(param.cuda_prec, true);
    qdpParam.pad = param.ga_pad;
    cudaGaugeField cudaQdpGauge(qdpParam);

    stopwatchStart();
    cudaQdpGauge.loadCPUField(qdpCpuGauge);
    double qdpGtime = stopwatchReadSeconds();
    printfQuda("QDP Gauge send time = %e seconds\n", qdpGtime);

    stopwatchStart();
    cudaQdpGauge.saveCPUField(qdpCpuGauge);
    double qdpGRtime = stopwatchReadSeconds();
    printfQuda("QDP Gauge restore time = %e seconds\n", qdpGRtime);
  }
#endif

  stopwatchStart();

  *cudaSpinor = *spinor;
  double sSendTime = stopwatchReadSeconds();
  printfQuda("Spinor send time = %e seconds\n", sSendTime);

  stopwatchStart();
  *spinor2 = *cudaSpinor;
  double sRecTime = stopwatchReadSeconds();
  printfQuda("Spinor receive time = %e seconds\n", sRecTime);

  double spinor_norm = blas::norm2(*spinor);
  double cuda_spinor_norm = blas::norm2(*cudaSpinor);
  double spinor2_norm = blas::norm2(*spinor2);

  printfQuda("Norm check: CPU = %e, CUDA = %e, CPU = %e\n", spinor_norm, cuda_spinor_norm, spinor2_norm);

  cpuColorSpinorField::Compare(*spinor, *spinor2, 1);
}

int main(int argc, char **argv) {
  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  init();
  packTest();
  end();

  finalizeComms();
}

