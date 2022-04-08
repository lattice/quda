#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <gauge_field.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <timer.h>

using namespace quda;

QudaGaugeParam param;
std::unique_ptr<ColorSpinorField> cudaSpinor;

void *qdpCpuGauge_p[4];
void *cpsCpuGauge_p;
std::unique_ptr<ColorSpinorField> spinor, spinor2;

ColorSpinorParam csParam;

int ODD_BIT = 0;
int DAGGER_BIT = 0;

QudaPrecision prec_cpu = QUDA_DOUBLE_PRECISION;

void init()
{

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
  param.ga_pad = xdim * ydim * zdim / 2;
#else
  param.ga_pad = 0;
#endif
  setDims(param.X);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) { qdpCpuGauge_p[dir] = safe_malloc(V * gauge_site_size * param.cpu_prec); }
  cpsCpuGauge_p = safe_malloc(4 * V * gauge_site_size * param.cpu_prec);

  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  csParam.pc_type = QUDA_4D_PC;
  for (int d = 0; d < 4; d++) csParam.x[d] = param.X[d];
  csParam.setPrecision(prec_cpu);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_NULL_FIELD_CREATE;
  csParam.location = QUDA_CPU_FIELD_LOCATION;

  spinor = std::make_unique<ColorSpinorField>(csParam);
  spinor2 = std::make_unique<ColorSpinorField>(csParam);

  spinor->Source(QUDA_RANDOM_SOURCE);

  initQuda(device_ordinal);

  setVerbosityQuda(QUDA_VERBOSE, "", stdout);

  csParam.setPrecision(prec, prec, true);
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.location = QUDA_CUDA_FIELD_LOCATION;

  cudaSpinor = std::make_unique<ColorSpinorField>(csParam);
}

void end()
{
  // release memory
  cudaSpinor.reset();
  spinor2.reset();
  spinor.reset();

  for (int dir = 0; dir < 4; dir++) host_free(qdpCpuGauge_p[dir]);
  host_free(cpsCpuGauge_p);
  endQuda();
}

void packTest()
{
  host_timer_t host_timer;

  printfQuda("Sending fields to GPU...\n");

#ifdef BUILD_CPS_INTERFACE
  {
    param.gauge_order = QUDA_CPS_WILSON_GAUGE_ORDER;

    GaugeFieldParam cpsParam(param, cpsCpuGauge_p);
    cpuGaugeField cpsCpuGauge(cpsParam);
    cpsParam.create = QUDA_NULL_FIELD_CREATE;
    cpsParam.reconstruct = param.reconstruct;
    cpsParam.setPrecision(param.cuda_prec, true);
    cpsParam.pad = param.ga_pad;
    cudaGaugeField cudaCpsGauge(cpsParam);

    host_timer.start();
    cudaCpsGauge.loadCPUField(cpsCpuGauge);
    host_timer.stop();
    printfQuda("CPS Gauge send time = %e seconds\n", host_timer.last());

    host_timer.start();
    cudaCpsGauge.saveCPUField(cpsCpuGauge);
    host_timer.stop();
    printfQuda("CPS Gauge restore time = %e seconds\n", host_timer.last());
  }
#endif

#ifdef BUILD_QDP_INTERFACE
  {
    param.gauge_order = QUDA_QDP_GAUGE_ORDER;

    GaugeFieldParam qdpParam(param, qdpCpuGauge_p);
    cpuGaugeField qdpCpuGauge(qdpParam);
    qdpParam.create = QUDA_NULL_FIELD_CREATE;
    qdpParam.reconstruct = param.reconstruct;
    qdpParam.setPrecision(param.cuda_prec, true);
    qdpParam.pad = param.ga_pad;
    cudaGaugeField cudaQdpGauge(qdpParam);

    host_timer.start();
    cudaQdpGauge.loadCPUField(qdpCpuGauge);
    host_timer.stop();
    printfQuda("QDP Gauge send time = %e seconds\n", host_timer.last());

    host_timer.start();
    cudaQdpGauge.saveCPUField(qdpCpuGauge);
    host_timer.stop();
    printfQuda("QDP Gauge restore time = %e seconds\n", host_timer.last());
  }
#endif

  host_timer.start();
  *cudaSpinor = *spinor;
  host_timer.stop();
  printfQuda("Spinor send time = %e seconds\n", host_timer.last());

  host_timer.start();
  *spinor2 = *cudaSpinor;
  host_timer.stop();
  printfQuda("Spinor receive time = %e seconds\n", host_timer.last());

  double spinor_norm = blas::norm2(*spinor);
  double cuda_spinor_norm = blas::norm2(*cudaSpinor);
  double spinor2_norm = blas::norm2(*spinor2);

  printfQuda("Norm check: CPU = %e, CUDA = %e, CPU = %e\n", spinor_norm, cuda_spinor_norm, spinor2_norm);

  ColorSpinorField::Compare(*spinor, *spinor2, 1);
}

int main(int argc, char **argv)
{
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
