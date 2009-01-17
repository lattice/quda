#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include <quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <field_quda.h>

FullGauge gauge;

void initQuda(int dev)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "No devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(stderr, "found device %d: %s\n", i, deviceProp.name);
  }

  if(dev<0) {
    dev = deviceCount - 1;
    //dev = 0;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.major < 1) {
    fprintf(stderr, "Device %d does not support CUDA.\n", dev);
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);
}

void loadQuda(void *h_gauge, QudaGaugeParam *param)
{
  gauge_param = param;
  gauge = loadGaugeField(h_gauge);
}

void endQuda()
{
  freeSpinorBuffer();
  freeGaugeField(gauge);
}

void invertQuda(void *h_x, void *h_b, QudaInvertParam *perf)
{

  if (perf->cuda_prec != QUDA_SINGLE_PRECISION) {
    printf("Sorry, only single precision supported\n");
    exit(-1);
  }

  if (perf->cpu_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }

  int slenh = Nh*spinorSiteSize;

  float spinorGiB = (float)slenh*sizeof(float) / (1 << 30);
  if (perf->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    spinorGiB *= (perf->inv_type == QUDA_CG_INVERTER ? 5 : 7);
  else
    spinorGiB *= (perf->inv_type == QUDA_CG_INVERTER ? 8 : 9);
  perf->spinorGiB = spinorGiB;

  perf->secs = 0;
  perf->gflops = 0;
  perf->iter = 0;

  float kappa = perf->kappa;

  FullSpinor b, x;
  ParitySpinor in = allocateParitySpinor(); // source vector
  ParitySpinor out = allocateParitySpinor(); // solution vector
  ParitySpinor tmp = allocateParitySpinor(); // temporary used when applying operator

  if (perf->solution_type == QUDA_MAT_SOLUTION) {
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
      b.even = allocateParitySpinor();
      b.odd = allocateParitySpinor();
    } else {
      b.even = out;
      b.odd = tmp;
    }

    if (perf->matpc_type == QUDA_MATPC_EVEN_EVEN) { x.odd = tmp; x.even = out; }
    else { x.even = tmp; x.odd = out; }

    loadSpinorField(b, h_b, perf->cpu_prec, perf->cuda_prec, perf->dirac_order);

    // multiply the source to get the mass normalization
    if (perf->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(2*kappa, (float *)b.even, slenh);
      axCuda(2*kappa, (float *)b.odd, slenh);
    }

    if (perf->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpayCuda(in, gauge, b.odd, 0, 0, b.even, kappa);
    } else {
      dslashXpayCuda(in, gauge, b.even, 1, 0, b.odd, kappa);
    }

  } else if (perf->solution_type == QUDA_MATPC_SOLUTION || 
	     perf->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION){
    loadParitySpinor(in, h_b, perf->cpu_prec, perf->cuda_prec, perf->dirac_order);

    // multiply the source to get the mass normalization
    if (perf->mass_normalization == QUDA_MASS_NORMALIZATION)
      if (perf->solution_type == QUDA_MATPC_SOLUTION) 
	axCuda(4*kappa*kappa, (float *)in, slenh);
      else
	axCuda(16*pow(kappa,4), (float *)in, slenh);
  }

  switch (perf->inv_type) {
  case QUDA_CG_INVERTER:
    if (perf->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda((float *)out, (float *)in, slenh);
      MatPCDagCuda(in, gauge, out, kappa, tmp, perf->matpc_type);
    }
    invertCgCuda(out, in, gauge, tmp, perf);
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (perf->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(out, in, gauge, tmp, perf, QUDA_DAG_YES);
      copyCuda((float *)in, (float *)out, slenh);
    }
    invertBiCGstabCuda(out, in, gauge, tmp, perf, QUDA_DAG_NO);
    break;
  default:
    printf("Inverter type %d not implemented\n", perf->inv_type);
    exit(-1);
  }

  if (perf->solution_type == QUDA_MAT_SOLUTION) {

    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      // qdp dirac fields are even-odd ordered
      b.even = in;
      loadSpinorField(b, h_b, perf->cpu_prec, perf->cuda_prec, perf->dirac_order);
    }

    if (perf->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpayCuda(x.odd, gauge, out, 1, 0, b.odd, kappa);
    } else {
      dslashXpayCuda(x.even, gauge, out, 0, 0, b.even, kappa);
    }

    retrieveSpinorField(h_x, x, perf->cpu_prec, perf->cuda_prec, perf->dirac_order);

    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeSpinorField(b);

  } else if (perf->solution_type == QUDA_MATPC_SOLUTION) {
    retrieveParitySpinor(h_x, out, perf->cpu_prec, perf->cuda_prec, perf->dirac_order);
  }

  freeParitySpinor(tmp);
  freeParitySpinor(in);
  freeParitySpinor(out);

  return;
}

