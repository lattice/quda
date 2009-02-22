#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include <invert_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <field_quda.h>

FullGauge gaugeSloppy; // sloppy gauge field
FullGauge gaugePrecise; // precise gauge field

void printGaugeParam(QudaGaugeParam *param) {

  printf("Gauge Params:\n");
  printf("X = %d\n", param->X);
  printf("Y = %d\n", param->Y);
  printf("Z = %d\n", param->Z);
  printf("T = %d\n", param->T);
  printf("anisotropy = %e\n", param->anisotropy);
  printf("gauge_order = %d\n", param->gauge_order);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("reconstruct = %d\n", param->reconstruct);
  printf("gauge_fix = %d\n", param->gauge_fix);
  printf("t_boundary = %d\n", param->t_boundary);
  printf("packed_size = %d\n", param->packed_size);
  printf("gaugeGiB = %e\n", param->gaugeGiB);
}

void printInvertParam(QudaInvertParam *param) {
  printf("kappa = %e\n", param->kappa);
  printf("mass_normalization = %d\n", param->mass_normalization);
  printf("inv_type = %d\n", param->inv_type);
  printf("tol = %e\n", param->tol);
  printf("iter = %d\n", param->iter);
  printf("maxiter = %d\n", param->maxiter);
  printf("matpc_type = %d\n", param->matpc_type);
  printf("solution_type = %d\n", param->solution_type);
  printf("preserve_source = %d\n", param->preserve_source);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("dirac_order = %d\n", param->dirac_order);
  printf("spinorGiB = %e\n", param->spinorGiB);
  printf("gflops = %e\n", param->gflops);
  printf("secs = %f\n", param->secs);
}

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

  cudaGauge.even = 0;
  cudaGauge.odd = 0;

  cudaHGauge.even = 0;
  cudaHGauge.odd = 0;

  hSpinor1.spinorHalf = 0;
  hSpinor1.spinorNorm = 0;

  hSpinor2.spinorHalf = 0;
  hSpinor2.spinorNorm = 0;
}

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  gauge_param = param;
  loadGaugeField(h_gauge);

  gaugePrecise = cudaGauge;
  gaugeSloppy = (param->cuda_prec == QUDA_HALF_PRECISION) ? cudaHGauge : cudaGauge;
}

void endQuda()
{
  freeSpinorBuffer();
  freeGaugeField();
}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger)
{
  ParitySpinor in = allocateParitySpinor();
  ParitySpinor out = allocateParitySpinor();

  loadParitySpinor(in, h_in, inv_param->cpu_prec, 
		   inv_param->cuda_prec, inv_param->dirac_order);
  
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    dslashCuda(out, cudaGauge, in, parity, dagger);
  } else if (inv_param->cuda_prec == QUDA_HALF_PRECISION) {
    dslashCuda(out, cudaHGauge, in, parity, dagger);
  }

  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, 
		       inv_param->cuda_prec, inv_param->dirac_order);

  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  ParitySpinor in = allocateParitySpinor();
  ParitySpinor out = allocateParitySpinor();
  ParitySpinor tmp = allocateParitySpinor();
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, 
		   inv_param->cuda_prec, inv_param->dirac_order);
  
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    MatPCCuda(out, cudaGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  } else {
    MatPCCuda(out, cudaHGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  }

  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, 
		       inv_param->cuda_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCDagQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  ParitySpinor in = allocateParitySpinor();
  ParitySpinor out = allocateParitySpinor();
  ParitySpinor tmp = allocateParitySpinor();
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, 
		   inv_param->cuda_prec, inv_param->dirac_order);
  
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    MatPCDagCuda(out, cudaGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  } else {
    MatPCDagCuda(out, cudaHGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  }

  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, 
		       inv_param->cuda_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  ParitySpinor in = allocateParitySpinor();
  ParitySpinor out = allocateParitySpinor();
  ParitySpinor tmp = allocateParitySpinor();
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, 
		   inv_param->cuda_prec, inv_param->dirac_order);
  
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    MatPCDagMatPCCuda(out, cudaGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  } else {
    MatPCDagMatPCCuda(out, cudaHGauge, in, inv_param->kappa, tmp, inv_param->matpc_type);
  }

  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, 
		       inv_param->cuda_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void invertQuda(void *h_x, void *h_b, QudaInvertParam *param)
{
  invert_param = param;

  if (param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    printf("Sorry, only double precision not yet supported\n");
    exit(-1);
  }

  if (param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }

  int slenh = Nh*spinorSiteSize;

  float spinorGiB = (float)slenh*sizeof(float) / (1 << 30);
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7);
  else
    spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9);
  param->spinorGiB = spinorGiB;

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  float kappa = param->kappa;

  FullSpinor b, x;
  ParitySpinor in = allocateParitySpinor(); // source vector
  ParitySpinor out = allocateParitySpinor(); // solution vector
  ParitySpinor tmp = allocateParitySpinor(); // temporary used when applying operator

  if (param->solution_type == QUDA_MAT_SOLUTION) {
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
      b.even = allocateParitySpinor();
      b.odd = allocateParitySpinor();
    } else {
      b.even = out;
      b.odd = tmp;
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) { x.odd = tmp; x.even = out; }
    else { x.even = tmp; x.odd = out; }

    loadSpinorField(b, h_b, param->cpu_prec, param->cuda_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(2*kappa, (float *)b.even, slenh);
      axCuda(2*kappa, (float *)b.odd, slenh);
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpaySCuda(in, gaugePrecise, b.odd, 0, 0, b.even, kappa);
    } else {
      dslashXpaySCuda(in, gaugePrecise, b.even, 1, 0, b.odd, kappa);
    }

  } else if (param->solution_type == QUDA_MATPC_SOLUTION || 
	     param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION){
    loadParitySpinor(in, h_b, param->cpu_prec, param->cuda_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION)
      if (param->solution_type == QUDA_MATPC_SOLUTION) 
	axCuda(4*kappa*kappa, (float *)in, slenh);
      else
	axCuda(16*pow(kappa,4), (float *)in, slenh);
  }

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda((float *)out, (float *)in, slenh);
      MatPCDagCuda(in, gaugePrecise, out, kappa, tmp, param->matpc_type);
    }
    invertCgCuda(out, in, gaugePrecise, tmp, param);
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(out, in, gaugeSloppy, gaugePrecise, tmp, param, QUDA_DAG_YES);
      copyCuda((float *)in, (float *)out, slenh);
    }
    invertBiCGstabCuda(out, in, gaugeSloppy, gaugePrecise, tmp, param, QUDA_DAG_NO);
    break;
  default:
    printf("Inverter type %d not implemented\n", param->inv_type);
    exit(-1);
  }

  if (param->solution_type == QUDA_MAT_SOLUTION) {

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      // qdp dirac fields are even-odd ordered
      b.even = in;
      loadSpinorField(b, h_b, param->cpu_prec, param->cuda_prec, param->dirac_order);
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpaySCuda(x.odd, gaugePrecise, out, 1, 0, b.odd, kappa);
    } else {
      dslashXpaySCuda(x.even, gaugePrecise, out, 0, 0, b.even, kappa);
    }

    retrieveSpinorField(h_x, x, param->cpu_prec, param->cuda_prec, param->dirac_order);

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeSpinorField(b);

  } else {
    retrieveParitySpinor(h_x, out, param->cpu_prec, param->cuda_prec, param->dirac_order);
  }

  freeParitySpinor(tmp);
  freeParitySpinor(in);
  freeParitySpinor(out);

  return;
}

