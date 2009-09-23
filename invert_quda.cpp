#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include <invert_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

#include <blas_reference.h>

FullGauge cudaGaugePrecise; // precise gauge field
FullGauge cudaGaugeSloppy; // sloppy gauge field

FullClover cudaCloverPrecise; // clover term
FullClover cudaCloverSloppy;

FullClover cudaCloverInvPrecise; // inverted clover term
FullClover cudaCloverInvSloppy;

void printGaugeParam(QudaGaugeParam *param) {

  printf("Gauge Params:\n");
  for (int d=0; d<4; d++) {
    printf("X[%d] = %d\n", d, param->X[d]);
  }
  printf("anisotropy = %e\n", param->anisotropy);
  printf("gauge_order = %d\n", param->gauge_order);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("reconstruct = %d\n", param->reconstruct);
  printf("cuda_prec_sloppy = %d\n", param->cuda_prec_sloppy);
  printf("reconstruct_sloppy = %d\n", param->reconstruct_sloppy);
  printf("gauge_fix = %d\n", param->gauge_fix);
  printf("t_boundary = %d\n", param->t_boundary);
  printf("packed_size = %d\n", param->packed_size);
  printf("gaugeGiB = %e\n", param->gaugeGiB);
}

void printInvertParam(QudaInvertParam *param) {
  printf("kappa = %e\n", param->kappa);
  printf("mass_normalization = %d\n", param->mass_normalization);
  printf("dslash_type = %d\n", param->dslash_type);
  printf("inv_type = %d\n", param->inv_type);
  printf("tol = %e\n", param->tol);
  printf("iter = %d\n", param->iter);
  printf("maxiter = %d\n", param->maxiter);
  printf("matpc_type = %d\n", param->matpc_type);
  printf("solution_type = %d\n", param->solution_type);
  printf("preserve_source = %d\n", param->preserve_source);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("cuda_prec_sloppy = %d\n", param->cuda_prec_sloppy);
  printf("dirac_order = %d\n", param->dirac_order);
  printf("spinorGiB = %e\n", param->spinorGiB);
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    printf("clover_cpu_prec = %d\n", param->clover_cpu_prec);
    printf("clover_cuda_prec = %d\n", param->clover_cuda_prec);
    printf("clover_cuda_prec_sloppy = %d\n", param->clover_cuda_prec_sloppy);
    printf("clover_order = %d\n", param->clover_order);
    printf("cloverGiB = %e\n", param->cloverGiB);
  }
  printf("gflops = %e\n", param->gflops);
  printf("secs = %f\n", param->secs);
  printf("verbosity = %d\n", param->verbosity);
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

  cudaGaugePrecise.even = NULL;
  cudaGaugePrecise.odd = NULL;

  cudaGaugeSloppy.even = NULL;
  cudaGaugeSloppy.odd = NULL;

  cudaCloverPrecise.even.clover = NULL;
  cudaCloverPrecise.odd.clover = NULL;

  cudaCloverSloppy.even.clover = NULL;
  cudaCloverSloppy.odd.clover = NULL;

  cudaCloverInvPrecise.even.clover = NULL;
  cudaCloverInvPrecise.odd.clover = NULL;

  cudaCloverInvSloppy.even.clover = NULL;
  cudaCloverInvSloppy.odd.clover = NULL;
}

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  gauge_param = param;

  gauge_param->packed_size = (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;

  createGaugeField(&cudaGaugePrecise, h_gauge, gauge_param->reconstruct, 
		   gauge_param->cuda_prec, gauge_param->X, gauge_param->anisotropy, gauge_param->blockDim);
  gauge_param->gaugeGiB = 2.0*cudaGaugePrecise.bytes/ (1 << 30);
  if (gauge_param->cuda_prec_sloppy != gauge_param->cuda_prec ||
      gauge_param->reconstruct_sloppy != gauge_param->reconstruct) {
    createGaugeField(&cudaGaugeSloppy, h_gauge, gauge_param->reconstruct_sloppy, 
		     gauge_param->cuda_prec_sloppy, gauge_param->X, gauge_param->anisotropy,
		     gauge_param->blockDim_sloppy);
    gauge_param->gaugeGiB += 2.0*cudaGaugeSloppy.bytes/ (1 << 30);
  } else {
    cudaGaugeSloppy = cudaGaugePrecise;
  }
}

void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  if (!h_clover && !h_clovinv) {
    printf("QUDA error: loadCloverQuda() called with neither clover term nor inverse\n");
    exit(-1);
  }
  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on CPU\n");
    exit(-1);
  }

  int X[4];
  for (int i=0; i<4; i++) {
    X[i] = gauge_param->X[i];
  }
  X[0] /= 2; // dimensions of the even-odd sublattice

  inv_param->cloverGiB = 0;

  if (h_clover) {
    cudaCloverPrecise = allocateCloverField(X, inv_param->clover_cuda_prec);
    loadCloverField(cudaCloverPrecise, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverPrecise.even.bytes / (1<<30);

    if (inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
	inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
	cudaCloverSloppy = allocateCloverField(X, inv_param->clover_cuda_prec_sloppy);
	loadCloverField(cudaCloverSloppy, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
	inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
      } else {
	cudaCloverSloppy = cudaCloverPrecise;
      }
    } // sloppy precision clover term not needed otherwise
  }

  cudaCloverInvPrecise = allocateCloverField(X, inv_param->clover_cuda_prec);
  if (!h_clovinv) {
    printf("QUDA error: clover term inverse not implemented yet\n");
    exit(-1);
  } else {
    loadCloverField(cudaCloverInvPrecise, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
  }
  inv_param->cloverGiB += 2.0*cudaCloverInvPrecise.even.bytes / (1<<30);

  if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
    cudaCloverInvSloppy = allocateCloverField(X, inv_param->clover_cuda_prec_sloppy);
    loadCloverField(cudaCloverInvSloppy, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
  } else {
    cudaCloverInvSloppy = cudaCloverInvPrecise;
  }
}

// discard clover term but keep the inverse
void discardCloverQuda(QudaInvertParam *inv_param)
{
  inv_param->cloverGiB -= 2.0*cudaCloverPrecise.even.bytes / (1<<30);
  freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) {
    inv_param->cloverGiB -= 2.0*cudaCloverSloppy.even.bytes / (1<<30);
    freeCloverField(&cudaCloverSloppy);
  }
}

void endQuda(void)
{
  freeSpinorBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
  if (cudaCloverPrecise.even.clover) freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) freeCloverField(&cudaCloverSloppy);
  if (cudaCloverInvPrecise.even.clover) freeCloverField(&cudaCloverInvPrecise);
  if (cudaCloverInvSloppy.even.clover) freeCloverField(&cudaCloverInvSloppy);
}

void checkPrecision(QudaInvertParam *param) {
  if (param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }
}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);

  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);
  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    dslashCuda(out, cudaGaugePrecise, in, parity, dagger);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    cloverDslashCuda(out, cudaGaugePrecise, cudaCloverInvPrecise, in, parity, dagger);
  } else {
    printf("QUDA error: unsupported dslash_type\n");
    exit(-1);
  }
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);
  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCCuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type, dagger);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    cloverMatPCCuda(out, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, in, inv_param->kappa,
		    tmp, inv_param->matpc_type, dagger);
  } else {
    printf("QUDA error: unsupported dslash_type\n");
    exit(-1);
  }
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);  
  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCDagMatPCCuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    cloverMatPCDagMatPCCuda(out, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, in, inv_param->kappa,
			    tmp, inv_param->matpc_type);
  } else {
    printf("QUDA error: unsupported dslash_type\n");
    exit(-1);
  }
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger) {
  checkPrecision(inv_param);

  FullSpinor in = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec);
  FullSpinor out = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec);

  loadSpinorField(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatCuda(out, cudaGaugePrecise, in, -inv_param->kappa, dagger);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
    cloverMatCuda(out, cudaGaugePrecise, cudaCloverPrecise, in, inv_param->kappa, tmp, dagger);
    freeParitySpinor(tmp);
  } else {
    printf("QUDA error: unsupported dslash_type\n");
    exit(-1);
  }
  retrieveSpinorField(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeSpinorField(out);
  freeSpinorField(in);
}

void invertQuda(void *h_x, void *h_b, QudaInvertParam *param)
{
  invert_param = param;

  checkPrecision(param);

  int slenh = cudaGaugePrecise.volume*spinorSiteSize;
  param->spinorGiB = (double)slenh*(param->cuda_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double): sizeof(float);
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(1<<30);
  else
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(1<<30);

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  double kappa = param->kappa;
  if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa /= cudaGaugePrecise.anisotropy;

  FullSpinor b, x;
  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, invert_param->cuda_prec); // source vector
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, invert_param->cuda_prec); // solution vector
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, invert_param->cuda_prec); // temporary used when applying operator

  if (param->solution_type == QUDA_MAT_SOLUTION) {
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
      b = allocateSpinorField(cudaGaugePrecise.X, invert_param->cuda_prec);
    } else {
      b.even = out;
      b.odd = tmp;
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN ||
	param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      x.odd = tmp;
      x.even = out;
    } else {
      x.even = tmp;
      x.odd = out;
    }

    loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b.even);
      axCuda(2.0*kappa, b.odd);
    }

    // cps uses a different anisotropy normalization
    if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      axCuda(1.0/gauge_param->anisotropy, b.even);
      axCuda(1.0/gauge_param->anisotropy, b.odd);
    }

    if (param->dslash_type == QUDA_WILSON_DSLASH) {
      if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
	// in = b_e + k D_eo b_o
	dslashXpayCuda(in, cudaGaugePrecise, b.odd, 0, 0, b.even, kappa);
      } else if (param->matpc_type == QUDA_MATPC_ODD_ODD) {
	// in = b_o + k D_oe b_e
	dslashXpayCuda(in, cudaGaugePrecise, b.even, 1, 0, b.odd, kappa);
      } else {
	printf("QUDA error: matpc_type not valid for plain Wilson\n");
	exit(-1);
      }
    } else if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
	// in = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
	ParitySpinor aux = tmp; // aliases b.odd when PRESERVE_SOURCE_NO is set
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, b.odd, 1);
	dslashXpayCuda(aux, cudaGaugePrecise, in, 0, 0, b.even, kappa);
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, aux, 0);
      } else if (param->matpc_type == QUDA_MATPC_ODD_ODD) {
	// in = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
	ParitySpinor aux = tmp; // aliases b.odd when PRESERVE_SOURCE_NO is set
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, b.even, 0);
	dslashXpayCuda(aux, cudaGaugePrecise, in, 1, 0, b.odd, kappa);
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, aux, 1);
      } else if (param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	// in = b_e + k D_eo A_oo^-1 b_o
	ParitySpinor aux = out; // aliases b.even when PRESERVE_SOURCE_NO is set
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, b.odd, 1);
	dslashXpayCuda(aux, cudaGaugePrecise, in, 0, 0, b.even, kappa);
	copyCuda(in, aux);
      } else if (param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	// in = b_o + k D_oe A_ee^-1 b_e
	ParitySpinor aux = out; // aliases b.even when PRESERVE_SOURCE_NO is set
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, b.even, 0);
	dslashXpayCuda(aux, cudaGaugePrecise, in, 1, 0, b.odd, kappa);
	copyCuda(in, aux);
      } else {
	printf("QUDA error: invalid matpc_type\n");
	exit(-1);
      }
    } else {
      printf("QUDA error: unsupported dslash_type\n");
      exit(-1);
    }

  } else if (param->solution_type == QUDA_MATPC_SOLUTION || 
	     param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION){
    loadParitySpinor(in, h_b, param->cpu_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION)
      if (param->solution_type == QUDA_MATPC_SOLUTION) 
	axCuda(4.0*kappa*kappa, in);
      else
	axCuda(16.0*pow(kappa,4), in);

    // cps uses a different anisotropy normalization
    if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
      if (param->solution_type == QUDA_MATPC_SOLUTION) 
	axCuda(pow(1.0/gauge_param->anisotropy, 2), in);
      else 
	axCuda(pow(1.0/gauge_param->anisotropy, 4), in);

  }

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(out, in);
      if (param->dslash_type == QUDA_WILSON_DSLASH) {
	MatPCCuda(in, cudaGaugePrecise, out, kappa, tmp, param->matpc_type, QUDA_DAG_YES);
      } else {
	cloverMatPCCuda(in, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, out, kappa, tmp,
			param->matpc_type, QUDA_DAG_YES);
      }
    }
    invertCgCuda(out, in, tmp, param);
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(out, in, tmp, param, QUDA_DAG_YES);
      copyCuda(in, out);
    }
    invertBiCGstabCuda(out, in, tmp, param, QUDA_DAG_NO);
    break;
  default:
    printf("Inverter type %d not implemented\n", param->inv_type);
    exit(-1);
  }

  if (param->solution_type == QUDA_MAT_SOLUTION) {

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      // qdp dirac fields are even-odd ordered
      b.even = in;
      loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);
    }

    if (param->dslash_type == QUDA_WILSON_DSLASH) {
      if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
	// x_o = b_o + k D_oe x_e
	dslashXpayCuda(x.odd, cudaGaugePrecise, out, 1, 0, b.odd, kappa);
      } else {
	// x_e = b_e + k D_eo x_o
	dslashXpayCuda(x.even, cudaGaugePrecise, out, 0, 0, b.even, kappa);
      }
    } else {
      if (param->matpc_type == QUDA_MATPC_EVEN_EVEN ||
	  param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	// x_o = A_oo^-1 (b_o + k D_oe x_e)
	ParitySpinor aux = b.even;
	dslashXpayCuda(aux, cudaGaugePrecise, out, 1, 0, b.odd, kappa);
	cloverCuda(x.odd, cudaGaugePrecise, cudaCloverInvPrecise, aux, 1);
      } else {
	// x_e = A_ee^-1 (b_e + k D_eo x_o)
	ParitySpinor aux = b.odd;
	dslashXpayCuda(aux, cudaGaugePrecise, out, 0, 0, b.even, kappa);
	cloverCuda(x.even, cudaGaugePrecise, cudaCloverInvPrecise, aux, 0);
      }
    }

    retrieveSpinorField(h_x, x, param->cpu_prec, param->dirac_order);

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeSpinorField(b);

  } else {
    retrieveParitySpinor(h_x, out, param->cpu_prec, param->dirac_order);
  }

  freeParitySpinor(tmp);
  freeParitySpinor(in);
  freeParitySpinor(out);

  return;
}
