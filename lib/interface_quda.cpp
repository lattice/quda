#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_quda.h>
#include <spinor_quda.h>
#include <clover_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>

#define spinorSiteSize 24 // real numbers per spinor

FullGauge cudaGaugePrecise; // precise gauge field
FullGauge cudaGaugeSloppy; // sloppy gauge field

FullClover cudaCloverPrecise; // clover term
FullClover cudaCloverSloppy;

FullClover cudaCloverInvPrecise; // inverted clover term
FullClover cudaCloverInvSloppy;

void initBlas(void);

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM

static void checkPrecision(QudaPrecision precision)
{
  if (precision == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }
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

  initBlas();
}

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  checkGaugeParam(param);

  param->packed_size = (param->reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;

  createGaugeField(&cudaGaugePrecise, h_gauge, param->cuda_prec, param->cpu_prec, param->gauge_order, param->reconstruct, param->gauge_fix,
		   param->t_boundary, param->X, param->anisotropy, param->ga_pad);
  param->gaugeGiB = 2.0*cudaGaugePrecise.bytes/ (1 << 30);
  if (param->cuda_prec_sloppy != param->cuda_prec ||
      param->reconstruct_sloppy != param->reconstruct) {
    createGaugeField(&cudaGaugeSloppy, h_gauge, param->cuda_prec_sloppy, param->cpu_prec, param->gauge_order,
		     param->reconstruct_sloppy, param->gauge_fix, param->t_boundary,
		     param->X, param->anisotropy, param->ga_pad);
    param->gaugeGiB += 2.0*cudaGaugeSloppy.bytes/ (1 << 30);
  } else {
    cudaGaugeSloppy = cudaGaugePrecise;
  }
}

/*
  Very limited functionailty here
  - no ability to dump the sloppy gauge field
  - really exposes how crap the current api is
*/
void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  restoreGaugeField(h_gauge, &cudaGaugePrecise, param->cpu_prec, param->gauge_order);
}

void loadCloverQuda(void *h_clover, void *h_clovinv, QudaGaugeParam *gauge_param, QudaInvertParam *inv_param)
{
  if (!h_clover && !h_clovinv) {
    printf("QUDA error: loadCloverQuda() called with neither clover term nor inverse\n");
    exit(-1);
  }
  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on CPU\n");
    exit(-1);
  }
  if (cudaGaugePrecise.even == NULL) {
    printf("QUDA error: gauge field must be loaded before clover\n");
    exit(-1);
  }
  if (inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
    printf("QUDA error: wrong dslash_type in loadCloverQuda()\n");
    exit(-1);
  }

  int X[4];
  for (int i=0; i<4; i++) {
    X[i] = gauge_param->X[i];
  }
  X[0] /= 2; // dimensions of the even-odd sublattice

  inv_param->cloverGiB = 0;

  if (h_clover) {
    allocateCloverField(&cudaCloverPrecise, X, inv_param->cl_pad, inv_param->clover_cuda_prec);
    loadCloverField(cudaCloverPrecise, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverPrecise.even.bytes / (1<<30);

    if (inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
	inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
	allocateCloverField(&cudaCloverSloppy, X, inv_param->cl_pad, inv_param->clover_cuda_prec_sloppy);
	loadCloverField(cudaCloverSloppy, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
	inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
      } else {
	cudaCloverSloppy = cudaCloverPrecise;
      }
    } // sloppy precision clover term not needed otherwise
  }

  allocateCloverField(&cudaCloverInvPrecise, X, inv_param->cl_pad, inv_param->clover_cuda_prec);
  if (!h_clovinv) {
    printf("QUDA error: clover term inverse not implemented yet\n");
    exit(-1);
  } else {
    loadCloverField(cudaCloverInvPrecise, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
  }
  inv_param->cloverGiB += 2.0*cudaCloverInvPrecise.even.bytes / (1<<30);

  if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
    allocateCloverField(&cudaCloverInvSloppy, X, inv_param->cl_pad, inv_param->clover_cuda_prec_sloppy);
    loadCloverField(cudaCloverInvSloppy, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
  } else {
    cudaCloverInvSloppy = cudaCloverInvPrecise;
  }
}

#if 0
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
#endif

void endQuda(void)
{
  freeSpinorBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
  if (cudaCloverPrecise.even.clover) freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) freeCloverField(&cudaCloverSloppy);
  if (cudaCloverInvPrecise.even.clover) freeCloverField(&cudaCloverInvPrecise);
  if (cudaCloverInvSloppy.even.clover) freeCloverField(&cudaCloverInvSloppy);
  endBlas();
}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger)
{
  checkPrecision(inv_param->cpu_prec);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);

  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    parity = (parity+1)%2;
    axCuda(cudaGaugePrecise.anisotropy, in);
  }

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
  checkPrecision(inv_param->cpu_prec);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);

  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;

  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCCuda(out, cudaGaugePrecise, in, kappa, tmp, inv_param->matpc_type, dagger);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    cloverMatPCCuda(out, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, in, kappa,
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
  checkPrecision(inv_param->cpu_prec);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);  

  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;

  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatPCDagMatPCCuda(out, cudaGaugePrecise, in, kappa, tmp, inv_param->matpc_type);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    cloverMatPCDagMatPCCuda(out, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, in, kappa,
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

void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger)
{
  checkPrecision(inv_param->cpu_prec);

  FullSpinor in = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
  FullSpinor out = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);

  loadSpinorField(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;

  if (inv_param->dslash_type == QUDA_WILSON_DSLASH) {
    MatCuda(out, cudaGaugePrecise, in, -kappa, dagger);
  } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec, inv_param->sp_pad);
    cloverMatCuda(out, cudaGaugePrecise, cudaCloverPrecise, in, kappa, tmp, dagger);
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
  checkInvertParam(param);
  checkPrecision(param->cpu_prec);

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
  if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= cudaGaugePrecise.anisotropy;

  FullSpinor b, x;
  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, param->cuda_prec, param->sp_pad); // source
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, param->cuda_prec, param->sp_pad); // solution
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, param->cuda_prec, param->sp_pad); // temporary

  if (param->solution_type == QUDA_MAT_SOLUTION) {
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
      b = allocateSpinorField(cudaGaugePrecise.X, param->cuda_prec, param->sp_pad);
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
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b.even);
      axCuda(2.0*kappa, b.odd);
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
	ParitySpinor aux = out; // aliases b.even when PRESERVE_SOURCE_NO is set
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, b.even, 0);
	dslashXpayCuda(aux, cudaGaugePrecise, in, 1, 0, b.odd, kappa);
	cloverCuda(in, cudaGaugePrecise, cudaCloverInvPrecise, aux, 1);
      } else if (param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	// in = b_e + k D_eo A_oo^-1 b_o
	ParitySpinor aux = tmp; // aliases b.odd when PRESERVE_SOURCE_NO is set
	cloverCuda(aux, cudaGaugePrecise, cudaCloverInvPrecise, b.odd, 1); // safe even when aux = b.odd
	dslashXpayCuda(in, cudaGaugePrecise, aux, 0, 0, b.even, kappa);
      } else if (param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	// in = b_o + k D_oe A_ee^-1 b_e
	ParitySpinor aux = out; // aliases b.even when PRESERVE_SOURCE_NO is set
	cloverCuda(aux, cudaGaugePrecise, cudaCloverInvPrecise, b.even, 0); // safe even when aux = b.even
	dslashXpayCuda(in, cudaGaugePrecise, aux, 1, 0, b.odd, kappa);
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
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      if (param->solution_type == QUDA_MATPC_SOLUTION)  {
	axCuda(4.0*kappa*kappa, in);
      } else {
	axCuda(16.0*pow(kappa,4), in);
      }
    } else if (param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      if (param->solution_type == QUDA_MATPC_SOLUTION)  {
	axCuda(2.0*kappa, in);
      } else {
	axCuda(4.0*kappa*kappa, in);
      }
    }
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
