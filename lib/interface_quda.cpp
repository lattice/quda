#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_quda.h>
#include <clover_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>

#include <color_spinor_field.h>
#include <dirac.h>

#include <iostream>

#define spinorSiteSize 24 // real numbers per spinor

FullGauge cudaGaugePrecise; // precise gauge field
FullGauge cudaGaugeSloppy; // sloppy gauge field

FullClover cudaCloverPrecise; // clover term
FullClover cudaCloverSloppy;

FullClover cudaCloverInvPrecise; // inverted clover term
FullClover cudaCloverInvSloppy;

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

void initQuda(int dev)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    errorQuda("No devices supporting CUDA");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(stderr, "QUDA: Found device %d: %s\n", i, deviceProp.name);
  }

  if (dev < 0) {
    dev = deviceCount - 1;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }

  fprintf(stderr, "QUDA: Using device %d: %s\n", dev, deviceProp.name);
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

void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  if (!h_clover && !h_clovinv) {
    errorQuda("loadCloverQuda() called with neither clover term nor inverse");
  }
  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }
  if (cudaGaugePrecise.even == NULL) {
    errorQuda("Gauge field must be loaded before clover");
  }
  if (inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
    errorQuda("Wrong dslash_type in loadCloverQuda()");
  }

  int X[4];
  for (int i=0; i<4; i++) {
    X[i] = cudaGaugePrecise.X[i];
  }

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
    errorQuda("Clover term inverse not implemented yet");
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
  cudaColorSpinorField::freeBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
  if (cudaCloverPrecise.even.clover) freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) freeCloverField(&cudaCloverSloppy);
  if (cudaCloverInvPrecise.even.clover) freeCloverField(&cudaCloverInvPrecise);
  if (cudaCloverInvSloppy.even.clover) freeCloverField(&cudaCloverInvSloppy);
  endBlas();
}

void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param) {
  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;

  if (inv_param->solution_type == QUDA_MAT_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSON_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVER_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else if (inv_param->solution_type == QUDA_MATPC_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSONPC_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVERPC_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else {
    errorQuda("Unsupported solution type %d", inv_param->solution_type);
  }

  diracParam.matpcType = inv_param->matpc_type;
  diracParam.gauge = &cudaGaugePrecise;
  diracParam.clover = &cudaCloverPrecise;
  diracParam.cloverInv = &cudaCloverInvPrecise;
  diracParam.kappa = kappa;
}

void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param) {
  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;

  if (inv_param->solution_type == QUDA_MAT_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSON_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVER_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else if (inv_param->solution_type == QUDA_MATPC_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSONPC_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVERPC_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else {
    errorQuda("Unsupported solution type %d", inv_param->solution_type);
  }

  diracParam.matpcType = inv_param->matpc_type;
  diracParam.gauge = &cudaGaugeSloppy;
  diracParam.clover = &cudaCloverSloppy;
  diracParam.cloverInv = &cudaCloverInvSloppy;
  diracParam.kappa = kappa;
}

void massRescale(double &kappa, QudaSolutionType solution_type, 
		 QudaMassNormalization mass_normalization, 
		 cudaColorSpinorField &b) {

  // multiply the source to get the mass normalization
  if (solution_type == QUDA_MAT_SOLUTION) {
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b);
    }
  } else if (solution_type == QUDA_MATPC_SOLUTION || 
	  solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
      if (solution_type == QUDA_MATPC_SOLUTION)  {
	axCuda(4.0*kappa*kappa, b);
      } else {
	axCuda(16.0*pow(kappa,4), b);
      }
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      if (solution_type == QUDA_MATPC_SOLUTION)  {
	axCuda(2.0*kappa, b);
      } else {
	axCuda(4.0*kappa*kappa, b);
      }
    }

  } else {
    errorQuda("Solution %d type not supported", solution_type);
  }

}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, QudaDagType dagger)
{
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  cpuParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);
 
  cudaColorSpinorField in(hIn, cudaParam);

  cudaParam.create = QUDA_NULL_CREATE;
  cudaColorSpinorField out(in, cudaParam);
  cudaColorSpinorField tmp;

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    parity = (parity+1)%2;
    axCuda(cudaGaugePrecise.anisotropy, in);
  }

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);
  if (diracParam.type == QUDA_WILSONPC_DIRAC || diracParam.type == QUDA_CLOVERPC_DIRAC) {
    tmp = cudaColorSpinorField(in, cudaParam);
    diracParam.tmp = &tmp;
  }

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->Dslash(out, in, parity, dagger); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut);// since this is a reference this won't work: hOut = out;
}

void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaDagType dagger)
{
  if (inv_param->solution_type == QUDA_MAT_SOLUTION) cudaGaugePrecise.X[0] *= 2;
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  if (inv_param->solution_type == QUDA_MAT_SOLUTION) {
    cudaGaugePrecise.X[0] /= 2;
    cpuParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;;
  } else {
    cpuParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  }
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);
  cudaColorSpinorField in(hIn, cudaParam);
  cudaParam.create = QUDA_NULL_CREATE;
  cudaColorSpinorField out(in, cudaParam);
  cudaColorSpinorField tmp;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);
  if (diracParam.type == QUDA_WILSONPC_DIRAC || diracParam.type == QUDA_CLOVERPC_DIRAC) {
    tmp = cudaColorSpinorField(in, cudaParam);
    diracParam.tmp = &tmp;
  }

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in, dagger); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut);// since this is a reference this won't work: hOut = out;
}

void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  if (inv_param->solution_type == QUDA_MAT_SOLUTION) cudaGaugePrecise.X[0] *= 2;
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  if (inv_param->solution_type == QUDA_MAT_SOLUTION) {
    cudaGaugePrecise.X[0] /= 2;
    cpuParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;;
  } else {
    cpuParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  }
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);
  cudaColorSpinorField in(hIn, cudaParam);
  cudaParam.create = QUDA_NULL_CREATE;
  cudaColorSpinorField out(in, cudaParam);
  cudaColorSpinorField tmp(in, cudaParam);

  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= cudaGaugePrecise.anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);
  if (diracParam.type == QUDA_WILSONPC_DIRAC || diracParam.type == QUDA_CLOVERPC_DIRAC) {
    tmp = cudaColorSpinorField(in, cudaParam);
    diracParam.tmp = &tmp;
  }

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut);// since this is a reference this won't work: hOut = out;
}

void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  checkInvertParam(param);
  int slenh = cudaGaugePrecise.volume*spinorSiteSize;
  param->spinorGiB = (double)slenh * (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  else
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  //FullSpinor b, x;
  ColorSpinorParam cpuParam(hp_b, *param, cudaGaugePrecise.X); // wrong dimensions
  cpuParam.fieldSubset = param->solution_type == QUDA_MATPC_SOLUTION ? 
    QUDA_PARITY_FIELD_SUBSET : QUDA_FULL_FIELD_SUBSET;
  ColorSpinorParam cudaParam(cpuParam, *param);

  cpuColorSpinorField h_b(cpuParam);
  cudaColorSpinorField b(h_b, cudaParam); // download source
  cudaParam.create = QUDA_NULL_CREATE;
  cudaColorSpinorField x(b, cudaParam); // solution
  cudaColorSpinorField tmp(b, cudaParam); // temporary
  
  cudaColorSpinorField in, out;

  // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param);
  if (diracParam.type == QUDA_WILSONPC_DIRAC || diracParam.type == QUDA_CLOVERPC_DIRAC) {
    tmp = cudaColorSpinorField(in, cudaParam);
    diracParam.tmp = &tmp;
  }

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator

  setDiracSloppyParam(diracParam, param);
  Dirac *diracSloppy = Dirac::create(diracParam);

  massRescale(diracParam.kappa, param->solution_type, param->mass_normalization, out);

  dirac->Prepare(in, out, x, b, param->solution_type);

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(out, in);
      dirac->M(in, out); // tmp?
    }
    invertCgCuda(*dirac, *diracSloppy, out, in, tmp, param);
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(*dirac, *diracSloppy, out, in, tmp, param, QUDA_DAG_YES);
      copyCuda(in, out);
    }
    invertBiCGstabCuda(*dirac, *diracSloppy, out, in, tmp, param, QUDA_DAG_NO);
    break;
  default:
    errorQuda("Inverter type %d not implemented", param->inv_type);
  }
  
  dirac->Reconstruct(x, b, param->solution_type);

  cpuParam.v = hp_x;
  cpuColorSpinorField h_x(cpuParam);
  out.saveCPUSpinorField(h_x);// since this is a reference this won't work: hOut = h_x;

  delete diracSloppy;
  delete dirac;

  return;
}
