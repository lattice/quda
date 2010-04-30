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

FullGauge cudaFatLinkPrecise;
FullGauge cudaFatLinkSloppy;

FullGauge cudaLongLinkPrecise;
FullGauge cudaLongLinkSloppy;

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
  static int initialized = 0;
  if (initialized){
    return ;
  }
  initialized = 1;

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


void loadGaugeQuda_general(void *h_gauge, QudaGaugeParam *param, void* _cudaLinkPrecise, void* _cudaLinkSloppy)
{
    checkGaugeParam(param);
    FullGauge* cudaLinkPrecise = (FullGauge*)_cudaLinkPrecise;
    FullGauge* cudaLinkSloppy = (FullGauge*)_cudaLinkSloppy;
    
    int packed_size;
    switch(param->reconstruct){
    case QUDA_RECONSTRUCT_8:
        packed_size = 8;
        break;
    case QUDA_RECONSTRUCT_12:
        packed_size = 12;
        break;
    case QUDA_RECONSTRUCT_NO:
        packed_size = 18;
        break;
    default:
        printf("ERROR: %s: reconstruct type not set, exitting\n", __FUNCTION__);
        exit(1);
    }

    param->packed_size = packed_size;

    createGaugeField(cudaLinkPrecise, h_gauge, param->cuda_prec, param->cpu_prec, param->gauge_order, param->reconstruct, param->gauge_fix,
		     param->t_boundary, param->X, param->anisotropy, param->ga_pad);

    param->gaugeGiB += 2.0*cudaLinkPrecise->bytes/ (1 << 30);
    if (param->cuda_prec_sloppy != param->cuda_prec ||
        param->reconstruct_sloppy != param->reconstruct) {
      createGaugeField(cudaLinkSloppy, h_gauge, param->cuda_prec_sloppy, param->cpu_prec, param->gauge_order,
		       param->reconstruct_sloppy, param->gauge_fix, param->t_boundary,
		       param->X, param->anisotropy, param->ga_pad);
      
      param->gaugeGiB += 2.0*cudaLinkSloppy->bytes/ (1 << 30);
    } else {
      *cudaLinkSloppy = *cudaLinkPrecise;
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

  if (inv_param->solver_type == QUDA_MAT_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSON_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVER_DIRAC;
    else if (inv_param->dslash_type == QUDA_STAGGERED_DSLASH)
	diracParam.type = QUDA_STAGGERED_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else if (inv_param->solver_type == QUDA_MATPC_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSONPC_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVERPC_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else {
    errorQuda("Unsupported solution type %d", inv_param->solver_type);
  }

  diracParam.matpcType = inv_param->matpc_type;
  diracParam.gauge = &cudaGaugePrecise;
  diracParam.clover = &cudaCloverPrecise;
  diracParam.cloverInv = &cudaCloverInvPrecise;
  diracParam.kappa = kappa;
  diracParam.mass = inv_param->mass;
  diracParam.verbose = inv_param->verbosity;

}

void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param) {
  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER)
    kappa *= cudaGaugePrecise.anisotropy;


  if (inv_param->solver_type == QUDA_MAT_SOLUTION) {
    if (inv_param->dslash_type == QUDA_WILSON_DSLASH) 
      diracParam.type = QUDA_WILSON_DIRAC;
    else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
      diracParam.type = QUDA_CLOVER_DIRAC;
    else if (inv_param->dslash_type == QUDA_STAGGERED_DSLASH)
	diracParam.type = QUDA_STAGGERED_DIRAC;
    else errorQuda("Unsupported dslash_type");
  } else if (inv_param->solver_type == QUDA_MATPC_SOLUTION) {
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
  diracParam.mass = inv_param->mass;
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

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    parity = (parity+1)%2;
    axCuda(cudaGaugePrecise.anisotropy, in);
  }

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);

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

  cpuParam.print(); std::cout << std::endl;
  cudaParam.print(); std::cout << std::endl;

  cpuColorSpinorField hIn(cpuParam);
  cudaColorSpinorField in(hIn, cudaParam);
  cudaParam.create = QUDA_NULL_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);

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

  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= cudaGaugePrecise.anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param);

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

  // temporary hack
  if (param->solution_type == QUDA_MAT_SOLUTION) cudaGaugePrecise.X[0] *= 2;
  ColorSpinorParam cpuParam(hp_b, *param, cudaGaugePrecise.X);
  if (param->solution_type == QUDA_MAT_SOLUTION) {
    cudaGaugePrecise.X[0] /= 2;
    cpuParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;;
  } else {
    cpuParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  }

  ColorSpinorParam cudaParam(cpuParam, *param);

  cpuColorSpinorField h_b(cpuParam);
  cudaColorSpinorField b(h_b, cudaParam); // download source

  std::cout << h_b.Volume() << " " << b.Volume() << std::endl;

  std::cout << "CPU source = " << norm2(h_b) << ", cuda copy = " << norm2(b) << std::endl;

  cudaParam.create = QUDA_ZERO_CREATE;
  cudaColorSpinorField x(cudaParam); // solution

  // if using preconditioning but solving the full system
  if (param->solver_type == QUDA_MATPC_SOLUTION && 
      param->solution_type == QUDA_MAT_SOLUTION) {
    cudaParam.x[0] /= 2;
    cudaParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
  }
  cudaColorSpinorField tmp(cudaParam); // temporary

  cudaColorSpinorField *in, *out;

  // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param);
  diracParam.verbose = QUDA_VERBOSE;
  //if (diracParam.type == QUDA_WILSONPC_DIRAC || diracParam.type == QUDA_CLOVERPC_DIRAC) {
  //tmp = cudaColorSpinorField(in, cudaParam);
  //diracParam.tmp = &tmp;
  //}

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator

  setDiracSloppyParam(diracParam, param);
  Dirac *diracSloppy = Dirac::create(diracParam);

  massRescale(diracParam.kappa, param->solution_type, param->mass_normalization, *out);

  std::cout << "Mass rescale done" << std::endl;

  dirac->Prepare(in, out, x, b, param->solution_type);

  std::cout << "Source preparation complete " << norm2(*in) << " " << norm2(b) << std::endl;
  std::cout << out->Volume() << " " << tmp.Volume() << " " << in->Volume() << " " << b.Volume() << std::endl;

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(*out, *in);
      dirac->M(*in, *out, QUDA_DAG_YES); // tmp?
    }
    invertCgCuda(*dirac, *diracSloppy, *out, *in, tmp, param);
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(*dirac, *diracSloppy, *out, *in, tmp, param, QUDA_DAG_YES);
      copyCuda(*in, *out);
    }
    invertBiCGstabCuda(*dirac, *diracSloppy, *out, *in, tmp, param, QUDA_DAG_NO);
    break;
  default:
    errorQuda("Inverter type %d not implemented", param->inv_type);
  }

  std::cout << "Solution = " << norm2(x) << std::endl;
  dirac->Reconstruct(x, b, param->solution_type);
  std::cout << "Solution = " << norm2(x) << std::endl;

  cpuParam.v = hp_x;
  cpuColorSpinorField h_x(cpuParam);
  x.saveCPUSpinorField(h_x);// since this is a reference this won't work: hOut = h_x;

  std::cout << "Solution = " << norm2(h_x) << std::endl;

  delete diracSloppy;
  delete dirac;
  
  return;
}



void invertQudaSt(void *hp_x, void *hp_b, QudaInvertParam *param)
{

  checkInvertParam(param);
  /*
  int slenh = cudaGaugePrecise.volume*spinorSiteSize;
  param->spinorGiB = (double)slenh * (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  else
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  */
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  
  ColorSpinorParam csParam;
  csParam.precision = param->cpu_prec;
  csParam.fieldType = QUDA_CPU_FIELD;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;  
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_ORDER;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  csParam.parity = param->in_parity;
  csParam.subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;

  if (param->in_parity == QUDA_FULL_PARITY){
    csParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;
    csParam.x[0] = param->gaugeParam->X[0];
  }else{
    csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
    csParam.x[0] = param->gaugeParam->X[0]/2;    
  }
  csParam.x[1] = param->gaugeParam->X[1];
  csParam.x[2] = param->gaugeParam->X[2];
  csParam.x[3] = param->gaugeParam->X[3];
  csParam.create = QUDA_REFERENCE_CREATE;
  csParam.v = hp_b;  
  cpuColorSpinorField h_b(csParam);
  
  csParam.v = hp_x;
  cpuColorSpinorField h_x(csParam);
  
  csParam.fieldType = QUDA_CUDA_FIELD;
  csParam.fieldOrder = QUDA_FLOAT2_ORDER;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;

  csParam.pad = param->sp_pad;
  csParam.precision = param->cuda_prec;
  csParam.create = QUDA_ZERO_CREATE;
  
  cudaColorSpinorField b(csParam);

 // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param);
  diracParam.fatGauge = &cudaFatLinkPrecise;
  diracParam.longGauge = &cudaLongLinkPrecise;
  
  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator

  diracParam.fatGauge = &cudaFatLinkSloppy;
  diracParam.longGauge = &cudaLongLinkSloppy;
  
  setDiracSloppyParam(diracParam, param);
  Dirac *diracSloppy = Dirac::create(diracParam);
  
  b= h_b; //send data from cpu to GPU

  csParam.create = QUDA_COPY_CREATE;  
  cudaColorSpinorField x(h_x, csParam); // solution  
  csParam.create = QUDA_ZERO_CREATE;
  cudaColorSpinorField tmp(csParam); // temporary
  invertCgCuda(*dirac, *diracSloppy, x, b, tmp, param);    
  
  x.saveCPUSpinorField(h_x);// since this is a reference this won't work: hOut = h_x;    
  
  delete diracSloppy;
  delete dirac;
  
  return;
}


void invertQudaStMultiMass(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			   double* offsets, int num_offsets, double* residue_sq)
{
  int i;
  
  if (num_offsets <= 0){
    return;
  }

  checkInvertParam(param);
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  double low_offset = offsets[0];
  int low_index = 0;
  for (int i=1;i < num_offsets;i++){
    if (offsets[i] < low_offset){
      low_offset = offsets[i];
      low_index = i;
    }
  }
  
  void* hp_x[num_offsets];
  void* hp_b = _hp_b;
  for(int i=0;i < num_offsets;i++){
    hp_x[i] = _hp_x[i];
  }
  
  if (low_index != 0){
    void* tmp = hp_x[0];
    hp_x[0] = hp_x[low_index] ;
    hp_x[low_index] = tmp;
    
    double tmp1 = offsets[0];
    offsets[0]= offsets[low_index];
    offsets[low_index] =tmp1;
  }
  
  
  ColorSpinorParam csParam;
  csParam.precision = param->cpu_prec;
  csParam.fieldType = QUDA_CPU_FIELD;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;  
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_ORDER;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  csParam.parity = param->in_parity;
  csParam.subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;

  if (param->in_parity == QUDA_FULL_PARITY){
    csParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;
    csParam.x[0] = param->gaugeParam->X[0];
  }else{
    csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
    csParam.x[0] = param->gaugeParam->X[0]/2;    
  }
  csParam.x[1] = param->gaugeParam->X[1];
  csParam.x[2] = param->gaugeParam->X[2];
  csParam.x[3] = param->gaugeParam->X[3];
  csParam.create = QUDA_REFERENCE_CREATE;
  csParam.v = hp_b;  
  cpuColorSpinorField h_b(csParam);
  
  cpuColorSpinorField* h_x[num_offsets];
  
  for(i=0;i < num_offsets; i++){
    csParam.v = hp_x[i];
    h_x[i] = new cpuColorSpinorField(csParam);
  }
  
  csParam.fieldType = QUDA_CUDA_FIELD;
  csParam.fieldOrder = QUDA_FLOAT2_ORDER;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;

  csParam.pad = param->sp_pad;
  csParam.precision = param->cuda_prec;
  csParam.create = QUDA_ZERO_CREATE;
  
  cudaColorSpinorField b(csParam);
  
  //set the mass in the invert_param
  param->mass = sqrt(offsets[0]/4);
  
 // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param);
  diracParam.verbose = QUDA_VERBOSE;
  diracParam.fatGauge = &cudaFatLinkPrecise;
  diracParam.longGauge = &cudaLongLinkPrecise;
  
  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator

  diracParam.fatGauge = &cudaFatLinkSloppy;
  diracParam.longGauge = &cudaLongLinkSloppy;
  
  setDiracSloppyParam(diracParam, param);
  Dirac *diracSloppy = Dirac::create(diracParam);
  
  b= h_b; //send data from cpu to GPU
  
  csParam.create = QUDA_ZERO_CREATE;
  cudaColorSpinorField* x[num_offsets]; // solution  
  for(i=0;i < num_offsets;i++){
    x[i] = new cudaColorSpinorField(csParam);
  }
  cudaColorSpinorField tmp(csParam); // temporary
  invertCgCudaMultiMass(*dirac, *diracSloppy, x, b, param, offsets, num_offsets, residue_sq);    
  
  for(i=0; i < num_offsets; i++){
    x[i]->saveCPUSpinorField(*h_x[i]);
  }
  
  for(i=0;i < num_offsets; i++){
    delete h_x[i];
    delete x[i];
  }
  delete diracSloppy;
  delete dirac;
  
  return;
}


