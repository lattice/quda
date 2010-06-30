#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <dslash_reference.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 0;
// clover-improved? (0 = plain Wilson, 1 = clover)
int clover_yes = 1;

// Pulled this out front so you can set once then forget
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec = QUDA_SINGLE_PRECISION;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

FullGauge gauge;
FullClover clover, cloverInv;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

double kappa = 1.0;
QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
QudaDagType dagger = QUDA_DAG_NO;     // apply Dslash or Dslash dagger?
int transfer = 0; // include transfer time in the benchmark?

const int LOOPS = 1000;

Dirac *dirac;

void init() {

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = 24;
  gauge_param.X[1] = 24;
  gauge_param.X[2] = 24;
  gauge_param.X[3] = 32;
  setDims(gauge_param.X);

  gauge_param.anisotropy = 2.3;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_8;
  gauge_param.reconstruct_sloppy = gauge_param.reconstruct;
  gauge_param.cuda_prec_sloppy = gauge_param.cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  inv_param.kappa = kappa;

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;

  gauge_param.ga_pad = 0;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  //gauge_param.ga_pad = 24*24*24;
  //inv_param.sp_pad = 24*24*24;
  //inv_param.cl_pad = 24*24*24;

  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  if (test_type == 2) {
    inv_param.solver_type = QUDA_MAT_SOLVER;
  } else {
    inv_param.solver_type = QUDA_MATPC_SOLVER;
  }

  if (clover_yes) {
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    if (test_type > 0) {
      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      hostCloverInv = hostClover; // fake it
    } else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
    }
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
  }

  inv_param.verbosity = QUDA_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  if (test_type < 2) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }    
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  
  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields...\n");

  construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  spinor->Source(QUDA_RANDOM_SOURCE);

  if (clover_yes) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    if (test_type == 2) {
      construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
    } else {
      construct_clover_field(hostCloverInv, norm, diag, inv_param.clover_cpu_prec);
    }
  }
  printfQuda("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);

  printfQuda("Sending gauge field to GPU\n");

  loadGaugeQuda(hostGauge, &gauge_param);
  gauge = cudaGaugePrecise;

  if (clover_yes) {
    printfQuda("Sending clover field to GPU\n");
    loadCloverQuda(hostClover, hostCloverInv, &inv_param);
    clover = cudaCloverPrecise;
    cloverInv = cudaCloverInvPrecise;
  }

  if (!transfer) {
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    if (test_type == 2) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);
    if (clover_yes) tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    std::cout << "Source: CPU = " << norm2(*spinor) << ", CUDA = " << 
      norm2(*cudaSpinor) << std::endl;

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param);
    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1 = tmp;
    diracParam.tmp2 = tmp2;
    
    dirac = Dirac::create(diracParam);

  } else {
    std::cout << "Source: CPU = " << norm2(*spinor) << std::endl;
  }
    
}

void end() {
  if (!transfer) {
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
    if (clover_yes) delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (clover_yes) {
    if (test_type == 2) free(hostClover);
    else free(hostCloverInv);
  }
  endQuda();
}

// execute kernel
double dslashCUDA() {

  printfQuda("Executing %d kernel loops...\n", LOOPS);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	dslashQuda(spinorOut->v, spinor->v, &inv_param, parity, dagger);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity, dagger);
      }
      break;
    case 1:
      if (transfer) {
	MatQuda(spinorOut->v, spinor->v, &inv_param, dagger);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor, dagger);
      }
      break;
    case 2:
      if (transfer) {
	MatQuda(spinorOut->v, spinor->v, &inv_param, dagger);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor, dagger);
      }
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds();
  printf("done.\n\n");

  return secs;
}

void dslashRef() {

  // FIXME: remove once reference clover is finished
  if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  } else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  }

  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:
    dslash(spinorRef->v, hostGauge, spinor->v, parity, dagger, 
	   inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  case 1:    
    matpc(spinorRef->v, hostGauge, spinor->v, kappa, inv_param.matpc_type, dagger, 
	  inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  case 2:
    mat(spinorRef->v, hostGauge, spinor->v, kappa, dagger, 
	inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

int main(int argc, char **argv)
{
  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  float sharedKB = 0;//(float)dslashCudaSharedBytes(inv_param.cuda_prec) / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);
  
  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int floats = test_type ? 2*(7*24+8*gauge_param.packed_size+24)+24 : 7*24+8*gauge_param.packed_size+24;
    if (clover_yes) {
      floats += test_type ? 72*2 : 72;
    }
    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/((secs/LOOPS)*(1<<30)));
    
    if (!transfer) {
      std::cout << "Results: CPU = " << norm2(*spinorRef) << ", CUDA = " << norm2(*cudaSpinorOut) << 
	", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    } else {
      std::cout << "Result: CPU = " << norm2(*spinorRef) << ", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    }
    
    cpuColorSpinorField::Compare(*spinorRef, *spinorOut);
  }    
  end();
}
