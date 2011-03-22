#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <wilson_dslash_reference.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
const int test_type = 1;

// Dirac operator type
const QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
//const QudaDslashType dslash_type = QUDA_CLOVER_WILSON_DSLASH;
//const QudaDslashType dslash_type = QUDA_TWISTED_MASS_DSLASH;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const QudaDagType dagger = QUDA_DAG_NO;     // apply Dslash or Dslash dagger?
const int transfer = 0; // include transfer time in the benchmark?

const int loops = 100;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec = QUDA_SINGLE_PRECISION;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

FullGauge gauge;
FullClover clover, cloverInv;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp1=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac;

void init() {

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = 24;
  gauge_param.X[1] = 24;
  gauge_param.X[2] = 24;
  gauge_param.X[3] = 24;
  setDims(gauge_param.X);

  gauge_param.anisotropy = 2.3;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_sloppy = gauge_param.reconstruct;
  gauge_param.cuda_prec_sloppy = gauge_param.cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.kappa = 0.1;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = 0.01;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
  }

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;

#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  gauge_param.ga_pad = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
#endif
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  //inv_param.sp_pad = 24*24*24;
  //inv_param.cl_pad = 24*24*24;

  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (test_type == 2) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  }

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    //if (test_type > 0) {
      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      hostCloverInv = hostClover; // fake it
      /*} else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      }*/
  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {

  }

  //inv_param.verbosity = QUDA_DEBUG_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
  }
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
  
  for (int d=0; d<3; d++) csParam.ghostDim[d] = false;
  csParam.ghostDim[3] = true;
  //csParam.verbose = QUDA_DEBUG_VERBOSE;

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  spinor->Source(QUDA_RANDOM_SOURCE);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
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

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
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
    tmp1 = new cudaColorSpinorField(csParam);
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
	dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      tmp2 = new cudaColorSpinorField(csParam);
    }

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    std::cout << "Source: CPU = " << norm2(*spinor) << ", CUDA = " << 
      norm2(*cudaSpinor) << std::endl;

    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.verbose = QUDA_DEBUG_VERBOSE;
    diracParam.tmp1 = tmp1;
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
    delete tmp1;
    delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (test_type == 2) free(hostClover);
    else free(hostCloverInv);
  }
  endQuda();

}

// execute kernel
double dslashCUDA() {

  /*
  if (!transfer) {
    if (test_type < 2) {
      dirac->Tune(*cudaSpinorOut, *cudaSpinor, *tmp1);
    } else {
      dirac->Tune(cudaSpinorOut->Even(), cudaSpinor->Even(), *tmp1);
    }
    }*/

  printfQuda("Executing %d kernel loops...\n", loops);
  fflush(stdout);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < loops; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	dslashQuda(spinorOut->v, spinor->v, &inv_param, parity);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 1:
    case 2:
      if (transfer) {
	MatQuda(spinorOut->v, spinor->v, &inv_param);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
      break;
    }
  }
    
  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000; //stopwatchReadSeconds();

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  printf("done.\n\n");

  return secs;
}

void dslashRef() {

  // FIXME: remove once reference clover is finished
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    } else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
    }
  }

  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_WILSON_DSLASH) {
    switch (test_type) {
    case 0:
      wil_dslash(spinorRef->v, hostGauge, spinor->v, parity, dagger, 
		 inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    case 1:    
      wil_matpc(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    case 2:
      wil_mat(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, dagger, 
	      inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    default:
      printf("Test type not defined\n");
      exit(-1);
    }
  } else { // twisted mass
    switch (test_type) {
    case 0:
      tm_dslash(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		parity, dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    case 1:    
      tm_matpc(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    case 2:
      tm_mat(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
      break;
    default:
      printf("Test type not defined\n");
      exit(-1);
    }
  }

  printf("done.\n");
}

int main(int argc, char **argv)
{

  int i;
  int tsize = 1; // defaults to 1
  for (i =1;i < argc; i++){
    if( strcmp(argv[i], "--tgridsize") == 0){
      if (i+1 >= argc){
	printf("Usage: %s <args>\n", argv[0]);
	printf("--tgridsize \t Set T comms grid size (default = 1)\n"); 
	exit(1);
      }     
      tsize =  atoi(argv[i+1]);
      if (tsize <= 0 ){
	errorQuda("Error: invalid T grid size");
      }
      i++;
      continue;
    }
  }

  int ndim=4, dims[] = {1, 1, 1, tsize};
  initCommsQuda(argc, argv, dims, ndim);

  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  
  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int floats = test_type ? 2*(7*24+8*gauge_param.reconstruct+24)+24 : 7*24+8*gauge_param.reconstruct+24;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      floats += test_type ? 72*2 : 72;
    }
    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/((secs/loops)*(1<<30)));
    
    if (!transfer) {
      std::cout << "Results: CPU = " << norm2(*spinorRef) << ", CUDA = " << norm2(*cudaSpinorOut) << 
	", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    } else {
      std::cout << "Result: CPU = " << norm2(*spinorRef) << ", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    }
    
    cpuColorSpinorField::Compare(*spinorRef, *spinorOut);
  }    
  end();

  endCommsQuda();
}
