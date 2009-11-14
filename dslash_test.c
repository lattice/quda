#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <dslash_reference.h>
#include <util_quda.h>
#include <test_util.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 1;
// clover-improved? (0 = plain Wilson, 1 = clover)
int clover_yes = 1;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge gauge;
FullClover clover, cloverInv;
FullSpinor cudaSpinor;
FullSpinor cudaSpinorOut;
ParitySpinor tmp;

void *hostGauge[4], *hostClover, *hostCloverInv;
void *spinor, *spinorEven, *spinorOdd;
void *spinorRef, *spinorRefEven, *spinorRefOdd;
void *spinorGPU, *spinorGPUEven, *spinorGPUOdd;

double kappa = 1.0;
int ODD_BIT = 1;
int DAGGER_BIT = 0;
int TRANSFER = 0; // include transfer time in the benchmark?

void init() {

  gaugeParam.X[0] = 24;
  gaugeParam.X[1] = 24;
  gaugeParam.X[2] = 24;
  gaugeParam.X[3] = 64;
  setDims(gaugeParam.X);

  gaugeParam.anisotropy = 2.3;

  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = QUDA_SINGLE_PRECISION;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;

  if (clover_yes) {
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
  }

  inv_param.kappa = kappa;

  inv_param.matpc_type = QUDA_MATPC_ODD_ODD;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_SINGLE_PRECISION;

  gaugeParam.ga_pad = 0;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  /*gaugeParam.ga_pad = 24*24*12;
  inv_param.sp_pad = 24*24*12;
  inv_param.cl_pad = 24*24*12;*/

  if (test_type == 2) inv_param.dirac_order = QUDA_DIRAC_ORDER;
  else inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (clover_yes) {
    inv_param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec = QUDA_SINGLE_PRECISION;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }
  inv_param.verbosity = QUDA_VERBOSE;

  gauge_param = &gaugeParam;
  invert_param = &inv_param;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gaugeParam.cpu_prec);

  if (clover_yes) {
    if (test_type > 0) {
      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      hostCloverInv = hostClover; // fake it
    } else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
    }
  }

  spinor = malloc(V*spinorSiteSize*inv_param.cpu_prec);
  spinorRef = malloc(V*spinorSiteSize*inv_param.cpu_prec);
  spinorGPU = malloc(V*spinorSiteSize*inv_param.cpu_prec);
  spinorEven = spinor;
  spinorRefEven = spinorRef;
  spinorGPUEven = spinorGPU;
  if (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) {
    spinorOdd = (void*)((double*)spinor + Vh*spinorSiteSize);
    spinorRefOdd = (void*)((double*)spinorRef + Vh*spinorSiteSize);
    spinorGPUOdd = (void*)((double*)spinorGPU + Vh*spinorSiteSize);
  } else {
    spinorOdd = (void*)((float*)spinor + Vh*spinorSiteSize);
    spinorRefOdd = (void*)((float*)spinorRef + Vh*spinorSiteSize);
    spinorGPUOdd = (void*)((float*)spinorGPU + Vh*spinorSiteSize);
  }

  printf("Randomizing fields... ");

  construct_gauge_field(hostGauge, 1, gaugeParam.cpu_prec);
  construct_spinor_field(spinor, 1, 0, 0, 0, inv_param.cpu_prec);

  if (clover_yes) {
    double norm = 0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    if (test_type == 2) {
      construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
    } else {
      construct_clover_field(hostCloverInv, norm, diag, inv_param.clover_cpu_prec);
    }
  }
  printf("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);

  loadGaugeQuda(hostGauge, &gaugeParam);
  gauge = cudaGaugePrecise;

  if (clover_yes) {
    loadCloverQuda(NULL, hostCloverInv, &inv_param);
    clover = cudaCloverPrecise;
    cloverInv = cudaCloverInvPrecise;
  }

  printf("Sending fields to GPU... "); fflush(stdout);

  if (!TRANSFER) {

    gaugeParam.X[0] /= 2;
    tmp = allocateParitySpinor(gaugeParam.X, inv_param.cuda_prec, inv_param.sp_pad);
    cudaSpinor = allocateSpinorField(gaugeParam.X, inv_param.cuda_prec, inv_param.sp_pad);
    cudaSpinorOut = allocateSpinorField(gaugeParam.X, inv_param.cuda_prec, inv_param.sp_pad);
    gaugeParam.X[0] *= 2;

    if (test_type < 2) {
      loadParitySpinor(cudaSpinor.even, spinorEven, inv_param.cpu_prec, 
		       inv_param.dirac_order);
    } else {
      loadSpinorField(cudaSpinor, spinor, inv_param.cpu_prec, 
		      inv_param.dirac_order);
    }

  }

}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (clover_yes) {
    if (test_type == 2) free(hostClover);
    else free(hostCloverInv);
  }
  free(spinorGPU);
  free(spinor);
  free(spinorRef);
  if (!TRANSFER) {
    freeSpinorField(cudaSpinorOut);
    freeSpinorField(cudaSpinor);
    freeParitySpinor(tmp);
  }
  endQuda();
}

double dslashCUDA() {

  // execute kernel
  const int LOOPS = 100;
  printf("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      if (TRANSFER) {
	dslashQuda(spinorOdd, spinorEven, &inv_param, ODD_BIT, DAGGER_BIT);
      } else if (!clover_yes) {
	dslashCuda(cudaSpinor.odd, gauge, cudaSpinor.even, ODD_BIT, DAGGER_BIT);
      } else {
	cloverDslashCuda(cudaSpinor.odd, gauge, cloverInv, cudaSpinor.even, ODD_BIT, DAGGER_BIT);    
      }
      break;
    case 1:
      if (TRANSFER) {
	MatPCQuda(spinorOdd, spinorEven, &inv_param, DAGGER_BIT);
      } else if (!clover_yes) {
	MatPCCuda(cudaSpinor.odd, gauge, cudaSpinor.even, kappa, tmp, inv_param.matpc_type, DAGGER_BIT);
      } else {
	cloverMatPCCuda(cudaSpinor.odd, gauge, clover, cloverInv, cudaSpinor.even, kappa, tmp,
			inv_param.matpc_type, DAGGER_BIT);
      }
      break;
    case 2:
      if (TRANSFER) {
	MatQuda(spinorGPU, spinor, &inv_param, DAGGER_BIT);
      } else if (!clover_yes) {
	MatCuda(cudaSpinorOut, gauge, cudaSpinor, kappa, DAGGER_BIT);
      } else {
	cloverMatCuda(cudaSpinorOut, gauge, clover, cudaSpinor, kappa, tmp, DAGGER_BIT);
      }
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds() / LOOPS;
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
    dslash(spinorRef, hostGauge, spinorEven, ODD_BIT, DAGGER_BIT, 
	   inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 1:    
    matpc(spinorRef, hostGauge, spinorEven, kappa, inv_param.matpc_type, DAGGER_BIT, 
	  inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 2:
    mat(spinorRef, hostGauge, spinor, kappa, DAGGER_BIT, 
	inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

void dslashTest() {

  init();
  
  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  float sharedKB = 0;//(float)dslashCudaSharedBytes(inv_param.cuda_prec) / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gaugeParam.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);
  
  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();
    
    if (!TRANSFER) {
      if (test_type < 2) 
	retrieveParitySpinor(spinorOdd, cudaSpinor.odd, inv_param.cpu_prec, inv_param.dirac_order);
      else 
	retrieveSpinorField(spinorGPU, cudaSpinorOut, inv_param.cpu_prec, inv_param.dirac_order);
    }
    
    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    int flops = test_type ? 1320*2 + 48 : 1320;
    int floats = test_type ? 2*(7*24+8*gaugeParam.packed_size+24)+24 : 7*24+8*gaugeParam.packed_size+24;
    if (clover_yes) {
      flops += test_type ? 504*2 : 504;
      floats += test_type ? 72*2 : 72;
    }
    printf("GFLOPS = %f\n", 1.0e-9*flops*Vh/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/(secs*(1<<30)));
    
    int res;
    if (test_type < 2) res = compare_floats(spinorOdd, spinorRef, Vh*4*3*2, 1e-4, inv_param.cpu_prec);
    else res = compare_floats(spinorGPU, spinorRef, V*4*3*2, 1e-4, inv_param.cpu_prec);
      
      printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");
      
      //if (test_type < 2) strong_check(spinorRef, spinorOdd, Vh, inv_param.cpu_prec);
      //else strong_check(spinorRef, spinorGPU, V, inv_param.cpu_prec);    
  }    
  end();
}

int main(int argc, char **argv) {
  dslashTest();
}
