#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <dslash_reference.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 0;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge gauge;
FullSpinor cudaSpinor;
FullSpinor cudaSpinorOut;
ParitySpinor tmp;

void *hostGauge[4];
void *spinor, *spinorRef, *spinorGPU;
void *spinorEven, *spinorOdd;
    
double kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
int TRANSFER = 0; // include transfer time in the benchmark?

void init() {

  gaugeParam.cpu_prec = QUDA_SINGLE_PRECISION;
  gaugeParam.reconstruct_precise = QUDA_RECONSTRUCT_8;
  gaugeParam.cuda_prec_precise = QUDA_SINGLE_PRECISION;
  gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_8;
  gaugeParam.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  gaugeParam.X = L1;
  gaugeParam.Y = L2;
  gaugeParam.Z = L3;
  gaugeParam.T = L4;
  gaugeParam.anisotropy = 2.3;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param = &gaugeParam;

  inv_param.cpu_prec = QUDA_SINGLE_PRECISION;
  inv_param.cuda_prec = QUDA_SINGLE_PRECISION;
  if (test_type == 2) inv_param.dirac_order = QUDA_DIRAC_ORDER;
  else inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.kappa = kappa;
  invert_param = &inv_param;

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // construct input fields
  for (int dir = 0; dir < 4; dir++)
    hostGauge[dir] = malloc(N*gaugeSiteSize*gSize);

  spinor = malloc(N*spinorSiteSize*sSize);
  spinorRef = malloc(N*spinorSiteSize*sSize);
  spinorGPU = malloc(N*spinorSiteSize*sSize);
  spinorEven = spinor;
  spinorOdd = spinor + Nh*spinorSiteSize;
    
  printf("Randomizing fields...");
  construct_gauge_field(hostGauge, 1, gaugeParam.cpu_prec);
  construct_spinor_field(spinor, 1, 0, 0, 0, inv_param.cpu_prec);

  printf("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);
  loadGaugeQuda(hostGauge, &gaugeParam);

  gauge = cudaGaugePrecise;

  printf("Sending fields to GPU..."); fflush(stdout);

  if (!TRANSFER) {
    cudaSpinor = allocateSpinorField(inv_param.cuda_prec);
    cudaSpinorOut = allocateSpinorField(inv_param.cuda_prec);
    tmp = allocateParitySpinor(inv_param.cuda_prec);
    
    loadSpinorField(cudaSpinor, spinor, inv_param.cpu_prec, 
		    inv_param.cuda_prec, inv_param.dirac_order);
    
    printf("\nnorm = %e\n", normCuda(cudaSpinor.even.spinor, Nh*24));
  }
}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
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

void dslashRef() {
  
  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:
    dslash_reference(spinorRef, hostGauge, spinorEven, ODD_BIT, DAGGER_BIT, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 1:    
    matpc(spinorRef, hostGauge, spinorEven, kappa, QUDA_MATPC_EVEN_EVEN, DAGGER_BIT, 
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

double dslashCUDA() {

  // execute kernel
  const int LOOPS = 20;
  printf("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      if (TRANSFER) dslashQuda(spinorOdd, spinorEven, &inv_param, ODD_BIT, DAGGER_BIT);
      else dslashCuda(cudaSpinor.odd, gauge, cudaSpinor.even, ODD_BIT, DAGGER_BIT);
      break;
    case 1:
      if (TRANSFER) MatPCQuda(spinorOdd, spinorEven, &inv_param);
      else MatPCCuda(cudaSpinor.odd, gauge, cudaSpinor.even, kappa, tmp, QUDA_MATPC_EVEN_EVEN);
      break;
    case 2:
      if (TRANSFER) MatQuda(spinorGPU, spinor, &inv_param);
      else MatCuda(cudaSpinorOut, gauge, cudaSpinor, kappa);
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

void strongCheck() {
  int len;
  void *spinorRes;
  if (test_type < 2) {
    len = Nh;
    spinorRes = spinorOdd;
  } else {
    len = N;
    spinorRes = spinorGPU;
  }

  printf("Reference:\n");
  printSpinorElement(spinorRef, 0, inv_param.cpu_prec); printf("...\n");
  printSpinorElement(spinorRef, len-1, inv_param.cpu_prec); printf("\n");    
    
  printf("\nCUDA:\n");
  printSpinorElement(spinorRes, 0, inv_param.cpu_prec); printf("...\n");
  printSpinorElement(spinorRes, len-1, inv_param.cpu_prec); printf("\n");

  //compare_spinor(spinorRef, spinorRes, len, inv_param.cpu_prec);
}


void dslashTest() {

  init();

  float spinorGiB = (float)Nh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  float sharedKB = (float)dslashCudaSharedBytes() / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gaugeParam.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);

  int attempts = 10000;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();
  
    if (!TRANSFER) {
      if (test_type < 2) retrieveParitySpinor(spinorOdd, cudaSpinor.odd, inv_param.cpu_prec, 
					      inv_param.cuda_prec, QUDA_DIRAC_ORDER);
      else retrieveSpinorField(spinorGPU, cudaSpinorOut, inv_param.cpu_prec, 
			       inv_param.cuda_prec, QUDA_DIRAC_ORDER);
    }
    // print timing information
    printf("%fms per loop\n", 1000*secs);
    int flops = test_type ? 1320*2 + 48 : 1320;
    int floats = test_type ? 2*(7*24+8*gaugeParam.packed_size+24)+24 : 7*24+8*gaugeParam.packed_size+24;
    printf("GFLOPS = %f\n", 1.0e-9*flops*Nh/secs);
    printf("GiB/s = %f\n\n", Nh*floats*sizeof(float)/(secs*(1<<30)));
    
    int res;
    if (test_type < 2) res = compare_floats(spinorOdd, spinorRef, Nh*4*3*2, 1e-4, inv_param.cpu_prec);
    else res = compare_floats(spinorGPU, spinorRef, N*4*3*2, 1e-4, inv_param.cpu_prec);
    printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");

    strongCheck();

    exit(0);
  }  

  end();

}

int main(int argc, char **argv) {
  dslashTest();
}
