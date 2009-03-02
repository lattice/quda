#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>
#include <field_quda.h>


// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 2;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge gauge;
FullSpinor cudaSpinor;
FullSpinor cudaSpinorOut;
ParitySpinor tmp;

float *hostGauge[4];
float *spinor, *spinorRef, *spinorGPU;
float *spinorEven, *spinorOdd;
    
float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
int TRANSFER = 0; // include transfer time in the benchmark?
    
void printSpinorHalfField(float *spinor) {
  printSpinor(&spinor[0*spinorSiteSize]);
  printf("...\n");
  printSpinor(&spinor[(Nh-1)*spinorSiteSize]);
  printf("\n");    
}

void printSpinorFullField(float *spinor) {
  printSpinor(&spinor[0*spinorSiteSize]);
  printf("...\n");
  printSpinor(&spinor[(N-1)*spinorSiteSize]);
  printf("\n");    
}

void init() {

  gaugeParam.cpu_prec = QUDA_SINGLE_PRECISION;
  gaugeParam.cuda_prec = QUDA_SINGLE_PRECISION;
  gaugeParam.X = L1;
  gaugeParam.Y = L2;
  gaugeParam.Z = L3;
  gaugeParam.T = L4;
  gaugeParam.anisotropy = 2.3;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
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

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    hostGauge[dir] = (float*)malloc(N*gaugeSiteSize*sizeof(float));
  }

  spinor = (float*)malloc(N*spinorSiteSize*sizeof(float));
  spinorRef = (float*)malloc(N*spinorSiteSize*sizeof(float));
  spinorGPU = (float*)malloc(N*spinorSiteSize*sizeof(float));
  spinorEven = spinor;
  spinorOdd = spinor + Nh*spinorSiteSize;
    
  printf("Randomizing fields...");
  constructGaugeField(hostGauge);
  constructSpinorField(spinor);

  printf("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);
  loadGaugeQuda((void*)hostGauge, &gaugeParam);

  if (gaugeParam.cuda_prec == QUDA_SINGLE_PRECISION) gauge = cudaGauge;
  else gauge = cudaHGauge;

  printf("Sending fields to GPU..."); fflush(stdout);

  if (!TRANSFER) {
    cudaSpinor = allocateSpinorField();
    cudaSpinorOut = allocateSpinorField();
    tmp = allocateParitySpinor();
    
    loadSpinorField(cudaSpinor, (void*)spinor, inv_param.cpu_prec, 
		    inv_param.cuda_prec, inv_param.dirac_order);
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
    dslashReference(spinorRef, hostGauge, spinorEven, ODD_BIT, DAGGER_BIT);
    break;
  case 1:    
    MatPC(spinorRef, hostGauge, spinorEven, kappa, QUDA_MATPC_EVEN_EVEN);
    break;
  case 2:
    Mat(spinorRef, hostGauge, spinor, kappa);
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
  if (test_type < 2) {
    printf("Reference:\n");
    printSpinorHalfField(spinorRef);
    
    printf("\nCUDA:\n");
    printSpinorHalfField(spinorOdd);
  } else {
    printf("Reference:\n");
    printSpinorHalfField(spinorRef);
    
    printf("\nCUDA:\n");
    printSpinorHalfField(spinorGPU);
  }

  int fail_check = 12;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[24];
  for (int i=0; i<24; i++) iter[i] = 0;

  if (test_type < 2) {
    for (int i=0; i<Nh; i++) {
      for (int j=0; j<24; j++) {
	int is = i*24+j;
	float diff = fabs(spinorRef[is]-spinorOdd[is]);
	for (int f=0; f<fail_check; f++)
	  if (diff > pow(10,-(f+1))) fail[f]++;
	//if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	if (diff > 1e-3) iter[j]++;
      }
    }
    
    for (int i=0; i<24; i++) printf("%d fails = %d\n", i, iter[i]);
    
    for (int f=0; f<fail_check; f++) {
      printf("%e Failures: %d / %d  = %e\n", pow(10,-(f+1)), fail[f], Nh*24, fail[f] / (float)(Nh*24));
    }
  } else {
    for (int i=0; i<N; i++) {
      for (int j=0; j<24; j++) {
	int is = i*24+j;
	float diff = fabs(spinorRef[is]-spinorGPU[is]);
	for (int f=0; f<fail_check; f++)
	  if (diff > pow(10,-(f+1))) fail[f]++;
	//if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	if (diff > 1e-3) iter[j]++;
      }
    }
    
    for (int i=0; i<24; i++) printf("%d fails = %d\n", i, iter[i]);
    
    for (int f=0; f<fail_check; f++) {
      printf("%e Failures: %d / %d  = %e\n", pow(10,-(f+1)), fail[f], N*24, fail[f] / (float)(N*24));
    }
  }

}


void dslashTest() {

  init();

  float spinorGiB = (float)Nh*spinorSiteSize*sizeof(float) / (1 << 30);
  float sharedKB = (float)dslashCudaSharedBytes() / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gaugeParam.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);

  int attempts = 10000;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();
  
    if (!TRANSFER) {
      if (test_type < 2) retrieveParitySpinor(spinorOdd, cudaSpinor.odd, QUDA_SINGLE_PRECISION, 
					      QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
      else retrieveSpinorField(spinorGPU, cudaSpinorOut, QUDA_SINGLE_PRECISION, 
			       QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
    }
    // print timing information
    printf("%fms per loop\n", 1000*secs);
    int flops = test_type ? 1320*2 + 48 : 1320;
    int floats = test_type ? 2*(7*24+8*gaugeParam.packed_size+24)+24 : 7*24+8*gaugeParam.packed_size+24;
    printf("GFLOPS = %f\n", 1.0e-9*flops*Nh/secs);
    printf("GiB/s = %f\n\n", Nh*floats*sizeof(float)/(secs*(1<<30)));
    
    int res;
    if (test_type < 2) res = compareFloats(spinorOdd, spinorRef, Nh*4*3*2, 1e-4);
    else res = compareFloats(spinorGPU, spinorRef, N*4*3*2, 1e-4);
    printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");

    strongCheck();

    exit(0);
  }  

  end();

}

int main(int argc, char **argv) {
  dslashTest();
}


