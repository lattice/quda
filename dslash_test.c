#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>
#include <field_quda.h>

#define FULL_WILSON 0

QudaGaugeParam param;
FullSpinor cudaSpinor;
ParitySpinor tmp;

float *hostGauge[4];
float *spinor, *spinorRef;
float *spinorEven, *spinorOdd;
    
float kappa = 1.0;
int ODD_BIT = 0;
int DAGGER_BIT = 0;
    
void printSpinorHalfField(float *spinor) {
  printSpinor(&spinor[0*spinorSiteSize]);
  printf("...\n");
  printSpinor(&spinor[(Nh-1)*spinorSiteSize]);
  printf("\n");    
}

void init() {

  cudaGauge.even = 0;
  cudaGauge.odd = 0;

  param.cpu_prec = QUDA_SINGLE_PRECISION;
  param.cuda_prec = QUDA_SINGLE_PRECISION;
  param.X = L1;
  param.Y = L2;
  param.Z = L3;
  param.T = L4;
  param.anisotropy = 2.3;
  param.reconstruct = QUDA_RECONSTRUCT_12;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param = &param;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
    hostGauge[dir] = (float*)malloc(N*gaugeSiteSize*sizeof(float));
  }

  spinor = (float*)malloc(N*spinorSiteSize*sizeof(float));
  spinorRef = (float*)malloc(N*spinorSiteSize*sizeof(float));
  spinorEven = spinor;
  spinorOdd = spinor + Nh*spinorSiteSize;
    
  printf("Randomizing fields...");
  constructGaugeField(hostGauge);
  constructSpinorField(spinor);
  printf("done.\n"); fflush(stdout);
  
  int dev = 0;
  cudaSetDevice(dev);

  printf("Sending fields to GPU..."); fflush(stdout);
  loadGaugeField(hostGauge);
  cudaSpinor = allocateSpinorField();
  tmp = allocateParitySpinor();

  loadSpinorField(cudaSpinor, (void*)spinor, QUDA_SINGLE_PRECISION, 
		  QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  free(spinor);
  free(spinorRef);
  freeSpinorField(cudaSpinor);
  freeSpinorBuffer();
  freeParitySpinor(tmp);
}

void dslashRef() {
  
  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  if (FULL_WILSON)
    MatPC(spinorRef, hostGauge, spinorEven, kappa, QUDA_MATPC_EVEN_EVEN);
  else
    dslashReference(spinorRef, hostGauge, spinorEven, ODD_BIT, DAGGER_BIT);
  printf("done.\n");
    
}

double dslashCUDA() {

  // execute kernel
  const int LOOPS = 200;
  printf("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < LOOPS; i++) {
    if (FULL_WILSON)
      MatPCCuda(cudaSpinor.odd, cudaGauge, cudaSpinor.even, kappa, tmp, QUDA_MATPC_EVEN_EVEN);
    else
      dslashCuda(cudaSpinor.odd, cudaGauge, cudaSpinor.even, ODD_BIT, DAGGER_BIT);
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

void dslashTest() {

  init();

  float spinorGiB = (float)Nh*spinorSiteSize*sizeof(float) / (1 << 30);
  float sharedKB = (float)dslashCudaSharedBytes() / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", param.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);

  int attempts = 100;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();
    retrieveParitySpinor(spinorOdd, cudaSpinor.odd, QUDA_SINGLE_PRECISION, 
			 QUDA_SINGLE_PRECISION, QUDA_DIRAC_ORDER);
    
    // print timing information
    printf("%fms per loop\n", 1000*secs);
    int flops = FULL_WILSON ? 1320*2 + 48 : 1320;
    int floats = FULL_WILSON ? 2*(7*24+8*param.packed_size+24)+24 : 7*24+8*param.packed_size+24;
    printf("GFLOPS = %f\n", 1.0e-9*flops*Nh/secs);
    printf("GiB/s = %f\n\n", Nh*floats*sizeof(float)/(secs*(1<<30)));
    
    int res = compareFloats(spinorOdd, spinorRef, Nh*4*3*2, 1e-4);
    printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");
  }  

  end();

}

int main(int argc, char **argv) {
  dslashTest();
}


  /*
  printf("Reference:\n");
  printSpinorHalfField(spinorRef);
    
  printf("\nCUDA:\n");
  printSpinorHalfField(spinorOdd);

  int fail_check = 12;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[24];
  for (int i=0; i<24; i++) iter[i] = 0;

  for (int i=0; i<Nh; i++) {
    for (int j=0; j<24; j++) {
      int is = i*24+j;
      float diff = fabs(spinorRef[is]-spinorOdd[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10,-(f+1))) fail[f]++;
      //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
      if (diff > 1e-6) iter[j]++;
    }
  }
    
  //for (int i=0; i<24; i++) printf("%d fails = %d\n", i, iter[i]);

  for (int f=0; f<fail_check; f++) {
    printf("%e Failures: %d / %d  = %e\n", pow(10,-(f+1)), fail[f], Nh*24, fail[f] / (float)(Nh*24));
  }

  if (stat != cudaSuccess)
    printf("Cuda failed with ERROR: %s\n", cudaGetErrorString(stat));
  */ 
