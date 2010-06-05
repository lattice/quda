#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include "test_util.h"
#include "gauge_quda.h"
#include "misc.h"
#include "fermion_force_reference.h"
#include "fermion_force_quda.h"
#include "dslash_quda.h"
#include "hw_quda.h"
#include <sys/time.h>

static FullGauge cudaSiteLink;
static FullMom cudaMom;
static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static void* siteLink;
static void* mom;
static void* refMom;
static void* hw; //the array of half_wilson_vector
static int X[4];

int verify_results = 0;

extern void initDslashCuda(FullGauge gauge);

static int sdim= 8;

int ODD_BIT = 1;
static int tdim = 8;

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
QudaPrecision hw_prec = QUDA_SINGLE_PRECISION;
QudaPrecision mom_prec = QUDA_SINGLE_PRECISION;

QudaPrecision cpu_hw_prec = QUDA_SINGLE_PRECISION;

typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { dcomplex e[3][3]; } dsu3_matrix;


int Z[4];
int V;
int Vh;
void
setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;
}

static void
fermion_force_init()
{ 
  int dev = 1;
    
  initQuda(dev);
  //cudaSetDevice(dev); CUERR;
    
  X[0] = gaugeParam.X[0] = sdim;
  X[1] = gaugeParam.X[1] = sdim;
  X[2] = gaugeParam.X[2] = sdim;
  X[3] = gaugeParam.X[3] = tdim;
    
  setDims(gaugeParam.X);
    
  gaugeParam.cpu_prec = link_prec;
  gaugeParam.cuda_prec = link_prec;
  gaugeParam.reconstruct = link_recon;
    
    
  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  siteLink = malloc(4*V*gaugeSiteSize* gSize);
  if (siteLink == NULL){
    fprintf(stderr, "ERROR: malloc failed for sitelink\n");
    exit(1);
  }
  createSiteLinkCPU(siteLink, gaugeParam.cpu_prec,1);

#if 1
  site_link_sanity_check(siteLink, V, gaugeParam.cpu_prec, &gaugeParam);
#endif

  mom = malloc(4*V*momSiteSize*gSize);
  if (mom == NULL){
    fprintf(stderr, "ERROR: malloc failed for mom\n");
    exit(1);
  }
  createMomCPU(mom,mom_prec);    
  memset(mom, 0, 4*V*momSiteSize*gSize);

  refMom = malloc(4*V*momSiteSize*gSize);
  if (refMom == NULL){
    fprintf(stderr, "ERROR: malloc failed for refMom\n");
    exit(1);
  }    
  memcpy(refMom, mom, 4*V*momSiteSize*gSize);
    
    
  hw = malloc(4*V*hwSiteSize*gSize);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);	
  }
  createHwCPU(hw, hw_prec);
    
    
  createLinkQuda(&cudaSiteLink, &gaugeParam);
  createMomQuda(&cudaMom, &gaugeParam);    
  cudaHw = createHwQuda(X, hw_prec);
    
  return;
}

static void 
fermion_force_end() 
{
  free(siteLink);
  free(mom);
  free(refMom);
  free(hw);
    
  freeLinkQuda(&cudaSiteLink);
  freeMomQuda(&cudaMom);
}


static void 
fermion_force_test(void) 
{
 
  fermion_force_init();
  initDslashConstants(cudaSiteLink, Vh, Vh);
  fermion_force_init_cuda(&gaugeParam);

    
  float eps= 0.02;
  float weight1 =1.0;
  float weight2 =1.0;
  float act_path_coeff[6];
    
  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;        
    
  loadMomToGPU(cudaMom, mom, &gaugeParam);
  loadLinkToGPU(cudaSiteLink, siteLink, &gaugeParam);
  loadHwToGPU(cudaHw, hw, cpu_hw_prec);

    
  if (verify_results){	
    fermion_force_reference(eps, weight1, weight2, act_path_coeff, hw, siteLink, refMom);
  }
    
    
  /*
   * The flops number comes from CPU implementation in MILC
   * function eo_fermion_force_twoterms_field(), fermion_force_asqtad.c
   *
   */
  int flops = 433968;

  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
    
  fermion_force_cuda(eps, weight1, weight2, act_path_coeff, cudaHw, cudaSiteLink, cudaMom, &gaugeParam);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  double secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
    
  storeMomToCPU(mom, cudaMom, &gaugeParam);
    
  int res;
  res = compare_floats(mom, refMom, 4*V*momSiteSize, 1e-5, gaugeParam.cpu_prec);
    
  strong_check_mom(mom, refMom, 4*V, gaugeParam.cpu_prec);
    
  printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
    
  int volume = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3];
  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printf("gpu time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);
    
  fermion_force_end();

  if (res == 0){//failed
    printf("\n");
    printf("Warning: you test failed. \n");
    printf("        Did you use --verify?\n");
    printf("        Did you check the GPU health by running cuda memtest?\n");
  }

    
}            


static void
display_test_info()
{
  printf("running the following fermion force computation test:\n");
    
  printf("link_precision           link_reconstruct           S_dimension         T_dimension\n");
  printf("%s                       %s                         %d                  %d \n", 
	 get_prec_str(link_prec),
	 get_recon_str(link_recon), 
	 sdim, tdim);
  return ;
    
}

static void
usage(char** argv )
{
  printf("Usage: %s <args>\n", argv[0]);
  printf("--gprec <double/single/half> \t Link precision\n"); 
  printf("--recon <8/12> \t\t\t Link reconstruction type\n"); 
  printf("--tdim \t\t\t\t Set T dimention size(default 24)\n"); 
  printf("--sdim \t\t\t\t Set spalce dimention size(default 16)\n"); 
  printf("--verify\n");
  printf("--help \t\t\t\t Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{
  int i;
  for (i =1;i < argc; i++){
	
    if( strcmp(argv[i], "--help")== 0){
      usage(argv);
    }
	
    if( strcmp(argv[i], "--gprec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--recon") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_recon =  get_recon(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--tdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      tdim =  atoi(argv[i+1]);
      if (tdim < 0 || tdim > 128){
	fprintf(stderr, "Error: invalid t dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }
    if( strcmp(argv[i], "--sdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      sdim =  atoi(argv[i+1]);
      if (sdim < 0 || sdim > 128){
	fprintf(stderr, "Error: invalid space dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }
    if( strcmp(argv[i], "--verify") == 0){
      verify_results=1;
      continue;	    
    }	
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }
    
  display_test_info();
    
  fermion_force_test();
    
    
  return 0;
}
