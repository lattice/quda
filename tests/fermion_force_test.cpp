#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include "test_util.h"
#include "gauge_quda.h"
#include "fat_force_quda.h"
#include "misc.h"
#include "fermion_force_reference.h"
#include "fermion_force_quda.h"
#include "hw_quda.h"
#include <sys/time.h>

int device = 0;
static FullGauge cudaSiteLink;
static FullMom cudaMom;
static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static void* siteLink;
static void* mom;
static void* refMom;
static void* hw; //the array of half_wilson_vector
static int X[4];

extern int gridsize_from_cmdline[];

int verify_results = 0;

#ifdef __cplusplus
extern "C" {
#endif

extern void initDslashCuda(FullGauge gauge);
extern void initDslashConstants(const FullGauge gauge, const int sp_stride);

#ifdef __cplusplus
}
#endif

int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;

extern QudaReconstructType link_recon;
QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
extern QudaPrecision prec;
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
  initQuda(device);
  //cudaSetDevice(dev); CUERR;
    
  X[0] = gaugeParam.X[0] = xdim;
  X[1] = gaugeParam.X[1] = ydim;
  X[2] = gaugeParam.X[2] = zdim;
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

  void* siteLink_2d[4];
  for(int i=0;i < 4;i++){
    siteLink_2d[i] = ((char*)siteLink) + i*V*gaugeSiteSize* gSize;
  }
  
  createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec,0);

#if 0
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
    
  //gaugeParam.site_ga_pad = gaugeParam.ga_pad = 0;
  //gaugeParam.reconstruct = link_recon;
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
  initDslashConstants(cudaSiteLink, Vh);
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
  loadLinkToGPU_gf(cudaSiteLink, siteLink, &gaugeParam);
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
    
  printf("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension\n");
  printf("%s                       %s                         %d/%d/%d                  %d \n", 
	 get_prec_str(link_prec),
	 get_recon_str(link_recon), 
	 xdim, ydim, zdim, tdim);
  return ;
    
}

static void
usage(char** argv )
{
  printf("Usage: %s <args>\n", argv[0]);
  printf("  --device <dev_id>               Set which device to run on\n");
  printf("  --gprec <double/single/half>    Link precision\n"); 
  printf("  --recon <8/12>                  Link reconstruction type\n"); 
  printf("  --sdim <n>                      Set spacial dimention\n");
  printf("  --tdim                          Set T dimention size(default 24)\n"); 
  printf("  --sdim                          Set spalce dimention size(default 16)\n"); 
  printf("  --verify                        Verify the GPU results using CPU results\n");
  printf("  --help                          Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{
  int i;
  for (i =1;i < argc; i++){
	
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    

    if( strcmp(argv[i], "--device") == 0){
        if (i+1 >= argc){
                usage(argv);
            }
            device =  atoi(argv[i+1]);
            if (device < 0){
                fprintf(stderr, "Error: invalid device number(%d)\n", device);
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

#ifdef MULTI_GPU
    initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
#endif

  link_recon = QUDA_RECONSTRUCT_12;
  link_prec = prec;

  display_test_info();
    
  fermion_force_test();


#ifdef MULTI_GPU
    endCommsQuda();
#endif
    
    
  return 0;
}
