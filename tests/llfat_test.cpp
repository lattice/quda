#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <quda_internal.h>

#include <quda.h>
#include <gauge_quda.h>
#include <dslash_quda.h>
#include <llfat_quda.h>

#include <test_util.h>
#include <llfat_reference.h>
#include "misc.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "mpicomm.h"
#include <mpi.h>
#endif


FullGauge cudaSiteLink;
FullGauge cudaFatLink;
FullStaple cudaStaple;
FullStaple cudaStaple1;
QudaGaugeParam gaugeParam;
void *fatlink, *sitelink[4], *reflink[4];

#ifdef MULTI_GPU
void* ghost_sitelink;
#endif

int verify_results = 0;

extern void initDslashCuda(FullGauge gauge);

#define DIM 24

int device = 0;
int ODD_BIT = 1;
int Z[4];
int V;
int Vh;
int Vs;
int Vsh;
extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
QudaPrecision  link_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision  cpu_link_prec = QUDA_DOUBLE_PRECISION;
size_t gSize;

typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { dcomplex e[3][3]; } dsu3_matrix;



void
setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;
  Vs = X[0]*X[1]*X[2];
  Vsh= Vs/2;
}

static void
llfat_init(void)
{ 
  initQuda(device);
  //cudaSetDevice(dev); CUERR;
    
  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);
    
  gaugeParam.cpu_prec = cpu_link_prec;
  gaugeParam.cuda_prec = link_prec;
        
  gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  int i;
  fatlink = malloc(4*V*gaugeSiteSize* gSize);
  if (fatlink == NULL){
    fprintf(stderr, "ERROR: malloc failed for fatlink\n");
    exit(1);
  }
  
  for(i=0;i < 4;i++){
    sitelink[i] = malloc(V*gaugeSiteSize* gSize);
    if (sitelink[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for sitelink[%d]\n", i);
      exit(1);
    }
  }

#ifdef MULTI_GPU
  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs
  ghost_sitelink = malloc(8*Vs*gaugeSiteSize*gSize);
  if (ghost_sitelink == NULL){
    printf("ERROR: malloc failed for ghost_sitelink \n");
    exit(1);
  }
#endif

  for(i=0;i < 4;i++){
    reflink[i] = malloc(V*gaugeSiteSize* gSize);
    if (reflink[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for reflink[%d]\n", i);
      exit(1);
    }
  }
    
    
  createSiteLinkCPU(sitelink, gaugeParam.cpu_prec, 1);
  
#ifdef MULTI_GPU
  exchange_cpu_sitelink(gaugeParam.X, sitelink, ghost_sitelink, gaugeParam.cpu_prec);
  
  gaugeParam.site_ga_pad = gaugeParam.ga_pad = 3*Vsh;
  gaugeParam.reconstruct = link_recon;
  createLinkQuda(&cudaSiteLink, &gaugeParam);
  loadLinkToGPU(cudaSiteLink, sitelink, ghost_sitelink, &gaugeParam);

  gaugeParam.staple_pad = 3*Vsh;
  createStapleQuda(&cudaStaple, &gaugeParam);
  createStapleQuda(&cudaStaple1, &gaugeParam);
#else
  gaugeParam.site_ga_pad = gaugeParam.ga_pad = Vsh;
  gaugeParam.reconstruct = link_recon;
  createLinkQuda(&cudaSiteLink, &gaugeParam);
  loadLinkToGPU(cudaSiteLink, sitelink, NULL, &gaugeParam);

  gaugeParam.staple_pad = Vsh;
  createStapleQuda(&cudaStaple, &gaugeParam);
  createStapleQuda(&cudaStaple1, &gaugeParam);
#endif
    

  gaugeParam.llfat_ga_pad = gaugeParam.ga_pad = Vsh;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  createLinkQuda(&cudaFatLink, &gaugeParam);
  
  initDslashConstants(cudaSiteLink, 0);
    
  return;
}

void 
llfat_end()  
{  
  int i;
  free(fatlink);
  for(i=0;i < 4 ;i++){
    free(sitelink[i]);
  }

#ifdef MULTI_GPU  
  free(ghost_sitelink);
#endif

  for(i=0;i < 4;i++){
    free(reflink[i]);
  }
  
  freeLinkQuda(&cudaSiteLink);
  freeLinkQuda(&cudaFatLink);
  freeStapleQuda(&cudaStaple);
  freeStapleQuda(&cudaStaple1);

#ifdef MULTI_GPU
  //exchange_cleanup();
#endif

}



static int
llfat_test(void) 
{
  llfat_init();


  float act_path_coeff_1[6];
  double act_path_coeff_2[6];
  
  for(int i=0;i < 6;i++){
    act_path_coeff_1[i]= 0.1*i;
    act_path_coeff_2[i]= 0.1*i;
  }
  

 

  void* act_path_coeff;    
  if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
    act_path_coeff = act_path_coeff_2;
  }else{
    act_path_coeff = act_path_coeff_1;	
  }
  if (verify_results){
#ifdef MULTI_GPU
    llfat_reference_mg(reflink, sitelink, ghost_sitelink, gaugeParam.cpu_prec, act_path_coeff);
#else
    llfat_reference(reflink, sitelink, gaugeParam.cpu_prec, act_path_coeff);
#endif
  }
  
  llfat_init_cuda(&gaugeParam);
  //The number comes from CPU implementation in MILC, fermion_links_helpers.c    
  int flops= 61632; 
    
  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
  llfat_cuda(cudaFatLink, cudaSiteLink, cudaStaple, cudaStaple1, &gaugeParam, act_path_coeff_2);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  double secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  
  gaugeParam.ga_pad = gaugeParam.llfat_ga_pad;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  storeLinkToCPU(fatlink, &cudaFatLink, &gaugeParam);    
 
  int i;
  void* myfatlink[4];
  for(i=0;i < 4;i++){
	myfatlink[i] = malloc(V*gaugeSiteSize*gSize);
	if(myfatlink[i] == NULL){
	  printf("Error: malloc failed for myfatlink[%d]\n", i);
	  exit(1);
	}
  }

 for(i=0;i < V; i++){
	for(int dir=0; dir< 4; dir++){
	  char* src = ((char*)fatlink)+ (4*i+dir)*gaugeSiteSize*gSize;
	  char* dst = ((char*)myfatlink[dir]) + i*gaugeSiteSize*gSize;
	  memcpy(dst, src, gaugeSiteSize*gSize);
	}
 }  



  int res=1;
  for(int i=0;i < 4;i++){
    res &= compare_floats(reflink[i], myfatlink[i], V*gaugeSiteSize, 1e-3, gaugeParam.cpu_prec);
  }
  int accuracy_level;
  
  accuracy_level = strong_check_link(reflink, myfatlink, V, gaugeParam.cpu_prec);  
    
  printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
  int volume = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3];
  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printfQuda("gpu time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);


  for(i=0;i < 4;i++){
	free(myfatlink[i]);
  }
  llfat_end();
    
  if (res == 0){//failed
    printfQuda("\n");
    printfQuda("Warning: your test failed. \n");
    printfQuda("	Did you use --verify?\n");
    printfQuda("	Did you check the GPU health by running cuda memtest?\n");
  }


  return accuracy_level;
}            


static void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d\n", 
	     get_prec_str(link_prec),
	     get_recon_str(link_recon), 
	     xdim, ydim, zdim,
	     tdim);
  return ;
  
}

static void
usage(char** argv )
{
  printfQuda("Usage: %s <args>\n", argv[0]);
  printfQuda("  --device <dev_id>               Set which device to run on\n");
  printfQuda("  --gprec <double/single/half>    Link precision\n"); 
  printfQuda("  --recon <8/12>                  Link reconstruction type\n"); 
  printfQuda("  --sdim <n>                      Set spacial dimention\n");
  printfQuda("  --tdim <n>                      Set T dimention size(default 24)\n"); 
  printfQuda("  --verify                        Verify the GPU results using CPU results\n");
  printfQuda("  --help                          Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{


  
  //default to 18 reconstruct, 8^3 x 8 
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=8;

  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }

    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      cpu_link_prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }	 

    if( strcmp(argv[i], "--verify") == 0){
      verify_results=1;
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

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  /*    
#ifdef MULTI_GPU
  MPI_Init(&argc, &argv);
  comm_init();
#endif
  */

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);


  display_test_info();

    
  int accuracy_level = llfat_test();
    
  printfQuda("accuracy_level=%d\n", accuracy_level);

  /*
#ifdef MULTI_GPU
  comm_cleanup();
#endif
  */

  endCommsQuda();

  int ret;
  if(accuracy_level >=3 ){
    ret = 0; 
  }else{
    ret = 1; //we delclare the test failed
  }

  return ret;
}


