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
#include <cuda.h>
#include <fat_force_quda.h>

#ifdef MULTI_GPU
#include "face_quda.h"
#include "mpicomm.h"
#include <mpi.h>
#endif

#define MAX(a,b) ((a)>(b)? (a):(b))

FullGauge cudaSiteLink;
FullGauge cudaFatLink;
FullStaple cudaStaple;
FullStaple cudaStaple1;
QudaGaugeParam gaugeParam;
void *fatlink, *sitelink[4], *reflink[4];

#ifdef MULTI_GPU
void* ghost_sitelink[4];
void* ghost_sitelink_diag[16];
#endif

int verify_results = 0;

extern void initDslashCuda(FullGauge gauge);

#define DIM 24

int device = 0;
int ODD_BIT = 1;
int Z[4];
int V;
int Vh;
int Vs[4];
int Vsh[4];
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;


extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision  prec;
QudaPrecision  cpu_prec = QUDA_DOUBLE_PRECISION;
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

  Vs[0] = Vs_x = X[1]*X[2]*X[3];
  Vs[1] = Vs_y = X[0]*X[2]*X[3];
  Vs[2] = Vs_z = X[0]*X[1]*X[3];
  Vs[3] = Vs_t = X[0]*X[1]*X[2];

  Vsh[0] = Vsh_x = Vs_x/2;
  Vsh[1] = Vsh_y = Vs_y/2;
  Vsh[2] = Vsh_z = Vs_z/2;
  Vsh[3] = Vsh_t = Vs_t/2;

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
    
  gaugeParam.cpu_prec = cpu_prec;
  gaugeParam.cuda_prec = prec;
        
  gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  int i;
#if (CUDA_VERSION >=4000)
  cudaMallocHost((void**)&fatlink,  4*V*gaugeSiteSize*gSize);
#else
  fatlink = malloc(4*V*gaugeSiteSize*gSize);
#endif
  if (fatlink == NULL){
    fprintf(stderr, "ERROR: malloc failed for fatlink\n");
    exit(1);
  }
  
  for(i=0;i < 4;i++){
#if (CUDA_VERSION >=4000)
    cudaMallocHost((void**)&sitelink[i], V*gaugeSiteSize* gSize);
#else
    sitelink[i] = malloc(V*gaugeSiteSize* gSize);
#endif
    if (sitelink[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for sitelink[%d]\n", i);
      exit(1);
    }
  }

#ifdef MULTI_GPU
  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  for(i=0;i < 4; i++){
    ghost_sitelink[i] = malloc(8*Vs[i]*gaugeSiteSize*gSize);
    if (ghost_sitelink[i] == NULL){
      printf("ERROR: malloc failed for ghost_sitelink[%d] \n",i);
      exit(1);
    }
  }

  /*
    nu |     |
       |_____|
          mu     
  */
  
  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
	ghost_sitelink_diag[nu*4+mu] = NULL;
      }else{
	//the other directions
	int dir1, dir2;
	for(dir1= 0; dir1 < 4; dir1++){
	  if(dir1 !=nu && dir1 != mu){
	    break;
	  }
	}
	for(dir2=0; dir2 < 4; dir2++){
	  if(dir2 != nu && dir2 != mu && dir2 != dir1){
	    break;
	  }
	}
	ghost_sitelink_diag[nu*4+mu] = malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
	if(ghost_sitelink_diag[nu*4+mu] == NULL){
	  errorQuda("malloc failed for ghost_sitelink_diag\n");
	}
	
	memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
      }

    }
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
  int Vh_2d_max = MAX(xdim*ydim/2, xdim*zdim/2);
  Vh_2d_max = MAX(Vh_2d_max, xdim*tdim/2);
  Vh_2d_max = MAX(Vh_2d_max, ydim*zdim/2);  
  Vh_2d_max = MAX(Vh_2d_max, ydim*tdim/2);  
  Vh_2d_max = MAX(Vh_2d_max, zdim*tdim/2);  
  
  gaugeParam.site_ga_pad = gaugeParam.ga_pad = 3*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + 4*Vh_2d_max;
  gaugeParam.reconstruct = link_recon;
  createLinkQuda(&cudaSiteLink, &gaugeParam);
  //loadLinkToGPU(cudaSiteLink, sitelink, &gaugeParam);

  gaugeParam.staple_pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
  createStapleQuda(&cudaStaple, &gaugeParam);
  createStapleQuda(&cudaStaple1, &gaugeParam);
#else
  gaugeParam.site_ga_pad = gaugeParam.ga_pad = Vsh_t;
  gaugeParam.reconstruct = link_recon;
  createLinkQuda(&cudaSiteLink, &gaugeParam);
  //loadLinkToGPU(cudaSiteLink, sitelink, NULL, NULL, &gaugeParam);

  gaugeParam.staple_pad = Vsh_t;
  createStapleQuda(&cudaStaple, &gaugeParam);
  createStapleQuda(&cudaStaple1, &gaugeParam);
#endif
   
  gaugeParam.llfat_ga_pad = gaugeParam.ga_pad = Vsh_t;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  createLinkQuda(&cudaFatLink, &gaugeParam);


  initDslashConstants(cudaSiteLink, 0);
    
  return;
}

void 
llfat_end()  
{  
  int i;
#if (CUDA_VERSION >= 4000)
  cudaFreeHost(fatlink);
  for(i=0;i < 4 ;i++){
    cudaFreeHost(sitelink[i]);
  }
#else
  free(fatlink);
  for(i=0;i < 4 ;i++){
    free(sitelink[i]);
  }

#endif

#ifdef MULTI_GPU  
  for(i=0;i < 4;i++){
    free(ghost_sitelink[i]);
  }
  for(i=0;i <4; i++){
    for(int j=0;j <4; j++){
      if (i==j){
	continue;
      }
      free(ghost_sitelink_diag[i*4+j]);
    }    
  }
#endif

  for(i=0;i < 4;i++){
    free(reflink[i]);
  }
  
  freeLinkQuda(&cudaSiteLink);
  freeLinkQuda(&cudaFatLink);
  freeStapleQuda(&cudaStaple);
  freeStapleQuda(&cudaStaple1);

#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif

  endQuda();
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

    int optflag = 0;
    exchange_cpu_sitelink(gaugeParam.X, sitelink, ghost_sitelink, ghost_sitelink_diag, gaugeParam.cpu_prec, optflag);


#ifdef MULTI_GPU
    llfat_reference_mg(reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, gaugeParam.cpu_prec, act_path_coeff);
    //llfat_reference(reflink, sitelink, gaugeParam.cpu_prec, act_path_coeff);
#else
    llfat_reference(reflink, sitelink, gaugeParam.cpu_prec, act_path_coeff);
#endif
  }
  
  llfat_init_cuda(&gaugeParam);
  //The number comes from CPU implementation in MILC, fermion_links_helpers.c    
  int flops= 61632; 

  struct timeval t0, t1, t2, t3;
  gettimeofday(&t0, NULL);
#ifdef MULTI_GPU
  gaugeParam.ga_pad = gaugeParam.site_ga_pad;
  gaugeParam.reconstruct = link_recon;
  loadLinkToGPU(cudaSiteLink, sitelink, &gaugeParam);
#else
  loadLinkToGPU(cudaSiteLink, sitelink, NULL, NULL, &gaugeParam);
#endif
  
  gettimeofday(&t1, NULL);  

  llfat_cuda(cudaFatLink, cudaSiteLink, cudaStaple, cudaStaple1, &gaugeParam, act_path_coeff_2);
  
  gettimeofday(&t2, NULL);
  storeLinkToCPU(fatlink, &cudaFatLink, &gaugeParam);
  gettimeofday(&t3, NULL);

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))
  double secs = TDIFF(t0,t3);
 
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

  printfQuda(" h2d=%f s, computation in gpu=%f s, d2h=%f s, total time=%f s\n", 
	     TDIFF(t0, t1), TDIFF(t1, t2), TDIFF(t2, t3), TDIFF(t0, t3));
  

  printfQuda(" h2d=%f s, computation in gpu=%f s, d2h=%f s, total time=%f s\n",
             TDIFF(t0, t1), TDIFF(t1, t2), TDIFF(t2, t3), TDIFF(t0, t3));
  

  return accuracy_level;
}            


static void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d\n", 
	     get_prec_str(prec),
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
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }

    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      cpu_prec =  get_prec(argv[i+1]);
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

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);


  display_test_info();

    
  int accuracy_level = llfat_test();
    
  printfQuda("accuracy_level=%d\n", accuracy_level);

  endCommsQuda();

  int ret;
  if(accuracy_level >=3 ){
    ret = 0; 
  }else{
    ret = 1; //we delclare the test failed
  }

  return ret;
}


