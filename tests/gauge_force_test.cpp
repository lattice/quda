#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <test_util.h>
#include <gauge_field.h>
#include "misc.h"
#include "gauge_force_reference.h"
#include "gauge_force_quda.h"
#include <sys/time.h>
#include "fat_force_quda.h"

extern void initDslashConstants(const cudaGaugeField& gauge, const int sp_stride);

extern int device;

static cudaGaugeField* cudaSiteLink = NULL;
static cudaGaugeField* cudaMom = NULL;
static QudaGaugeParam gaugeParam;
static cpuGaugeField* siteLink = NULL;
static cpuGaugeField* mom = NULL;
static cpuGaugeField* refMom = NULL;

static int verify_results = 0;
extern int tdim;
extern QudaPrecision prec;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern void usage(char** argv);

int Z[4];
int V;
int Vh;

extern QudaReconstructType link_recon;
QudaPrecision  link_prec = QUDA_SINGLE_PRECISION;

extern int gridsize_from_cmdline[];


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
gauge_force_init()
{ 
    initQuda(device);
    
    gaugeParam.X[0] = xdim;
    gaugeParam.X[1] = ydim;
    gaugeParam.X[2] = zdim;
    gaugeParam.X[3] = tdim;
    
    setDims(gaugeParam.X);
    
    gaugeParam.cpu_prec = link_prec;
    gaugeParam.cuda_prec = link_prec;
    gaugeParam.reconstruct = link_recon;
   
    gaugeParam.type = QUDA_WILSON_LINKS; // in this context, just means these are site links   
 
    gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

    GaugeFieldParam gParam(0, gaugeParam);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.pad = 0;
    siteLink = new cpuGaugeField(gParam);

    // this is a hack to have site link generated in 2d 
    // then copied to 1d array in "MILC" format
    void* siteLink_2d[4];
    for(int i=0;i < 4;i++){
      siteLink_2d[i] = malloc(siteLink->Volume()*gaugeSiteSize*gaugeParam.cpu_prec);
    }

    // fills the gauge field with random numbers
    createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec, 0);
    
    //copy the 2d sitelink to 1d milc format 
    for(int dir=0;dir < 4; dir++){
      for(int i=0;i < siteLink->Volume(); i++){
	char* src =  ((char*)siteLink_2d[dir]) + i * gaugeSiteSize* gaugeParam.cpu_prec;
	char* dst =  ((char*)siteLink->Gauge_p()) + (4*i+dir)*gaugeSiteSize*gaugeParam.cpu_prec ;
	memcpy(dst, src, gaugeSiteSize*gaugeParam.cpu_prec);
      }
    }
    
    for(int i=0;i < 4;i++){
      free(siteLink_2d[i]);
    }
#if 0
    site_link_sanity_check(siteLink, V, gaugeParam.cpu_prec, &gaugeParam);
#endif

    gParam.precision = gaugeParam.cuda_prec;
    gParam.reconstruct = link_recon;
    cudaSiteLink = new cudaGaugeField(gParam);
    
    gParam.reconstruct = QUDA_RECONSTRUCT_10;
    gParam.precision = gaugeParam.cpu_prec;
    gParam.create =QUDA_ZERO_FIELD_CREATE;
    mom = new cpuGaugeField(gParam);    
    refMom = new cpuGaugeField(gParam);
    
    
    //initiaze some data in mom
    createMomCPU(mom->Gauge_p(),  gaugeParam.cpu_prec);    
    
    memcpy(refMom->Gauge_p(), mom->Gauge_p(), 4*mom->Volume()*momSiteSize*gaugeParam.cpu_prec);
    
    gParam.precision = gaugeParam.cuda_prec;
    cudaMom = new cudaGaugeField(gParam);
    
    return;
}

static void 
gauge_force_end() 
{
  delete siteLink;
  delete mom;
  
  delete cudaSiteLink;
  delete cudaMom;
  delete refMom;
  
  endQuda();
}


static int
gauge_force_test(void) 
{
  gauge_force_init();
  

    int path_dir_x[][5] = {
	{1, 7, 6 },
        {6, 7, 1 },
        {2, 7, 5 },
        {5, 7, 2 },
        {3, 7, 4 },
        {4, 7, 3 },
        {0, 1, 7, 7, 6 },
        {1, 7, 7, 6, 0 },
        {6, 7, 7, 1, 0 },
        {0, 6, 7, 7, 1 },
        {0, 2, 7, 7, 5 },
        {2, 7, 7, 5, 0 },
        {5, 7, 7, 2, 0 },
        {0, 5, 7, 7, 2 },
        {0, 3, 7, 7, 4 },
        {3, 7, 7, 4, 0 },
        {4, 7, 7, 3, 0 },
        {0, 4, 7, 7, 3 },
        {6, 6, 7, 1, 1 },
        {1, 1, 7, 6, 6 },
        {5, 5, 7, 2, 2 },
        {2, 2, 7, 5, 5 },
        {4, 4, 7, 3, 3 },
        {3, 3, 7, 4, 4 },
        {1, 2, 7, 6, 5 },
        {5, 6, 7, 2, 1 },
        {1, 5, 7, 6, 2 },
        {2, 6, 7, 5, 1 },
        {6, 2, 7, 1, 5 },
        {5, 1, 7, 2, 6 },
        {6, 5, 7, 1, 2 },
        {2, 1, 7, 5, 6 },
        {1, 3, 7, 6, 4 },
        {4, 6, 7, 3, 1 },
        {1, 4, 7, 6, 3 },
        {3, 6, 7, 4, 1 },
        {6, 3, 7, 1, 4 },
        {4, 1, 7, 3, 6 },
        {6, 4, 7, 1, 3 },
        {3, 1, 7, 4, 6 },
        {2, 3, 7, 5, 4 },
        {4, 5, 7, 3, 2 },
        {2, 4, 7, 5, 3 },
        {3, 5, 7, 4, 2 },
        {5, 3, 7, 2, 4 },
        {4, 2, 7, 3, 5 },
        {5, 4, 7, 2, 3 },
        {3, 2, 7, 4, 5 },
    };
    
    
    int length[]={
	3, 
	3, 
	3, 
	3, 
	3, 
	3, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
	5, 
    };
    
    float loop_coeff_f[]={
      1.1,
      1.2,
      1.3,
      1.4,
      1.5,
      1.6,
      2.5,
      2.6,
      2.7,
      2.8,
      2.9,
      3.0,
      3.1,
      3.2,
      3.3,
      3.4,
      3.5,
      3.6,
      3.7,
      3.8,
      3.9,
      4.0,
      4.1,
      4.2,
      4.3,
      4.4,
      4.5,
      4.6,
      4.7,
      4.8,
      4.9,
      5.0,
      5.1,
      5.2,
      5.3,
      5.4,
      5.5,
      5.6,
      5.7,
      5.8,
      5.9,
      5.0,
      6.1,
      6.2,
      6.3,
      6.4,
      6.5,
      6.6,
    };

    double loop_coeff_d[sizeof(loop_coeff_f)/sizeof(float)];
    for(unsigned int i=0;i < sizeof(loop_coeff_f)/sizeof(float); i++){
      loop_coeff_d[i] = loop_coeff_f[i];
    }
    
    void* loop_coeff;
    if(gaugeParam.cuda_prec == QUDA_SINGLE_PRECISION){
      loop_coeff = (void*)&loop_coeff_f[0];
    }else{
      loop_coeff = loop_coeff_d;
    }

    int path_dir_y[][5] = {
        { 2 ,6 ,5 },
        { 5 ,6 ,2 },
        { 3 ,6 ,4 },
        { 4 ,6 ,3 },
        { 0 ,6 ,7 },
        { 7 ,6 ,0 },
        { 1 ,2 ,6 ,6 ,5 },
        { 2 ,6 ,6 ,5 ,1 },
        { 5 ,6 ,6 ,2 ,1 },
        { 1 ,5 ,6 ,6 ,2 },
        { 1 ,3 ,6 ,6 ,4 },
        { 3 ,6 ,6 ,4 ,1 },
        { 4 ,6 ,6 ,3 ,1 },
        { 1 ,4 ,6 ,6 ,3 },
        { 1 ,0 ,6 ,6 ,7 },
        { 0 ,6 ,6 ,7 ,1 },
        { 7 ,6 ,6 ,0 ,1 },
        { 1 ,7 ,6 ,6 ,0 },
        { 5 ,5 ,6 ,2 ,2 },
        { 2 ,2 ,6 ,5 ,5 },
        { 4 ,4 ,6 ,3 ,3 },
        { 3 ,3 ,6 ,4 ,4 },
        { 7 ,7 ,6 ,0 ,0 },
        { 0 ,0 ,6 ,7 ,7 },
        { 2 ,3 ,6 ,5 ,4 },
        { 4 ,5 ,6 ,3 ,2 },
        { 2 ,4 ,6 ,5 ,3 },
        { 3 ,5 ,6 ,4 ,2 },
        { 5 ,3 ,6 ,2 ,4 },
        { 4 ,2 ,6 ,3 ,5 },
        { 5 ,4 ,6 ,2 ,3 },
        { 3 ,2 ,6 ,4 ,5 },
        { 2 ,0 ,6 ,5 ,7 },
        { 7 ,5 ,6 ,0 ,2 },
        { 2 ,7 ,6 ,5 ,0 },
        { 0 ,5 ,6 ,7 ,2 },
        { 5 ,0 ,6 ,2 ,7 },
        { 7 ,2 ,6 ,0 ,5 },
        { 5 ,7 ,6 ,2 ,0 },
        { 0 ,2 ,6 ,7 ,5 },
        { 3 ,0 ,6 ,4 ,7 },
        { 7 ,4 ,6 ,0 ,3 },
        { 3 ,7 ,6 ,4 ,0 },
        { 0 ,4 ,6 ,7 ,3 },
        { 4 ,0 ,6 ,3 ,7 },
        { 7 ,3 ,6 ,0 ,4 },
        { 4 ,7 ,6 ,3 ,0 },
        { 0 ,3 ,6 ,7 ,4 }
    };
    
    int path_dir_z[][5] = {	
	{ 3 ,5 ,4 },
        { 4 ,5 ,3 },
        { 0 ,5 ,7 },
        { 7 ,5 ,0 },
        { 1 ,5 ,6 },
        { 6 ,5 ,1 },
        { 2 ,3 ,5 ,5 ,4 },
        { 3 ,5 ,5 ,4 ,2 },
        { 4 ,5 ,5 ,3 ,2 },
        { 2 ,4 ,5 ,5 ,3 },
        { 2 ,0 ,5 ,5 ,7 },
        { 0 ,5 ,5 ,7 ,2 },
        { 7 ,5 ,5 ,0 ,2 },
        { 2 ,7 ,5 ,5 ,0 },
        { 2 ,1 ,5 ,5 ,6 },
        { 1 ,5 ,5 ,6 ,2 },
        { 6 ,5 ,5 ,1 ,2 },
        { 2 ,6 ,5 ,5 ,1 },
        { 4 ,4 ,5 ,3 ,3 },
        { 3 ,3 ,5 ,4 ,4 },
        { 7 ,7 ,5 ,0 ,0 },
        { 0 ,0 ,5 ,7 ,7 },
        { 6 ,6 ,5 ,1 ,1 },
        { 1 ,1 ,5 ,6 ,6 },
        { 3 ,0 ,5 ,4 ,7 },
        { 7 ,4 ,5 ,0 ,3 },
        { 3 ,7 ,5 ,4 ,0 },
        { 0 ,4 ,5 ,7 ,3 },
        { 4 ,0 ,5 ,3 ,7 },
        { 7 ,3 ,5 ,0 ,4 },
        { 4 ,7 ,5 ,3 ,0 },
        { 0 ,3 ,5 ,7 ,4 },
        { 3 ,1 ,5 ,4 ,6 },
        { 6 ,4 ,5 ,1 ,3 },
        { 3 ,6 ,5 ,4 ,1 },
        { 1 ,4 ,5 ,6 ,3 },
        { 4 ,1 ,5 ,3 ,6 },
        { 6 ,3 ,5 ,1 ,4 },
        { 4 ,6 ,5 ,3 ,1 },
        { 1 ,3 ,5 ,6 ,4 },
        { 0 ,1 ,5 ,7 ,6 },
        { 6 ,7 ,5 ,1 ,0 },
        { 0 ,6 ,5 ,7 ,1 },
        { 1 ,7 ,5 ,6 ,0 },
        { 7 ,1 ,5 ,0 ,6 },
        { 6 ,0 ,5 ,1 ,7 },
        { 7 ,6 ,5 ,0 ,1 },
        { 1 ,0 ,5 ,6 ,7 }
    };
    
    int path_dir_t[][5] = {
        { 0 ,4 ,7 },
        { 7 ,4 ,0 },
        { 1 ,4 ,6 },
        { 6 ,4 ,1 },
        { 2 ,4 ,5 },
        { 5 ,4 ,2 },
        { 3 ,0 ,4 ,4 ,7 },
        { 0 ,4 ,4 ,7 ,3 },
        { 7 ,4 ,4 ,0 ,3 },
        { 3 ,7 ,4 ,4 ,0 },
        { 3 ,1 ,4 ,4 ,6 },
        { 1 ,4 ,4 ,6 ,3 },
        { 6 ,4 ,4 ,1 ,3 },
        { 3 ,6 ,4 ,4 ,1 },
        { 3 ,2 ,4 ,4 ,5 },
        { 2 ,4 ,4 ,5 ,3 },
        { 5 ,4 ,4 ,2 ,3 },
        { 3 ,5 ,4 ,4 ,2 },
        { 7 ,7 ,4 ,0 ,0 },
        { 0 ,0 ,4 ,7 ,7 },
        { 6 ,6 ,4 ,1 ,1 },
        { 1 ,1 ,4 ,6 ,6 },
        { 5 ,5 ,4 ,2 ,2 },
        { 2 ,2 ,4 ,5 ,5 },
        { 0 ,1 ,4 ,7 ,6 },
        { 6 ,7 ,4 ,1 ,0 },
        { 0 ,6 ,4 ,7 ,1 },
        { 1 ,7 ,4 ,6 ,0 },
        { 7 ,1 ,4 ,0 ,6 },
        { 6 ,0 ,4 ,1 ,7 },
        { 7 ,6 ,4 ,0 ,1 },
        { 1 ,0 ,4 ,6 ,7 },
        { 0 ,2 ,4 ,7 ,5 },
        { 5 ,7 ,4 ,2 ,0 },
        { 0 ,5 ,4 ,7 ,2 },
        { 2 ,7 ,4 ,5 ,0 },
        { 7 ,2 ,4 ,0 ,5 },
        { 5 ,0 ,4 ,2 ,7 },
        { 7 ,5 ,4 ,0 ,2 },
        { 2 ,0 ,4 ,5 ,7 },
        { 1 ,2 ,4 ,6 ,5 },
        { 5 ,6 ,4 ,2 ,1 },
        { 1 ,5 ,4 ,6 ,2 },
        { 2 ,6 ,4 ,5 ,1 },
        { 6 ,2 ,4 ,1 ,5 },
        { 5 ,1 ,4 ,2 ,6 },
        { 6 ,5 ,4 ,1 ,2 },
        { 2 ,1 ,4 ,5 ,6 }
    };
    
    int max_length = 6;

    

    initDslashConstants(*cudaSiteLink, 0);
    gauge_force_init_cuda(&gaugeParam, max_length); 
    
    double eb3 = 0.3;
    int num_paths = sizeof(path_dir_x)/sizeof(path_dir_x[0]);

    int i;
    
    int** input_path;
    input_path = (int**)malloc(num_paths*sizeof(int*));
    if (input_path == NULL){
	printf("ERORR: malloc failed for input path\n");
	exit(1);
    }
    for(i=0;i < num_paths;i++){
	input_path[i] = (int*)malloc(length[i]*sizeof(int));
	if (input_path[i] == NULL){
	    printf("ERROR: malloc failed for input_path[%d]\n", i);
	    exit(1);
	}
    }
    
    // download the momentum field to the GPU
    cudaMom->loadCPUField(*mom, QUDA_CPU_FIELD_LOCATION);

    // download the gauge field to the GPU
    cudaSiteLink->loadCPUField(*siteLink, QUDA_CPU_FIELD_LOCATION);
    
#define CX 
#define CY 
#define CZ 
#define CT 
    
    if (verify_results){
	
#ifdef CX
	for(i=0;i < num_paths;i++){
	    memcpy(input_path[i], path_dir_x, length[i]*sizeof(int));
	}
	gauge_force_reference(refMom->Gauge_p(), 0, eb3, siteLink->Gauge_p(), gaugeParam.cpu_prec, input_path, length, loop_coeff, num_paths);
#endif

#ifdef CY	
	for(i=0;i < num_paths;i++){
	    memcpy(input_path[i], path_dir_y, length[i]*sizeof(int));
	}
	gauge_force_reference(refMom->Gauge_p(), 1, eb3, siteLink->Gauge_p(), gaugeParam.cpu_prec, input_path, length, loop_coeff, num_paths);
#endif	

#ifdef CZ	
	for(i=0;i < num_paths;i++){
	    memcpy(input_path[i], path_dir_z, length[i]*sizeof(int));
	}
	gauge_force_reference(refMom->Gauge_p(), 2, eb3, siteLink->Gauge_p(), gaugeParam.cpu_prec, input_path, length, loop_coeff, num_paths);
#endif
	
#ifdef CT	
	for(i=0;i < num_paths;i++){
	    memcpy(input_path[i], path_dir_t, length[i]*sizeof(int));
	}
	gauge_force_reference(refMom->Gauge_p(), 3, eb3, siteLink->Gauge_p(), gaugeParam.cpu_prec, input_path, length, loop_coeff, num_paths);
#endif
	
    }
    
      
    //The number comes from CPU implementation in MILC, gauge_force_imp.c
    int flops=153004;
    
    

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    
#ifdef CX
    for(i=0;i < num_paths;i++){
	memcpy(input_path[i], path_dir_x, length[i]*sizeof(int));
    }
    gauge_force_cuda(*cudaMom, 0, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff, num_paths, max_length);
#endif

#ifdef CY
    for(i=0;i < num_paths;i++){
	memcpy(input_path[i], path_dir_y, length[i]*sizeof(int));
    }
    gauge_force_cuda(*cudaMom, 1, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff, num_paths, max_length);
#endif

#ifdef CZ
    for(i=0;i < num_paths;i++){
	memcpy(input_path[i], path_dir_z, length[i]*sizeof(int));
    }
    gauge_force_cuda(*cudaMom, 2, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff, num_paths, max_length);
#endif
    
#ifdef CT
    for(i=0;i < num_paths;i++){
      memcpy(input_path[i], path_dir_t, length[i]*sizeof(int));
    }       
    gauge_force_cuda(*cudaMom, 3, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff, num_paths, max_length);
#endif
    
    
    gettimeofday(&t1, NULL);
    double secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);

    // copy the new momentum back on the CPU
    cudaMom->saveCPUField(*mom, QUDA_CPU_FIELD_LOCATION);
    
    int res;
    res = compare_floats(mom->Gauge_p(), refMom->Gauge_p(), 4*mom->Volume()*momSiteSize, 1e-3, gaugeParam.cpu_prec);
    
    int accuracy_level;
    accuracy_level = strong_check_mom(mom->Gauge_p(), refMom->Gauge_p(), 4*mom->Volume(), gaugeParam.cpu_prec);
    
    printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
    
    int volume = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3];
    double perf = 1.0* flops*volume/(secs*1024*1024*1024);
    printf("gpu time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);
    
    for(i=0;i < num_paths; i++){
      free(input_path[i]);
    }
    free(input_path);

    gauge_force_end();

    if (res == 0){//failed
        printf("\n");
        printf("Warning: you test failed. \n");
        printf("        Did you use --verify?\n");
        printf("        Did you check the GPU health by running cuda memtest?\n");
    }
    
    return accuracy_level;
}            


static void
display_test_info()
{
    printf("running the following test:\n");
    
    printf("link_precision           link_reconstruct           space_dim(x/y/z)              T_dimension\n");
    printf("%s                       %s                         %d/%d/%d                       %d\n", 
	   get_prec_str(link_prec),
	   get_recon_str(link_recon), 
	   xdim,ydim,zdim, tdim);
    return ;
    
}

void
usage_extra(char** argv )
{
  printf("Extra options:\n");
  printf("    --verify                                  # Verify the GPU results using CPU results\n");
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
      
      if( strcmp(argv[i], "--verify") == 0){
	verify_results=1;
	continue;	    
      }	
      
      fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
      usage(argv);
    }
    
    
    link_prec = prec;
#ifdef MULTI_GPU
    initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
#endif

    display_test_info();
    
    int accuracy_level = gauge_force_test();
    printfQuda("accuracy_level=%d\n", accuracy_level);

#ifdef MULTI_GPU
    endCommsQuda();
#endif
    
    int ret;
    if(accuracy_level >=3 ){
      ret = 0; 
    }else{
      ret = 1; //we delclare the test failed
    }

    return ret;
}
