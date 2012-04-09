#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include "test_util.h"
#include "gauge_field.h"
#include "fat_force_quda.h"
#include "misc.h"
#include "fermion_force_reference.h"
#include "fermion_force_quda.h"
#include "hw_quda.h"
#include <sys/time.h>

extern void usage(char** argv);
extern int device;
cudaGaugeField *cudaGauge = NULL;
cpuGaugeField *cpuGauge = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom = NULL;
cpuGaugeField *refMom = NULL;

static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static void* hw; //the array of half_wilson_vector

extern int gridsize_from_cmdline[];

int verify_results = 0;

int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;

extern QudaReconstructType link_recon;
QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
extern QudaPrecision prec;
QudaPrecision hw_prec = QUDA_SINGLE_PRECISION;
QudaPrecision mom_prec = QUDA_SINGLE_PRECISION;

QudaPrecision cpu_hw_prec = QUDA_SINGLE_PRECISION;

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

extern void initLatticeConstants(const LatticeField &gauge);

static void
fermion_force_init()
{ 
  initQuda(device);
  //cudaSetDevice(dev); CUERR;
    
  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;
  setDims(gaugeParam.X);

  gaugeParam.cpu_prec = link_prec;
  gaugeParam.cuda_prec = link_prec;
  gaugeParam.reconstruct = link_recon;
    
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  GaugeFieldParam gParam(0, gaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;

  cpuGauge = new cpuGaugeField(gParam);
    
  // this is a hack to have site link generated in 2d 
  // then copied to 1d array in "MILC" format
  void* siteLink_2d[4];
  for(int i=0;i < 4;i++){
    siteLink_2d[i] = malloc(cpuGauge->Volume()*gaugeSiteSize*gaugeParam.cpu_prec);
    if (siteLink_2d[i] == NULL){
      errorQuda("ERROR: malloc failed for siteLink_2d\n");
    }
  }
    
  // fills the gauge field with random numbers
  createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec, 1);

  //copy the 2d sitelink to 1d milc format 
  for(int dir=0;dir < 4; dir++){
    for(int i=0;i < cpuGauge->Volume(); i++){
      char* src =  ((char*)siteLink_2d[dir]) + i * gaugeSiteSize* gaugeParam.cpu_prec;
      char* dst =  ((char*)cpuGauge->Gauge_p()) + (4*i+dir)*gaugeSiteSize*gaugeParam.cpu_prec ;
      memcpy(dst, src, gaugeSiteSize*gaugeParam.cpu_prec);
    }
  }

  for(int i=0;i < 4;i++){
    free(siteLink_2d[i]);
  }
  
#if 0
  site_link_sanity_check(siteLink, V, gaugeParam.cpu_prec, &gaugeParam);
#endif

  //gaugeParam.site_ga_pad = gaugeParam.ga_pad = 0;
  //gaugeParam.reconstruct = link_recon;

  gParam.precision = gaugeParam.cuda_prec;
  gParam.reconstruct = link_recon;
  cudaGauge = new cudaGaugeField(gParam);

  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.precision = gaugeParam.cpu_prec;
  cpuMom = new cpuGaugeField(gParam);
  refMom = new cpuGaugeField(gParam);

  createMomCPU(cpuMom->Gauge_p(), mom_prec);    
  //memset(cpuMom->Gauge_p(), 0, 4*cpuMom->Volume()*momSiteSize*gaugeParam.cpu_prec);

  memcpy(refMom->Gauge_p(), cpuMom->Gauge_p(), 4*cpuMom->Volume()*momSiteSize*gaugeParam.cpu_prec);
    
  gParam.precision = gaugeParam.cuda_prec;
  cudaMom = new cudaGaugeField(gParam);
    
  hw = malloc(4*cpuGauge->Volume()*hwSiteSize*gaugeParam.cpu_prec);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);	
  }
  createHwCPU(hw, hw_prec);

  cudaHw = createHwQuda(gaugeParam.X, hw_prec);
    
  return;
}

static void 
fermion_force_end() 
{
  delete cudaMom;
  delete cudaGauge;
  
  delete cpuGauge;
  delete cpuMom;
  delete refMom;

  freeHwQuda(cudaHw);
  free(hw);
  
  endQuda();
}


static int 
fermion_force_test(void) 
{
 
  fermion_force_init();
  initLatticeConstants(*cudaGauge);
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
    
  // download the momentum field to the GPU
  cudaMom->loadCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  // download the gauge field to the GPU
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);

  loadHwToGPU(cudaHw, hw, cpu_hw_prec);

    
  if (verify_results){	
    fermion_force_reference(eps, weight1, weight2, act_path_coeff, hw, cpuGauge->Gauge_p(), refMom->Gauge_p());
  }
    
    
  /*
   * The flops number comes from CPU implementation in MILC
   * function eo_fermion_force_twoterms_field(), fermion_force_asqtad.c
   *
   */
  int flops = 433968;

  struct timeval t0, t1;
  cudaDeviceSynchronize();    

  gettimeofday(&t0, NULL);
  fermion_force_cuda(eps, weight1, weight2, act_path_coeff, cudaHw, *cudaGauge, *cudaMom, &gaugeParam);
  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);
  double secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
    
  // copy the new momentum back on the CPU
  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  int res;
  res = compare_floats(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume()*momSiteSize, 1e-5, gaugeParam.cpu_prec);
    
  int accuracy_level;
  accuracy_level =  strong_check_mom(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume(), gaugeParam.cpu_prec);
  
  printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
    
  int volume = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3];
  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printf("GPU runtime =%.2f ms, kernel performance= %.2f GFLOPS\n", secs*1000, perf);
    
  fermion_force_end();

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
  printf("running the following fermion force computation test:\n");
    
  printf("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension\n");
  printf("%s                       %s                         %d/%d/%d                  %d \n", 
	 get_prec_str(link_prec),
	 get_recon_str(link_recon), 
	 xdim, ydim, zdim, tdim);
  return ;
    
}

void
usage_extra(char** argv )
{
  printf("Extra options: \n");
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

#ifdef MULTI_GPU
    initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
#endif

  link_prec = prec;

  display_test_info();
  
  int accuracy_level = fermion_force_test();
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
