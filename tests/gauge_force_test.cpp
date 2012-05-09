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
#include <dslash_quda.h>

#ifdef MULTI_GPU
#include <face_quda.h>
#endif

extern int device;

static QudaGaugeParam qudaGaugeParam;
QudaGaugeFieldOrder gauge_order =  QUDA_QDP_GAUGE_ORDER;
static int verify_results = 0;
extern int tdim;
extern QudaPrecision prec;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern void usage(char** argv);
extern bool tune;

int attempts = 1;

extern QudaReconstructType link_recon;
QudaPrecision  link_prec = QUDA_SINGLE_PRECISION;

extern int gridsize_from_cmdline[];


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



static int
gauge_force_test(void) 
{
  int max_length = 6;    
  
  initQuda(device);
  
  qudaGaugeParam = newQudaGaugeParam();
  
  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;
  
  setDims(qudaGaugeParam.X);
  
  qudaGaugeParam.anisotropy = 1.0;
  qudaGaugeParam.cpu_prec = link_prec;
  qudaGaugeParam.cuda_prec = link_prec;
  qudaGaugeParam.reconstruct = link_recon;  
  qudaGaugeParam.type = QUDA_WILSON_LINKS; // in this context, just means these are site links   
  
  qudaGaugeParam.gauge_order = gauge_order;
  
  int gSize = qudaGaugeParam.cpu_prec;
    
  void* sitelink;  
  void* sitelink_1d;
  
#ifdef GPU_DIRECT
  if (cudaMallocHost(&sitelink_1d, 4*V*gaugeSiteSize*gSize) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for sitelink_1d\n");
  }
#else
  sitelink_1d= malloc(4*V*gaugeSiteSize*gSize);
#endif
  if(sitelink_1d == NULL){
    printf("ERROR: malloc failed for sitelink_1d\n");
    exit(1);
  }
  
  // this is a hack to have site link generated in 2d 
  // then copied to 1d array in "MILC" format
  void* sitelink_2d[4];
  for(int i=0;i < 4;i++){
#ifdef GPU_DIRECT
    if(cudaMallocHost(&sitelink_2d[i], V*gaugeSiteSize*qudaGaugeParam.cpu_prec) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for sitelink_2d\n");
  }
#else
    sitelink_2d[i] = malloc(V*gaugeSiteSize*qudaGaugeParam.cpu_prec);
#endif
  }
  
  // fills the gauge field with random numbers
  createSiteLinkCPU(sitelink_2d, qudaGaugeParam.cpu_prec, 0);
  
  //copy the 2d sitelink to 1d milc format 
  
  for(int dir = 0; dir < 4; dir++){
    for(int i=0; i < V; i++){
      char* src =  ((char*)sitelink_2d[dir]) + i * gaugeSiteSize* qudaGaugeParam.cpu_prec;
      char* dst =  ((char*)sitelink_1d) + (4*i+dir)*gaugeSiteSize*qudaGaugeParam.cpu_prec ;
      memcpy(dst, src, gaugeSiteSize*qudaGaugeParam.cpu_prec);
    }
  }
  if (qudaGaugeParam.gauge_order ==  QUDA_MILC_GAUGE_ORDER){ 
    sitelink =  sitelink_1d;    
  }else{ //QUDA_QDP_GAUGE_ORDER
    sitelink = (void**)sitelink_2d;
  }  
  
#ifdef MULTI_GPU
  void* sitelink_ex;
  void* sitelink_ex_2d[4];
  void* sitelink_ex_1d;

  if (cudaMallocHost((void**)&sitelink_ex_1d, 4*V_ex*gaugeSiteSize*gSize) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for sitelink_ex_1d\n");
  }
  for(int i=0;i < 4;i++){
    if (cudaMallocHost((void**)&sitelink_ex_2d[i], V_ex*gaugeSiteSize*gSize) == cudaErrorMemoryAllocation) {
      errorQuda("ERROR: cudaMallocHost failed for sitelink_ex_2d\n");
    }
    if(sitelink_ex_2d[i] == NULL){
      errorQuda("ERROR; allocate sitelink_ex[%d] failed\n", i);
    }
  }

  int X1= Z[0];
  int X2= Z[1];
  int X3= Z[2];
  int X4= Z[3];

  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }
    
    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;

    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
      continue;
    }
    
    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;
    
    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)sitelink_2d[dir];
      char* dst = (char*)sitelink_ex_2d[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir
  }//i
  
  
  for(int dir = 0; dir < 4; dir++){
    for(int i=0; i < V_ex; i++){
      char* src =  ((char*)sitelink_ex_2d[dir]) + i * gaugeSiteSize* qudaGaugeParam.cpu_prec;
      char* dst =  ((char*)sitelink_ex_1d) + (4*i+dir)*gaugeSiteSize*qudaGaugeParam.cpu_prec ;
      memcpy(dst, src, gaugeSiteSize*qudaGaugeParam.cpu_prec);
    }
  }
  
  if(qudaGaugeParam.gauge_order == QUDA_QDP_GAUGE_ORDER){
    sitelink_ex = sitelink_ex_2d;
  }else{
    sitelink_ex = sitelink_ex_1d;
  }

#endif
  


  void* mom = malloc(4*V*momSiteSize*gSize);
  void* refmom = malloc(4*V*momSiteSize*gSize);
  if(mom == NULL || refmom == NULL){
    printf("ERROR: malloc failed for mom/refmom\n");
    exit(1);
  }    
  memset(mom, 0, 4*V*momSiteSize*gSize);
  //initiaze some data in cpuMom
  createMomCPU(mom,  qudaGaugeParam.cpu_prec);      
  memcpy(refmom, mom, 4*V*momSiteSize*qudaGaugeParam.cpu_prec);
  
  
  double loop_coeff_d[sizeof(loop_coeff_f)/sizeof(float)];
  for(unsigned int i=0;i < sizeof(loop_coeff_f)/sizeof(float); i++){
    loop_coeff_d[i] = loop_coeff_f[i];
  }
    
  void* loop_coeff;
  if(qudaGaugeParam.cuda_prec == QUDA_SINGLE_PRECISION){
    loop_coeff = (void*)&loop_coeff_f[0];
  }else{
    loop_coeff = loop_coeff_d;
  }
  double eb3 = 0.3;
  int num_paths = sizeof(path_dir_x)/sizeof(path_dir_x[0]);
  
  int** input_path_buf[4];
  for(int dir =0; dir < 4; dir++){
    input_path_buf[dir] = (int**)malloc(num_paths*sizeof(int*));
    if (input_path_buf[dir] == NULL){
      printf("ERORR: malloc failed for input path\n");
      exit(1);
    }
    
    for(int i=0;i < num_paths;i++){
      input_path_buf[dir][i] = (int*)malloc(length[i]*sizeof(int));
      if (input_path_buf[dir][i] == NULL){
	printf("ERROR: malloc failed for input_path_buf[dir][%d]\n", i);
	exit(1);
      }
      if(dir == 0) memcpy(input_path_buf[dir][i], path_dir_x[i], length[i]*sizeof(int));
      else if(dir ==1) memcpy(input_path_buf[dir][i], path_dir_y[i], length[i]*sizeof(int));
      else if(dir ==2) memcpy(input_path_buf[dir][i], path_dir_z[i], length[i]*sizeof(int));
      else if(dir ==3) memcpy(input_path_buf[dir][i], path_dir_t[i], length[i]*sizeof(int));
    }
  }

  if (tune) {
    printfQuda("Tuning...\n");
    setDslashTuning(QUDA_TUNE_YES, QUDA_VERBOSE);
  }
  
  struct timeval t0, t1;
  double timeinfo[3];
  /* Multiple execution to exclude warmup time in the first run*/
  for (int i =0;i < attempts; i++){
    gettimeofday(&t0, NULL);
#ifdef MULTI_GPU
    computeGaugeForceQuda(mom, sitelink_ex,  input_path_buf, length,
			  loop_coeff, num_paths, max_length, eb3,
			  &qudaGaugeParam, timeinfo);
    
#else
    computeGaugeForceQuda(mom, sitelink,  input_path_buf, length,
			  loop_coeff, num_paths, max_length, eb3,
			  &qudaGaugeParam, timeinfo);
#endif  
    gettimeofday(&t1, NULL);
  }
  
  double total_time = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  //The number comes from CPU implementation in MILC, gauge_force_imp.c
  int flops=153004;
    
  if (verify_results){	
    for(int i = 0;i < attempts;i++){
#ifdef MULTI_GPU
      //last arg=0 means no optimization for communication, i.e. exchange data in all directions 
      //even they are not partitioned
      int R[4] = {2, 2, 2, 2};
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, (void**)sitelink_ex_2d,
			       QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0);    
      gauge_force_reference(refmom, eb3, sitelink_2d, sitelink_ex_2d, qudaGaugeParam.cpu_prec,
			    input_path_buf, length, loop_coeff, num_paths);
#else
      gauge_force_reference(refmom, eb3, sitelink_2d, NULL, qudaGaugeParam.cpu_prec,
			    input_path_buf, length, loop_coeff, num_paths);
#endif
    }
  }
  
  int res;
  res = compare_floats(mom, refmom, 4*V*momSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
  
  int accuracy_level;
  accuracy_level = strong_check_mom(mom, refmom, 4*V, qudaGaugeParam.cpu_prec);
  
  printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
  
  double perf = 1.0* flops*V/(total_time*1e+9);
  double kernel_perf = 1.0*flops*V/(timeinfo[1]*1e+9);
  printf("init and cpu->gpu time: %.2f ms, kernel time: %.2f ms, gpu->cpu and cleanup time: %.2f  total time =%.2f ms\n", 
	 timeinfo[0]*1e+3, timeinfo[1]*1e+3, timeinfo[2]*1e+3, total_time*1e+3);
  printf("kernel performance: %.2f GFLOPS, overall performance : %.2f GFOPS\n", kernel_perf, perf);
  
  for(int dir = 0; dir < 4; dir++){
    for(int i=0;i < num_paths; i++){
      free(input_path_buf[dir][i]);
    }
    free(input_path_buf[dir]);
  }
  
#ifdef GPU_DIRECT  
  cudaFreeHost(sitelink_1d);
#else
  free(sitelink_1d);
#endif
  for(int dir=0;dir < 4;dir++){
#ifdef GPU_DIRECT
    cudaFreeHost(sitelink_2d[dir]);
#else
    free(sitelink_2d[dir]);
#endif
  }
  
#ifdef MULTI_GPU  
  cudaFreeHost(sitelink_ex_1d);
  for(int dir=0; dir < 4; dir++){
    cudaFreeHost(sitelink_ex_2d[dir]);
  }
#endif


  free(mom);
  free(refmom);
  endQuda();
  
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
    
    printf("link_precision           link_reconstruct           space_dim(x/y/z)              T_dimension        Gauge_order    Attempts\n");
    printf("%s                       %s                         %d/%d/%d                       %d                  %s           %d\n",  
	   get_prec_str(link_prec),
	   get_recon_str(link_recon), 
	   xdim,ydim,zdim, tdim, 
	   get_gauge_order_str(gauge_order),
	   attempts);
    return ;
    
}

void
usage_extra(char** argv )
{
  printf("Extra options:\n");
  printf("    --gauge-order  <qdp/milc>                 # Gauge storing order in CPU\n");
  printf("    --attempts  <n>                           # Number of tests\n");
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

      if( strcmp(argv[i], "--gauge-order") == 0){
	if(i+1 >= argc){
	  usage(argv);
	}
	
	if(strcmp(argv[i+1], "milc") == 0){
	  gauge_order = QUDA_MILC_GAUGE_ORDER;
	}else if(strcmp(argv[i+1], "qdp") == 0){
	  gauge_order = QUDA_QDP_GAUGE_ORDER;
	}else{
	  fprintf(stderr, "Error: unsupported gauge-field order\n");
	  exit(1);
	}
	i++;
	continue;
      }
      if( strcmp(argv[i], "--attempts") == 0){
	if(i+1 >= argc){
	  usage(argv);
	}
	
	attempts = atoi(argv[i+1]);
	if(attempts <= 0){
	  printf("ERROR: invalid number of attempts(%d)\n", attempts);
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
