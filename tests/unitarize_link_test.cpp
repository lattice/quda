#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quda.h"
#include "gauge_field.h"
#include "test_util.h"
#include "llfat_reference.h"
#include "misc.h"
#include "util_quda.h"
#include "llfat_quda.h"
#include "fat_force_quda.h"
#include "hisq_links_quda.h"
#include "dslash_quda.h"
#include "hisq_force_quda.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "comm_quda.h"
#endif

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

extern void usage(char** argv);
static int verify_results = 0;

extern int device;
int Z[4];
int V;
int Vh;
int Vs[4];
int Vsh[4];

static int V_ex;
static int Vh_ex;


static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-5;
static double max_allowed_error = 1e-12;
static bool check_unitarization = true;


extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
static QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

static size_t gSize;

void
setDims(const int *X) {
  V = 1;
  for (int dir=0; dir<4; ++dir) {
    V *= X[dir];
    Z[dir] = X[dir];
  }
  Vh = V/2;

  Vs[0] =  X[1]*X[2]*X[3];
  Vs[1] =  X[0]*X[2]*X[3];
  Vs[2] =  X[0]*X[1]*X[3];
  Vs[3] =  X[0]*X[1]*X[2];
  for(int dir=0; dir<4; ++dir) Vsh[dir] = Vs[dir]/2;
  V_ex = 1;
  for (int d=0; d< 4; d++) {
    V_ex *= X[d]+4;
  }
  Vh_ex = V_ex/2;
  return;
}

static int
unitarize_link_test()
{

  QudaGaugeParam qudaGaugeParam = newQudaGaugeParam();

  initQuda(0);

  cpu_prec = prec;
  gSize = cpu_prec;  
  qudaGaugeParam.anisotropy = 1.0;
  
  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);
  
  QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
  QudaReconstructType link_recon = QUDA_RECONSTRUCT_NO;

  qudaGaugeParam.cpu_prec  = link_prec;
  qudaGaugeParam.cuda_prec = link_prec;
  qudaGaugeParam.reconstruct = link_recon;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;


  hisq::fermion_force::hisqForceInitCuda(&qudaGaugeParam);
  
  qudaGaugeParam.t_boundary  	   = QUDA_PERIODIC_T;
  qudaGaugeParam.anisotropy  	   = 1.0;
  qudaGaugeParam.cuda_prec_sloppy   = prec;
  qudaGaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  qudaGaugeParam.gauge_fix   	   = QUDA_GAUGE_FIXED_NO;
  qudaGaugeParam.ga_pad      	   = 0;
  qudaGaugeParam.packed_size 	   = 0;
  qudaGaugeParam.gaugeGiB    	   = 0;
  qudaGaugeParam.flag              = false;

   
  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = prec;
  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.type=QUDA_WILSON_LINKS;
  qudaGaugeParam.reconstruct = link_recon;
  qudaGaugeParam.flag = QUDA_FAT_PRESERVE_CPU_GAUGE
    | QUDA_FAT_PRESERVE_GPU_GAUGE
    | QUDA_FAT_PRESERVE_COMM_MEM;

  setFatLinkPadding(QUDA_COMPUTE_FAT_STANDARD, &qudaGaugeParam);
 
  GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.pad = 0;
  gParam.create    = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = QUDA_WILSON_LINKS;
  gParam.order     = QUDA_MILC_GAUGE_ORDER;
  cpuGaugeField *cpuOutLink  = new cpuGaugeField(gParam);

  gParam.pad         = 0;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = QUDA_WILSON_LINKS;
  gParam.order       = QUDA_QDP_GAUGE_ORDER;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGaugeField *cudaFatLink = new cudaGaugeField(gParam);
  cudaGaugeField *cudaULink   = new cudaGaugeField(gParam);  

  initCommonConstants(*cudaFatLink);

  void* fatlink = (void*)malloc(4*V*gaugeSiteSize*gSize);
  if(fatlink == NULL){
    errorQuda("ERROR: allocating fatlink failed\n");
  }
  
  void* sitelink[4];
  for(int i=0;i < 4;i++){
    cudaMallocHost((void**)&sitelink[i], V*gaugeSiteSize*gSize);
    if(sitelink[i] == NULL){
      errorQuda("ERROR; allocate sitelink[%d] failed\n", i);
    }
  }
  
  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 1);

  double act_path_coeff[6];
  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;


  //only record the last call's performance
  //the first one is for creating the cpu/cuda data structures
  
  if(gauge_order == QUDA_QDP_GAUGE_ORDER){
    computeFatLinkQuda(fatlink, sitelink, act_path_coeff, &qudaGaugeParam,
			   QUDA_COMPUTE_FAT_STANDARD);
  } // gauge order is QDP_GAUGE_ORDER

  cpuOutLink->setGauge((void**)fatlink);
  cudaFatLink->loadCPUField(*cpuOutLink, QUDA_CPU_FIELD_LOCATION);
 

 
  hisq::setUnitarizeLinksConstants(unitarize_eps,
				   max_allowed_error,
				   reunit_allow_svd,
				   reunit_svd_only,
				   svd_rel_error,
				   svd_abs_error);
 
  hisq::setUnitarizeLinksPadding(0,0);

  int* num_failures_dev;
  if(cudaMalloc(&num_failures_dev, sizeof(int)) != cudaSuccess){
	errorQuda("cudaMallo failed for num_failures_dev\n");
  }
  cudaMemset(num_failures_dev, 0, sizeof(int));

  struct timeval t0, t1;

  gettimeofday(&t0,NULL);
  hisq::unitarizeLinksCuda(qudaGaugeParam,*cudaFatLink, cudaULink, num_failures_dev);
  cudaThreadSynchronize();
  gettimeofday(&t1,NULL);

  int num_failures=0;
  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);

 delete cudaFatLink;
 delete cudaULink;
 for(int dir=0; dir<4; ++dir) cudaFreeHost(sitelink[dir]);
  cudaFree(num_failures_dev); 
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif
  endQuda();
   
  printfQuda("Unitarization time: %g ms\n", TDIFF(t0,t1)*1000); 
  return num_failures;
}

static void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       algorithm     max allowed error\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d               %s             %g \n", 
	     get_prec_str(prec),
	     get_recon_str(link_recon), 
	     xdim, ydim, zdim, tdim, 
	     get_unitarization_str(reunit_svd_only),
	     max_allowed_error);

#ifdef MULTI_GPU
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
             commDimPartitioned(0),
             commDimPartitioned(1),
             commDimPartitioned(2),
             commDimPartitioned(3));
#endif

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
  for (i=1; i<argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  display_test_info();
  int num_failures = unitarize_link_test();
  printfQuda("Number of failures = %d\n", num_failures);
  if(num_failures > 0){
    printfQuda("Failure rate = %lf%\n", num_failures/(4.0*V));
    printfQuda("You may want to increase your error tolerance or vary the unitarization parameters\n");
  }
  endCommsQuda();

  return EXIT_SUCCESS;
}


