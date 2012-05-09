#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include "test_util.h"
#include "gauge_field.h"
#include "fat_force_quda.h"
#include "misc.h"
#include "hisq_force_reference.h"
#include "hisq_force_quda.h"
#include "hisq_force_utils.h"
#include "hw_quda.h"
#include <sys/time.h>

using namespace quda::fermion_force;
extern void usage(char** argv);
static int device = 0;
cudaGaugeField *cudaGauge = NULL;
cpuGaugeField  *cpuGauge  = NULL;

cudaGaugeField *cudaForce = NULL;
cpuGaugeField  *cpuForce = NULL;

cpuGaugeField *cpuReference = NULL;

static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static void* hw; // the array of half_wilson_vector


cpuGaugeField *cpuOprod = NULL;
cudaGaugeField *cudaOprod = NULL;


int verify_results = 0;
int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
QudaPrecision hw_prec = QUDA_SINGLE_PRECISION;
QudaPrecision cpu_hw_prec = QUDA_SINGLE_PRECISION;
QudaPrecision mom_prec = QUDA_SINGLE_PRECISION;

void setPrecision(QudaPrecision precision)
{
  link_prec   = precision;
  hw_prec     = precision;
  cpu_hw_prec = precision;
  mom_prec    = precision;
  
  return;
}


void initLatticeConstants(const LatticeField &lat);
void initGaugeConstants(const cudaGaugeField &gauge);


// allocate memory
// set the layout, etc.
static void
hisq_force_init()
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

  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  GaugeFieldParam gParam(0, gaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;

  cpuGauge = new cpuGaugeField(gParam);

  // this is a hack to get the gauge field to appear as a void** rather than void*
  void* siteLink_2d[4];
    for(int i=0;i < 4;i++){
       siteLink_2d[i] = ((char*)cpuGauge->Gauge_p()) + i*cpuGauge->Volume()* gaugeSiteSize* gaugeParam.cpu_prec;
     }

  // fills the gauge field with random numbers
  createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec, 0);

  gParam.precision = gaugeParam.cuda_prec;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGauge = new cudaGaugeField(gParam);

  // create the force matrix
  // cannot reconstruct, since the force matrix is not in SU(3)
  gParam.precision = gaugeParam.cpu_prec;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuForce = new cpuGaugeField(gParam); 
  memset(cpuForce->Gauge_p(), 0, 4*cpuForce->Volume()*gaugeSiteSize*gaugeParam.cpu_prec);

  cpuReference = new cpuGaugeField(gParam);
  memset(cpuReference->Gauge_p(), 0, 4*cpuReference->Volume()*gaugeSiteSize*gaugeParam.cpu_prec);
  


  gParam.precision = gaugeParam.cuda_prec;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForce = new cudaGaugeField(gParam); 
  cudaMemset((void**)(cudaForce->Gauge_p()), 0, cudaForce->Bytes());

  cudaForce->loadCPUField(*cpuForce, QUDA_CPU_FIELD_LOCATION);
  


  gParam.precision = gaugeParam.cuda_prec;

  hw = malloc(4*cpuGauge->Volume()*hwSiteSize*gaugeParam.cpu_prec);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);
  }

  createHwCPU(hw, hw_prec);
  cudaHw = createHwQuda(gaugeParam.X, hw_prec);



  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.precision = gaugeParam.cpu_prec;
  cpuOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuOprod->Gauge_p(), hw_prec, QUDA_MILC_GAUGE_ORDER);


  gParam.precision = hw_prec;
  cudaOprod = new cudaGaugeField(gParam);

  checkCudaError();	
  
  return;
}


static void 
hisq_force_end()
{
  delete cudaForce;
  delete cudaOprod;
  delete cudaGauge;
  freeHwQuda(cudaHw);
  
  delete cpuGauge;
  delete cpuForce;
  delete cpuOprod;
  delete cpuReference;
  free(hw);

  endQuda();
  return;
}

static void 
hisq_force_test()
{
  hisq_force_init();

  initLatticeConstants(*cudaGauge);
  initGaugeConstants(*cudaGauge);
  hisqForceInitCuda(&gaugeParam);

  float act_path_coeff[6];
  act_path_coeff[0] = 0.625000;
  // act_path_coeff[1] = -0.058479;
  act_path_coeff[1] = 0.0; // set Naik term to zero, temporarily
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;

  double d_act_path_coeff[6];
  for(int i=0; i<6; ++i){
    d_act_path_coeff[i] = act_path_coeff[i];
  }


  // copy the gauge field to the GPU
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);

  // copy the outer product field to the GPU
  cudaOprod->loadCPUField(*cpuOprod, QUDA_CPU_FIELD_LOCATION);

  loadHwToGPU(cudaHw, hw, cpu_hw_prec);


  const double unitarize_eps = 1e-5;
  const double hisq_force_filter = 5e-5;
  const double max_det_error = 1e-12;
  const bool allow_svd = true;
  const bool svd_only = false;
  const double svd_rel_err = 1e-8;
  const double svd_abs_err = 1e-8;

  setUnitarizeForceConstants(unitarize_eps, hisq_force_filter, max_det_error, allow_svd, svd_only, svd_rel_err, svd_abs_err);
  
  // First of all we fatten the links on the GPU
  hisqStaplesForceCuda(d_act_path_coeff, gaugeParam, *cudaOprod, *cudaGauge, cudaForce);

  cudaDeviceSynchronize();


  checkCudaError();
  cudaForce->saveCPUField(*cpuForce, QUDA_CPU_FIELD_LOCATION);


  int* unitarization_failed_dev; 
  cudaMalloc((void**)&unitarization_failed_dev, sizeof(int));

  unitarizeForceCuda(gaugeParam, *cudaForce, *cudaGauge, cudaOprod, unitarization_failed_dev); // output is written to cudaOprod.
  if(verify_results){
    unitarizeForceCPU(gaugeParam, *cpuForce, *cpuGauge, cpuOprod);
  }
  cudaDeviceSynchronize();
  checkCudaError();
  cudaFree(unitarization_failed_dev);

  cudaOprod->saveCPUField(*cpuReference, QUDA_CPU_FIELD_LOCATION); 

  int res;
  res = compare_floats(cpuOprod->Gauge_p(), cpuOprod->Gauge_p(), 4*cpuReference->Volume()*gaugeSiteSize, 1e-5, gaugeParam.cpu_prec);

  printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");

  hisq_force_end();
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

  setPrecision(prec);

  display_test_info();
    
  hisq_force_test();


#ifdef MULTI_GPU
    endCommsQuda();
#endif
    
    
  return EXIT_SUCCESS;
}

