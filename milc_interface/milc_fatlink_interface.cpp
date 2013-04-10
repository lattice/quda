#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <string.h>
#include <quda.h>
#include <util_quda.h>
#include <test_util.h>
#include "../tests/blas_reference.h"
#include "../tests/staggered_dslash_reference.h"
#include <gauge_field.h>
#include "external_headers/quda_milc_interface.h"

#include "include/milc_utilities.h"

#include "include/milc_timer.h"




// No need to do this if I return a pointer


namespace milc_interface {

void copyGaugeField(int volume, QudaPrecision prec, void* src, void* dst)
{
  const int realSize = getRealSize(prec);
  const int siteSize = 18;
  memcpy(dst, src, 4*volume*siteSize*realSize);

  return;
}


void assignQDPGaugeField(const int dim[4], QudaPrecision precision, void* src, void** dst)
{
  const int matrix_size = 18*getRealSize(precision);
  const int volume = getVolume(dim);

  for(int i=0; i<volume; ++i){
    for(int dir=0; dir<4; ++dir){
	    char* dst_ptr = (char*)dst[dir];
	    memcpy(dst_ptr + i*matrix_size, (char*)src + (i*4 + dir)*matrix_size, matrix_size);
    } // end loop over directions
  } // loop over the extended volume
  return;
}

} // namespace milc_interface





void qudaLoadFatLink(int precision, QudaFatLinkArgs_t fatlink_args, const double act_path_coeff[6], void* inlink, void* outlink)
{
  using namespace milc_interface;

  milc_interface::Timer timer("qudaLoadFatLink");
#ifndef TIME_INTERFACE
  timer.mute();
#endif
  const bool usePinnedMemory = (fatlink_args.use_pinned_memory) ? true : false;

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  GridInfo local_latt_info(local_dim);
  const int volume = local_latt_info.getVolume();
#ifdef MULTI_GPU  
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
  //QudaComputeFatMethod method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
#else
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
#endif

  const QudaPrecision prec = (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
#ifdef MULTI_GPU
  void* local_inlink[4];
  if(method == QUDA_COMPUTE_FAT_STANDARD){
    for(int dir=0; dir<4; ++dir){
      allocateColorField(volume, prec, usePinnedMemory, local_inlink[dir]);
    }
    assignQDPGaugeField(local_dim, prec, inlink, local_inlink);
  }else if(method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME){	
    int extended_dim[4] = {local_dim[0]+4, local_dim[1]+4, local_dim[2]+4, local_dim[3]+4};
    for(int dir=0; dir<4; ++dir) allocateColorField(getVolume(extended_dim), prec, usePinnedMemory, local_inlink[dir]);
    assignExtendedQDPGaugeField(local_dim, prec, inlink, local_inlink);
  }
#else
  void* local_inlink = inlink;
#endif

  void* local_outlink;
  if(usePinnedMemory){
    allocateColorField(4*volume, prec, usePinnedMemory, local_outlink); 
  }else{
    local_outlink = outlink;
  }

  QudaGaugeParam param = newQudaGaugeParam();
  // Make sure all parameters are initialised, even those that aren't needed
  for(int dir=0; dir<4; ++dir) param.X[dir] = local_dim[dir];
  param.t_boundary  	   = QUDA_PERIODIC_T;
  param.anisotropy  	   = 1.0;
  param.cuda_prec_sloppy   = prec;
  param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  param.gauge_fix   	   = QUDA_GAUGE_FIXED_NO;
  param.ga_pad      	   = 0;
  param.gaugeGiB    	   = 0;


  param.type        = QUDA_WILSON_LINKS; // an unfortunate misnomer
  param.reconstruct = QUDA_RECONSTRUCT_NO; // change this so it is read in at run time
  param.cpu_prec    = prec;
  param.cuda_prec   = prec;
#ifdef MULTI_GPU
  param.gauge_order = QUDA_QDP_GAUGE_ORDER; // In the next release, we will fix this.
#else 
  param.gauge_order = QUDA_MILC_GAUGE_ORDER;
#endif

  timer.check("Setup and data load");
  computeFatLinkQuda(local_outlink, (void**)local_inlink, const_cast<double*>(act_path_coeff), &param, method);

  if(usePinnedMemory){
    copyGaugeField(volume, prec, local_outlink, outlink);
    cudaFreeHost(local_outlink);
  }

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir){
   if(usePinnedMemory){
     cudaFreeHost(local_inlink[dir]);
   }else{
     free(local_inlink[dir]);
   }     
  }
#endif


  return;
}


#include <llfat_quda.h>
#include <hisq_links_quda.h>
#include <dslash_quda.h> // for initCommonConstants
#include <fat_force_quda.h> // for loadLinkToGPU and loadLinkToGPU_ex

// I need to be careful here about using pinned memory. 
// If I use pinned memory, then I need to explicitly copy the input and output data arrays. 
// Otherwise, I can just copy pointers.
void qudaLoadUnitarizedLink(int precision, QudaFatLinkArgs_t fatlink_args, const double path_coeff[6], void* inlink, void* fatlink, void* ulink)
{
  using namespace milc_interface;

  milc_interface::Timer timer("qudaLoadUnitarizedLink");
#ifndef TIME_INTERFACE
  timer.mute();
#endif
  //printf(" %s enters\n", __FUNCTION__);

  // Initialize unitarization parameters
  {
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    quda::setUnitarizeLinksConstants(unitarize_eps, max_error,
			       reunit_allow_svd, reunit_svd_only,
			       svd_rel_error, svd_abs_error);
  }

  const bool usePinnedMemory = (fatlink_args.use_pinned_memory) ? true : false;

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  GridInfo local_latt_info(local_dim);
  const int volume = local_latt_info.getVolume();

  const QudaPrecision prec = (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;


  QudaComputeFatMethod method;

#ifdef MULTI_GPU
//  method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
  method = QUDA_COMPUTE_FAT_STANDARD;
  void* local_inlink[4];
  if(method == QUDA_COMPUTE_FAT_STANDARD){
    for(int dir=0; dir<4; ++dir) allocateColorField(volume, prec, usePinnedMemory, local_inlink[dir]);
    assignQDPGaugeField(local_dim, prec, inlink, local_inlink);
  }else if(method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME){
    int extended_dim[4] = {local_dim[0]+4, local_dim[1]+4, local_dim[2]+4, local_dim[3]+4};
    for(int dir=0; dir<4; ++dir) allocateColorField(getVolume(extended_dim), prec, usePinnedMemory, local_inlink[dir]);
    assignExtendedQDPGaugeField(local_dim, prec, inlink, local_inlink);
  }
#else
  method = QUDA_COMPUTE_FAT_STANDARD;
  void* local_inlink = inlink;
#endif

  // Allocate memory for the fatlink and unitarized link if 
  // necessary. Otherwise assign pointers.
  void* local_fatlink;
  void* local_ulink;
  if(usePinnedMemory){
    allocateColorField(4*volume, prec, usePinnedMemory, local_fatlink); 
    allocateColorField(4*volume, prec, usePinnedMemory, local_ulink); 
  }else{
    local_fatlink = fatlink;
    local_ulink   = ulink;
  }

  QudaGaugeParam param = newQudaGaugeParam();
  // Make sure all parameters are initialised, even those that aren't needed
  for(int dir=0; dir<4; ++dir) param.X[dir] = local_dim[dir];
  param.t_boundary  	   = QUDA_PERIODIC_T;
  param.anisotropy  	   = 1.0;
  param.cuda_prec_sloppy   = prec;
  param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  param.gauge_fix   	   = QUDA_GAUGE_FIXED_NO;
  param.ga_pad      	   = 0;
  param.gaugeGiB    	   = 0;
  param.preserve_gauge     = false;


  param.type        = QUDA_WILSON_LINKS; // an unfortunate misnomer
  param.reconstruct = (fatlink_args.su3_source) ? QUDA_RECONSTRUCT_12 : QUDA_RECONSTRUCT_NO;
  param.cpu_prec    = prec;
  param.cuda_prec   = prec;
#ifdef MULTI_GPU
  param.gauge_order = QUDA_QDP_GAUGE_ORDER; // In the next release, we will fix this.
#else 
  param.gauge_order = QUDA_MILC_GAUGE_ORDER;
#endif


  // Much of this code was previously in computeFatLinkQuda
  // I have added cpuUnitarizeLink mainly for clarity. 
  // It only points to local_ulink
  static cpuGaugeField* cpuFatLink=NULL, *cpuInLink=NULL,    *cpuUnitarizedLink=NULL;
  static cudaGaugeField* cudaFatLink=NULL, *cudaInLink=NULL, *cudaUnitarizedLink=NULL;
  const int preserve_gauge = param.preserve_gauge;

  QudaGaugeParam qudaGaugeParam_ex_buf;
  QudaGaugeParam* qudaGaugeParam_ex = &qudaGaugeParam_ex_buf;
  memcpy(qudaGaugeParam_ex, &param, sizeof(QudaGaugeParam));

  for(int dir=0; dir<4; ++dir){ qudaGaugeParam_ex->X[dir] = param.X[dir]+4; }


  // fat-link padding
  setFatLinkPadding(method, &param);
  qudaGaugeParam_ex->llfat_ga_pad = param.llfat_ga_pad;
  qudaGaugeParam_ex->staple_pad   = param.staple_pad;
  qudaGaugeParam_ex->site_ga_pad  = param.site_ga_pad;
  
  GaugeFieldParam gParam(0, param);


  // create the host fatlink
  if(cpuFatLink == NULL && fatlink != NULL){
    gParam.create = QUDA_REFERENCE_FIELD_CREATE;
    gParam.link_type = QUDA_GENERAL_LINKS;
    gParam.order = QUDA_MILC_GAUGE_ORDER;
    gParam.gauge = local_fatlink;
    cpuFatLink = new cpuGaugeField(gParam);
    if(cpuFatLink == NULL){
      errorQuda("ERROR: Creating cpuFatLink failed\n");
    }
  }else if(fatlink != NULL){
    cpuFatLink->setGauge((void**)local_fatlink);
  }

   // create the host fatlink
  if(cpuUnitarizedLink == NULL){
    gParam.create = QUDA_REFERENCE_FIELD_CREATE;
    gParam.link_type = QUDA_GENERAL_LINKS;
    gParam.order = QUDA_MILC_GAUGE_ORDER;
    gParam.gauge = local_ulink;
    cpuUnitarizedLink = new cpuGaugeField(gParam);
    if(cpuUnitarizedLink == NULL){
      errorQuda("ERROR: Creating cpuFatLink failed\n");
    }
  }else{
    cpuUnitarizedLink->setGauge((void**)local_ulink);
  }

  // create the device fatlink 
  if(cudaFatLink == NULL){
    gParam.pad    = param.llfat_ga_pad;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.link_type = QUDA_GENERAL_LINKS;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    cudaFatLink = new cudaGaugeField(gParam);
  }
  
  // create the device unitarize link - same format as the fatlink
  if(cudaUnitarizedLink == NULL){
    gParam.pad    = param.llfat_ga_pad;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.link_type = QUDA_GENERAL_LINKS;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    cudaUnitarizedLink = new cudaGaugeField(gParam);
  }

  quda::setUnitarizeLinksPadding(param.llfat_ga_pad,param.llfat_ga_pad);

  // create the host sitelink	
  if(cpuInLink == NULL){
    gParam.pad = 0; 
    gParam.create    = QUDA_REFERENCE_FIELD_CREATE;
    gParam.link_type = param.type;
    gParam.order     = param.gauge_order;
    gParam.gauge     = local_inlink;
    if(method != QUDA_COMPUTE_FAT_STANDARD){
      for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam_ex->X[dir];	
    }
    cpuInLink      = new cpuGaugeField(gParam);
    if(cpuInLink == NULL){
      errorQuda("ERROR: Creating cpuInLink failed\n");
    }
  }else{
    cpuInLink->setGauge((void**)local_inlink);
  }

  if(cudaInLink == NULL){
    gParam.pad         = param.site_ga_pad;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = param.type;
    gParam.reconstruct = param.reconstruct;      
    cudaInLink = new cudaGaugeField(gParam);
  }

#define QUDA_VER ((10000*QUDA_VERSION_MAJOR) + (100*QUDA_VERSION_MINOR) + QUDA_VERSION_SUBMINOR)
#if (QUDA_VER > 400)
  initLatticeConstants(*cudaFatLink);
#else
  initCommonConstants(*cudaFatLink);
#endif

  if(method == QUDA_COMPUTE_FAT_STANDARD){
    llfat_init_cuda(&param);
    param.ga_pad = param.site_ga_pad;
    if(param.gauge_order == QUDA_QDP_GAUGE_ORDER){
      loadLinkToGPU(cudaInLink, cpuInLink, &param);
    }else{
#ifdef MULTI_GPU
      errorQuda("Only QDP-ordered site links are supported in the multi-gpu standard fattening code\n");
#else
      cudaInLink->loadCPUField(*cpuInLink, QUDA_CPU_FIELD_LOCATION);
#endif
    }
  }else{
    llfat_init_cuda_ex(qudaGaugeParam_ex);
	
#ifdef MULTI_GPU
    int R[4] = {2, 2, 2, 2}; 
    exchange_cpu_sitelink_ex(param.X, R, (void**)cpuInLink->Gauge_p(), QUDA_QDP_GAUGE_ORDER, param.cpu_prec, 1);
#endif
    qudaGaugeParam_ex->ga_pad = qudaGaugeParam_ex->site_ga_pad;
    if(param.gauge_order == QUDA_QDP_GAUGE_ORDER){ 
      loadLinkToGPU_ex(cudaInLink, cpuInLink);
    }else{
      cudaInLink->loadCPUField(*cpuInLink, QUDA_CPU_FIELD_LOCATION);
    }
  } // Initialise and load siteLinks

  timer.check("Setup and data load");
  //printf(" %s starts to compute\n", __FUNCTION__);
  // time the subroutines in computeFatLinkCore
  struct timeval time_array[4];
  // Actually do the fattening
  computeFatLinkCore(cudaInLink, const_cast<double*>(path_coeff), &param, method, cudaFatLink, time_array);

  timer.check("computeFatLinkCore");
 
  //printf(" %s finished compute\n", __FUNCTION__);

  int num_failures=0;
  int* num_failures_dev;
  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));

  if(num_failures_dev == NULL){
    errorQuda("cudaMalloc fialed for dev_pointer\n");
  }
  timer.check();
  quda::unitarizeLinksCuda(param, *cudaFatLink, cudaUnitarizedLink, num_failures_dev); // unitarize on the gpu

  timer.check("unitarizeLinksCuda");
  
  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(num_failures_dev); 

  if(num_failures>0){
    errorQuda("Error in the unitarization component of the hisq fattening\n"); 
    exit(1);
  } 




  // copy the fatlink back to the cpu
  if(fatlink != NULL){
#ifdef MULTI_GPU
    storeLinkToCPU(cpuFatLink, cudaFatLink, &param);
#else
    cudaFatLink->saveCPUField(*cpuFatLink, QUDA_CPU_FIELD_LOCATION);
#endif
    cudaThreadSynchronize(); checkCudaError();
    if(usePinnedMemory) copyGaugeField(volume, prec, local_fatlink, fatlink); 
  }
  // copy the unitarized link back to the cpu

#ifdef MULTI_GPU
  storeLinkToCPU(cpuUnitarizedLink, cudaUnitarizedLink, &param); 
#else
  cudaUnitarizedLink->saveCPUField(*cpuUnitarizedLink, QUDA_CPU_FIELD_LOCATION);
#endif
  cudaThreadSynchronize(); checkCudaError();
  if(usePinnedMemory) copyGaugeField(volume, prec, local_ulink, ulink);


  if (!(preserve_gauge & QUDA_FAT_PRESERVE_CPU_GAUGE) ){
    if(cpuFatLink){ delete cpuFatLink; cpuFatLink = NULL; } 
    delete cpuInLink; cpuInLink = NULL;
    delete cpuUnitarizedLink; cpuUnitarizedLink = NULL;
  }  

  if (!(preserve_gauge & QUDA_FAT_PRESERVE_GPU_GAUGE) ){
    delete cudaFatLink; cudaFatLink = NULL;
    delete cudaInLink; cudaInLink = NULL;
    delete cudaUnitarizedLink; cudaUnitarizedLink = NULL;
   }
  
  if(usePinnedMemory){
    if(fatlink != NULL) cudaFreeHost(local_fatlink);
    cudaFreeHost(local_ulink);
  }

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir){
    if(usePinnedMemory){
      cudaFreeHost(local_inlink[dir]);
    }else{
      free(local_inlink[dir]);
    }
  }
#endif

  return;
}
