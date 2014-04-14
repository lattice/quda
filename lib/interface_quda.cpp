#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <quda.h>
#include <quda_fortran.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <fat_force_quda.h>
#include <hisq_links_quda.h>
#include <algorithm>
#include <staggered_oprod.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>

#ifdef NUMA_AFFINITY
#include <numa_affinity.h>
#endif

#include <cuda.h>
#include "face_quda.h"

#ifdef MULTI_GPU
extern void exchange_cpu_sitelink_ex(int* X, int *R, void** sitelink, QudaGaugeFieldOrder cpu_order,
    QudaPrecision gPrecision, int optflag, int geom);
#endif // MULTI_GPU

#include <ks_force_quda.h>

#ifdef GPU_GAUGE_FORCE
#include <gauge_force_quda.h>
#endif
#include <gauge_update_quda.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

#define spinorSiteSize 24 // real numbers per spinor

#define MAX_GPU_NUM_PER_NODE 16

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM


int numa_affinity_enabled = 1;

using namespace quda;

cudaGaugeField *gaugePrecise = NULL;
cudaGaugeField *gaugeSloppy = NULL;
cudaGaugeField *gaugePrecondition = NULL;

// It's important that these alias the above so that constants are set correctly in Dirac::Dirac()
cudaGaugeField *&gaugeFatPrecise = gaugePrecise;
cudaGaugeField *&gaugeFatSloppy = gaugeSloppy;
cudaGaugeField *&gaugeFatPrecondition = gaugePrecondition;

cudaGaugeField *gaugeLongPrecise = NULL;
cudaGaugeField *gaugeLongSloppy = NULL;
cudaGaugeField *gaugeLongPrecondition = NULL;

cudaCloverField *cloverPrecise = NULL;
cudaCloverField *cloverSloppy = NULL;
cudaCloverField *cloverPrecondition = NULL;

cudaCloverField *cloverInvPrecise = NULL;
cudaCloverField *cloverInvSloppy = NULL;
cudaCloverField *cloverInvPrecondition = NULL;

cudaGaugeField *momResident = NULL;
cudaGaugeField *extendedGaugeResident = NULL;

cudaDeviceProp deviceProp;
cudaStream_t *streams;

static bool initialized = false;

//!< Profiler for initQuda
static TimeProfile profileInit("initQuda");

//!< Profile for loadGaugeQuda / saveGaugeQuda
static TimeProfile profileGauge("loadGaugeQuda");

//!< Profile for loadCloverQuda
static TimeProfile profileClover("loadCloverQuda");

//!< Profiler for invertQuda
static TimeProfile profileInvert("invertQuda");

//!< Profiler for invertMultiShiftQuda
static TimeProfile profileMulti("invertMultiShiftQuda");

//!< Profiler for invertMultiShiftMixedQuda
static TimeProfile profileMultiMixed("invertMultiShiftMixedQuda");

//!< Profiler for computeFatLinkQuda
static TimeProfile profileFatLink("computeKSLinkQuda");

//!< Profiler for computeGaugeForceQuda
static TimeProfile profileGaugeForce("computeGaugeForceQuda");

//!<Profiler for updateGaugeFieldQuda 
static TimeProfile profileGaugeUpdate("updateGaugeFieldQuda");

//!<Profiler for createExtendedGaugeField
static TimeProfile profileExtendedGauge("createExtendedGaugeField");


//!<Profiler for createClover>
static TimeProfile profileCloverCreate("createCloverQuda");

//!<Profiler for computeCloverDerivative
static TimeProfile profileCloverDerivative("computeCloverDerivativeQuda");

//!<Profiler for computeCloverSigmaTrace
static TimeProfile profileCloverTrace("computeCloverTraceQuda");

//!<Profiler for computeStaggeredOprodQuda
static TimeProfile profileStaggeredOprod("computeStaggeredOprodQuda");

//!<Profiler for computeAsqtadForceQuda
static TimeProfile profileAsqtadForce("computeAsqtadForceQuda");

//!<Profiler for computeAsqtadForceQuda
static TimeProfile profileHISQForce("computeHISQForceQuda");

//!<Profiler for computeHISQForceCompleteQuda
static TimeProfile profileHISQForceComplete("computeHISQForceCompleteQuda");

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");

void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[], FILE *outfile)
{
  setVerbosity(verbosity);
  setOutputPrefix(prefix);
  setOutputFile(outfile);
}


typedef struct {
  int ndim;
  int dims[QUDA_MAX_DIM];
} LexMapData;

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */
static int lex_rank_from_coords(const int *coords, void *fdata)
{
  LexMapData *md = static_cast<LexMapData *>(fdata);

  int rank = coords[0];
  for (int i = 1; i < md->ndim; i++) {
    rank = md->dims[i] * rank + coords[i];
  }
  return rank;
}

#ifdef QMP_COMMS
/**
 * For QMP, we use the existing logical topology if already declared.
 */
static int qmp_rank_from_coords(const int *coords, void *fdata)
{
  return QMP_get_node_number_from(coords);
}
#endif


static bool comms_initialized = false;

void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)
{
  if (nDim != 4) {
    errorQuda("Number of communication grid dimensions must be 4");
  }

  if (!func) {

#if QMP_COMMS
    if (QMP_logical_topology_is_declared()) {
      if (QMP_get_logical_number_of_dimensions() != 4) {
        errorQuda("QMP logical topology must have 4 dimensions");
      }
      for (int i=0; i<nDim; i++) {
        int qdim = QMP_get_logical_dimensions()[i];
        if(qdim != dims[i]) {
          errorQuda("QMP logical dims[%d]=%d does not match dims[%d]=%d argument", i, qdim, i, dims[i]);
        }
      }
      fdata = NULL;
      func = qmp_rank_from_coords;
    } else {
      warningQuda("QMP logical topology is undeclared; using default lexicographical ordering");
#endif

      LexMapData map_data;
      map_data.ndim = nDim;
      for (int i=0; i<nDim; i++) {
        map_data.dims[i] = dims[i];
      }
      fdata = (void *) &map_data;
      func = lex_rank_from_coords;

#if QMP_COMMS
    }
#endif      

  }
  comm_init(nDim, dims, func, fdata);
  comms_initialized = true;
}


static void init_default_comms()
{
#if defined(QMP_COMMS)
  if (QMP_logical_topology_is_declared()) {
    int ndim = QMP_get_logical_number_of_dimensions();
    const int *dims = QMP_get_logical_dimensions();
    initCommsGridQuda(ndim, dims, NULL, NULL);
  } else {
    errorQuda("initQuda() called without prior call to initCommsGridQuda(),"
        " and QMP logical topology has not been declared");
  }
#elif defined(MPI_COMMS)
  errorQuda("When using MPI for communications, initCommsGridQuda() must be called before initQuda()");
#else // single-GPU
  const int dims[4] = {1, 1, 1, 1};
  initCommsGridQuda(4, dims, NULL, NULL);
#endif
}


/*
 * Set the device that QUDA uses.
 */
void initQudaDevice(int dev) {

  //static bool initialized = false;
  if (initialized) return;
  initialized = true;

#if defined(GPU_DIRECT) && defined(MULTI_GPU) && (CUDA_VERSION == 4000)
  //check if CUDA_NIC_INTEROP is set to 1 in the enviroment
  // not needed for CUDA >= 4.1
  char* cni_str = getenv("CUDA_NIC_INTEROP");
  if(cni_str == NULL){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set");
  }
  int cni_int = atoi(cni_str);
  if (cni_int != 1){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set to 1");    
  }
#endif

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    errorQuda("No CUDA devices found");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProp, i);
    checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Found device %d: %s\n", i, deviceProp.name);
    }
  }

#ifdef MULTI_GPU
  if (dev < 0) {
    if (!comms_initialized) {
      errorQuda("initDeviceQuda() called with a negative device ordinal, but comms have not been initialized");
    }
    dev = comm_gpuid();
  }
#else
  if (dev < 0 || dev >= 16) errorQuda("Invalid device number %d", dev);
#endif

  cudaGetDeviceProperties(&deviceProp, dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    printfQuda("Using device %d: %s\n", dev, deviceProp.name);
  }
#ifndef USE_QDPJIT
  cudaSetDevice(dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
#endif

#ifdef NUMA_AFFINITY
  if(numa_affinity_enabled){
    setNumaAffinity(dev);
  }
#endif

  // if the device supports host-mapped memory, then enable this
#ifndef USE_QDPJIT
  if(deviceProp.canMapHostMemory) cudaSetDeviceFlags(cudaDeviceMapHost);
  checkCudaError();
#endif

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaGetDeviceProperties(&deviceProp, dev);
}


/*
 * Any persistent memory allocations that QUDA uses are done here.
 */
void initQudaMemory()
{
  if (!comms_initialized) init_default_comms();

  streams = new cudaStream_t[Nstream];

#if (CUDA_VERSION >= 5050)
  int greatestPriority;
  int leastPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

  for (int i=0; i<Nstream-1; i++) {
    cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, greatestPriority);
  }
  cudaStreamCreateWithPriority(&streams[Nstream-1], cudaStreamDefault, leastPriority);
#else
  for (int i=0; i<Nstream; i++) {
    cudaStreamCreate(&streams[i]);
  }
#endif

  checkCudaError();
  createDslashEvents();
  createStaggeredOprodEvents();  

  initBlas();

  loadTuneCache(getVerbosity());
}


void initQuda(int dev)
{
  profileInit.Start(QUDA_PROFILE_TOTAL);

  // initialize communications topology, if not already done explicitly via initCommsGridQuda()
  if (!comms_initialized) init_default_comms();

  // set the device that QUDA uses
  initQudaDevice(dev);

  // set the persistant memory allocations that QUDA uses (Blas, streams, etc.)
  initQudaMemory();

  profileInit.Stop(QUDA_PROFILE_TOTAL);
}


void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  //printfQuda("loadGaugeQuda use_resident_gauge = %d phase=%d\n", 
  //param->use_resident_gauge, param->staggered_phase_applied);

  profileGauge.Start(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  checkGaugeParam(param);

  profileGauge.Start(QUDA_PROFILE_INIT);  
  // Set the specific input parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);

  // if we are using half precision then we need to compute the fat
  // link maximum while still on the cpu
  // FIXME get a kernel for this
  if ((param->cuda_prec == QUDA_HALF_PRECISION ||
        param->cuda_prec_sloppy == QUDA_HALF_PRECISION ||
        param->cuda_prec_precondition == QUDA_HALF_PRECISION) &&
      param->type == QUDA_ASQTAD_FAT_LINKS)
    gauge_param.compute_fat_link_max = true;

  GaugeField *in = (param->location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<GaugeField*>(new cpuGaugeField(gauge_param)) : 
    static_cast<GaugeField*>(new cudaGaugeField(gauge_param));

  // if not preserving then copy the gauge field passed in
  cudaGaugeField *precise = NULL;
  // switch the parameters for creating the mirror precise cuda gauge field
  gauge_param.create = QUDA_NULL_FIELD_CREATE;
  gauge_param.precision = param->cuda_prec;
  gauge_param.reconstruct = param->reconstruct;
  gauge_param.pad = param->ga_pad;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

  precise = new cudaGaugeField(gauge_param);

  if (param->use_resident_gauge) {
    if(gaugePrecise == NULL) errorQuda("No resident gauge field");
    // copy rather than point at to ensure that the padded region is filled in
    precise->copy(*gaugePrecise);
    precise->exchangeGhost();
    delete gaugePrecise;
    gaugePrecise = NULL;
    profileGauge.Stop(QUDA_PROFILE_INIT);  
  } else {
    profileGauge.Stop(QUDA_PROFILE_INIT);  
    profileGauge.Start(QUDA_PROFILE_H2D);  
    precise->copy(*in);
    profileGauge.Stop(QUDA_PROFILE_H2D);  
  }

  param->gaugeGiB += precise->GBytes();

  // creating sloppy fields isn't really compute, but it is work done on the gpu
  profileGauge.Start(QUDA_PROFILE_COMPUTE); 

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.precision = param->cuda_prec_sloppy;
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
      gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *sloppy = NULL;
  if (param->cuda_prec != param->cuda_prec_sloppy) {
    sloppy = new cudaGaugeField(gauge_param);
    sloppy->copy(*precise);
    param->gaugeGiB += sloppy->GBytes();
  } else {
    sloppy = precise;
  }

  // switch the parameters for creating the mirror preconditioner cuda gauge field
  gauge_param.precision = param->cuda_prec_precondition;
  gauge_param.reconstruct = param->reconstruct_precondition;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
      gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *precondition = NULL;
  if (param->cuda_prec_sloppy != param->cuda_prec_precondition) {
    precondition = new cudaGaugeField(gauge_param);
    precondition->copy(*sloppy);
    param->gaugeGiB += precondition->GBytes();
  } else {
    precondition = sloppy;
  }

  profileGauge.Stop(QUDA_PROFILE_COMPUTE); 

  switch (param->type) {
    case QUDA_WILSON_LINKS:
      //if (gaugePrecise) errorQuda("Precise gauge field already allocated");
      gaugePrecise = precise;
      //if (gaugeSloppy) errorQuda("Sloppy gauge field already allocated");
      gaugeSloppy = sloppy;
      //if (gaugePrecondition) errorQuda("Precondition gauge field already allocated");
      gaugePrecondition = precondition;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      if (gaugeFatPrecise) errorQuda("Precise gauge fat field already allocated");
      gaugeFatPrecise = precise;
      if (gaugeFatSloppy) errorQuda("Sloppy gauge fat field already allocated");
      gaugeFatSloppy = sloppy;
      if (gaugeFatPrecondition) errorQuda("Precondition gauge fat field already allocated");
      gaugeFatPrecondition = precondition;
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      if (gaugeLongPrecise) errorQuda("Precise gauge long field already allocated");
      gaugeLongPrecise = precise;
      if (gaugeLongSloppy) errorQuda("Sloppy gauge long field already allocated");
      gaugeLongSloppy = sloppy;
      if (gaugeLongPrecondition) errorQuda("Precondition gauge long field already allocated");
      gaugeLongPrecondition = precondition;
      break;
    default:
      errorQuda("Invalid gauge type");   
  }

  profileGauge.Start(QUDA_PROFILE_FREE);  
  delete in;
  profileGauge.Stop(QUDA_PROFILE_FREE);  

  profileGauge.Stop(QUDA_PROFILE_TOTAL);
}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.Start(QUDA_PROFILE_TOTAL);

  if (param->location != QUDA_CPU_FIELD_LOCATION) 
    errorQuda("Non-cpu output location not yet supported");

  if (!initialized) errorQuda("QUDA not initialized");
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);
  cpuGaugeField cpuGauge(gauge_param);
  cudaGaugeField *cudaGauge = NULL;
  switch (param->type) {
    case QUDA_WILSON_LINKS:
      cudaGauge = gaugePrecise;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      cudaGauge = gaugeFatPrecise;
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      cudaGauge = gaugeLongPrecise;
      break;
    default:
      errorQuda("Invalid gauge type");   
  }

  profileGauge.Start(QUDA_PROFILE_D2H);  
  cudaGauge->saveCPUField(cpuGauge, QUDA_CPU_FIELD_LOCATION);
  profileGauge.Stop(QUDA_PROFILE_D2H);  

  profileGauge.Stop(QUDA_PROFILE_TOTAL);
}


void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  profileClover.Start(QUDA_PROFILE_TOTAL);

  bool device_calc = false; // calculate clover and inverse on the device?

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if (!initialized) errorQuda("QUDA not initialized");

  if (!h_clover && !h_clovinv) {
    printfQuda("clover_coeff: %lf\n", inv_param->clover_coeff);
    if(inv_param->clover_coeff != 0){
      device_calc = true;
    }else{
      errorQuda("loadCloverQuda() called with neither clover term nor inverse");
    }
  }


  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }
  if (gaugePrecise == NULL) {
    errorQuda("Gauge field must be loaded before clover");
  }
  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH)) {
    errorQuda("Wrong dslash_type in loadCloverQuda()");
  }

  // determines whether operator is preconditioned when calling invertQuda()
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE ||
      inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);

  // determines whether operator is preconditioned when calling MatQuda() or MatDagMatQuda()
  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool asymmetric = (inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
      inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  // We issue a warning only when it seems likely that the user is screwing up:

  // uninverted clover term is required when applying unpreconditioned operator,
  // but note that dslashQuda() is always preconditioned
  if (!h_clover && !pc_solve && !pc_solution) {
    //warningQuda("Uninverted clover term not loaded");
  }

  // uninverted clover term is also required for "asymmetric" preconditioning
  if (!h_clover && pc_solve && pc_solution && asymmetric && !device_calc) {
    warningQuda("Uninverted clover term not loaded");
  }

  CloverFieldParam clover_param;
  CloverField *in, *inInv;

  if(!device_calc){
    // create a param for the cpu clover field
    profileClover.Start(QUDA_PROFILE_INIT);
    CloverFieldParam cpuParam;
    cpuParam.nDim = 4;
    for (int i=0; i<4; i++) cpuParam.x[i] = gaugePrecise->X()[i];
    cpuParam.precision = inv_param->clover_cpu_prec;
    cpuParam.order = inv_param->clover_order;
    cpuParam.direct = h_clover ? true : false;
    cpuParam.inverse = h_clovinv ? true : false;
    cpuParam.clover = h_clover;
    cpuParam.norm = 0;
    cpuParam.cloverInv = h_clovinv;
    cpuParam.invNorm = 0;
    cpuParam.create = QUDA_REFERENCE_FIELD_CREATE;
    cpuParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    cpuParam.twisted = false;
    cpuParam.mu2 = 0.;

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      cpuParam.direct = true;
      cpuParam.inverse = false;
      cpuParam.cloverInv = NULL;
      cpuParam.clover = h_clover;
      in = (inv_param->clover_location == QUDA_CPU_FIELD_LOCATION) ?
        static_cast<CloverField*>(new cpuCloverField(cpuParam)) : 
        static_cast<CloverField*>(new cudaCloverField(cpuParam));

      cpuParam.cloverInv = h_clovinv;
      cpuParam.clover = NULL;
      cpuParam.twisted = true;
      cpuParam.direct = true;
      cpuParam.inverse = false;
      cpuParam.mu2 = 4.*inv_param->kappa*inv_param->kappa*inv_param->mu*inv_param->mu;

      inInv = (inv_param->clover_location == QUDA_CPU_FIELD_LOCATION) ?
        static_cast<CloverField*>(new cpuCloverField(cpuParam)) : 
        static_cast<CloverField*>(new cudaCloverField(cpuParam));
    } else {
      in = (inv_param->clover_location == QUDA_CPU_FIELD_LOCATION) ?
        static_cast<CloverField*>(new cpuCloverField(cpuParam)) : 
        static_cast<CloverField*>(new cudaCloverField(cpuParam));
    }

    clover_param.nDim = 4;
    for (int i=0; i<4; i++) clover_param.x[i] = gaugePrecise->X()[i];
    clover_param.setPrecision(inv_param->clover_cuda_prec);
    clover_param.pad = inv_param->cl_pad;
    clover_param.direct = h_clover ? true : false;
    clover_param.inverse = (h_clovinv || pc_solve) ? true : false;
    clover_param.create = QUDA_NULL_FIELD_CREATE;
    clover_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      clover_param.direct = true;
      clover_param.inverse = false;
      cloverPrecise = new cudaCloverField(clover_param);
      clover_param.direct = false;
      clover_param.inverse = true;
      clover_param.twisted = true;
      cloverInvPrecise = new cudaCloverField(clover_param);
      clover_param.twisted = false;
    } else {
      cloverPrecise = new cudaCloverField(clover_param);
    }

    profileClover.Stop(QUDA_PROFILE_INIT);

    profileClover.Start(QUDA_PROFILE_H2D);
    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      cloverPrecise->copy(*in, false);
      cloverInvPrecise->copy(*in, true);
      cloverInvert(*cloverInvPrecise, inv_param->compute_clover_trlog, QUDA_CUDA_FIELD_LOCATION);
    } else {
      cloverPrecise->copy(*in, h_clovinv ? true : false);
    }

    profileClover.Stop(QUDA_PROFILE_H2D);
  } else {
    profileClover.Start(QUDA_PROFILE_COMPUTE);

    createCloverQuda(inv_param);

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      cloverInvert(*cloverInvPrecise, inv_param->compute_clover_trlog, QUDA_CUDA_FIELD_LOCATION);
      if (inv_param->compute_clover_trlog) {
        inv_param->trlogA[0] = cloverInvPrecise->TrLog()[0];
        inv_param->trlogA[1] = cloverInvPrecise->TrLog()[1];
      }
    }
    profileClover.Stop(QUDA_PROFILE_COMPUTE);
  }

  // inverted clover term is required when applying preconditioned operator
  if ((!h_clovinv && pc_solve) && inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    profileClover.Start(QUDA_PROFILE_COMPUTE);
    cloverInvert(*cloverPrecise, inv_param->compute_clover_trlog, QUDA_CUDA_FIELD_LOCATION);
    profileClover.Stop(QUDA_PROFILE_COMPUTE);
    if (inv_param->compute_clover_trlog) {
      inv_param->trlogA[0] = cloverPrecise->TrLog()[0];
      inv_param->trlogA[1] = cloverPrecise->TrLog()[1];
    }
  }

  if (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH)
    inv_param->cloverGiB = cloverPrecise->GBytes();
  else
    inv_param->cloverGiB = cloverPrecise->GBytes() + cloverInvPrecise->GBytes();

  clover_param.norm    = 0;
  clover_param.invNorm = 0;
  clover_param.mu2 = 0.;
  clover_param.nDim = 4;
  for(int dir=0; dir<4; ++dir) clover_param.x[dir] = gaugePrecise->X()[dir];
  clover_param.pad = inv_param->cl_pad;
  clover_param.siteSubset = QUDA_FULL_SITE_SUBSET;
  clover_param.create = QUDA_NULL_FIELD_CREATE;
  clover_param.direct = true;
  clover_param.inverse = true;

  // create the mirror sloppy clover field
  if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
    profileClover.Start(QUDA_PROFILE_INIT);
    clover_param.setPrecision(inv_param->clover_cuda_prec_sloppy);

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      clover_param.direct = false;
      clover_param.inverse = true;
      clover_param.twisted = true;
      clover_param.mu2 = 4.*inv_param->kappa*inv_param->kappa*inv_param->mu*inv_param->mu;
      cloverInvSloppy = new cudaCloverField(clover_param); 
      cloverInvSloppy->copy(*cloverInvPrecise, true);
      clover_param.direct = true;
      clover_param.inverse = false;
      clover_param.twisted = false;
      cloverSloppy = new cudaCloverField(clover_param); 
      cloverSloppy->copy(*cloverPrecise);
      inv_param->cloverGiB += cloverSloppy->GBytes() + cloverInvSloppy->GBytes();
    } else {
      cloverSloppy = new cudaCloverField(clover_param); 
      cloverSloppy->copy(*cloverPrecise);
      inv_param->cloverGiB += cloverSloppy->GBytes();
    }
    profileClover.Stop(QUDA_PROFILE_INIT);
  } else {
    cloverSloppy = cloverPrecise;
    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
      cloverInvSloppy = cloverInvPrecise;
  }

  // create the mirror preconditioner clover field
  if (inv_param->clover_cuda_prec_sloppy != inv_param->clover_cuda_prec_precondition &&
      inv_param->clover_cuda_prec_precondition != QUDA_INVALID_PRECISION) {
    profileClover.Start(QUDA_PROFILE_INIT);
    clover_param.setPrecision(inv_param->clover_cuda_prec_precondition);
    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      clover_param.direct = true;
      clover_param.inverse = false;
      clover_param.twisted = false;
      cloverPrecondition = new cudaCloverField(clover_param); 
      cloverPrecondition->copy(*cloverSloppy);
      clover_param.direct = false;
      clover_param.inverse = true;
      clover_param.twisted = true;
      cloverInvPrecondition = new cudaCloverField(clover_param); 
      cloverInvPrecondition->copy(*cloverInvSloppy, true);
      inv_param->cloverGiB += cloverPrecondition->GBytes() + cloverInvPrecondition->GBytes();
    } else {
      cloverPrecondition = new cudaCloverField(clover_param);
      cloverPrecondition->copy(*cloverSloppy);
      inv_param->cloverGiB += cloverPrecondition->GBytes();
    }
    profileClover.Stop(QUDA_PROFILE_INIT);
  } else {
    cloverPrecondition = cloverSloppy;
    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
      cloverInvPrecondition = cloverInvSloppy;
  }

  // need to copy back the odd inverse field into the application clover field
  if (!h_clovinv && pc_solve && !device_calc) {
    // copy the inverted clover term into host application order on the device
    clover_param.setPrecision(inv_param->clover_cpu_prec);
    clover_param.direct = false;
    clover_param.inverse = true;
    clover_param.order = inv_param->clover_order;

    // this isn't really "epilogue" but this label suffices
    profileClover.Start(QUDA_PROFILE_EPILOGUE);
    cudaCloverField hack(clover_param);
    hack.copy(*cloverPrecise);
    profileClover.Stop(QUDA_PROFILE_EPILOGUE);

    // copy the odd components into the host application's clover field
    profileClover.Start(QUDA_PROFILE_D2H);
    cudaMemcpy((char*)(in->V(false))+in->Bytes()/2, (char*)(hack.V(true))+hack.Bytes()/2, 
        in->Bytes()/2, cudaMemcpyDeviceToHost);
    profileClover.Stop(QUDA_PROFILE_D2H);

    checkCudaError();
  }

  if(!device_calc)
  {
    delete in; // delete object referencing input field

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
      delete inInv;
  }

  popVerbosity();

  profileClover.Stop(QUDA_PROFILE_TOTAL);
}

void freeGaugeQuda(void) 
{  
  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
  if (gaugePrecise) delete gaugePrecise;

  gaugePrecondition = NULL;
  gaugeSloppy = NULL;
  gaugePrecise = NULL;

  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
  if (gaugeLongPrecise) delete gaugeLongPrecise;

  gaugeLongPrecondition = NULL;
  gaugeLongSloppy = NULL;
  gaugeLongPrecise = NULL;

  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
  if (gaugeFatPrecise) delete gaugeFatPrecise;

  gaugeFatPrecondition = NULL;
  gaugeFatSloppy = NULL;
  gaugeFatPrecise = NULL;

  if (extendedGaugeResident) {
    delete extendedGaugeResident;
    extendedGaugeResident = NULL;
  }
}

// just free the sloppy fields used in mixed-precision solvers
void freeSloppyGaugeQuda(void) 
{  
  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;

  gaugePrecondition = NULL;
  gaugeSloppy = NULL;

  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;

  gaugeLongPrecondition = NULL;
  gaugeLongSloppy = NULL;

  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;

  gaugeFatPrecondition = NULL;
  gaugeFatSloppy = NULL;
}


void freeCloverQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");
  if (cloverPrecondition != cloverSloppy && cloverPrecondition) delete cloverPrecondition;
  if (cloverSloppy != cloverPrecise && cloverSloppy) delete cloverSloppy;
  if (cloverPrecise) delete cloverPrecise;

  cloverPrecondition = NULL;
  cloverSloppy = NULL;
  cloverPrecise = NULL;

  if (cloverInvPrecise != NULL) {
     if (cloverInvPrecondition != cloverInvSloppy && cloverInvPrecondition) delete cloverInvPrecondition;
     if (cloverInvSloppy != cloverInvPrecise && cloverInvSloppy) delete cloverInvSloppy;
     if (cloverInvPrecise) delete cloverInvPrecise;

     cloverInvPrecondition = NULL;
     cloverInvSloppy = NULL;
     cloverInvPrecise = NULL;
  }
}

void endQuda(void)
{
  profileEnd.Start(QUDA_PROFILE_TOTAL);

  if (!initialized) return;

  LatticeField::freeBuffer();
  cudaColorSpinorField::freeBuffer();
  cudaColorSpinorField::freeGhostBuffer();
  cpuColorSpinorField::freeGhostBuffer();
  FaceBuffer::flushPinnedCache();
  freeGaugeQuda();
  freeCloverQuda();

  endBlas();

  if (streams) {
    for (int i=0; i<Nstream; i++) cudaStreamDestroy(streams[i]);
    delete []streams;
    streams = NULL;
  }
  destroyDslashEvents();

  destroyStaggeredOprodEvents();

  saveTuneCache(getVerbosity());

#if (!defined(USE_QDPJIT) && !defined(GPU_COMMS))
  // end this CUDA context
  cudaDeviceReset();
#endif

  initialized = false;

  comm_finalize();
  comms_initialized = false;

  profileEnd.Stop(QUDA_PROFILE_TOTAL);

  // print out the profile information of the lifetime of the library
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    profileInit.Print();
    profileGauge.Print();
    profileCloverCreate.Print();
    profileClover.Print();
    profileInvert.Print();
    profileMulti.Print();
    profileMultiMixed.Print();
    profileFatLink.Print();
    profileGaugeForce.Print();
    profileGaugeUpdate.Print();
    profileExtendedGauge.Print();
    profileCloverDerivative.Print();
    profileCloverTrace.Print();
    profileStaggeredOprod.Print();
    profileAsqtadForce.Print();
    profileHISQForce.Print();
    profileEnd.Print();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }

  assertAllMemFree();
}


namespace quda {

  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    double kappa = inv_param->kappa;
    if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      kappa *= gaugePrecise->Anisotropy();
    }

    switch (inv_param->dslash_type) {
      case QUDA_WILSON_DSLASH:
        diracParam.type = pc ? QUDA_WILSONPC_DIRAC : QUDA_WILSON_DIRAC;
        break;
      case QUDA_CLOVER_WILSON_DSLASH:
        diracParam.type = pc ? QUDA_CLOVERPC_DIRAC : QUDA_CLOVER_DIRAC;
        break;
      case QUDA_DOMAIN_WALL_DSLASH:
        diracParam.type = pc ? QUDA_DOMAIN_WALLPC_DIRAC : QUDA_DOMAIN_WALL_DIRAC;
        diracParam.Ls = inv_param->Ls;
        break;
      case QUDA_STAGGERED_DSLASH:
        diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
        break;
      case QUDA_ASQTAD_DSLASH:
        diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
        break;
      case QUDA_TWISTED_MASS_DSLASH:
        diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
        if (inv_param->twist_flavor == QUDA_TWIST_MINUS || inv_param->twist_flavor == QUDA_TWIST_PLUS)  
        {
          diracParam.Ls = 1;
          diracParam.epsilon = 0.0;
        }
        else 
        {
          diracParam.Ls = 2;
          diracParam.epsilon = inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ? inv_param->epsilon : 0.0;
        } 
        break;
      case QUDA_TWISTED_CLOVER_DSLASH:
        diracParam.type = pc ? QUDA_TWISTED_CLOVERPC_DIRAC : QUDA_TWISTED_CLOVER_DIRAC;
        if (inv_param->twist_flavor == QUDA_TWIST_MINUS || inv_param->twist_flavor == QUDA_TWIST_PLUS)  
        {
          diracParam.Ls = 1;
          diracParam.epsilon = 0.0;
        }
        else 
        {
          diracParam.Ls = 2;
          diracParam.epsilon = inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ? inv_param->epsilon : 0.0;
        } 
        break;
      default:
        errorQuda("Unsupported dslash_type %d", inv_param->dslash_type);
    }

    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = gaugePrecise;
    diracParam.fatGauge = gaugeFatPrecise;
    diracParam.longGauge = gaugeLongPrecise;    
    diracParam.clover = cloverPrecise;
    diracParam.cloverInv = cloverInvPrecise;
    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }
  }


  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugeSloppy;
    diracParam.fatGauge = gaugeFatSloppy;
    diracParam.longGauge = gaugeLongSloppy;    
    diracParam.clover = cloverSloppy;
    diracParam.cloverInv = cloverInvSloppy;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }

  }

  // The preconditioner currently mimicks the sloppy operator with no comms
  void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugePrecondition;
    diracParam.fatGauge = gaugeFatPrecondition;
    diracParam.longGauge = gaugeLongPrecondition;    
    diracParam.clover = cloverPrecondition;
    diracParam.cloverInv = cloverInvPrecondition;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 0; // comms are always off
    }
  
    // In the preconditioned staggered CG allow a different dlsash type in the preconditioning
    if(inv_param->inv_type == QUDA_PCG_INVERTER && inv_param->dslash_type == QUDA_ASQTAD_DSLASH
       && inv_param->dslash_type_precondition == QUDA_STAGGERED_DSLASH) {
       diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
       diracParam.gauge = gaugeFatPrecondition;
    }
  }

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    setDiracPreParam(diracPreParam, &param, pc_solve);

    d = Dirac::create(diracParam); // create the Dirac operator   
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
  }

  void massRescale(QudaDslashType dslash_type, double &kappa, double &mass, 
      QudaSolutionType solution_type, 
      QudaMassNormalization mass_normalization, cudaColorSpinorField &b)
  {   
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
      double nin = norm2(b);
      printfQuda("Mass rescale: norm of source in = %g\n", nin);
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH || dslash_type == QUDA_STAGGERED_DSLASH) {
      switch (solution_type) {
        case QUDA_MAT_SOLUTION:
        case QUDA_MATPC_SOLUTION:
          if (mass_normalization == QUDA_KAPPA_NORMALIZATION) axCuda(2.0*mass, b);
          break;
        case QUDA_MATDAG_MAT_SOLUTION:
        case QUDA_MATPCDAG_MATPC_SOLUTION:
          if (mass_normalization == QUDA_KAPPA_NORMALIZATION) axCuda(4.0*mass*mass, b);
          break;
        default:
          errorQuda("Not implemented");
      }
      return;
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    switch (solution_type) {
      case QUDA_MAT_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION ||
            mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          axCuda(2.0*kappa, b);
        }
        break;
      case QUDA_MATDAG_MAT_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION ||
            mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          axCuda(4.0*kappa*kappa, b);
        }
        break;
      case QUDA_MATPC_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION) {
          axCuda(4.0*kappa*kappa, b);
        } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          axCuda(2.0*kappa, b);
        }
        break;
      case QUDA_MATPCDAG_MATPC_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION) {
          axCuda(16.0*pow(kappa,4), b);
        } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          axCuda(4.0*kappa*kappa, b);
        }
        break;
      default:
        errorQuda("Solution type %d not supported", solution_type);
    }

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
      double nin = norm2(b);
      printfQuda("Mass rescale: norm of source out = %g\n", nin);
    }

  }

  void massRescaleCoeff(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
      QudaMassNormalization mass_normalization, double &coeff)
  {    
    if (dslash_type == QUDA_ASQTAD_DSLASH || dslash_type == QUDA_STAGGERED_DSLASH) {
      if (mass_normalization != QUDA_MASS_NORMALIZATION) {
        errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
      }
      return;
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    switch (solution_type) {
      case QUDA_MAT_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION ||
            mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          coeff *= 2.0*kappa;
        }
        break;
      case QUDA_MATDAG_MAT_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION ||
            mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          coeff *= 4.0*kappa*kappa;
        }
        break;
      case QUDA_MATPC_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION) {
          coeff *= 4.0*kappa*kappa;
        } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          coeff *= 2.0*kappa;
        }
        break;
      case QUDA_MATPCDAG_MATPC_SOLUTION:
        if (mass_normalization == QUDA_MASS_NORMALIZATION) {
          coeff*=16.0*pow(kappa,4);
        } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
          coeff*=4.0*kappa*kappa;
        }
        break;
      default:
        errorQuda("Solution type %d not supported", solution_type);
    }

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
  }
}

/*void QUDA_DiracField(QUDA_DiracParam *param) {

  }*/

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{
  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH))) 
    errorQuda("Clover field not allocated");
  if (cloverInvPrecise == NULL && inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    errorQuda("Clover field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1);

  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  if (inv_param->mass_normalization == QUDA_KAPPA_NORMALIZATION && 
      (inv_param->dslash_type == QUDA_STAGGERED_DSLASH ||
       inv_param->dslash_type == QUDA_ASQTAD_DSLASH) ) 
    axCuda(1.0/(2.0*inv_param->mass), in);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    axCuda(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->Dslash(out, in, parity); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH))) 
    errorQuda("Clover field not allocated");
  if (cloverInvPrecise == NULL && inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
        inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH))) 
    errorQuda("Clover field not allocated");
  if (cloverInvPrecise == NULL && inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));  

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  //  double kappa = inv_param->kappa;
  //  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= gaugePrecise->anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(1.0/pow(2.0*kappa,4), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
        inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

quda::cudaGaugeField* checkGauge(QudaInvertParam *param) {
  quda::cudaGaugeField *cudaGauge = NULL;
  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    if (gaugePrecise == NULL) errorQuda("Precise gauge field doesn't exist");
    if (gaugeSloppy == NULL) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == NULL) errorQuda("Precondition gauge field doesn't exist");
    cudaGauge = gaugePrecise;
  } else {
    if (gaugeFatPrecise == NULL) errorQuda("Precise gauge fat field doesn't exist");
    if (gaugeFatSloppy == NULL) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == NULL) errorQuda("Precondition gauge fat field doesn't exist");

    if (gaugeLongPrecise == NULL) errorQuda("Precise gauge long field doesn't exist");
    if (gaugeLongSloppy == NULL) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == NULL) errorQuda("Precondition gauge long field doesn't exist");
    cudaGauge = gaugeFatPrecise;
  }
  return cudaGauge;
}


void cloverQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int inverse)
{
  pushVerbosity(inv_param->verbosity);

  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL) errorQuda("Clover field not allocated");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH))
    errorQuda("Cannot apply the clover term for a non Wilson-clover or Twisted-mass-clover dslash");

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1);

  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    axCuda(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);
	//FIXME: Do we need this for twisted clover???
  DiracCloverPC dirac(diracParam); // create the Dirac operator
  if (!inverse) dirac.Clover(out, in, parity); // apply the clover operator
  else dirac.CloverInv(out, in, parity);

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  /*for (int i=0; i<in_h->Volume(); i++) {
    ((cpuColorSpinorField*)out_h)->PrintVector(i);
    }*/

  delete out_h;
  delete in_h;

  popVerbosity();
}


void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  profileInvert.Start(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  checkInvertParam(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || 
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || 
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || 
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || 
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert.Start(QUDA_PROFILE_H2D);

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = hp_x;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); 

  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
        (param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }

    x = new cudaColorSpinorField(*h_x, cudaParam); // solution  
  } else { // zero initial guess
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam); // solution
  }

  profileInvert.Stop(QUDA_PROFILE_H2D);

  double nb = norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = norm2(*h_b);
    double nh_x = norm2(*h_x);
    double nx = norm2(*x);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    axCuda(1.0/sqrt(nb), *b);
    axCuda(1.0/sqrt(nb), *x);
  }

  setTuning(param->tune);

  dirac.prepare(in, out, *x, *b, param->solution_type);
  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    double nout = norm2(*out);
    printfQuda("Prepared source = %g\n", nin);   
    printfQuda("Prepared solution = %g\n", nout);   
  }

  massRescale(param->dslash_type, param->kappa, param->mass, 
      param->solution_type, param->mass_normalization, *in);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    printfQuda("Prepared source post mass rescale = %g\n", nin);   
  }

  // solution_type specifies *what* system is to be solved.
  // solve_type specifies *how* the system is to be solved.
  //
  // We have the following four cases (plus preconditioned variants):
  //
  // solution_type    solve_type    Effect
  // -------------    ----------    ------
  // MAT              DIRECT        Solve Ax=b
  // MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
  // MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
  // MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
  //
  // We generally require that the solution_type and solve_type
  // preconditioning match.  As an exception, the unpreconditioned MAT
  // solution_type may be used with any solve_type, including
  // DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
  // preconditioned source and reconstruction of the full solution are
  // taken care of by Dirac::prepare() and Dirac::reconstruct(),
  // respectively.

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }

  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }

  if (mat_solution && !direct_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    copyCuda(*in, *out);
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  } else {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = norm2(*x);
    printfQuda("Solution = %g\n",nx);
  }
  dirac.reconstruct(*x, *b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    axCuda(sqrt(nb), *x);
  }

  profileInvert.Start(QUDA_PROFILE_D2H);
  *h_x = *x;
  profileInvert.Stop(QUDA_PROFILE_D2H);

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = norm2(*x);
    double nh_x = norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
  }

  delete h_b;
  delete h_x;
  delete b;
  delete x;

  delete d;
  delete dSloppy;
  delete dPre;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());

  profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

cudaColorSpinorField *solutionResident = NULL;

void invertMDQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  profileInvert.Start(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  checkInvertParam(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || 
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || 
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || 
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || 
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert.Start(QUDA_PROFILE_H2D);

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = hp_x;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); 

  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
        (param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }

    x = new cudaColorSpinorField(*h_x, cudaParam); // solution  
  } else { // zero initial guess
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam); // solution
  }

  profileInvert.Stop(QUDA_PROFILE_H2D);

  double nb = norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = norm2(*h_b);
    double nh_x = norm2(*h_x);
    double nx = norm2(*x);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    axCuda(1.0/sqrt(nb), *b);
    axCuda(1.0/sqrt(nb), *x);
  }

  setTuning(param->tune);

  dirac.prepare(in, out, *x, *b, param->solution_type);
  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    double nout = norm2(*out);
    printfQuda("Prepared source = %g\n", nin);   
    printfQuda("Prepared solution = %g\n", nout);   
  }

  massRescale(param->dslash_type, param->kappa, param->mass, 
      param->solution_type, param->mass_normalization, *in);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    printfQuda("Prepared source post mass rescale = %g\n", nin);   
  }

  // solution_type specifies *what* system is to be solved.
  // solve_type specifies *how* the system is to be solved.
  //
  // We have the following four cases (plus preconditioned variants):
  //
  // solution_type    solve_type    Effect
  // -------------    ----------    ------
  // MAT              DIRECT        Solve Ax=b
  // MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
  // MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
  // MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
  //
  // We generally require that the solution_type and solve_type
  // preconditioning match.  As an exception, the unpreconditioned MAT
  // solution_type may be used with any solve_type, including
  // DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
  // preconditioned source and reconstruction of the full solution are
  // taken care of by Dirac::prepare() and Dirac::reconstruct(),
  // respectively.

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }

  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }

  if (mat_solution && !direct_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    copyCuda(*in, *out);
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  } else {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = norm2(*x);
    printfQuda("Solution = %g\n",nx);
  }
  dirac.reconstruct(*x, *b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    axCuda(sqrt(nb), *x);
  }

  if (solutionResident) 
    delete solutionResident;
  //errorQuda("solutionResident already allocated");
  cudaParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  cudaParam.x[0] *= 2;
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  solutionResident = new cudaColorSpinorField(cudaParam);

  dirac.Dslash(solutionResident->Odd(), solutionResident->Even(), QUDA_ODD_PARITY);

  profileInvert.Start(QUDA_PROFILE_D2H);
  *h_x = *x;
  profileInvert.Stop(QUDA_PROFILE_D2H);

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = norm2(*x);
    double nh_x = norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
  }

  delete h_b;
  delete h_x;
  delete b;
  delete x;

  delete d;
  delete dSloppy;
  delete dPre;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());

  profileInvert.Stop(QUDA_PROFILE_TOTAL);
}


/*! 
 * Generic version of the multi-shift solver. Should work for
 * most fermions. Note that offset[0] is not folded into the mass parameter.
 *
 * At present, the solution_type must be MATDAG_MAT or MATPCDAG_MATPC,
 * and solve_type must be NORMOP or NORMOP_PC.  The solution and solve
 * preconditioning have to match.
 */
void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param)
{
  profileMulti.Start(QUDA_PROFILE_TOTAL);

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  if (param->num_offset > QUDA_MAX_MULTI_SHIFT) 
    errorQuda("Number of shifts %d requested greater than QUDA_MAX_MULTI_SHIFT %d", 
        param->num_offset, QUDA_MAX_MULTI_SHIFT);

  pushVerbosity(param->verbosity);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  if (mat_solution) {
    errorQuda("Multi-shift solver does not support MAT or MATPC solution types");
  }
  if (direct_solve) {
    errorQuda("Multi-shift solver does not support DIRECT or DIRECT_PC solve types");
  }
  if (pc_solution & !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!pc_solution & pc_solve) {
    errorQuda("In multi-shift solver, a preconditioned (PC) solve_type requires a PC solution_type");
  }

  // No of GiB in a checkerboard of a spinor
  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if( !pc_solve) param->spinorGiB *= 2; // Double volume for non PC solve

  // **** WARNING *** this may not match implementation... 
  if( param->inv_type == QUDA_CG_INVERTER ) { 
    // CG-M needs 5 vectors for the smallest shift + 2 for each additional shift
    param->spinorGiB *= (5 + 2*(param->num_offset-1))/(double)(1<<30);
  } else {
    errorQuda("QUDA only currently supports multi-shift CG");
    // BiCGStab-M needs 7 for the original shift + 2 for each additional shift + 1 auxiliary
    // (Jegerlehner hep-lat/9612014 eq (3.13)
    param->spinorGiB *= (7 + 2*(param->num_offset-1))/(double)(1<<30);
  }

  // Timing and FLOP counters
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  for (int i=0; i<param->num_offset-1; i++) {
    for (int j=i+1; j<param->num_offset; j++) {
      if (param->offset[i] > param->offset[j])
        errorQuda("Offsets must be ordered from smallest to largest");
    }
  }

  // Host pointers for x, take a copy of the input host pointers
  void** hp_x;
  hp_x = new void* [ param->num_offset ];

  void* hp_b = _hp_b;
  for(int i=0;i < param->num_offset;i++){
    hp_x[i] = _hp_x[i];
  }

  // Create the matrix.
  // The way this works is that createDirac will create 'd' and 'dSloppy'
  // which are global. We then grab these with references...
  //
  // Balint: Isn't there a nice construction pattern we could use here? This is 
  // expedient but yucky.
  //  DiracParam diracParam; 
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || 
      param->dslash_type == QUDA_STAGGERED_DSLASH){
    param->mass = sqrt(param->offset[0]/4);  
  }

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;

  cudaColorSpinorField *b = NULL;   // Cuda RHS
  cudaColorSpinorField **x = NULL;  // Cuda Solutions

  // Grab the dimension array of the input gauge field.
  const int *X = ( param->dslash_type == QUDA_ASQTAD_DSLASH ) ? 
    gaugeFatPrecise->X() : gaugePrecise->X();

  // This creates a ColorSpinorParam struct, from the host data
  // pointer, the definitions in param, the dimensions X, and whether
  // the solution is on a checkerboard instruction or not. These can
  // then be used as 'instructions' to create the actual
  // ColorSpinorField
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorField **h_x = new ColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  for(int i=0; i < param->num_offset; i++) { 
    cpuParam.v = hp_x[i];
    h_x[i] = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  }

  profileMulti.Start(QUDA_PROFILE_H2D);
  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it
  profileMulti.Stop(QUDA_PROFILE_H2D);

  // Create the solution fields filled with zero
  x = new cudaColorSpinorField* [ param->num_offset ];
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int i=0; i < param->num_offset; i++) { 
    x[i] = new cudaColorSpinorField(cudaParam);
  }

  // Check source norms
  double nb = norm2(*b);
  if (nb==0.0) errorQuda("Solution has zero norm");

  if(getVerbosity() >= QUDA_VERBOSE ) {
    double nh_b = norm2(*h_b);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
  }

  // rescale the source vector to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    axCuda(1.0/sqrt(nb), *b);
  }

  setTuning(param->tune);

  massRescale(param->dslash_type, param->kappa, param->mass,
      param->solution_type, param->mass_normalization, *b);
  double *unscaled_shifts = new double [param->num_offset];
  for(int i=0; i < param->num_offset; i++){ 
    unscaled_shifts[i] = param->offset[i];
    massRescaleCoeff(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, param->offset[i]);
  }

  // use multi-shift CG
  {
    DiracMdagM m(dirac), mSloppy(diracSloppy);
    SolverParam solverParam(*param);
    MultiShiftCG cg_m(m, mSloppy, solverParam, profileMulti);
    cg_m(x, *b);  
    solverParam.updateInvertParam(*param);
  }

  // experimenting with Minimum residual extrapolation
  /*
     cudaColorSpinorField **q = new cudaColorSpinorField* [ param->num_offset ];
     cudaColorSpinorField **z = new cudaColorSpinorField* [ param->num_offset ];
     cudaColorSpinorField tmp(cudaParam);

     for(int i=0; i < param->num_offset; i++) {
     cudaParam.create = QUDA_ZERO_FIELD_CREATE;
     q[i] = new cudaColorSpinorField(cudaParam);
     cudaParam.create = QUDA_COPY_FIELD_CREATE;
     z[i] = new cudaColorSpinorField(*x[i], cudaParam);
     }

     for(int i=0; i < param->num_offset; i++) {
     dirac.setMass(sqrt(param->offset[i]/4));  
     DiracMdagM m(dirac);
     MinResExt mre(m, profileMulti);
     copyCuda(tmp, *b);
     mre(*x[i], tmp, z, q, param -> num_offset);
     dirac.setMass(sqrt(param->offset[0]/4));  
     }

     for(int i=0; i < param->num_offset; i++) {
     delete q[i];
     delete z[i];
     }
     delete []q;
     delete []z;
     */

  // check each shift has the desired tolerance and use sequential CG to refine

  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField r(*b, cudaParam);
  for(int i=0; i < param->num_offset; i++) { 
    double rsd_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->true_res_hq_offset[i] : 0;

    double tol_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->tol_hq_offset[i] : 0;

    // refine if either L2 or heavy quark residual tolerances have not been met
    if (param->true_res_offset[i] > param->tol_offset[i] || rsd_hq > tol_hq) {
      if (getVerbosity() >= QUDA_VERBOSE) 
        printfQuda("Refining shift %d: L2 residual %e / %e, heavy quark %e / %e (actual / requested)\n",
            i, param->true_res_offset[i], param->tol_offset[i], rsd_hq, tol_hq);

      // for staggered the shift is just a change in mass term (FIXME: for twisted mass also)
      if (param->dslash_type == QUDA_ASQTAD_DSLASH || 
          param->dslash_type == QUDA_STAGGERED_DSLASH) { 
        dirac.setMass(sqrt(param->offset[i]/4));  
        diracSloppy.setMass(sqrt(param->offset[i]/4));  
      }

      DiracMdagM m(dirac), mSloppy(diracSloppy);

      // need to curry in the shift if we are not doing staggered
      if (param->dslash_type != QUDA_ASQTAD_DSLASH &&
          param->dslash_type != QUDA_STAGGERED_DSLASH) { 
        m.shift = param->offset[i];
        mSloppy.shift = param->offset[i];
      }

      SolverParam solverParam(*param);
      solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
      solverParam.tol = param->tol_offset[i]; // set L2 tolerance
      solverParam.tol_hq = param->tol_hq_offset[i]; // set heavy quark tolerance

      CG cg(m, mSloppy, solverParam, profileMulti);
      cg(*x[i], *b);        

      solverParam.true_res_offset[i] = solverParam.true_res;
      solverParam.true_res_hq_offset[i] = solverParam.true_res_hq;
      solverParam.updateInvertParam(*param,i);

      if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
          param->dslash_type == QUDA_STAGGERED_DSLASH) { 
        dirac.setMass(sqrt(param->offset[0]/4)); // restore just in case
        diracSloppy.setMass(sqrt(param->offset[0]/4)); // restore just in case
      }
    }
  }

  // restore shifts -- avoid side effects
  for(int i=0; i < param->num_offset; i++) { 
    param->offset[i] = unscaled_shifts[i];
  }

  delete [] unscaled_shifts;

  profileMulti.Start(QUDA_PROFILE_D2H);
  for(int i=0; i < param->num_offset; i++) { 
    if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) { // rescale the solution 
      axCuda(sqrt(nb), *x[i]);
    }

    if (getVerbosity() >= QUDA_VERBOSE){
      double nx = norm2(*x[i]);
      printfQuda("Solution %d = %g\n", i, nx);
    }

    *h_x[i] = *x[i];
  }
  profileMulti.Stop(QUDA_PROFILE_D2H);

  for(int i=0; i < param->num_offset; i++){ 
    delete h_x[i];
    delete x[i];
  }

  delete h_b;
  delete b;

  delete [] h_x;
  delete [] x;

  delete [] hp_x;

  delete d;
  delete dSloppy;
  delete dPre;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());

  profileMulti.Stop(QUDA_PROFILE_TOTAL);
}


/*
 * Hacked multi-shift solver for Wilson RHMC molecular dynamics
 * FIXME!!
 */
void invertMultiShiftMDQuda(void **_hp_xe, void **_hp_xo, void **_hp_ye, void **_hp_yo, 
    void *_hp_b, QudaInvertParam *param)
{
  profileMulti.Start(QUDA_PROFILE_TOTAL);

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  if (param->num_offset > QUDA_MAX_MULTI_SHIFT) 
    errorQuda("Number of shifts %d requested greater than QUDA_MAX_MULTI_SHIFT %d", 
        param->num_offset, QUDA_MAX_MULTI_SHIFT);

  pushVerbosity(param->verbosity);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  if (mat_solution) {
    errorQuda("Multi-shift solver does not support MAT or MATPC solution types");
  }
  if (direct_solve) {
    errorQuda("Multi-shift solver does not support DIRECT or DIRECT_PC solve types");
  }
  if (pc_solution & !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!pc_solution & pc_solve) {
    errorQuda("In multi-shift solver, a preconditioned (PC) solve_type requires a PC solution_type");
  }

  // No of GiB in a checkerboard of a spinor
  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if( !pc_solve) param->spinorGiB *= 2; // Double volume for non PC solve

  // **** WARNING *** this may not match implementation... 
  if( param->inv_type == QUDA_CG_INVERTER ) { 
    // CG-M needs 5 vectors for the smallest shift + 2 for each additional shift
    param->spinorGiB *= (5 + 2*(param->num_offset-1))/(double)(1<<30);
  } else {
    errorQuda("QUDA only currently supports multi-shift CG");
    // BiCGStab-M needs 7 for the original shift + 2 for each additional shift + 1 auxiliary
    // (Jegerlehner hep-lat/9612014 eq (3.13)
    param->spinorGiB *= (7 + 2*(param->num_offset-1))/(double)(1<<30);
  }

  // Timing and FLOP counters
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  for (int i=0; i<param->num_offset-1; i++) {
    for (int j=i+1; j<param->num_offset; j++) {
      if (param->offset[i] > param->offset[j])
        errorQuda("Offsets must be ordered from smallest to largest");
    }
  }

  // Host pointers for x, take a copy of the input host pointers
  void **hp_xe = new void* [ param->num_offset ];
  void **hp_xo = new void* [ param->num_offset ];
  void **hp_ye = new void* [ param->num_offset ];
  void **hp_yo = new void* [ param->num_offset ];

  void* hp_b = _hp_b;
  for(int i=0;i < param->num_offset;i++){
    hp_xe[i] = _hp_xe[i];
    hp_xo[i] = _hp_xo[i];
    hp_ye[i] = _hp_ye[i];
    hp_yo[i] = _hp_yo[i];
  }

  // Create the matrix.
  // The way this works is that createDirac will create 'd' and 'dSloppy'
  // which are global. We then grab these with references...
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || 
      param->dslash_type == QUDA_STAGGERED_DSLASH){
    param->mass = sqrt(param->offset[0]/4);  
  }

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;

  cudaColorSpinorField *b = NULL;   // Cuda RHS
  cudaColorSpinorField **xe = NULL;  // Cuda Solutions
  cudaColorSpinorField *xo, *ye, *yo = NULL;  // Cuda Solutions

  // Grab the dimension array of the input gauge field.
  const int *X = ( param->dslash_type == QUDA_ASQTAD_DSLASH ) ? 
    gaugeFatPrecise->X() : gaugePrecise->X();

  // This creates a ColorSpinorParam struct, from the host data
  // pointer, the definitions in param, the dimensions X, and whether
  // the solution is on a checkerboard instruction or not. These can
  // then be used as 'instructions' to create the actual
  // ColorSpinorField
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorField **h_xe = new ColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  ColorSpinorField **h_xo = new ColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  ColorSpinorField **h_ye = new ColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  ColorSpinorField **h_yo = new ColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  for(int i=0; i < param->num_offset; i++) { 
    cpuParam.v = hp_xe[i];
    h_xe[i] = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    cpuParam.v = hp_xo[i];
    h_xo[i] = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    cpuParam.v = hp_ye[i];
    h_ye[i] = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    cpuParam.v = hp_yo[i];
    h_yo[i] = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  }

  profileMulti.Start(QUDA_PROFILE_H2D);
  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it
  profileMulti.Stop(QUDA_PROFILE_H2D);

  // Create the solution fields filled with zero
  xe = new cudaColorSpinorField* [ param->num_offset ];
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int i=0; i < param->num_offset; i++) { 
    xe[i] = new cudaColorSpinorField(cudaParam);
  }

  xo = new cudaColorSpinorField(cudaParam);
  ye = new cudaColorSpinorField(cudaParam);
  yo = new cudaColorSpinorField(cudaParam);

  // Check source norms
  double nb = norm2(*b);
  if (nb==0.0) errorQuda("Solution has zero norm");

  if(getVerbosity() >= QUDA_VERBOSE ) {
    double nh_b = norm2(*h_b);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
  }

  // rescale the source vector to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    axCuda(1.0/sqrt(nb), *b);
  }

  setTuning(param->tune);

  massRescale(param->dslash_type, param->kappa, param->mass,
      param->solution_type, param->mass_normalization, *b);
  double *unscaled_shifts = new double [param->num_offset];
  for(int i=0; i < param->num_offset; i++){ 
    unscaled_shifts[i] = param->offset[i];
    massRescaleCoeff(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, param->offset[i]);
  }

  // use multi-shift CG
  {
    DiracMdagM m(dirac), mSloppy(diracSloppy);
    SolverParam solverParam(*param);
    MultiShiftCG cg_m(m, mSloppy, solverParam, profileMulti);
    cg_m(xe, *b);  
    solverParam.updateInvertParam(*param);
  }

  // check each shift has the desired tolerance and use sequential CG to refine

  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField r(*b, cudaParam);
  for(int i=0; i < param->num_offset; i++) { 
    double rsd_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->true_res_hq_offset[i] : 0;

    double tol_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->tol_hq_offset[i] : 0;

    // refine if either L2 or heavy quark residual tolerances have not been met
    if (param->true_res_offset[i] > param->tol_offset[i] || rsd_hq > tol_hq) {
      if (getVerbosity() >= QUDA_VERBOSE) 
        printfQuda("Refining shift %d: L2 residual %e / %e, heavy quark %e / %e (actual / requested)\n",
            i, param->true_res_offset[i], param->tol_offset[i], rsd_hq, tol_hq);

      // for staggered the shift is just a change in mass term (FIXME: for twisted mass also)
      if (param->dslash_type == QUDA_ASQTAD_DSLASH || 
          param->dslash_type == QUDA_STAGGERED_DSLASH) { 
        dirac.setMass(sqrt(param->offset[i]/4));  
        diracSloppy.setMass(sqrt(param->offset[i]/4));  
      }

      DiracMdagM m(dirac), mSloppy(diracSloppy);

      // need to curry in the shift if we are not doing staggered
      if (param->dslash_type != QUDA_ASQTAD_DSLASH &&
          param->dslash_type != QUDA_STAGGERED_DSLASH) { 
        m.shift = param->offset[i];
        mSloppy.shift = param->offset[i];
      }

      SolverParam solverParam(*param);
      solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
      solverParam.tol = param->tol_offset[i]; // set L2 tolerance
      solverParam.tol_hq = param->tol_hq_offset[i]; // set heavy quark tolerance

      CG cg(m, mSloppy, solverParam, profileMulti);
      cg(*xe[i], *b);        

      solverParam.updateInvertParam(*param);
      param->true_res_offset[i] = param->true_res;
      param->true_res_hq_offset[i] = param->true_res_hq;

      if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
          param->dslash_type == QUDA_STAGGERED_DSLASH) { 
        dirac.setMass(sqrt(param->offset[0]/4)); // restore just in case
        diracSloppy.setMass(sqrt(param->offset[0]/4)); // restore just in case
      }
    }
  }

  // restore shifts -- avoid side effects
  for(int i=0; i < param->num_offset; i++) { 
    param->offset[i] = unscaled_shifts[i];
  }

  delete [] unscaled_shifts;

  profileMulti.Start(QUDA_PROFILE_D2H);
  for(int i=0; i < param->num_offset; i++) { 
    if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) { // rescale the solution 
      axCuda(sqrt(nb), *xe[i]);
    }

    if (getVerbosity() >= QUDA_VERBOSE){
      double nx = norm2(*xe[i]);
      printfQuda("Solution %d = %g\n", i, nx);
    }

    dirac.Dslash(*xo, *xe[i], QUDA_ODD_PARITY);
    dirac.M(*ye, *xe[i]);
    dirac.Dagger(QUDA_DAG_YES);
    dirac.Dslash(*yo, *ye, QUDA_ODD_PARITY);
    dirac.Dagger(QUDA_DAG_NO);

    *h_xe[i] = *xe[i];
    *h_xo[i] = *xo;
    *h_ye[i] = *ye;
    *h_yo[i] = *yo;
  }
  profileMulti.Stop(QUDA_PROFILE_D2H);

  for(int i=0; i < param->num_offset; i++){ 
    delete h_xe[i];
    delete h_xo[i];
    delete h_ye[i];
    delete h_yo[i];
    delete xe[i];
  }

  delete h_b;
  delete b;

  delete [] h_xe;
  delete [] h_xo;
  delete [] h_ye;
  delete [] h_yo;

  delete [] xe;
  delete xo;
  delete ye;
  delete yo;

  delete [] hp_xe;
  delete [] hp_xo;
  delete [] hp_ye;
  delete [] hp_yo;

  delete d;
  delete dSloppy;
  delete dPre;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());

  profileMulti.Stop(QUDA_PROFILE_TOTAL);
}


#ifdef GPU_FATLINK 
/*   @method  
 *   QUDA_COMPUTE_FAT_STANDARD: standard method (default)
 *   QUDA_COMPUTE_FAT_EXTENDED_VOLUME, extended volume method
 *
 */
#include <sys/time.h>

void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param)
{
  int* X    = param->X;
#ifdef MULTI_GPU
  int Vsh_x = X[1]*X[2]*X[3]/2;
  int Vsh_y = X[0]*X[2]*X[3]/2;
  int Vsh_z = X[0]*X[1]*X[3]/2;
#endif
  int Vsh_t = X[0]*X[1]*X[2]/2;

  int E[4];
  for (int i=0; i<4; i++) E[i] = X[i] + 4;

  // fat-link padding 
  param->llfat_ga_pad = Vsh_t;

  // site-link padding
  if(method ==  QUDA_COMPUTE_FAT_STANDARD) {
#ifdef MULTI_GPU
    int Vh_2d_max = MAX(X[0]*X[1]/2, X[0]*X[2]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[0]*X[3]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[1]*X[2]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[1]*X[3]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[2]*X[3]/2);
    param->site_ga_pad = 3*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + 4*Vh_2d_max;
#else
    param->site_ga_pad = Vsh_t;
#endif
  } else {
    param->site_ga_pad = (E[0]*E[1]*E[2]/2)*3;
  }
  param->ga_pad = param->site_ga_pad;

  // staple padding
  if(method == QUDA_COMPUTE_FAT_STANDARD) {
#ifdef MULTI_GPU
    param->staple_pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
#else
    param->staple_pad = 3*Vsh_t;
#endif
  } else {
    param->staple_pad = (E[0]*E[1]*E[2]/2)*3;
  }

  return;
}


namespace quda {
  void computeFatLinkCore(cudaGaugeField* cudaSiteLink, double* act_path_coeff,
      QudaGaugeParam* qudaGaugeParam, QudaComputeFatMethod method,
      cudaGaugeField* cudaFatLink, 
      cudaGaugeField* cudaLongLink,
      TimeProfile &profile)
  {

    profile.Start(QUDA_PROFILE_INIT);
    const int flag = qudaGaugeParam->preserve_gauge;
    GaugeFieldParam gParam(0,*qudaGaugeParam);

    if (method == QUDA_COMPUTE_FAT_STANDARD) {
      for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam->X[dir];
    } else {
      for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam->X[dir] + 4;
    }

    static cudaGaugeField* cudaStapleField=NULL, *cudaStapleField1=NULL;
    if (cudaStapleField == NULL || cudaStapleField1 == NULL) {
      gParam.pad    = qudaGaugeParam->staple_pad;
      gParam.create = QUDA_NULL_FIELD_CREATE;
      gParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gParam.geometry = QUDA_SCALAR_GEOMETRY; // only require a scalar matrix field for the staple
      gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
#ifdef MULTI_GPU
      if(method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME) gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
#else
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
#endif
      cudaStapleField  = new cudaGaugeField(gParam);
      cudaStapleField1 = new cudaGaugeField(gParam);
    }
    profile.Stop(QUDA_PROFILE_INIT);

    profile.Start(QUDA_PROFILE_COMPUTE);
    if (method == QUDA_COMPUTE_FAT_STANDARD) {
      llfat_cuda(cudaFatLink, cudaLongLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, qudaGaugeParam, act_path_coeff);
    } else { //method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME
      llfat_cuda_ex(cudaFatLink, cudaLongLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, qudaGaugeParam, act_path_coeff);
    }
    profile.Stop(QUDA_PROFILE_COMPUTE);


    profile.Start(QUDA_PROFILE_FREE);
    if (!(flag & QUDA_FAT_PRESERVE_GPU_GAUGE) ){
      delete cudaStapleField; cudaStapleField = NULL;
      delete cudaStapleField1; cudaStapleField1 = NULL;
    }
    profile.Stop(QUDA_PROFILE_FREE);


    return;
  }
} // namespace quda


void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink, double *path_coeff, QudaGaugeParam *param, QudaComputeFatMethod method)
{

  profileFatLink.Start(QUDA_PROFILE_TOTAL);
  profileFatLink.Start(QUDA_PROFILE_INIT);
  // Initialize unitarization parameters
  if(ulink){
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

  cudaGaugeField* cudaFatLink        = NULL;
  cudaGaugeField* cudaLongLink       = NULL;
  cudaGaugeField* cudaUnitarizedLink = NULL;
  cudaGaugeField* cudaInLinkEx       = NULL;

  QudaGaugeParam qudaGaugeParam_ex_buf;
  QudaGaugeParam* qudaGaugeParam_ex = &qudaGaugeParam_ex_buf;
  memcpy(qudaGaugeParam_ex, param, sizeof(QudaGaugeParam));
  for(int dir=0; dir<4; ++dir){ qudaGaugeParam_ex->X[dir] = param->X[dir]+4; }

  // fat-link padding
  setFatLinkPadding(method, param);
  qudaGaugeParam_ex->llfat_ga_pad = param->llfat_ga_pad;
  qudaGaugeParam_ex->staple_pad   = param->staple_pad;
  qudaGaugeParam_ex->site_ga_pad  = param->site_ga_pad;

  GaugeFieldParam gParam(0, *param);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  // create the host fatlink
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.gauge = fatlink;
  cpuGaugeField cpuFatLink(gParam);
  gParam.gauge = longlink;
  cpuGaugeField cpuLongLink(gParam);
  gParam.gauge = ulink;
  cpuGaugeField cpuUnitarizedLink(gParam);

  // create the device fatlink 
  gParam.pad    = param->llfat_ga_pad;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaFatLink = new cudaGaugeField(gParam);
  if(longlink) cudaLongLink = new cudaGaugeField(gParam);
  if(ulink){
    cudaUnitarizedLink = new cudaGaugeField(gParam);
    quda::setUnitarizeLinksPadding(param->llfat_ga_pad,param->llfat_ga_pad);
  }
  // create the host sitelink  
  gParam.pad = 0; 
  gParam.create    = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = param->type;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.gauge     = inlink;
  cpuGaugeField cpuInLink(gParam);


  gParam.pad         = param->site_ga_pad;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param->type;
  gParam.reconstruct = param->reconstruct;    
  gParam.order       = (param->reconstruct == QUDA_RECONSTRUCT_12) ? QUDA_FLOAT4_GAUGE_ORDER : QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField* cudaInLink = new cudaGaugeField(gParam);

  if(method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME){
    for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam_ex->X[dir];
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    cudaInLinkEx = new cudaGaugeField(gParam);
  }

#define QUDA_VER ((10000*QUDA_VERSION_MAJOR) + (100*QUDA_VERSION_MINOR) + QUDA_VERSION_SUBMINOR)
#if (QUDA_VER > 400)
  profileFatLink.Stop(QUDA_PROFILE_INIT);
  initLatticeConstants(*cudaFatLink, profileFatLink);
  profileFatLink.Start(QUDA_PROFILE_INIT);
#else
  initCommonConstants(*cudaFatLink);
#endif

  cudaGaugeField* inlinkPtr;
  if(method == QUDA_COMPUTE_FAT_STANDARD){
    llfat_init_cuda(param);
    param->ga_pad = param->site_ga_pad;
    inlinkPtr = cudaInLink;
  }else{
    llfat_init_cuda_ex(qudaGaugeParam_ex);
    inlinkPtr = cudaInLinkEx;
  }
  profileFatLink.Stop(QUDA_PROFILE_INIT);

  profileFatLink.Start(QUDA_PROFILE_H2D);
  cudaInLink->loadCPUField(cpuInLink, QUDA_CPU_FIELD_LOCATION);
  profileFatLink.Stop(QUDA_PROFILE_H2D);

  if(method != QUDA_COMPUTE_FAT_STANDARD){
    profileFatLink.Start(QUDA_PROFILE_COMMS);
    int R[4] = {2, 2, 2, 2}; 
    copyExtendedGauge(*cudaInLinkEx, *cudaInLink, QUDA_CUDA_FIELD_LOCATION);
#ifdef MULTI_GPU
    cudaInLinkEx->exchangeExtendedGhost(R,true); // instead of exchange_cpu_sitelink_ex 
#endif
    profileFatLink.Stop(QUDA_PROFILE_COMMS);
  } // Initialise and load siteLinks

  quda::computeFatLinkCore(inlinkPtr, const_cast<double*>(path_coeff), param, method, cudaFatLink, cudaLongLink, profileFatLink);

  if(ulink){
    profileFatLink.Start(QUDA_PROFILE_INIT);
    int num_failures=0;
    int* num_failures_dev;
    cudaMalloc((void**)&num_failures_dev, sizeof(int));
    cudaMemset(num_failures_dev, 0, sizeof(int));
    if(num_failures_dev == NULL) errorQuda("cudaMalloc fialed for dev_pointer\n");
    profileFatLink.Stop(QUDA_PROFILE_INIT);

    profileFatLink.Start(QUDA_PROFILE_COMPUTE);
    quda::unitarizeLinksCuda(*param, *cudaFatLink, cudaUnitarizedLink, num_failures_dev); // unitarize on the gpu
    profileFatLink.Stop(QUDA_PROFILE_COMPUTE);


    profileFatLink.Start(QUDA_PROFILE_D2H); 
    cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
    profileFatLink.Stop(QUDA_PROFILE_D2H); 
    cudaFree(num_failures_dev); 
    if(num_failures>0){
      errorQuda("Error in the unitarization component of the hisq fattening\n"); 
      exit(1);
    }
    profileFatLink.Start(QUDA_PROFILE_D2H);
    cudaUnitarizedLink->saveCPUField(cpuUnitarizedLink, QUDA_CPU_FIELD_LOCATION);
    profileFatLink.Stop(QUDA_PROFILE_D2H);
  }

  profileFatLink.Start(QUDA_PROFILE_D2H);
  if(fatlink) cudaFatLink->saveCPUField(cpuFatLink, QUDA_CPU_FIELD_LOCATION);
  if(longlink) cudaLongLink->saveCPUField(cpuLongLink, QUDA_CPU_FIELD_LOCATION);
  profileFatLink.Stop(QUDA_PROFILE_D2H);

  profileFatLink.Start(QUDA_PROFILE_FREE);
  if(longlink) delete cudaLongLink;
  delete cudaFatLink; 
  delete cudaInLink; 
  delete cudaUnitarizedLink; 
  if(cudaInLinkEx) delete cudaInLinkEx; 
  profileFatLink.Stop(QUDA_PROFILE_FREE);

  profileFatLink.Stop(QUDA_PROFILE_TOTAL);

  return;
}

#endif

int getGaugePadding(GaugeFieldParam& param){
  int pad = 0;
#ifdef MULTI_GPU
  int volume = param.x[0]*param.x[1]*param.x[2]*param.x[3];
  int face_size[4];
  for(int dir=0; dir<4; ++dir) face_size[dir] = (volume/param.x[dir])/2;
  pad = *std::max_element(face_size, face_size+4);
#endif

  return pad;
}

#if 0 
  int
computeGaugeForceQuda(void* mom, void* sitelink,  int*** input_path_buf, int* path_length,
    void* loop_coeff, int num_paths, int max_length, double eb3,
    QudaGaugeParam* qudaGaugeParam, double* timeinfo)
{
#ifdef GPU_GAUGE_FORCE

  profileGaugeForce.Start(QUDA_PROFILE_TOTAL);
  profileGaugeForce.Start(QUDA_PROFILE_INIT); 

#ifdef MULTI_GPU
  int E[4];
  QudaGaugeParam qudaGaugeParam_ex_buf;
  QudaGaugeParam* qudaGaugeParam_ex=&qudaGaugeParam_ex_buf;
  memcpy(qudaGaugeParam_ex, qudaGaugeParam, sizeof(QudaGaugeParam));
  for (int d=0; d<4; d++) E[d] = qudaGaugeParam_ex->X[d] = qudaGaugeParam->X[d] + 4;
#endif

  GaugeFieldParam gParam(0, *qudaGaugeParam);
#ifdef MULTI_GPU
  GaugeFieldParam gParam_ex(0, *qudaGaugeParam_ex);
  GaugeFieldParam& gParamSL = gParam_ex;  
#else
  GaugeFieldParam& gParamSL = gParam;
  int* X = qudaGaugeParam->X;
#endif

  gParamSL.pad = 0;
#ifndef MULTI_GPU
  gParamSL.create = QUDA_REFERENCE_FIELD_CREATE;
  gParamSL.gauge = sitelink;
  cpuGaugeField *cpuSiteLink = new cpuGaugeField(gParamSL);
#else
  // need to get host gauge field into extended order (can be MILC or QDP)
  GaugeFieldParam appParam = gParam;
  appParam.order = gParam.order;
  appParam.create = QUDA_REFERENCE_FIELD_CREATE;
  appParam.gauge = sitelink;
  appParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cpuGaugeField appLink(appParam);

  gParamSL.order = QUDA_MILC_GAUGE_ORDER;
  gParamSL.create = QUDA_ZERO_FIELD_CREATE;
  gParamSL.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cpuGaugeField *cpuSiteLink = new cpuGaugeField(gParamSL);

  copyExtendedGauge(*cpuSiteLink, appLink, QUDA_CPU_FIELD_LOCATION);

  int R[4] = {2, 2, 2, 2}; // radius of the extended region in each dimension / direction

  profileGaugeForce.Stop(QUDA_PROFILE_INIT);
  profileGaugeForce.Start(QUDA_PROFILE_COMMS);
  cpuSiteLink->exchangeExtendedGhost(R);
  //exchange_cpu_sitelink_ex(qudaGaugeParam->X, R, (void**)cpuSiteLink->Gauge_p(), 
  //			   cpuSiteLink->Order(), qudaGaugeParam->cpu_prec, 1, 4);

  profileGaugeForce.Stop(QUDA_PROFILE_COMMS);
  profileGaugeForce.Start(QUDA_PROFILE_INIT);
#endif

  gParamSL.create = QUDA_ZERO_FIELD_CREATE;
  gParamSL.pad = 0;
  gParamSL.reconstruct = qudaGaugeParam->reconstruct;
  gParamSL.order = (qudaGaugeParam->reconstruct == QUDA_RECONSTRUCT_NO || 
      qudaGaugeParam->cuda_prec == QUDA_DOUBLE_PRECISION) ? 
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

  cudaGaugeField* cudaSiteLink = new cudaGaugeField(gParamSL);  

  qudaGaugeParam->site_ga_pad = gParamSL.pad;//need to record this value

  GaugeFieldParam &gParamMom = gParam;
  gParamMom.pad = 0;
  gParamMom.order = qudaGaugeParam->gauge_order;
  // FIXME - test program uses MILC for mom but can use QDP for gauge
  if (gParamMom.order == QUDA_QDP_GAUGE_ORDER) gParamMom.order = QUDA_MILC_GAUGE_ORDER;
  gParamMom.precision = qudaGaugeParam->cpu_prec;

  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParamMom.create = QUDA_REFERENCE_FIELD_CREATE;
  if (gParamMom.order == QUDA_TIFR_GAUGE_ORDER) {
    gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  } else {
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  }

  gParamMom.gauge=mom;

  cpuGaugeField* cpuMom = new cpuGaugeField(gParamMom);              

  gParamMom.pad = 0;
  gParamMom.create = QUDA_NULL_FIELD_CREATE;  
  gParamMom.order = QUDA_FLOAT2_GAUGE_ORDER;

  gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;

  gParamMom.precision = qudaGaugeParam->cuda_prec;

  cudaGaugeField* cudaMom = new cudaGaugeField(gParamMom);
  qudaGaugeParam->mom_ga_pad = gParamMom.pad; //need to record this value
  profileGaugeForce.Stop(QUDA_PROFILE_INIT);

  initLatticeConstants(*cudaMom, profileGaugeForce);

  gauge_force_init_cuda(qudaGaugeParam, max_length); 

  profileGaugeForce.Start(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  loadLinkToGPU_ex(cudaSiteLink, cpuSiteLink);
#else  
  loadLinkToGPU(cudaSiteLink, cpuSiteLink, qudaGaugeParam);    
#endif
  cudaMom->loadCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  profileGaugeForce.Stop(QUDA_PROFILE_H2D);

  // actually do the computation
  profileGaugeForce.Start(QUDA_PROFILE_COMPUTE);
  gauge_force_cuda(*cudaMom, eb3, *cudaSiteLink, qudaGaugeParam, input_path_buf, 
      path_length, loop_coeff, num_paths, max_length);
  profileGaugeForce.Stop(QUDA_PROFILE_COMPUTE);

  profileGaugeForce.Start(QUDA_PROFILE_D2H);

  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  profileGaugeForce.Stop(QUDA_PROFILE_D2H);

  profileGaugeForce.Start(QUDA_PROFILE_FREE);
  delete cpuSiteLink;
  delete cpuMom;
  delete cudaSiteLink;
  delete cudaMom;
  profileGaugeForce.Stop(QUDA_PROFILE_FREE);

  profileGaugeForce.Stop(QUDA_PROFILE_TOTAL);

  if(timeinfo){
    timeinfo[0] = profileGaugeForce.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = profileGaugeForce.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = profileGaugeForce.Last(QUDA_PROFILE_D2H);
  }

  checkCudaError();
#else
  errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  return 0;  
}

#else

  int
computeGaugeForceQuda(void* mom, void* siteLink,  int*** input_path_buf, int* path_length,
    double* loop_coeff, int num_paths, int max_length, double eb3,
    QudaGaugeParam* qudaGaugeParam, double* timeinfo)
{

  /*printfQuda("GaugeForce: use_resident_gauge = %d, make_resident_gauge = %d\n", 
    qudaGaugeParam->use_resident_gauge, qudaGaugeParam->make_resident_gauge);
    printfQuda("GaugeForce: use_resident_mom = %d, make_resident_mom = %d\n", 
    qudaGaugeParam->use_resident_mom, qudaGaugeParam->make_resident_mom);*/

#ifdef GPU_GAUGE_FORCE
  profileGaugeForce.Start(QUDA_PROFILE_TOTAL);
  profileGaugeForce.Start(QUDA_PROFILE_INIT); 

  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(0, *qudaGaugeParam);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.pad = 0;

#ifdef MULTI_GPU
  GaugeFieldParam gParamEx(gParam);
  for (int d=0; d<4; d++) gParamEx.x[d] = gParam.x[d] + 4;
#endif

  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = siteLink;
  cpuGaugeField *cpuSiteLink = new cpuGaugeField(gParam);

  cudaGaugeField* cudaSiteLink = NULL;

  if (qudaGaugeParam->use_resident_gauge) {
    if (!gaugePrecise) errorQuda("No resident gauge field to use");
    cudaSiteLink = gaugePrecise;
    profileGaugeForce.Stop(QUDA_PROFILE_INIT); 
    printfQuda("GaugeForce: Using resident gauge field\n");
  } else {
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = qudaGaugeParam->reconstruct;
    gParam.order = (qudaGaugeParam->reconstruct == QUDA_RECONSTRUCT_NO || 
        qudaGaugeParam->cuda_prec == QUDA_DOUBLE_PRECISION) ? 
      QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

    cudaSiteLink = new cudaGaugeField(gParam);  
    profileGaugeForce.Stop(QUDA_PROFILE_INIT); 

    profileGaugeForce.Start(QUDA_PROFILE_H2D);
    cudaSiteLink->loadCPUField(*cpuSiteLink, QUDA_CPU_FIELD_LOCATION);    
    profileGaugeForce.Stop(QUDA_PROFILE_H2D);
  }

  profileGaugeForce.Start(QUDA_PROFILE_INIT); 

#ifndef MULTI_GPU
  cudaGaugeField *cudaGauge = cudaSiteLink;
#else

  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.reconstruct = qudaGaugeParam->reconstruct;
  gParamEx.order = (qudaGaugeParam->reconstruct == QUDA_RECONSTRUCT_NO || 
      qudaGaugeParam->cuda_prec == QUDA_DOUBLE_PRECISION) ? 
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  qudaGaugeParam->site_ga_pad = gParamEx.pad;//need to record this value

  cudaGaugeField *cudaGauge = new cudaGaugeField(gParamEx);

  copyExtendedGauge(*cudaGauge, *cudaSiteLink, QUDA_CUDA_FIELD_LOCATION);
  int R[4] = {2, 2, 2, 2}; // radius of the extended region in each dimension / direction

  profileGaugeForce.Stop(QUDA_PROFILE_INIT); 

  profileGaugeForce.Start(QUDA_PROFILE_COMMS);
  cudaGauge->exchangeExtendedGhost(R);
  profileGaugeForce.Stop(QUDA_PROFILE_COMMS);
  profileGaugeForce.Start(QUDA_PROFILE_INIT); 
#endif

  GaugeFieldParam &gParamMom = gParam;
  gParamMom.order = qudaGaugeParam->gauge_order;
  // FIXME - test program always uses MILC for mom but can use QDP for gauge
  if (gParamMom.order == QUDA_QDP_GAUGE_ORDER) gParamMom.order = QUDA_MILC_GAUGE_ORDER;
  gParamMom.precision = qudaGaugeParam->cpu_prec;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParamMom.create = QUDA_REFERENCE_FIELD_CREATE;
  gParamMom.gauge=mom;
  if (gParamMom.order == QUDA_TIFR_GAUGE_ORDER) gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  else gParamMom.reconstruct = QUDA_RECONSTRUCT_10;

  cpuGaugeField* cpuMom = new cpuGaugeField(gParamMom);              

  cudaGaugeField* cudaMom = NULL;
  if (qudaGaugeParam->use_resident_mom) {
    if (!gaugePrecise) errorQuda("No resident momentum field to use");
    cudaMom = momResident;
    printfQuda("GaugeForce: Using resident mom field\n");
    profileGaugeForce.Stop(QUDA_PROFILE_INIT);
  } else {
    gParamMom.create = QUDA_ZERO_FIELD_CREATE;  
    gParamMom.order = QUDA_FLOAT2_GAUGE_ORDER;
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
    gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
    gParamMom.precision = qudaGaugeParam->cuda_prec;
    cudaMom = new cudaGaugeField(gParamMom);
    profileGaugeForce.Stop(QUDA_PROFILE_INIT);

    profileGaugeForce.Start(QUDA_PROFILE_H2D);
    cudaMom->loadCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
    profileGaugeForce.Stop(QUDA_PROFILE_H2D);
  }

  initLatticeConstants(*cudaMom, profileGaugeForce);

  profileGaugeForce.Start(QUDA_PROFILE_CONSTANT);
  qudaGaugeParam->mom_ga_pad = gParamMom.pad; //need to set this (until we use order classes)
  gauge_force_init_cuda(qudaGaugeParam, max_length); 
  profileGaugeForce.Stop(QUDA_PROFILE_CONSTANT);

  // actually do the computation
  profileGaugeForce.Start(QUDA_PROFILE_COMPUTE);
  gauge_force_cuda(*cudaMom, eb3, *cudaGauge, qudaGaugeParam, input_path_buf, 
      path_length, loop_coeff, num_paths, max_length);
  profileGaugeForce.Stop(QUDA_PROFILE_COMPUTE);

  // still need to copy this back even when preserving
  profileGaugeForce.Start(QUDA_PROFILE_D2H);
  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  profileGaugeForce.Stop(QUDA_PROFILE_D2H);

  profileGaugeForce.Start(QUDA_PROFILE_FREE);
  if (qudaGaugeParam->make_resident_gauge) {
    if (gaugePrecise && gaugePrecise != cudaSiteLink) delete gaugePrecise;
    gaugePrecise = cudaSiteLink;
  } else {
    delete cudaSiteLink;
  }

  if (qudaGaugeParam->make_resident_mom) {
    if (momResident && momResident != cudaMom) delete momResident;
    momResident = cudaMom;
  } else {
    delete cudaMom;
  }

  delete cpuSiteLink;
  delete cpuMom;

#ifdef MULTI_GPU
  delete cudaGauge;
#endif
  profileGaugeForce.Stop(QUDA_PROFILE_FREE);

  profileGaugeForce.Stop(QUDA_PROFILE_TOTAL);

  if(timeinfo){
    timeinfo[0] = profileGaugeForce.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = profileGaugeForce.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = profileGaugeForce.Last(QUDA_PROFILE_D2H);
  }

  checkCudaError();
#else
  errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  return 0;  
}

#endif



void createCloverQuda(QudaInvertParam* invertParam)
{
  profileCloverCreate.Start(QUDA_PROFILE_TOTAL);
  profileCloverCreate.Start(QUDA_PROFILE_INIT);
  if(!cloverPrecise){
    printfQuda("About to create cloverPrecise\n");
    CloverFieldParam cloverParam;
    cloverParam.nDim = 4;
    for(int dir=0; dir<4; ++dir) cloverParam.x[dir] = gaugePrecise->X()[dir];
    cloverParam.setPrecision(invertParam->clover_cuda_prec);
    cloverParam.pad = invertParam->cl_pad;
    cloverParam.direct = true;
    cloverParam.inverse = true;
    cloverParam.norm    = 0;
    cloverParam.invNorm = 0;
    cloverParam.twisted = false;
    cloverParam.create = QUDA_NULL_FIELD_CREATE;
    cloverParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    cloverParam.setPrecision(invertParam->cuda_prec);
    if (invertParam->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    {
      cloverParam.direct = true;
      cloverParam.inverse = false;
      cloverPrecise = new cudaCloverField(cloverParam);
      cloverParam.inverse = true;
      cloverParam.direct = false;
      cloverParam.twisted = true;
      cloverParam.mu2 = 4.*invertParam->kappa*invertParam->kappa*invertParam->mu*invertParam->mu;
      cloverInvPrecise = new cudaCloverField(cloverParam);	//FIXME Only with tmClover
    } else {
      cloverPrecise = new cudaCloverField(cloverParam);
    } 
  }

  int y[4];
  for(int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 4;
  int pad = 0;
  GaugeFieldParam gParamEx(y, gaugePrecise->Precision(), QUDA_RECONSTRUCT_NO,
      pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_NO);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = gaugePrecise->Order();
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gaugePrecise->TBoundary();
  gParamEx.nFace = 1;

  cudaGaugeField *cudaGaugeExtended = NULL;
  if (extendedGaugeResident) {
    cudaGaugeExtended = extendedGaugeResident;
    profileCloverCreate.Stop(QUDA_PROFILE_INIT);
  } else {
    cudaGaugeExtended = new cudaGaugeField(gParamEx);

    // copy gaugePrecise into the extended device gauge field
    copyExtendedGauge(*cudaGaugeExtended, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);
    int R[4] = {2,2,2,2}; // radius of the extended region in each dimension / direction
#if 1
    profileCloverCreate.Stop(QUDA_PROFILE_INIT);
    profileCloverCreate.Start(QUDA_PROFILE_COMMS);
    cudaGaugeExtended->exchangeExtendedGhost(R,true);
    profileCloverCreate.Stop(QUDA_PROFILE_COMMS);
#else

    GaugeFieldParam gParam(gaugePrecise->X(), gaugePrecise->Precision(), QUDA_RECONSTRUCT_NO,
        pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_NO);
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.order = QUDA_MILC_GAUGE_ORDER;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.t_boundary = gaugePrecise->TBoundary();
    gParam.nFace = 1;

    // create an extended gauge field on the hose
    for(int dir=0; dir<4; ++dir) gParam.x[dir] += 4;
    cpuGaugeField cpuGaugeExtended(gParam);
    cudaGaugeExtended->saveCPUField(cpuGaugeExtended, QUDA_CPU_FIELD_LOCATION);

    profileCloverCreate.Stop(QUDA_PROFILE_INIT);
    // communicate data
    profileCloverCreate.Start(QUDA_PROFILE_COMMS);
    //exchange_cpu_sitelink_ex(const_cast<int*>(gaugePrecise->X()), R, (void**)cpuGaugeExtended.Gauge_p(),
    //			   cpuGaugeExtended.Order(),cpuGaugeExtended.Precision(), 0, 4);
    cpuGaugeExtended.exchangeExtendedGhost(R,true);

    cudaGaugeExtended->loadCPUField(cpuGaugeExtended, QUDA_CPU_FIELD_LOCATION);
    profileCloverCreate.Stop(QUDA_PROFILE_COMMS);
#endif
  }

  profileCloverCreate.Start(QUDA_PROFILE_COMPUTE);
  computeClover(*cloverPrecise, *cudaGaugeExtended, invertParam->clover_coeff, QUDA_CUDA_FIELD_LOCATION);

  if (invertParam->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    computeClover(*cloverInvPrecise, *cudaGaugeExtended, invertParam->clover_coeff, QUDA_CUDA_FIELD_LOCATION);	//FIXME Only with tmClover

  profileCloverCreate.Stop(QUDA_PROFILE_COMPUTE);

  profileCloverCreate.Stop(QUDA_PROFILE_TOTAL);

  // FIXME always preserve the extended gauge
  extendedGaugeResident = cudaGaugeExtended;

  return;
}

void* createGaugeField(void* gauge, int geometry, QudaGaugeParam* param)
{

  GaugeFieldParam gParam(0,*param);
  if(geometry == 1){
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
  }else if(geometry == 4){ 
    gParam.geometry = QUDA_VECTOR_GEOMETRY;
  }else{
    errorQuda("Only scalar and vector geometries are supported\n");
  }
  gParam.pad = 0;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.gauge = gauge;
  gParam.link_type = QUDA_GENERAL_LINKS;


  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaGaugeField* cudaGauge = new cudaGaugeField(gParam);
  if(gauge){
    gParam.order = QUDA_MILC_GAUGE_ORDER;
    gParam.create = QUDA_REFERENCE_FIELD_CREATE;
    cpuGaugeField cpuGauge(gParam);
    cudaGauge->loadCPUField(cpuGauge,QUDA_CPU_FIELD_LOCATION);
  }
  return cudaGauge;
}


void saveGaugeField(void* gauge, void* inGauge, QudaGaugeParam* param){

  cudaGaugeField* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);

  GaugeFieldParam gParam(0,*param);
  gParam.geometry = cudaGauge->Geometry();
  gParam.pad = 0;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.gauge = gauge;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.create = QUDA_REFERENCE_FIELD_CREATE; 

  cpuGaugeField cpuGauge(gParam);
  cudaGauge->saveCPUField(cpuGauge,QUDA_CPU_FIELD_LOCATION);
}


void* createExtendedGaugeField(void* gauge, int geometry, QudaGaugeParam* param)
{
  profileExtendedGauge.Start(QUDA_PROFILE_TOTAL);

  if (param->use_resident_gauge && extendedGaugeResident && geometry == 4) {
    profileExtendedGauge.Stop(QUDA_PROFILE_TOTAL);
    return extendedGaugeResident;
  }

  profileExtendedGauge.Start(QUDA_PROFILE_INIT);

  QudaFieldGeometry geom = QUDA_INVALID_GEOMETRY;
  if (geometry == 1) {
    geom = QUDA_SCALAR_GEOMETRY;
  } else if(geometry == 4) {
    geom = QUDA_VECTOR_GEOMETRY;
  } else {
    errorQuda("Only scalar and vector geometries are supported");
  }

  cpuGaugeField* cpuGauge;
  cudaGaugeField* cudaGauge;


  // Create the unextended cpu field 
  GaugeFieldParam gParam(0, *param);
  gParam.order          =  QUDA_MILC_GAUGE_ORDER;
  gParam.pad            = 0;
  gParam.link_type      = param->type;
  gParam.ghostExchange  = QUDA_GHOST_EXCHANGE_NO;
  gParam.create         = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge          = gauge;
  gParam.geometry       = geom;

  if(gauge){
    cpuGauge  = new cpuGaugeField(gParam);
    // Create the unextended GPU field 
    gParam.order  = QUDA_FLOAT2_GAUGE_ORDER;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    cudaGauge     = new cudaGaugeField(gParam);
    profileExtendedGauge.Stop(QUDA_PROFILE_INIT);

    // load the data into the unextended device field 
    profileExtendedGauge.Start(QUDA_PROFILE_H2D);
    cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
    profileExtendedGauge.Stop(QUDA_PROFILE_H2D);

    profileExtendedGauge.Start(QUDA_PROFILE_INIT);
  }

  QudaGaugeParam param_ex;
  memcpy(&param_ex, param, sizeof(QudaGaugeParam));
  for(int dir=0; dir<4; ++dir) param_ex.X[dir] = param->X[dir]+4;
  GaugeFieldParam gParam_ex(0, param_ex);
  gParam_ex.link_type     = param->type; 
  gParam_ex.geometry      = geom;
  gParam_ex.order         = QUDA_FLOAT2_GAUGE_ORDER;
  gParam_ex.create        = QUDA_ZERO_FIELD_CREATE;
  gParam_ex.pad           = 0;
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  // create the extended gauge field
  cudaGaugeField* cudaGaugeEx = new cudaGaugeField(gParam_ex);

  // copy data from the interior into the border region
  if(gauge) copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);

  profileExtendedGauge.Stop(QUDA_PROFILE_INIT);
  if(gauge){
    int R[4] = {2,2,2,2};
    // communicate 
    profileExtendedGauge.Start(QUDA_PROFILE_COMMS);
    cudaGaugeEx->exchangeExtendedGhost(R, true);
    profileExtendedGauge.Stop(QUDA_PROFILE_COMMS);
    delete cpuGauge;
    delete cudaGauge;
  }
  profileExtendedGauge.Stop(QUDA_PROFILE_TOTAL);

  return cudaGaugeEx;
}

// extend field on the GPU
void extendGaugeField(void* out, void* in){
  cudaGaugeField* inGauge   = reinterpret_cast<cudaGaugeField*>(in);
  cudaGaugeField* outGauge  = reinterpret_cast<cudaGaugeField*>(out);

  copyExtendedGauge(*outGauge, *inGauge, QUDA_CUDA_FIELD_LOCATION);

  int R[4] = {2,2,2,2};
  outGauge->exchangeExtendedGhost(R,true);

  return;
}



void destroyQudaGaugeField(void* gauge){
  cudaGaugeField* g = reinterpret_cast<cudaGaugeField*>(gauge);
  delete g;
}


void computeCloverTraceQuda(void *out,
    void *clov,
    int mu,
    int nu,
    int dim[4])
{

  profileCloverTrace.Start(QUDA_PROFILE_TOTAL);


  cudaGaugeField* cudaGauge = reinterpret_cast<cudaGaugeField*>(out);

  if(cloverPrecise){
    computeCloverSigmaTrace(*cudaGauge, *cloverPrecise, mu, nu,  QUDA_CUDA_FIELD_LOCATION);
    //computeCloverSigmaTrace(*cudaGauge, cudaClover, mu, nu,  QUDA_CUDA_FIELD_LOCATION);
  }else{
    errorQuda("cloverPrecise not set\n");
  }
  profileCloverTrace.Stop(QUDA_PROFILE_TOTAL);
  return;
}


void computeCloverDerivativeQuda(void* out,
    void* gauge,
    void* oprod,
    int mu, int nu,
    double coeff,
    QudaParity parity,
    QudaGaugeParam* param,
    int conjugate)
{
  profileCloverDerivative.Start(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  profileCloverDerivative.Start(QUDA_PROFILE_INIT);
#ifndef USE_EXTENDED_VOLUME
#define USE_EXTENDED_VOLUME
#endif

  // create host fields
  GaugeFieldParam gParam(0, *param);
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.pad = 0;
  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  //  gParam.gauge = out;
  //  cpuGaugeField cpuOut(gParam);
#ifndef USE_EXTENDED_VOLUME
  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.gauge = oprod;
  cpuGaugeField cpuOprod(gParam);

  gParam.geometry = QUDA_VECTOR_GEOMETRY;
  gParam.link_type = QUDA_SU3_LINKS;
  gParam.gauge = gauge;
  cpuGaugeField cpuGauge(gParam);
#endif

  /*
  // create device fields
  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  //  gParam.pad = getGaugePadding(gParam);
  gParam.pad = 0;
  gParam.ghostExchange  = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  //  cudaGaugeField cudaOut(gParam);
  */

#ifndef USE_EXTENDED_VOLUME
  cudaGaugeField cudaOprod(gParam);

  gParam.geometry = QUDA_VECTOR_GEOMETRY;
  gParam.link_type = QUDA_SU3_LINKS;
  cudaGaugeField cudaGauge(gParam);
#endif
  profileCloverDerivative.Stop(QUDA_PROFILE_INIT);

  cudaGaugeField* cudaOut = reinterpret_cast<cudaGaugeField*>(out);
  cudaGaugeField* gPointer = reinterpret_cast<cudaGaugeField*>(gauge);
  cudaGaugeField* oPointer = reinterpret_cast<cudaGaugeField*>(oprod);


  profileCloverDerivative.Start(QUDA_PROFILE_COMPUTE);
  cloverDerivative(*cudaOut, *gPointer, *oPointer, mu, nu, coeff, parity, conjugate);
  profileCloverDerivative.Stop(QUDA_PROFILE_COMPUTE);


  profileCloverDerivative.Start(QUDA_PROFILE_D2H);

  //  saveGaugeField(out, cudaOut, param);
  //  cudaOut->saveCPUField(cpuOut, QUDA_CPU_FIELD_LOCATION);
  profileCloverDerivative.Stop(QUDA_PROFILE_D2H);
  checkCudaError();

  //  delete cudaOut;

  profileCloverDerivative.Stop(QUDA_PROFILE_TOTAL);


  return;
}

void computeKSOprodQuda(void* oprod,
    void* fermion,
    double coeff,
    int X[4],
    QudaPrecision prec)

{
  using namespace quda;       

  cudaGaugeField* cudaOprod;
  cudaColorSpinorField* cudaQuark;

  const int Ls = 1;
  const int Ninternal = 6;
#ifdef BUILD_TIFR_INTERFACE
  const int Nface = 1;
#else
  const int Nface = 3;  
#endif
  FaceBuffer fB(X, 4, Ninternal, Nface, prec, Ls);
  cudaOprod = reinterpret_cast<cudaGaugeField*>(oprod);
  cudaQuark = reinterpret_cast<cudaColorSpinorField*>(fermion);

  double new_coeff[2] = {0,0}; 
  new_coeff[0] = coeff;
  // Operate on even-parity sites
  computeStaggeredOprod(*cudaOprod, *cudaOprod, *cudaQuark, fB, 0, new_coeff);

  // Operator on odd-parity sites
  computeStaggeredOprod(*cudaOprod, *cudaOprod, *cudaQuark, fB, 1, new_coeff);

  return;
}


void computeStaggeredForceQuda(void* cudaMom, void* qudaQuark, double coeff)
{
  bool use_resident_solution = false;
  if (solutionResident) {
    qudaQuark = solutionResident;
    use_resident_solution = true;
  } else {
    errorQuda("No input quark field defined");
  }

  if (momResident) {
    cudaMom = momResident;
  } else {
    errorQuda("No input momentum defined");
  }

  if (!gaugePrecise) {
    errorQuda("No resident gauge field");
  }

  int pad = 0;
  GaugeFieldParam oParam(gaugePrecise->X(), gaugePrecise->Precision(), QUDA_RECONSTRUCT_NO, 
      pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_NO);
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.order  = QUDA_FLOAT2_GAUGE_ORDER;
  oParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  oParam.t_boundary = QUDA_PERIODIC_T;
  oParam.nFace = 1;

  // create temporary field for quark-field outer product
  cudaGaugeField cudaOprod(oParam);

  // compute quark-field outer product
  computeKSOprodQuda(&cudaOprod, qudaQuark, coeff,
      const_cast<int*>(gaugePrecise->X()),
      gaugePrecise->Precision());

  cudaGaugeField* mom = reinterpret_cast<cudaGaugeField*>(cudaMom);

  completeKSForce(*mom, cudaOprod, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);

  if (use_resident_solution) {
    delete solutionResident;
    solutionResident = NULL;
  }

  return;
}


void computeAsqtadForceQuda(void* const milc_momentum,
    const double act_path_coeff[6],
    const void* const one_link_src[4],
    const void* const naik_src[4],
    const void* const link,
    const QudaGaugeParam* gParam)
{

#ifdef GPU_HISQ_FORCE
  using namespace quda::fermion_force;
  profileAsqtadForce.Start(QUDA_PROFILE_TOTAL);
  profileAsqtadForce.Start(QUDA_PROFILE_INIT);

  cudaGaugeField *cudaGauge = NULL;
  cpuGaugeField *cpuGauge = NULL;
  cudaGaugeField *cudaInForce = NULL;
  cpuGaugeField *cpuOneLinkInForce = NULL;
  cpuGaugeField *cpuNaikInForce = NULL;
  cudaGaugeField *cudaOutForce = NULL;
  cudaGaugeField *cudaMom = NULL;
  cpuGaugeField *cpuMom = NULL;

#ifdef MULTI_GPU
  cudaGaugeField *cudaGauge_ex = NULL;
  cudaGaugeField *cudaInForce_ex = NULL;
  cudaGaugeField *cudaOutForce_ex = NULL;
#endif

  GaugeFieldParam param(0, *gParam);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.anisotropy = 1.0;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  param.t_boundary = QUDA_PERIODIC_T;
  param.nFace = 1;

  param.link_type = QUDA_GENERAL_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.create = QUDA_REFERENCE_FIELD_CREATE;

  // create host fields
  param.gauge = (void*)link;
  cpuGauge = new cpuGaugeField(param);

  param.order = QUDA_QDP_GAUGE_ORDER;  
  param.gauge = (void*)one_link_src;
  cpuOneLinkInForce = new cpuGaugeField(param);

  param.gauge = (void*)naik_src;
  cpuNaikInForce = new cpuGaugeField(param);

  param.order = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.gauge = milc_momentum;
  cpuMom = new cpuGaugeField(param);

  // create device fields
  param.create = QUDA_NULL_FIELD_CREATE;
  param.link_type = QUDA_GENERAL_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.order =  QUDA_FLOAT2_GAUGE_ORDER;

  cudaGauge    = new cudaGaugeField(param);
  cudaInForce  = new cudaGaugeField(param);
  cudaOutForce = new cudaGaugeField(param);

  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  cudaMom = new cudaGaugeField(param);

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir) param.x[dir] += 4;
  param.link_type = QUDA_GENERAL_LINKS;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.reconstruct = QUDA_RECONSTRUCT_NO;

  cudaGauge_ex    = new cudaGaugeField(param);
  cudaInForce_ex  = new cudaGaugeField(param);
  cudaOutForce_ex = new cudaGaugeField(param);
#endif
  profileAsqtadForce.Stop(QUDA_PROFILE_INIT);

#ifdef MULTI_GPU
  int R[4] = {2, 2, 2, 2};
#endif

  profileAsqtadForce.Start(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  profileAsqtadForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaInForce_ex->Gauge_p()), 0, cudaInForce_ex->Bytes());
  copyExtendedGauge(*cudaGauge_ex, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGauge_ex->exchangeExtendedGhost(R,true);
#endif

  profileAsqtadForce.Start(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuOneLinkInForce, QUDA_CPU_FIELD_LOCATION);
  profileAsqtadForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaInForce_ex->Gauge_p()), 0, cudaInForce_ex->Bytes());
  copyExtendedGauge(*cudaInForce_ex, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce_ex->exchangeExtendedGhost(R,true);
#endif

  cudaMemset((void**)(cudaOutForce->Gauge_p()), 0, cudaOutForce->Bytes());
  profileAsqtadForce.Start(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaOutForce_ex->Gauge_p()), 0, cudaOutForce_ex->Bytes());
  hisqStaplesForceCuda(act_path_coeff, *gParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex);
#else
  hisqStaplesForceCuda(act_path_coeff, *gParam, *cudaInForce, *cudaGauge, cudaOutForce);
#endif
  profileAsqtadForce.Stop(QUDA_PROFILE_COMPUTE);

  profileAsqtadForce.Start(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuNaikInForce, QUDA_CPU_FIELD_LOCATION); 
#ifdef MULTI_GPU
  copyExtendedGauge(*cudaInForce_ex, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce_ex->exchangeExtendedGhost(R,true);
#endif
  profileAsqtadForce.Stop(QUDA_PROFILE_H2D);

  profileAsqtadForce.Start(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex);
  completeKSForce(*cudaMom, *cudaOutForce_ex, *cudaGauge_ex, QUDA_CUDA_FIELD_LOCATION);
#else
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *cudaInForce, *cudaGauge, cudaOutForce);
  hisqCompleteForceCuda(*gParam, *cudaOutForce, *cudaGauge, cudaMom);
#endif
  profileAsqtadForce.Stop(QUDA_PROFILE_COMPUTE);

  profileAsqtadForce.Start(QUDA_PROFILE_D2H);
  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  profileAsqtadForce.Stop(QUDA_PROFILE_D2H);

  profileAsqtadForce.Start(QUDA_PROFILE_FREE);
  delete cudaInForce;
  delete cudaOutForce;
  delete cudaGauge;
  delete cudaMom;
#ifdef MULTI_GPU
  delete cudaInForce_ex;
  delete cudaOutForce_ex;
  delete cudaGauge_ex;
#endif

  delete cpuGauge;
  delete cpuOneLinkInForce;
  delete cpuNaikInForce;
  delete cpuMom;

  profileAsqtadForce.Stop(QUDA_PROFILE_FREE);

  profileAsqtadForce.Stop(QUDA_PROFILE_TOTAL);
  return;

#else
  errorQuda("HISQ force has not been built");
#endif

}

void 
computeHISQForceCompleteQuda(void* const milc_momentum,
                             const double level2_coeff[6],
                             const double fat7_coeff[6],
                             void** quark_array,
                             int num_terms,
                             double** quark_coeff,
                             const void* const w_link,
                             const void* const v_link,
                             const void* const u_link,
                             const QudaGaugeParam* gParam)
{

  void* oprod[2];

/*
  computeStaggeredOprodQuda(void** oprod,
    void** fermion,
    int num_terms,
    double** coeff,
    QudaGaugeParam* gParam)

  computeHISQForceQuda(milc_momentum,
                       level2_coeff,
                       fat7_coeff,
                       staple_src,
                       one_link_src,
                       naik_src,
                       w_link, 
                       v_link,
                       u_link,
                       gParam);

*/
  return;
}




  void
computeHISQForceQuda(void* const milc_momentum,
    const double level2_coeff[6],
    const double fat7_coeff[6],
    const void* const staple_src[4],
    const void* const one_link_src[4],
    const void* const naik_src[4],
    const void* const w_link,
    const void* const v_link,
    const void* const u_link,
    const QudaGaugeParam* gParam)
{
#ifdef GPU_HISQ_FORCE
  using namespace quda::fermion_force;
  profileHISQForce.Start(QUDA_PROFILE_TOTAL);
  profileHISQForce.Start(QUDA_PROFILE_INIT); 

  double act_path_coeff[6] = {0,1,level2_coeff[2],level2_coeff[3],level2_coeff[4],level2_coeff[5]};
  // You have to look at the MILC routine to understand the following
  // Basically, I have already absorbed the one-link coefficient

  GaugeFieldParam param(0, *gParam);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.order  = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS; 
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.gauge = (void*)milc_momentum;
  cpuGaugeField* cpuMom = new cpuGaugeField(param);

  param.create = QUDA_ZERO_FIELD_CREATE;
  param.order  = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField* cudaMom = new cudaGaugeField(param);

  param.order = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_GENERAL_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.gauge = (void*)w_link;
  cpuGaugeField cpuWLink(param);
  param.gauge = (void*)v_link;
  cpuGaugeField cpuVLink(param);
  param.gauge = (void*)u_link;
  cpuGaugeField cpuULink(param);
  param.create = QUDA_ZERO_FIELD_CREATE;

  param.ghostExchange =  QUDA_GHOST_EXCHANGE_NO;
  param.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField* cudaGauge = new cudaGaugeField(param);

  cpuGaugeField* cpuStapleForce;
  cpuGaugeField* cpuOneLinkForce;
  cpuGaugeField* cpuNaikForce;

  param.order = QUDA_QDP_GAUGE_ORDER;
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.gauge = (void*)staple_src;
  cpuStapleForce = new cpuGaugeField(param);
  param.gauge = (void*)one_link_src;
  cpuOneLinkForce = new cpuGaugeField(param);
  param.gauge = (void*)naik_src;
  cpuNaikForce = new cpuGaugeField(param);
  param.create = QUDA_ZERO_FIELD_CREATE;

  param.ghostExchange =  QUDA_GHOST_EXCHANGE_NO;
  param.link_type = QUDA_GENERAL_LINKS; 
  param.precision = gParam->cpu_prec;

  param.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField* cudaInForce  = new cudaGaugeField(param);

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir) param.x[dir] += 4;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaGaugeField* cudaGaugeEx = new cudaGaugeField(param);
  cudaGaugeField* cudaInForceEx = new cudaGaugeField(param);
  cudaGaugeField* cudaOutForceEx = new cudaGaugeField(param);
  cudaGaugeField* gaugePtr = cudaGaugeEx;
  cudaGaugeField* inForcePtr = cudaInForceEx;
  cudaGaugeField* outForcePtr = cudaOutForceEx;
#else
  cudaGaugeField* cudaOutForce = new cudaGaugeField(param);
  cudaGaugeField* gaugePtr = cudaGauge;
  cudaGaugeField* inForcePtr = cudaInForce;
  cudaGaugeField* outForcePtr = cudaOutForce;
#endif


  {
    // default settings for the unitarization
    const double unitarize_eps = 1e-14;
    const double hisq_force_filter = 5e-5;
    const double max_det_error = 1e-10;
    const bool   allow_svd = true;
    const bool   svd_only = false;
    const double svd_rel_err = 1e-8;
    const double svd_abs_err = 1e-8;

    setUnitarizeForceConstants(unitarize_eps, 
        hisq_force_filter, 
        max_det_error, 
        allow_svd, 
        svd_only, 
        svd_rel_err, 
        svd_abs_err);
  }
  profileHISQForce.Stop(QUDA_PROFILE_INIT); 


  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(cpuWLink, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  int R[4] = {2, 2, 2, 2};
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#endif

  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuStapleForce, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaInForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuOneLinkForce, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaOutForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaOutForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#else 
  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaOutForce->loadCPUField(*cpuOneLinkForce, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
#endif

  profileHISQForce.Start(QUDA_PROFILE_COMPUTE);
  hisqStaplesForceCuda(act_path_coeff, *gParam, *inForcePtr, *gaugePtr, outForcePtr);
  profileHISQForce.Stop(QUDA_PROFILE_COMPUTE);

  // Load naik outer product
  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuNaikForce, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaInForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#endif

  // Compute Naik three-link term
  profileHISQForce.Start(QUDA_PROFILE_COMPUTE);
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *inForcePtr, *gaugePtr, outForcePtr);
  profileHISQForce.Stop(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  cudaOutForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#endif
  // load v-link
  profileHISQForce.Start(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(cpuVLink, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#endif
  // Done with cudaInForce. It becomes the output force. Oops!
  profileHISQForce.Start(QUDA_PROFILE_INIT);
  int numFailures = 0;
  int* numFailuresDev;

  if(cudaMalloc((void**)&numFailuresDev, sizeof(int)) == cudaErrorMemoryAllocation){
    errorQuda("cudaMalloc failed for numFailuresDev\n");
  }
  cudaMemset(numFailuresDev, 0, sizeof(int));
  profileHISQForce.Stop(QUDA_PROFILE_INIT);


  profileHISQForce.Start(QUDA_PROFILE_COMPUTE);
  unitarizeForceCuda(*outForcePtr, *gaugePtr, inForcePtr, numFailuresDev);
  profileHISQForce.Stop(QUDA_PROFILE_COMPUTE);
  profileHISQForce.Start(QUDA_PROFILE_D2H);
  cudaMemcpy(&numFailures, numFailuresDev, sizeof(int), cudaMemcpyDeviceToHost);
  profileHISQForce.Stop(QUDA_PROFILE_D2H);
  cudaFree(numFailuresDev); 

  if(numFailures>0){
    errorQuda("Error in the unitarization component of the hisq fermion force\n"); 
    exit(1);
  } 
  cudaMemset((void**)(outForcePtr->Gauge_p()), 0, outForcePtr->Bytes());
  // read in u-link
  profileHISQForce.Start(QUDA_PROFILE_COMPUTE);
  cudaGauge->loadCPUField(cpuULink, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  profileHISQForce.Start(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.Stop(QUDA_PROFILE_COMMS);
#endif
  // Compute Fat7-staple term 
  profileHISQForce.Start(QUDA_PROFILE_COMPUTE);
  hisqStaplesForceCuda(fat7_coeff, *gParam, *inForcePtr, *gaugePtr, outForcePtr);
  hisqCompleteForceCuda(*gParam, *outForcePtr, *gaugePtr, cudaMom);
  profileHISQForce.Stop(QUDA_PROFILE_COMPUTE);

  profileHISQForce.Start(QUDA_PROFILE_D2H);
  // Close the paths, make anti-hermitian, and store in compressed format
  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  profileHISQForce.Stop(QUDA_PROFILE_D2H);



  profileHISQForce.Start(QUDA_PROFILE_FREE);

  delete cpuStapleForce;
  delete cpuOneLinkForce;
  delete cpuNaikForce;
  delete cpuMom;

  delete cudaInForce;
  delete cudaGauge;
  delete cudaMom;

#ifdef MULTI_GPU
  delete cudaInForceEx;
  delete cudaOutForceEx;
  delete cudaGaugeEx;
#else
  delete cudaOutForce;
#endif
  profileHISQForce.Stop(QUDA_PROFILE_FREE);
  profileHISQForce.Stop(QUDA_PROFILE_TOTAL);
  return;
#else 
  errorQuda("HISQ force has not been built");
#endif    
}

void computeStaggeredOprodQuda(void** oprod,   
    void** fermion,
    int num_terms,
    double** coeff,
    QudaGaugeParam* param)
{
  using namespace quda;
  profileStaggeredOprod.Start(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  profileStaggeredOprod.Start(QUDA_PROFILE_INIT);
  GaugeFieldParam oParam(0, *param);

  oParam.nDim = 4;
  oParam.nFace = 0; 
  // create the host outer-product field
  oParam.pad = 0;
  oParam.create = QUDA_REFERENCE_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.order = QUDA_QDP_GAUGE_ORDER;
  oParam.gauge = oprod[0];
  cpuGaugeField cpuOprod0(oParam);

  oParam.gauge = oprod[1];
  cpuGaugeField cpuOprod1(oParam);

  // create the device outer-product field
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField cudaOprod0(oParam);
  cudaGaugeField cudaOprod1(oParam);
  profileStaggeredOprod.Stop(QUDA_PROFILE_INIT); 
  initLatticeConstants(cudaOprod0, profileStaggeredOprod);



  profileStaggeredOprod.Start(QUDA_PROFILE_H2D);
  cudaOprod0.loadCPUField(cpuOprod0,QUDA_CPU_FIELD_LOCATION);
  cudaOprod1.loadCPUField(cpuOprod1,QUDA_CPU_FIELD_LOCATION);
  profileStaggeredOprod.Stop(QUDA_PROFILE_H2D);


  profileStaggeredOprod.Start(QUDA_PROFILE_INIT);



  ColorSpinorParam qParam;
  qParam.nColor = 3;
  qParam.nSpin = 1;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.precision = oParam.precision;
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];

  // create the device quark field
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  cudaColorSpinorField cudaQuark(qParam); 

  // create the host quark field
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;

  const int Ls = 1;
  const int Ninternal = 6;
  FaceBuffer faceBuffer1(cudaOprod0.X(), 4, Ninternal, 3, cudaOprod0.Precision(), Ls);
  FaceBuffer faceBuffer2(cudaOprod0.X(), 4, Ninternal, 3, cudaOprod0.Precision(), Ls);
  profileStaggeredOprod.Stop(QUDA_PROFILE_INIT);

  // loop over different quark fields
  for(int i=0; i<num_terms; ++i){

    profileStaggeredOprod.Start(QUDA_PROFILE_INIT);
    qParam.v = fermion[i];
    cpuColorSpinorField cpuQuark(qParam); // create host quark field
    profileStaggeredOprod.Stop(QUDA_PROFILE_INIT);

    profileStaggeredOprod.Start(QUDA_PROFILE_H2D);
    cudaQuark = cpuQuark;
    profileStaggeredOprod.Stop(QUDA_PROFILE_H2D);

    profileStaggeredOprod.Start(QUDA_PROFILE_COMPUTE);
    // Operate on even-parity sites
    computeStaggeredOprod(cudaOprod0, cudaOprod1, cudaQuark, faceBuffer1, 0, coeff[i]);

    // Operate on odd-parity sites
    computeStaggeredOprod(cudaOprod0, cudaOprod1, cudaQuark, faceBuffer2, 1, coeff[i]);
    profileStaggeredOprod.Stop(QUDA_PROFILE_COMPUTE);
  }


  // copy the outer product field back to the host
  profileStaggeredOprod.Start(QUDA_PROFILE_D2H);
  cudaOprod0.saveCPUField(cpuOprod0,QUDA_CPU_FIELD_LOCATION);
  cudaOprod1.saveCPUField(cpuOprod1,QUDA_CPU_FIELD_LOCATION);
  profileStaggeredOprod.Stop(QUDA_PROFILE_D2H); 


  profileStaggeredOprod.Stop(QUDA_PROFILE_TOTAL);

  checkCudaError();
  return;
}


/*
   void computeStaggeredOprodQuda(void** oprod,   
   void** fermion,
   int num_terms,
   double** coeff,
   QudaGaugeParam* param)
   {
   using namespace quda;
   profileStaggeredOprod.Start(QUDA_PROFILE_TOTAL);

   checkGaugeParam(param);

   profileStaggeredOprod.Start(QUDA_PROFILE_INIT);
   GaugeFieldParam oParam(0, *param);

   oParam.nDim = 4;
   oParam.nFace = 0; 
// create the host outer-product field
oParam.pad = 0;
oParam.create = QUDA_REFERENCE_FIELD_CREATE;
oParam.link_type = QUDA_GENERAL_LINKS;
oParam.reconstruct = QUDA_RECONSTRUCT_NO;
oParam.order = QUDA_QDP_GAUGE_ORDER;
oParam.gauge = oprod[0];
cpuGaugeField cpuOprod0(oParam);

oParam.gauge = oprod[1];
cpuGaugeField cpuOprod1(oParam);

// create the device outer-product field
oParam.create = QUDA_ZERO_FIELD_CREATE;
oParam.order = QUDA_FLOAT2_GAUGE_ORDER;
cudaGaugeField cudaOprod0(oParam);
cudaGaugeField cudaOprod1(oParam);
initLatticeConstants(cudaOprod0, profileStaggeredOprod);

profileStaggeredOprod.Stop(QUDA_PROFILE_INIT); 


profileStaggeredOprod.Start(QUDA_PROFILE_H2D);
cudaOprod0.loadCPUField(cpuOprod0,QUDA_CPU_FIELD_LOCATION);
cudaOprod1.loadCPUField(cpuOprod1,QUDA_CPU_FIELD_LOCATION);
profileStaggeredOprod.Stop(QUDA_PROFILE_H2D);



ColorSpinorParam qParam;
qParam.nColor = 3;
qParam.nSpin = 1;
qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
qParam.nDim = 4;
qParam.precision = oParam.precision;
qParam.pad = 0;
for(int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];

// create the device quark field
qParam.create = QUDA_NULL_FIELD_CREATE;
qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
cudaColorSpinorField cudaQuark(qParam); 


cudaColorSpinorField**dQuark = new cudaColorSpinorField*[num_terms];
for(int i=0; i<num_terms; ++i){
dQuark[i] = new cudaColorSpinorField(qParam);
}

double* new_coeff  = new double[num_terms];

// create the host quark field
qParam.create = QUDA_REFERENCE_FIELD_CREATE;
qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
for(int i=0; i<num_terms; ++i){
  qParam.v = fermion[i];
  cpuColorSpinorField cpuQuark(qParam);
  *(dQuark[i]) = cpuQuark;
  new_coeff[i] = coeff[i][0];
}



// loop over different quark fields
for(int i=0; i<num_terms; ++i){
  computeKSOprodQuda(&cudaOprod0, dQuark[i], new_coeff[i], oParam.x, oParam.precision);
}



// copy the outer product field back to the host
profileStaggeredOprod.Start(QUDA_PROFILE_D2H);
cudaOprod0.saveCPUField(cpuOprod0,QUDA_CPU_FIELD_LOCATION);
cudaOprod1.saveCPUField(cpuOprod1,QUDA_CPU_FIELD_LOCATION);
profileStaggeredOprod.Stop(QUDA_PROFILE_D2H); 


for(int i=0; i<num_terms; ++i){
  delete dQuark[i];
}
delete[] dQuark;
delete[] new_coeff;

profileStaggeredOprod.Stop(QUDA_PROFILE_TOTAL);

checkCudaError();
return;
}
*/





void updateGaugeFieldQuda(void* gauge, 
    void* momentum, 
    double dt, 
    int conj_mom,
    int exact,
    QudaGaugeParam* param)
{
  profileGaugeUpdate.Start(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  profileGaugeUpdate.Start(QUDA_PROFILE_INIT);  
  GaugeFieldParam gParam(0, *param);

  // create the host fields
  gParam.pad = 0;
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = QUDA_SU3_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.gauge = gauge;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cpuGaugeField *cpuGauge = new cpuGaugeField(gParam);

  if (gParam.order == QUDA_TIFR_GAUGE_ORDER) {
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  } else {
    gParam.reconstruct = QUDA_RECONSTRUCT_10;
  }
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;

  gParam.gauge = momentum;

  cpuGaugeField *cpuMom = new cpuGaugeField(gParam);

  // create the device fields 
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;

  cudaGaugeField *cudaMom = !param->use_resident_mom ? new cudaGaugeField(gParam) : NULL;

  gParam.pad = param->ga_pad;
  gParam.link_type = QUDA_SU3_LINKS;
  gParam.reconstruct = param->reconstruct;

  cudaGaugeField *cudaInGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : NULL;
  cudaGaugeField *cudaOutGauge = new cudaGaugeField(gParam);

  profileGaugeUpdate.Stop(QUDA_PROFILE_INIT);  

  profileGaugeUpdate.Start(QUDA_PROFILE_H2D);

  /*printfQuda("UpdateGaugeFieldQuda use_resident_gauge = %d, make_resident_gauge = %d\n", 
    param->use_resident_gauge, param->make_resident_gauge);
    printfQuda("UpdateGaugeFieldQuda use_resident_mom = %d, make_resident_mom = %d\n", 
    param->use_resident_mom, param->make_resident_mom);*/

  if (!param->use_resident_gauge) {   // load fields onto the device
    cudaInGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  } else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = NULL;
  } 

  if (!param->use_resident_mom) {
    cudaMom->loadCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  } else {
    if (!momResident) errorQuda("No resident mom field allocated");
    cudaMom = momResident;
    momResident = NULL;
  }

  profileGaugeUpdate.Stop(QUDA_PROFILE_H2D);
  
  // perform the update
  profileGaugeUpdate.Start(QUDA_PROFILE_COMPUTE);
  updateGaugeField(*cudaOutGauge, dt, *cudaInGauge, *cudaMom, 
      (bool)conj_mom, (bool)exact);
  profileGaugeUpdate.Stop(QUDA_PROFILE_COMPUTE);

  // copy the gauge field back to the host
  profileGaugeUpdate.Start(QUDA_PROFILE_D2H);
  cudaOutGauge->saveCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  profileGaugeUpdate.Stop(QUDA_PROFILE_D2H);

  profileGaugeUpdate.Stop(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != NULL) delete gaugePrecise;
    gaugePrecise = cudaOutGauge;
  } else {
    delete cudaOutGauge;
  }

  if (param->make_resident_mom) {
    if (momResident != NULL && momResident != cudaMom) delete momResident;
    momResident = cudaMom;
  } else {
    delete cudaMom;
  }

  delete cudaInGauge;
  delete cpuMom;
  delete cpuGauge;

  checkCudaError();
  return;
}




/*
   The following functions are for the Fortran interface.
   */

void init_quda_(int *dev) { initQuda(*dev); }
void init_quda_device_(int *dev) { initQudaDevice(*dev); }
void init_quda_memory_() { initQudaMemory(); }
void end_quda_() { endQuda(); }
void load_gauge_quda_(void *h_gauge, QudaGaugeParam *param) { loadGaugeQuda(h_gauge, param); }
void free_gauge_quda_() { freeGaugeQuda(); }
void free_sloppy_gauge_quda_() { freeSloppyGaugeQuda(); }
void load_clover_quda_(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param) 
{ loadCloverQuda(h_clover, h_clovinv, inv_param); }
void free_clover_quda_(void) { freeCloverQuda(); }
void dslash_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
    QudaParity *parity) { dslashQuda(h_out, h_in, inv_param, *parity); }
void clover_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
    QudaParity *parity, int *inverse) { cloverQuda(h_out, h_in, inv_param, *parity, *inverse); }
void mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param)
{ MatQuda(h_out, h_in, inv_param); }
void mat_dag_mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param)
{ MatDagMatQuda(h_out, h_in, inv_param); }
void invert_quda_(void *hp_x, void *hp_b, QudaInvertParam *param) 
{ invertQuda(hp_x, hp_b, param); }    
void invert_md_quda_(void *hp_x, void *hp_b, QudaInvertParam *param) 
{ invertMDQuda(hp_x, hp_b, param); }    
void new_quda_gauge_param_(QudaGaugeParam *param) {
  *param = newQudaGaugeParam();
}
void new_quda_invert_param_(QudaInvertParam *param) {
  *param = newQudaInvertParam();
}
void update_gauge_field_quda_(void *gauge, void *momentum, double *dt, 
    bool *conj_mom, bool *exact, 
    QudaGaugeParam *param) {
  updateGaugeFieldQuda(gauge, momentum, *dt, (int)*conj_mom, (int)*exact, param);
}

int compute_gauge_force_quda_(void *mom, void *gauge,  int *input_path_buf, int *path_length,
    double *loop_coeff, int *num_paths, int *max_length, double *dt,
    QudaGaugeParam *param) {

  // fortran uses multi-dimenional arrays which we have convert into an array of pointers to pointers 
  const int dim = 4;
  int ***input_path = (int***)safe_malloc(dim*sizeof(int**));
  for (int i=0; i<dim; i++) {
    input_path[i] = (int**)safe_malloc(*num_paths*sizeof(int*));
    for (int j=0; j<*num_paths; j++) {
      input_path[i][j] = (int*)safe_malloc(path_length[j]*sizeof(int));
      for (int k=0; k<path_length[j]; k++) {
        input_path[i][j][k] = input_path_buf[(i* (*num_paths) + j)* (*max_length) + k];
      }
    }
  }

  computeGaugeForceQuda(mom, gauge, input_path, path_length, loop_coeff, *num_paths, *max_length, *dt, param, 0);

  for (int i=0; i<dim; i++) {
    for (int j=0; j<*num_paths; j++) { host_free(input_path[i][j]); }
    host_free(input_path[i]);
  }
  host_free(input_path);

  return 0;
}

void compute_staggered_force_quda_(void* cudaMom, void* qudaQuark, double *coeff) {
  computeStaggeredForceQuda(cudaMom, qudaQuark, *coeff);
}

// apply the staggered phases
void apply_staggered_phase_quda_() {
  printfQuda("applying staggered phase\n");
  if (gaugePrecise) {
    gaugePrecise->applyStaggeredPhase();
  } else {
    errorQuda("No persistent gauge field");
  }
}

// remove the staggered phases
void remove_staggered_phase_quda_() {
  printfQuda("removing staggered phase\n");
  if (gaugePrecise) {
    gaugePrecise->removeStaggeredPhase();
  } else {
    errorQuda("No persistent gauge field");
  }
  cudaDeviceSynchronize();
}

/**
 * BQCD wants a node mapping with x varying fastest.
 */
static int bqcd_rank_from_coords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = dims[i] * rank + coords[i];
  }
  return rank;
}

void comm_set_gridsize_(int *grid)
{
#ifdef MULTI_GPU
  initCommsGridQuda(4, grid, bqcd_rank_from_coords, static_cast<void *>(grid));
#endif
}

/**
 * Exposed due to poor derived MPI datatype performance with GPUDirect RDMA
 */
void set_kernel_pack_t_(int* pack) 
{
  bool pack_ = *pack ? true : false;
  setKernelPackT(pack_);
}

