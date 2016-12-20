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
#include <ritz_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <lanczos_quda.h>
#include <color_spinor_field.h>
#include <eig_variables.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <fat_force_quda.h>
#include <unitarization_links.h>
#include <algorithm>
#include <staggered_oprod.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>

#include <multigrid.h>

#ifdef NUMA_NVML
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

#include <gauge_tools.h>
#include <contractQuda.h>

#include <momentum.h>


using namespace quda;

static int R[4] = {0, 0, 0, 0};
// setting this to false prevents redundant halo exchange but isn't yet compatible with HISQ / ASQTAD kernels
static bool redundant_comms = false;

//for MAGMA lib:
#include <blas_magma.h>

static bool InitMagma = false;

void openMagma() {

  if (!InitMagma) {
    BlasMagmaArgs::OpenMagma();
    InitMagma = true;
  } else {
    printfQuda("\nMAGMA library was already initialized..\n");
  }

}

void closeMagma(){

  if (InitMagma) {
    BlasMagmaArgs::CloseMagma();
    InitMagma = false;
  } else {
    printfQuda("\nMAGMA library was not initialized..\n");
  }

  return;
}

cudaGaugeField *gaugePrecise = NULL;
cudaGaugeField *gaugeSloppy = NULL;
cudaGaugeField *gaugePrecondition = NULL;
cudaGaugeField *gaugeExtended = NULL;

// It's important that these alias the above so that constants are set correctly in Dirac::Dirac()
cudaGaugeField *&gaugeFatPrecise = gaugePrecise;
cudaGaugeField *&gaugeFatSloppy = gaugeSloppy;
cudaGaugeField *&gaugeFatPrecondition = gaugePrecondition;
cudaGaugeField *&gaugeFatExtended = gaugeExtended;


cudaGaugeField *gaugeLongExtended = NULL;
cudaGaugeField *gaugeLongPrecise = NULL;
cudaGaugeField *gaugeLongSloppy = NULL;
cudaGaugeField *gaugeLongPrecondition = NULL;

cudaGaugeField *gaugeSmeared = NULL;

cudaCloverField *cloverPrecise = NULL;
cudaCloverField *cloverSloppy = NULL;
cudaCloverField *cloverPrecondition = NULL;

cudaGaugeField *momResident = NULL;
cudaGaugeField *extendedGaugeResident = NULL;

std::vector<cudaColorSpinorField*> solutionResident;

// Mapped memory buffer used to hold unitarization failures
static int *num_failures_h = NULL;
static int *num_failures_d = NULL;

cudaDeviceProp deviceProp;
cudaStream_t *streams;
#ifdef PTHREADS
pthread_mutex_t pthread_mutex;
#endif

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

//!< Profiler for computeFatLinkQuda
static TimeProfile profileFatLink("computeKSLinkQuda");

//!< Profiler for computeGaugeForceQuda
static TimeProfile profileGaugeForce("computeGaugeForceQuda");

//!<Profiler for updateGaugeFieldQuda
static TimeProfile profileGaugeUpdate("updateGaugeFieldQuda");

//!<Profiler for createExtendedGaugeField
static TimeProfile profileExtendedGauge("createExtendedGaugeField");


//!<Profiler for computeCloverForceQuda
static TimeProfile profileCloverForce("computeCloverForceQuda");

//!<Profiler for computeStaggeredOprodQuda
static TimeProfile profileStaggeredOprod("computeStaggeredOprodQuda");

//!<Profiler for computeAsqtadForceQuda
static TimeProfile profileAsqtadForce("computeAsqtadForceQuda");

//!<Profiler for computeAsqtadForceQuda
static TimeProfile profileHISQForce("computeHISQForceQuda");

//!<Profiler for computeHISQForceCompleteQuda
static TimeProfile profileHISQForceComplete("computeHISQForceCompleteQuda");

//!<Profiler for plaqQuda
static TimeProfile profilePlaq("plaqQuda");

//!< Profiler for APEQuda
static TimeProfile profileAPE("APEQuda");

//!< Profiler for STOUTQuda
static TimeProfile profileSTOUT("STOUTQuda");

//!< Profiler for projectSU3Quda
static TimeProfile profileProject("projectSU3Quda");

//!< Profiler for staggeredPhaseQuda
static TimeProfile profilePhase("staggeredPhaseQuda");

//!< Profiler for contractions
static TimeProfile profileContract("contractQuda");

//!< Profiler for contractions
static TimeProfile profileCovDev("covDevQuda");

//!< Profiler for contractions
static TimeProfile profileMomAction("momActionQuda");

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");

//!< Profiler for GaugeFixing
static TimeProfile GaugeFixFFTQuda("GaugeFixFFTQuda");
static TimeProfile GaugeFixOVRQuda("GaugeFixOVRQuda");



//!< Profiler for toal time spend between init and end
static TimeProfile profileInit2End("initQuda-endQuda",false);

namespace quda {
  void printLaunchTimer();
}

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

  LexMapData map_data;
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


#define STR_(x) #x
#define STR(x) STR_(x)
  static const std::string quda_version = STR(QUDA_VERSION_MAJOR) "." STR(QUDA_VERSION_MINOR) "." STR(QUDA_VERSION_SUBMINOR);
#undef STR
#undef STR_

extern char* gitversion;

/*
 * Set the device that QUDA uses.
 */
void initQudaDevice(int dev) {
  //static bool initialized = false;
  if (initialized) return;
  initialized = true;

  profileInit2End.TPSTART(QUDA_PROFILE_TOTAL);
  profileInit.TPSTART(QUDA_PROFILE_TOTAL);

  if (getVerbosity() >= QUDA_SUMMARIZE) {
#ifdef GITVERSION
    printfQuda("QUDA %s (git %s)\n",quda_version.c_str(),gitversion);
#else
    printfQuda("QUDA %s\n",quda_version.c_str());
#endif
  }

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


#if ((CUDA_VERSION >= 6000) && defined NUMA_NVML)
  char *enable_numa_env = getenv("QUDA_ENABLE_NUMA");
  if (enable_numa_env && strcmp(enable_numa_env, "0") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling numa_affinity\n");
  }
  else{
    setNumaAffinityNVML(dev);
  }
#endif



  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaGetDeviceProperties(&deviceProp, dev);

  { // determine if we will do CPU or GPU data reordering (default is GPU)
    char *reorder_str = getenv("QUDA_REORDER_LOCATION");

    if (!reorder_str || (strcmp(reorder_str,"CPU") && strcmp(reorder_str,"cpu")) ) {
      warningQuda("Data reordering done on GPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      reorder_location_set(QUDA_CUDA_FIELD_LOCATION);
    } else {
      warningQuda("Data reordering done on CPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      reorder_location_set(QUDA_CPU_FIELD_LOCATION);
    }
  }

  profileInit.TPSTOP(QUDA_PROFILE_TOTAL);
}

/*
 * Any persistent memory allocations that QUDA uses are done here.
 */
void initQudaMemory()
{
  profileInit.TPSTART(QUDA_PROFILE_TOTAL);

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
#ifdef GPU_STAGGERED_OPROD
  createStaggeredOprodEvents();
#endif
  blas::init();

  // initalize the memory pool allocators
  pool::init();

  num_failures_h = static_cast<int*>(mapped_malloc(sizeof(int)));
  cudaHostGetDevicePointer(&num_failures_d, num_failures_h, 0);

  loadTuneCache();

  for (int d=0; d<4; d++) R[d] = 2 * (redundant_comms || commDimPartitioned(d));

  profileInit.TPSTOP(QUDA_PROFILE_TOTAL);
}

void initQuda(int dev)
{
  // initialize communications topology, if not already done explicitly via initCommsGridQuda()
  if (!comms_initialized) init_default_comms();

  // set the device that QUDA uses
  initQudaDevice(dev);

  // set the persistant memory allocations that QUDA uses (Blas, streams, etc.)
  initQudaMemory();

#ifdef PTHREADS
  pthread_mutexattr_t mutex_attr;
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&pthread_mutex, &mutex_attr);
#endif
}


void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  checkGaugeParam(param);

  profileGauge.TPSTART(QUDA_PROFILE_INIT);
  // Set the specific input parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);

  // if we are using half precision then we need to compute the fat
  // link maximum while still on the cpu
  // FIXME get a kernel for this
  if (param->type == QUDA_ASQTAD_FAT_LINKS)
    gauge_param.compute_fat_link_max = true;

  GaugeField *in = (param->location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<GaugeField*>(new cpuGaugeField(gauge_param)) :
    static_cast<GaugeField*>(new cudaGaugeField(gauge_param));

  // free any current gauge field before new allocations to reduce memory overhead
  switch (param->type) {
    case QUDA_WILSON_LINKS:
      if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
      if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
      if (gaugePrecise && !param->use_resident_gauge) delete gaugePrecise;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
      if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
      if (gaugeFatPrecise && !param->use_resident_gauge) delete gaugeFatPrecise;
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
      if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
      if (gaugeLongPrecise) delete gaugeLongPrecise;
      break;
    default:
      errorQuda("Invalid gauge type %d", param->type);
  }

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
    profileGauge.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    profileGauge.TPSTOP(QUDA_PROFILE_INIT);
    profileGauge.TPSTART(QUDA_PROFILE_H2D);
    precise->copy(*in);
    profileGauge.TPSTOP(QUDA_PROFILE_H2D);
  }

  param->gaugeGiB += precise->GBytes();

  // creating sloppy fields isn't really compute, but it is work done on the gpu
  profileGauge.TPSTART(QUDA_PROFILE_COMPUTE);

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.precision = param->cuda_prec_sloppy;
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION ||
      gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *sloppy = NULL;
  if (param->cuda_prec != param->cuda_prec_sloppy ||
      param->reconstruct != param->reconstruct_sloppy) {
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
  if (param->cuda_prec_sloppy != param->cuda_prec_precondition ||
      param->reconstruct_sloppy != param->reconstruct_precondition) {
    precondition = new cudaGaugeField(gauge_param);
    precondition->copy(*sloppy);
    param->gaugeGiB += precondition->GBytes();
  } else {
    precondition = sloppy;
  }

  // create an extended preconditioning field
  cudaGaugeField* extended = NULL;
  if(param->overlap){
    int R[4]; // domain-overlap widths in different directions
    for(int i=0; i<4; ++i){
      R[i] = param->overlap*commDimPartitioned(i);
      gauge_param.x[i] += 2*R[i];
    }
    // the extended field does not require any ghost padding
    gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    extended = new cudaGaugeField(gauge_param);

    // copy the unextended preconditioning field into the interior of the extended field
    copyExtendedGauge(*extended, *precondition, QUDA_CUDA_FIELD_LOCATION);
    // now perform communication and fill the overlap regions
    extended->exchangeExtendedGhost(R);
  }

  profileGauge.TPSTOP(QUDA_PROFILE_COMPUTE);

  switch (param->type) {
    case QUDA_WILSON_LINKS:
      gaugePrecise = precise;
      gaugeSloppy = sloppy;
      gaugePrecondition = precondition;

      if(param->overlap) gaugeExtended = extended;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      gaugeFatPrecise = precise;
      gaugeFatSloppy = sloppy;
      gaugeFatPrecondition = precondition;

      if(param->overlap){
        if(gaugeFatExtended) errorQuda("Extended gauge fat field already allocated");
	gaugeFatExtended = extended;
      }
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      gaugeLongPrecise = precise;
      gaugeLongSloppy = sloppy;
      gaugeLongPrecondition = precondition;

      if(param->overlap){
        if(gaugeLongExtended) errorQuda("Extended gauge long field already allocated");
   	gaugeLongExtended = extended;
      }
      break;
    default:
      errorQuda("Invalid gauge type %d", param->type);
  }


  profileGauge.TPSTART(QUDA_PROFILE_FREE);
  delete in;
  profileGauge.TPSTOP(QUDA_PROFILE_FREE);

  profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.TPSTART(QUDA_PROFILE_TOTAL);

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

  profileGauge.TPSTART(QUDA_PROFILE_D2H);
  cudaGauge->saveCPUField(cpuGauge);
  profileGauge.TPSTOP(QUDA_PROFILE_D2H);

  profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
}


void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  if (!gaugePrecise) errorQuda("Cannot call loadCloverQuda with no resident gauge field");

  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  profileClover.TPSTART(QUDA_PROFILE_INIT);
  bool device_calc = false; // calculate clover and inverse on the device?

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if (!initialized) errorQuda("QUDA not initialized");

  if ( (!h_clover && !h_clovinv) || inv_param->compute_clover ) {
    device_calc = true;
    if (inv_param->clover_coeff == 0.0) errorQuda("called with neither clover term nor inverse and clover coefficient not set");
    if (gaugePrecise->Anisotropy() != 1.0) errorQuda("cannot compute anisotropic clover field");
  }

  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION)  errorQuda("Half precision not supported on CPU");
  if (gaugePrecise == NULL) errorQuda("Gauge field must be loaded before clover");
  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH)) {
    errorQuda("Wrong dslash_type %d in loadCloverQuda()", inv_param->dslash_type);
  }

  // determines whether operator is preconditioned when calling invertQuda()
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE ||
      inv_param->solve_type == QUDA_NORMOP_PC_SOLVE ||
      inv_param->solve_type == QUDA_NORMERR_PC_SOLVE );

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

  bool twisted = inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? true : false;
#ifdef DYNAMIC_CLOVER
  bool dynamic_clover = twisted ? true : false; // dynamic clover only supported on twisted clover currently
#else
  bool dynamic_clover = false;
#endif

  CloverFieldParam clover_param;
  clover_param.nDim = 4;
  clover_param.twisted = twisted;
  clover_param.mu2 = twisted ? 4.*inv_param->kappa*inv_param->kappa*inv_param->mu*inv_param->mu : 0.0;
  clover_param.siteSubset = QUDA_FULL_SITE_SUBSET;
  for (int i=0; i<4; i++) clover_param.x[i] = gaugePrecise->X()[i];
  clover_param.pad = inv_param->cl_pad;
  clover_param.create = QUDA_NULL_FIELD_CREATE;
  clover_param.norm = nullptr;
  clover_param.invNorm = nullptr;
  clover_param.setPrecision(inv_param->clover_cuda_prec);
  clover_param.direct = h_clover || device_calc ? true : false;
  clover_param.inverse = (h_clovinv || pc_solve) && !dynamic_clover ? true : false;

  cloverPrecise = new cudaCloverField(clover_param);

  CloverField *in = nullptr;

  if (!device_calc || inv_param->return_clover || inv_param->return_clover_inverse) {
    // create a param for the cpu clover field
    CloverFieldParam inParam(clover_param);
    inParam.precision = inv_param->clover_cpu_prec;
    inParam.order = inv_param->clover_order;
    inParam.direct = h_clover ? true : false;
    inParam.inverse = h_clovinv ? true : false;
    inParam.clover = h_clover;
    inParam.cloverInv = h_clovinv;
    inParam.create = QUDA_REFERENCE_FIELD_CREATE;
    in = (inv_param->clover_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<CloverField*>(new cpuCloverField(inParam)) :
      static_cast<CloverField*>(new cudaCloverField(inParam));
  }
  profileClover.TPSTOP(QUDA_PROFILE_INIT);

  if (!device_calc) {
    profileClover.TPSTART(QUDA_PROFILE_H2D);
    cloverPrecise->copy(*in, h_clovinv && !inv_param->compute_clover_inverse ? true : false);
    profileClover.TPSTOP(QUDA_PROFILE_H2D);
  } else {
    profileClover.TPSTOP(QUDA_PROFILE_TOTAL);
    createCloverQuda(inv_param);
    profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  }

  // inverted clover term is required when applying preconditioned operator
  if ((!h_clovinv || inv_param->compute_clover_inverse) && pc_solve) {
    profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
    if (!dynamic_clover) {
      cloverInvert(*cloverPrecise, inv_param->compute_clover_trlog, QUDA_CUDA_FIELD_LOCATION);
      if (inv_param->compute_clover_trlog) {
	inv_param->trlogA[0] = cloverPrecise->TrLog()[0];
	inv_param->trlogA[1] = cloverPrecise->TrLog()[1];
      }
    }
    profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  inv_param->cloverGiB = cloverPrecise->GBytes();

  clover_param.direct = true;
  clover_param.inverse = dynamic_clover ? false : true;

  // create the mirror sloppy clover field
  if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
    profileClover.TPSTART(QUDA_PROFILE_INIT);

    clover_param.setPrecision(inv_param->clover_cuda_prec_sloppy);
    cloverSloppy = new cudaCloverField(clover_param);
    cloverSloppy->copy(*cloverPrecise, clover_param.inverse);

    inv_param->cloverGiB += cloverSloppy->GBytes();
    profileClover.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    cloverSloppy = cloverPrecise;
  }

  // create the mirror preconditioner clover field
  if (inv_param->clover_cuda_prec_sloppy != inv_param->clover_cuda_prec_precondition &&
      inv_param->clover_cuda_prec_precondition != QUDA_INVALID_PRECISION) {
    profileClover.TPSTART(QUDA_PROFILE_INIT);

    clover_param.setPrecision(inv_param->clover_cuda_prec_precondition);
    cloverPrecondition = new cudaCloverField(clover_param);
    cloverPrecondition->copy(*cloverSloppy, clover_param.inverse);

    inv_param->cloverGiB += cloverPrecondition->GBytes();
    profileClover.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    cloverPrecondition = cloverSloppy;
  }

  // if requested, copy back the clover / inverse field
  if ( inv_param->return_clover || inv_param->return_clover_inverse ) {
    if (!h_clover && !h_clovinv) errorQuda("Requested clover field return but no clover host pointers set");

    // copy the inverted clover term into host application order on the device
    clover_param.setPrecision(inv_param->clover_cpu_prec);
    clover_param.direct = (h_clover && inv_param->return_clover);
    clover_param.inverse = (h_clovinv && inv_param->return_clover_inverse);

    // this isn't really "epilogue" but this label suffices
    profileClover.TPSTART(QUDA_PROFILE_EPILOGUE);
    cudaCloverField *hack = nullptr;
    if (!dynamic_clover) {
      clover_param.order = inv_param->clover_order;
      hack = new cudaCloverField(clover_param);
      hack->copy(*cloverPrecise); // FIXME this can lead to an redundant copies if we're not copying back direct + inverse
    } else {
      cudaCloverField *hackOfTheHack = new cudaCloverField(clover_param);	// Hack of the hack
      hackOfTheHack->copy(*cloverPrecise, false);
      cloverInvert(*hackOfTheHack, inv_param->compute_clover_trlog, QUDA_CUDA_FIELD_LOCATION);
      if (inv_param->compute_clover_trlog) {
	inv_param->trlogA[0] = cloverPrecise->TrLog()[0];
	inv_param->trlogA[1] = cloverPrecise->TrLog()[1];
      }
      clover_param.order = inv_param->clover_order;
      hack = new cudaCloverField(clover_param);
      hack->copy(*hackOfTheHack); // FIXME this can lead to an redundant copies if we're not copying back direct + inverse
      delete hackOfTheHack;
    }
    profileClover.TPSTOP(QUDA_PROFILE_EPILOGUE);

    // copy the field into the host application's clover field
    profileClover.TPSTART(QUDA_PROFILE_D2H);
    if (inv_param->return_clover) {
      qudaMemcpy((char*)(in->V(false)), (char*)(hack->V(false)), in->Bytes(), cudaMemcpyDeviceToHost);
    }
    if (inv_param->return_clover_inverse) {
      qudaMemcpy((char*)(in->V(true)), (char*)(hack->V(true)), in->Bytes(), cudaMemcpyDeviceToHost);
    }
    profileClover.TPSTOP(QUDA_PROFILE_D2H);

    delete hack;
    checkCudaError();
  }

  profileClover.TPSTART(QUDA_PROFILE_FREE);
  if (in) delete in; // delete object referencing input field
  profileClover.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  profileClover.TPSTOP(QUDA_PROFILE_TOTAL);
}

void loadSloppyCloverQuda(QudaPrecision prec_sloppy, QudaPrecision prec_precondition)
{

  if (cloverPrecise) {
    // create the mirror sloppy clover field
    CloverFieldParam clover_param(*cloverPrecise);
    clover_param.setPrecision(prec_sloppy);

    if (cloverPrecise->V(false) != cloverPrecise->V(true)) {
      clover_param.direct = true;
      clover_param.inverse = true;
    } else {
      clover_param.direct = false;
      clover_param.inverse = true;
    }

    if (cloverSloppy) errorQuda("cloverSloppy already exists");

    if (clover_param.precision != cloverPrecise->Precision()) {
      cloverSloppy = new cudaCloverField(clover_param);
      cloverSloppy->copy(*cloverPrecise, clover_param.inverse);
    } else {
      cloverSloppy = cloverPrecise;
    }

    // switch the parameteres for creating the mirror preconditioner clover field
    clover_param.setPrecision(prec_precondition);

    if (cloverPrecondition) errorQuda("cloverPrecondition already exists");

    // create the mirror preconditioner clover field
    if (clover_param.precision != cloverSloppy->Precision()) {
      cloverPrecondition = new cudaCloverField(clover_param);
      cloverPrecondition->copy(*cloverSloppy, clover_param.inverse);
    } else {
      cloverPrecondition = cloverSloppy;
    }
  }

}

void freeGaugeQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
  if (gaugePrecise) delete gaugePrecise;
  if (gaugeExtended) delete gaugeExtended;

  gaugePrecondition = NULL;
  gaugeSloppy = NULL;
  gaugePrecise = NULL;
  gaugeExtended = NULL;

  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
  if (gaugeLongPrecise) delete gaugeLongPrecise;
  if (gaugeLongExtended) delete gaugeLongExtended;

  gaugeLongPrecondition = NULL;
  gaugeLongSloppy = NULL;
  gaugeLongPrecise = NULL;
  gaugeLongExtended = NULL;

  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
  if (gaugeFatPrecise) delete gaugeFatPrecise;

  gaugeFatPrecondition = NULL;
  gaugeFatSloppy = NULL;
  gaugeFatPrecise = NULL;
  gaugeFatExtended = NULL;

  if (gaugeSmeared) delete gaugeSmeared;

  gaugeSmeared = NULL;
  // Need to merge extendedGaugeResident and gaugeFatPrecise/gaugePrecise
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


void loadSloppyGaugeQuda(QudaPrecision prec_sloppy, QudaPrecision prec_precondition)
{
  // first do SU3 links (if they exist)
  if (gaugePrecise) {
    GaugeFieldParam gauge_param(*gaugePrecise);
    gauge_param.setPrecision(prec_sloppy);
    //gauge_param.reconstruct = param->reconstruct_sloppy; // FIXME

    if (gaugeSloppy) errorQuda("gaugeSloppy already exists");

    if (gauge_param.precision != gaugePrecise->Precision() ||
	gauge_param.reconstruct != gaugePrecise->Reconstruct()) {
      gaugeSloppy = new cudaGaugeField(gauge_param);
      gaugeSloppy->copy(*gaugePrecise);
    } else {
      gaugeSloppy = gaugePrecise;
    }

    // switch the parameters for creating the mirror preconditioner cuda gauge field
    gauge_param.setPrecision(prec_precondition);
    //gauge_param.reconstruct = param->reconstruct_precondition; // FIXME

    if (gaugePrecondition) errorQuda("gaugePrecondition already exists");

    if (gauge_param.precision != gaugeSloppy->Precision() ||
	gauge_param.reconstruct != gaugeSloppy->Reconstruct()) {
      gaugePrecondition = new cudaGaugeField(gauge_param);
      gaugePrecondition->copy(*gaugeSloppy);
    } else {
      gaugePrecondition = gaugeSloppy;
    }
  }

  // fat links (if they exist)
  if (gaugeFatPrecise) {
    GaugeFieldParam gauge_param(*gaugeFatPrecise);

    if (gaugeFatSloppy != gaugeSloppy) {
      gauge_param.setPrecision(prec_sloppy);
      //gauge_param.reconstruct = param->reconstruct_sloppy; // FIXME

      if (gaugeFatSloppy) errorQuda("gaugeFatSloppy already exists");
      if (gaugeFatSloppy != gaugeFatPrecise) delete gaugeFatSloppy;

      if (gauge_param.precision != gaugeFatPrecise->Precision() ||
	  gauge_param.reconstruct != gaugeFatPrecise->Reconstruct()) {
	gaugeFatSloppy = new cudaGaugeField(gauge_param);
	gaugeFatSloppy->copy(*gaugeFatPrecise);
      } else {
	gaugeFatSloppy = gaugeFatPrecise;
      }
    }

    if (gaugeFatPrecondition != gaugePrecondition) {
      // switch the parameters for creating the mirror preconditioner cuda gauge field
      gauge_param.setPrecision(prec_precondition);
      //gauge_param.reconstruct = param->reconstruct_precondition; // FIXME

      if (gaugeFatPrecondition) errorQuda("gaugeFatPrecondition already exists\n");

      if (gauge_param.precision != gaugeFatSloppy->Precision() ||
	  gauge_param.reconstruct != gaugeFatSloppy->Reconstruct()) {
	gaugeFatPrecondition = new cudaGaugeField(gauge_param);
	gaugeFatPrecondition->copy(*gaugeFatSloppy);
      } else {
	gaugeFatPrecondition = gaugeFatSloppy;
      }
    }
  }

  // long links (if they exist)
  if (gaugeLongPrecise) {
    GaugeFieldParam gauge_param(*gaugeLongPrecise);
    gauge_param.setPrecision(prec_sloppy);
    //gauge_param.reconstruct = param->reconstruct_sloppy; // FIXME

    if (gaugeLongSloppy) errorQuda("gaugeLongSloppy already exists");
    if (gaugeLongSloppy != gaugeLongPrecise) delete gaugeLongSloppy;

    if (gauge_param.precision != gaugeLongPrecise->Precision() ||
	gauge_param.reconstruct != gaugeLongPrecise->Reconstruct()) {
      gaugeLongSloppy = new cudaGaugeField(gauge_param);
      gaugeLongSloppy->copy(*gaugeLongPrecise);
    } else {
      gaugeLongSloppy = gaugeLongPrecise;
    }

    // switch the parameters for creating the mirror preconditioner cuda gauge field
    gauge_param.setPrecision(prec_precondition);
    //gauge_param.reconstruct = param->reconstruct_precondition; // FIXME

    if (gaugeLongPrecondition) warningQuda("gaugeLongPrecondition already exists\n");

    if (gauge_param.precision != gaugeLongSloppy->Precision() ||
	gauge_param.reconstruct != gaugeLongSloppy->Reconstruct()) {
      gaugeLongPrecondition = new cudaGaugeField(gauge_param);
      gaugeLongPrecondition->copy(*gaugeLongSloppy);
    } else {
      gaugeLongPrecondition = gaugeLongSloppy;
    }
  }
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
}

void freeSloppyCloverQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");
  if (cloverPrecondition != cloverSloppy && cloverPrecondition) delete cloverPrecondition;
  if (cloverSloppy != cloverPrecise && cloverSloppy) delete cloverSloppy;

  cloverPrecondition = NULL;
  cloverSloppy = NULL;
}

void endQuda(void)
{
  profileEnd.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) return;

  freeGaugeQuda();
  freeCloverQuda();

  for (unsigned int i=0; i<solutionResident.size(); i++) {
    if(solutionResident[i]) delete solutionResident[i];
  }
  solutionResident.clear();
  if(momResident) delete momResident;

  LatticeField::freeBuffer(0);
  LatticeField::freeBuffer(1);
  cudaColorSpinorField::freeBuffer(0);
  cudaColorSpinorField::freeBuffer(1);
  cudaColorSpinorField::freeGhostBuffer();
  cpuColorSpinorField::freeGhostBuffer();
  FaceBuffer::flushPinnedCache();

  blas::end();

  pool::flush_pinned();
  pool::flush_device();

  host_free(num_failures_h);
  num_failures_h = NULL;
  num_failures_d = NULL;

  if (streams) {
    for (int i=0; i<Nstream; i++) cudaStreamDestroy(streams[i]);
    delete []streams;
    streams = NULL;
  }
  destroyDslashEvents();

#ifdef GPU_STAGGERED_OPROD
  destroyStaggeredOprodEvents();
#endif

  saveTuneCache();
  saveProfile();

  initialized = false;

  comm_finalize();
  comms_initialized = false;

  profileEnd.TPSTOP(QUDA_PROFILE_TOTAL);
  profileInit2End.TPSTOP(QUDA_PROFILE_TOTAL);

  // print out the profile information of the lifetime of the library
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    profileInit.Print();
    profileGauge.Print();
    profileClover.Print();
    profileInvert.Print();
    profileMulti.Print();
    profileFatLink.Print();
    profileGaugeForce.Print();
    profileGaugeUpdate.Print();
    profileExtendedGauge.Print();
    profileCloverForce.Print();
    profileStaggeredOprod.Print();
    profileAsqtadForce.Print();
    profileHISQForce.Print();
    profileContract.Print();
    profileCovDev.Print();
    profilePlaq.Print();
    profileAPE.Print();
    profileSTOUT.Print();
    profileProject.Print();
    profilePhase.Print();
    profileMomAction.Print();
    profileEnd.Print();

    profileInit2End.Print();
    TimeProfile::PrintGlobal();

    printLaunchTimer();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }

  assertAllMemFree();

#if (!defined(USE_QDPJIT) && !defined(GPU_COMMS))
  // end this CUDA context
  cudaDeviceReset();
#endif

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
    case QUDA_DOMAIN_WALL_4D_DSLASH:
      if(pc) {
	diracParam.type = QUDA_DOMAIN_WALL_4DPC_DIRAC;
	diracParam.Ls = inv_param->Ls;
      } else errorQuda("For 4D type of DWF dslash, pc must be turned on, %d", inv_param->dslash_type);
      break;
    case QUDA_MOBIUS_DWF_DSLASH:
      if (inv_param->Ls > QUDA_MAX_DWF_LS)
	errorQuda("Length of Ls dimension %d greater than QUDA_MAX_DWF_LS %d", inv_param->Ls, QUDA_MAX_DWF_LS);
      diracParam.type = pc ? QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC : QUDA_MOBIUS_DOMAIN_WALL_DIRAC;
      diracParam.Ls = inv_param->Ls;
      memcpy(diracParam.b_5, inv_param->b_5, sizeof(double)*inv_param->Ls);
      memcpy(diracParam.c_5, inv_param->c_5, sizeof(double)*inv_param->Ls);
      break;
    case QUDA_STAGGERED_DSLASH:
      diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
      break;
    case QUDA_ASQTAD_DSLASH:
      diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
      break;
    case QUDA_TWISTED_MASS_DSLASH:
      diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
      if (inv_param->twist_flavor == QUDA_TWIST_MINUS || inv_param->twist_flavor == QUDA_TWIST_PLUS) {
	diracParam.Ls = 1;
	diracParam.epsilon = 0.0;
      } else {
	diracParam.Ls = 2;
	diracParam.epsilon = inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ? inv_param->epsilon : 0.0;
      }
      break;
    case QUDA_TWISTED_CLOVER_DSLASH:
      diracParam.type = pc ? QUDA_TWISTED_CLOVERPC_DIRAC : QUDA_TWISTED_CLOVER_DIRAC;
      if (inv_param->twist_flavor == QUDA_TWIST_MINUS || inv_param->twist_flavor == QUDA_TWIST_PLUS)  {
	diracParam.Ls = 1;
	diracParam.epsilon = 0.0;
      } else {
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
    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on
  }


  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugeSloppy;
    diracParam.fatGauge = gaugeFatSloppy;
    diracParam.longGauge = gaugeLongSloppy;
    diracParam.clover = cloverSloppy;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }

  }

  // The preconditioner currently mimicks the sloppy operator with no comms
  void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc, bool comms)
  {
    setDiracParam(diracParam, inv_param, pc);

    if(inv_param->overlap){
      diracParam.gauge = gaugeExtended;
      diracParam.fatGauge = gaugeFatExtended;
      diracParam.longGauge = gaugeLongExtended;
    }else{
      diracParam.gauge = gaugePrecondition;
      diracParam.fatGauge = gaugeFatPrecondition;
      diracParam.longGauge = gaugeLongPrecondition;
    }
    diracParam.clover = cloverPrecondition;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = comms ? 1 : 0;
    }

    // In the preconditioned staggered CG allow a different dslash type in the preconditioning
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
    setDiracPreParam(diracPreParam, &param, pc_solve, false);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
  }

  static double unscaled_shifts[QUDA_MAX_MULTI_SHIFT];

  void massRescale(cudaColorSpinorField &b, QudaInvertParam &param) {

    double kappa5 = (0.5/(5.0 + param.m5));
    double kappa = (param.dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
		    param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
		    param.dslash_type == QUDA_MOBIUS_DWF_DSLASH) ? kappa5 : param.kappa;

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", param.mass_normalization);
      double nin = blas::norm2(b);
      printfQuda("Mass rescale: norm of source in = %g\n", nin);
    }

    // staggered dslash uses mass normalization internally
    if (param.dslash_type == QUDA_ASQTAD_DSLASH || param.dslash_type == QUDA_STAGGERED_DSLASH) {
      switch (param.solution_type) {
        case QUDA_MAT_SOLUTION:
        case QUDA_MATPC_SOLUTION:
          if (param.mass_normalization == QUDA_KAPPA_NORMALIZATION) blas::ax(2.0*param.mass, b);
          break;
        case QUDA_MATDAG_MAT_SOLUTION:
        case QUDA_MATPCDAG_MATPC_SOLUTION:
          if (param.mass_normalization == QUDA_KAPPA_NORMALIZATION) blas::ax(4.0*param.mass*param.mass, b);
          break;
        default:
          errorQuda("Not implemented");
      }
      return;
    }

    for(int i=0; i<param.num_offset; i++) {
      unscaled_shifts[i] = param.offset[i];
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    switch (param.solution_type) {
      case QUDA_MAT_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION ||
            param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(2.0*kappa, b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 2.0*kappa;
        }
        break;
      case QUDA_MATDAG_MAT_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION ||
            param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 4.0*kappa*kappa;
        }
        break;
      case QUDA_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 4.0*kappa*kappa;
        } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(2.0*kappa, b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 2.0*kappa;
        }
        break;
      case QUDA_MATPCDAG_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  blas::ax(16.0*pow(kappa,4), b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 16.0*pow(kappa,4);
        } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 4.0*kappa*kappa;
        }
        break;
      default:
        errorQuda("Solution type %d not supported", param.solution_type);
    }

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", param.mass_normalization);
      double nin = blas::norm2(b);
      printfQuda("Mass rescale: norm of source out = %g\n", nin);
    }

  }
}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{
  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  if (inv_param->mass_normalization == QUDA_KAPPA_NORMALIZATION &&
      (inv_param->dslash_type == QUDA_STAGGERED_DSLASH ||
       inv_param->dslash_type == QUDA_ASQTAD_DSLASH) )
    blas::ax(1.0/(2.0*inv_param->mass), in);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    blas::ax(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH && inv_param->dagger) {
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    cudaColorSpinorField tmp1(in, cudaParam);
    ((DiracTwistedCloverPC*) dirac)->TwistCloverInv(tmp1, in, (parity+1)%2); // apply the clover-twist
    dirac->Dslash(out, tmp1, parity); // apply the operator
  } else {
    dirac->Dslash(out, in, parity); // apply the operator
  }

  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

void dslashQuda_4dpc(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int test_type)
{
  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH )
    setKernelPackT(true);
  else
    errorQuda("This type of dslashQuda operator is defined for QUDA_DOMAIN_WALL_$D_DSLASH and QUDA_MOBIUS_DWF_DSLASH only");

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
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
    blas::ax(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  DiracDomainWall4DPC dirac(diracParam); // create the Dirac operator
  printfQuda("kappa for QUDA input : %e\n",inv_param->kappa);
  switch (test_type) {
    case 0:
      dirac.Dslash4(out, in, parity);
      break;
    case 1:
      dirac.Dslash5(out, in, parity);
      break;
    case 2:
      dirac.Dslash5inv(out, in, parity, inv_param->kappa);
      break;
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

void dslashQuda_mdwf(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int test_type)
{
  if ( inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH)
    setKernelPackT(true);
  else
    errorQuda("This type of dslashQuda operator is defined for QUDA_DOMAIN_WALL_$D_DSLASH and QUDA_MOBIUS_DWF_DSLASH only");

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
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
    blas::ax(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  DiracMobiusPC dirac(diracParam); // create the Dirac operator
  switch (test_type) {
    case 0:
      dirac.Dslash4(out, in, parity);
      break;
    case 1:
      dirac.Dslash5(out, in, parity);
      break;
    case 2:
      dirac.Dslash4pre(out, in, parity);
      break;
    case 3:
      dirac.Dslash5inv(out, in, parity);
      break;
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
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
      blas::ax(0.25/(kappa*kappa), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      blas::ax(0.5/kappa, out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
        inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      blas::ax(0.5/kappa, out);
    }
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (cloverPrecise == NULL && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE){
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
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
      blas::ax(1.0/pow(2.0*kappa,4), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      blas::ax(0.25/(kappa*kappa), out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
        inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      blas::ax(0.25/(kappa*kappa), out);
    }
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE){
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

namespace quda{
bool canReuseResidentGauge(QudaInvertParam *param){
  return (gaugePrecise != NULL) and param->cuda_prec == gaugePrecise->Precision();
}
}

void checkClover(QudaInvertParam *param) {

  if (param->dslash_type != QUDA_CLOVER_WILSON_DSLASH && param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    return;
  }

  if (param->cuda_prec != cloverPrecise->Precision()) {
    errorQuda("Solve precision %d doesn't match clover precision %d", param->cuda_prec, cloverPrecise->Precision());
  }

  if (param->cuda_prec_sloppy != cloverSloppy->Precision() ||
      param->cuda_prec_precondition != cloverPrecondition->Precision()) {
    freeSloppyCloverQuda();
    loadSloppyCloverQuda(param->cuda_prec_sloppy, param->cuda_prec_precondition);
  }

  if (cloverPrecise == NULL) errorQuda("Precise gauge field doesn't exist");
  if (cloverSloppy == NULL) errorQuda("Sloppy gauge field doesn't exist");
  if (cloverPrecondition == NULL) errorQuda("Precondition gauge field doesn't exist");
}

quda::cudaGaugeField* checkGauge(QudaInvertParam *param) {

  if (param->cuda_prec != gaugePrecise->Precision()) {
    errorQuda("Solve precision %d doesn't match gauge precision %d", param->cuda_prec, gaugePrecise->Precision());
  }

  quda::cudaGaugeField *cudaGauge = NULL;
  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    if (param->cuda_prec_sloppy != gaugeSloppy->Precision() ||
	param->cuda_prec_precondition != gaugePrecondition->Precision()) {
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(param->cuda_prec_sloppy, param->cuda_prec_precondition);
    }

    if (gaugePrecise == NULL) errorQuda("Precise gauge field doesn't exist");
    if (gaugeSloppy == NULL) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == NULL) errorQuda("Precondition gauge field doesn't exist");
    if (param->overlap) {
      if (gaugeExtended == NULL) errorQuda("Extended gauge field doesn't exist");
    }
    cudaGauge = gaugePrecise;
  } else {
    if (param->cuda_prec_sloppy != gaugeFatSloppy->Precision() ||
	param->cuda_prec_precondition != gaugeFatPrecondition->Precision() ||
	param->cuda_prec_sloppy != gaugeLongSloppy->Precision() ||
	param->cuda_prec_precondition != gaugeLongPrecondition->Precision()) {
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(param->cuda_prec_sloppy, param->cuda_prec_precondition);
    }

    if (gaugeFatPrecise == NULL) errorQuda("Precise gauge fat field doesn't exist");
    if (gaugeFatSloppy == NULL) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == NULL) errorQuda("Precondition gauge fat field doesn't exist");
    if (param->overlap) {
      if(gaugeFatExtended == NULL) errorQuda("Extended gauge fat field doesn't exist");
    }

    if (gaugeLongPrecise == NULL) errorQuda("Precise gauge long field doesn't exist");
    if (gaugeLongSloppy == NULL) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == NULL) errorQuda("Precondition gauge long field doesn't exist");
    if (param->overlap) {
      if(gaugeLongExtended == NULL) errorQuda("Extended gauge long field doesn't exist");
    }
    cudaGauge = gaugeFatPrecise;
  }

  checkClover(param);

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
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
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
    blas::ax(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);
	//FIXME: Do we need this for twisted clover???
  DiracCloverPC dirac(diracParam); // create the Dirac operator
  if (!inverse) dirac.Clover(out, in, parity); // apply the clover operator
  else dirac.CloverInv(out, in, parity);

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  /*for (int i=0; i<in_h->Volume(); i++) {
    ((cpuColorSpinorField*)out_h)->PrintVector(i);
    }*/

  delete out_h;
  delete in_h;

  popVerbosity();
}


void lanczosQuda(int k0, int m, void *hp_Apsi, void *hp_r, void *hp_V,
                 void *hp_alpha, void *hp_beta, QudaEigParam *eig_param)
{
  QudaInvertParam *param;
  param = eig_param->invert_param;

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  checkEigParam(eig_param);

  bool pc_solution = (param->solution_type == QUDA_MATPC_DAG_SOLUTION) ||
                     (param->solution_type == QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION);

  // create the dirac operator
  DiracParam diracParam;
  setDiracParam(diracParam, param, pc_solution);
  Dirac *d = Dirac::create(diracParam); // create the Dirac operator

  Dirac &dirac = *d;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  cudaColorSpinorField *r = NULL;
  cudaColorSpinorField *Apsi = NULL;
  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_r, *param, X, pc_solution);
  ColorSpinorField *h_r = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
                          static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
                          static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = hp_Apsi;
  ColorSpinorField *h_Apsi = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
                             static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
                             static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  //Make Eigen vector data set
  cpuColorSpinorField **h_Eig_Vec;
  h_Eig_Vec =(cpuColorSpinorField **)safe_malloc( m*sizeof(cpuColorSpinorField*));
  for( int k = 0 ; k < m ; k++)
  {
    cpuParam.v = ((double**)hp_V)[k];
    h_Eig_Vec[k] = new cpuColorSpinorField(cpuParam);
  }

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  r = new cudaColorSpinorField(*h_r, cudaParam);
  Apsi = new cudaColorSpinorField(*h_Apsi, cudaParam);

  double cpu;
  double gpu;

  if (getVerbosity() >= QUDA_VERBOSE) {
    cpu = blas::norm2(*h_r);
    gpu = blas::norm2(*r);
    printfQuda("r vector CPU %1.14e CUDA %1.14e\n", cpu, gpu);
    cpu = blas::norm2(*h_Apsi);
    gpu = blas::norm2(*Apsi);
    printfQuda("Apsi vector CPU %1.14e CUDA %1.14e\n", cpu, gpu);
  }

  // download Eigen vector set
  cudaColorSpinorField **Eig_Vec;
  Eig_Vec = (cudaColorSpinorField **)safe_malloc( m*sizeof(cudaColorSpinorField*));

  for( int k = 0 ; k < m ; k++)
  {
    Eig_Vec[k] = new cudaColorSpinorField(*h_Eig_Vec[k], cudaParam);
    if (getVerbosity() >= QUDA_VERBOSE) {
      cpu = blas::norm2(*h_Eig_Vec[k]);
      gpu = blas::norm2(*Eig_Vec[k]);
      printfQuda("Eig_Vec[%d] CPU %1.14e CUDA %1.14e\n", k, cpu, gpu);
    }
  }
  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  if(eig_param->RitzMat_lanczos == QUDA_MATPC_DAG_SOLUTION)
  {
    DiracMdag mat(dirac);
    RitzMat ritz_mat(mat,*eig_param);
    Eig_Solver *eig_solve = Eig_Solver::create(*eig_param, ritz_mat, profileInvert);
    (*eig_solve)((double*)hp_alpha, (double*)hp_beta, Eig_Vec, *r, *Apsi, k0, m);
    delete eig_solve;
  }
  else if(eig_param->RitzMat_lanczos == QUDA_MATPCDAG_MATPC_SOLUTION)
  {
    DiracMdagM mat(dirac);
    RitzMat ritz_mat(mat,*eig_param);
    Eig_Solver *eig_solve = Eig_Solver::create(*eig_param, ritz_mat, profileInvert);
    (*eig_solve)((double*)hp_alpha, (double*)hp_beta, Eig_Vec, *r, *Apsi, k0, m);
    delete eig_solve;
  }
  else if(eig_param->RitzMat_lanczos == QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION)
  {
    DiracMdagM mat(dirac);
    RitzMat ritz_mat(mat,*eig_param);
    Eig_Solver *eig_solve = Eig_Solver::create(*eig_param, ritz_mat, profileInvert);
    (*eig_solve)((double*)hp_alpha, (double*)hp_beta, Eig_Vec, *r, *Apsi, k0, m);
    delete eig_solve;
  }
  else
  {
    errorQuda("invalid ritz matrix type\n");
    exit(0);
  }

  //Write back calculated eigen vector
  profileInvert.TPSTART(QUDA_PROFILE_D2H);
  for( int k = 0 ; k < m ; k++)
  {
    *h_Eig_Vec[k] = *Eig_Vec[k];
  }
  *h_r = *r;
  *h_Apsi = *Apsi;
  profileInvert.TPSTOP(QUDA_PROFILE_D2H);


  delete h_r;
  delete h_Apsi;
  for( int k = 0 ; k < m ; k++)
  {
    delete Eig_Vec[k];
    delete h_Eig_Vec[k];
  }
  host_free(Eig_Vec);
  host_free(h_Eig_Vec);

  delete d;

  popVerbosity();

  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

multigrid_solver::multigrid_solver(QudaMultigridParam &mg_param, TimeProfile &profile)
  : profile(profile) {
  profile.TPSTART(QUDA_PROFILE_INIT);
  QudaInvertParam *param = mg_param.invert_param;

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkMultigridParam(&mg_param);

  // check MG params (needs to go somewhere else)
  if (mg_param.n_level > QUDA_MAX_MG_LEVEL)
    errorQuda("Requested MG levels %d greater than allowed maximum %d", mg_param.n_level, QUDA_MAX_MG_LEVEL);
  for (int i=0; i<mg_param.n_level; i++) {
    if (mg_param.smoother_solve_type[i] != QUDA_DIRECT_SOLVE && mg_param.smoother_solve_type[i] != QUDA_DIRECT_PC_SOLVE)
      errorQuda("Unsupported smoother solve type %d on level %d", mg_param.smoother_solve_type[i], i);
  }
  if (param->solve_type != QUDA_DIRECT_SOLVE)
    errorQuda("Outer MG solver can only use QUDA_DIRECT_SOLVE at present");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaMultigridParam(&mg_param);
  mg_param.secs = 0;
  mg_param.gflops = 0;

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool outer_pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);

  // create the dirac operators for the fine grid

  // this is the Dirac operator we use for inter-grid residual computation
  // at present this doesn't support preconditioning
  DiracParam diracParam;
  setDiracSloppyParam(diracParam, param, outer_pc_solve);
  d = Dirac::create(diracParam);
  m = new DiracM(*d);

  // this is the Dirac operator we use for smoothing
  DiracParam diracSmoothParam;
  bool fine_grid_pc_solve = (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE) ||
    (mg_param.smoother_solve_type[0] == QUDA_NORMOP_PC_SOLVE);
  setDiracSloppyParam(diracSmoothParam, param, fine_grid_pc_solve);
  dSmooth = Dirac::create(diracSmoothParam);
  mSmooth = new DiracM(*dSmooth);

  // this is the Dirac operator we use for sloppy smoothing (we use the preconditioner fields for this)
  DiracParam diracSmoothSloppyParam;
  setDiracPreParam(diracSmoothSloppyParam, param, fine_grid_pc_solve, true);
  dSmoothSloppy = Dirac::create(diracSmoothSloppyParam);;
  mSmoothSloppy = new DiracM(*dSmoothSloppy);

  printfQuda("Creating vector of null space fields of length %d\n", mg_param.n_vec[0]);

  ColorSpinorParam cpuParam(0, *param, cudaGauge->X(), pc_solution, QUDA_CPU_FIELD_LOCATION);
  cpuParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuParam.precision = param->cuda_prec_sloppy;
  B.resize(mg_param.n_vec[0]);
  for (int i=0; i<mg_param.n_vec[0]; i++) B[i] = new cpuColorSpinorField(cpuParam);

  // fill out the MG parameters for the fine level
  mgParam = new MGParam(mg_param, B, *m, *mSmooth, *mSmoothSloppy);

  mg = new MG(*mgParam, profile);
  mgParam->updateInvertParam(*param);
  profile.TPSTOP(QUDA_PROFILE_INIT);
}

void* newMultigridQuda(QudaMultigridParam *mg_param) {
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  openMagma();

  multigrid_solver *mg = new multigrid_solver(*mg_param, profileInvert);

  closeMagma();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  saveProfile(__func__);
  flushProfile();
  return static_cast<void*>(mg);
}

void destroyMultigridQuda(void *mg) {
  delete static_cast<multigrid_solver*>(mg);
}


void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);
  bool norm_error_solve = (param->solve_type == QUDA_NORMERR_SOLVE) ||
    (param->solve_type == QUDA_NORMERR_PC_SOLVE);

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

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = hp_x;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

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

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  double nb = blas::norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = blas::norm2(*h_b);
    double nh_x = blas::norm2(*h_x);
    double nx = blas::norm2(*x);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    blas::ax(1.0/sqrt(nb), *b);
    blas::ax(1.0/sqrt(nb), *x);
  }

  massRescale(*static_cast<cudaColorSpinorField*>(b), *param);

  dirac.prepare(in, out, *x, *b, param->solution_type);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = blas::norm2(*in);
    double nout = blas::norm2(*out);
    printfQuda("Prepared source = %g\n", nin);
    printfQuda("Prepared solution = %g\n", nout);
  }

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = blas::norm2(*in);
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
  // MAT              NORMERR       Solve (A A^dag) y = b, then x = A^dag y
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

  if (!mat_solution && norm_error_solve) {
    errorQuda("Normal-error solve requires Mat solution");
  }

  if (param->inv_type_precondition == QUDA_MG_INVERTER && (!direct_solve || !mat_solution))
      errorQuda("Multigrid preconditioning only supported for direct solves");

  if (mat_solution && !direct_solve && !norm_error_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    blas::copy(*in, *out);
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
  } else if (!norm_error_solve) {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  } else { // norm_error_solve
    DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    cudaColorSpinorField tmp(*out);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(tmp, *in); // y = (M M^\dag) b
    dirac.Mdag(*out, tmp);  // x = M^dag y
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = blas::norm2(*x);
   printfQuda("Solution = %g\n",nx);
  }

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);
  dirac.reconstruct(*x, *b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    blas::ax(sqrt(nb), *x);
  }
  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  if (!param->make_resident_solution) {
    profileInvert.TPSTART(QUDA_PROFILE_D2H);
    *h_x = *x;
    profileInvert.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);
  if (param->make_resident_solution) {
    for (unsigned int i=0; i<solutionResident.size(); i++) {
      if (solutionResident[i]) delete solutionResident[i];
    }
    solutionResident.resize(1);

    solutionResident[0] = static_cast<cudaColorSpinorField*>(x);
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = blas::norm2(*x);
    double nh_x = blas::norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
  }
  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileInvert.TPSTART(QUDA_PROFILE_FREE);

  delete h_b;
  delete h_x;
  delete b;
  if (!param->make_resident_solution) delete x;

  delete d;
  delete dSloppy;
  delete dPre;

  profileInvert.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}


/*!
 * Generic version of the multi-shift solver. Should work for
 * most fermions. Note that offset[0] is not folded into the mass parameter.
 *
 * At present, the solution_type must be MATDAG_MAT or MATPCDAG_MATPC,
 * and solve_type must be NORMOP or NORMOP_PC.  The solution and solve
 * preconditioning have to match.
 */
void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param)
{

  // currently that code is just a copy of invertQuda and cannot work

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);
  bool norm_error_solve = (param->solve_type == QUDA_NORMERR_SOLVE) ||
    (param->solve_type == QUDA_NORMERR_PC_SOLVE);

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

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  // std::vector<ColorSpinorField*> b;  // Cuda Solutions
  // b.resize(param->num_src);
  // std::vector<ColorSpinorField*> x;  // Cuda Solutions
  // x.resize(param->num_src);
  ColorSpinorField* in;  // = NULL;
  //in.resize(param->num_src);
  ColorSpinorField* out;  // = NULL;
  //out.resize(param->num_src);

  // for(int i=0;i < param->num_src;i++){
  //   in[i] = NULL;
  //   out[i] = NULL;
  // }

  const int *X = cudaGauge->X();


  // Host pointers for x, take a copy of the input host pointers
  void** hp_x;
  hp_x = new void* [ param->num_src ];

  void** hp_b;
  hp_b = new void* [param->num_src];

  for(int i=0;i < param->num_src;i++){
    hp_x[i] = _hp_x[i];
    hp_b[i] = _hp_b[i];
  }

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b[0], *param, X, pc_solution, param->input_location);
  std::vector<ColorSpinorField*> h_b;
  h_b.resize(param->num_src);
  for(int i=0; i < param->num_src; i++) {
    cpuParam.v = hp_b[i]; //MW seems wird in the loop
    h_b[i] = ColorSpinorField::Create(cpuParam);
  }

 // cpuParam.v = hp_x;
  cpuParam.location = param->output_location;
  std::vector<ColorSpinorField*> h_x;
  h_x.resize(param->num_src);
//
  for(int i=0; i < param->num_src; i++) {
    cpuParam.v = hp_x[i]; //MW seems wird in the loop
    h_x[i] = ColorSpinorField::Create(cpuParam);
  }


  // MW currently checked until here

  // download source
  printfQuda("Setup b\n");
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.is_composite = true;
  cudaParam.composite_dim = param->num_src;

  printfQuda("Create b \n");
  ColorSpinorField *b = ColorSpinorField::Create(cudaParam);




  for(int i=0; i < param->num_src; i++) {
    b->Component(i) = *h_b[i];
  }
  printfQuda("Done b \n");

    ColorSpinorField *x;
  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
        (param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }
    cudaParam.is_composite = true;
    cudaParam.is_component = false;
    cudaParam.composite_dim = param->num_src;

    x = ColorSpinorField::Create(cudaParam);
    for(int i=0; i < param->num_src; i++) {
      x->Component(i) = *h_x[i];
    }

  } else { // zero initial guess
    // Create the solution fields filled with zero
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
      printfQuda("Create x \n");
    x = ColorSpinorField::Create(cudaParam);
      printfQuda("Done x \n");
 // solution
  }

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  double * nb = new double[param->num_src];
  for(int i=0; i < param->num_src; i++) {
    nb[i] = blas::norm2(b->Component(i));
    printfQuda("Source %i: CPU = %g, CUDA copy = %g\n", i, nb[i], nb[i]);
    if (nb[i]==0.0) errorQuda("Source has zero norm");

    if (getVerbosity() >= QUDA_VERBOSE) {
      double nh_b = blas::norm2(*h_b[i]);
      double nh_x = blas::norm2(*h_x[i]);
      double nx = blas::norm2(x->Component(i));
      printfQuda("Source %i: CPU = %g, CUDA copy = %g\n", i, nh_b, nb[i]);
      printfQuda("Solution %i: CPU = %g, CUDA copy = %g\n", i, nh_x, nx);
    }
  }

  // MW checked until here do far

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    for(int i=0; i < param->num_src; i++) {
      blas::ax(1.0/sqrt(nb[i]), b->Component(i));
      blas::ax(1.0/sqrt(nb[i]), x->Component(i));
    }
  }

  for(int i=0; i < param->num_src; i++) {
    massRescale(dynamic_cast<cudaColorSpinorField&>( b->Component(i) ), *param);
  }

  // MW: need to check what dirac.prepare does
  // for now let's just try looping of num_rhs already here???
  // for(int i=0; i < param->num_src; i++) {
    dirac.prepare(in, out, *x, *b, param->solution_type);
for(int i=0; i < param->num_src; i++) {
    if (getVerbosity() >= QUDA_VERBOSE) {
      double nin = blas::norm2((in->Component(i)));
      double nout = blas::norm2((out->Component(i)));
      printfQuda("Prepared source %i = %g\n", i, nin);
      printfQuda("Prepared solution %i = %g\n", i, nout);
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      double nin = blas::norm2(in->Component(i));
      printfQuda("Prepared source %i post mass rescale = %g\n", i, nin);
    }
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
    // MAT              NORMERR       Solve (A A^dag) y = b, then x = A^dag y
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

    if (!mat_solution && norm_error_solve) {
      errorQuda("Normal-error solve requires Mat solution");
    }

    if (param->inv_type_precondition == QUDA_MG_INVERTER && (pc_solve || pc_solution || !direct_solve || !mat_solution))
      errorQuda("Multigrid preconditioning only supported for direct non-red-black solve");

    if (mat_solution && !direct_solve && !norm_error_solve) { // prepare source: b' = A^dag b
      for(int i=0; i < param->num_src; i++) {
        cudaColorSpinorField tmp((in->Component(i)));
        dirac.Mdag(in->Component(i), tmp);
      }
    } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
      DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      solve->solve(*out,*in);
      for(int i=0; i < param->num_src; i++) {
        blas::copy(in->Component(i), out->Component(i));
      }
      solverParam.updateInvertParam(*param);
      delete solve;
    }

    if (direct_solve) {
      DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      solve->solve(*out,*in);
      solverParam.updateInvertParam(*param);
      delete solve;
    } else if (!norm_error_solve) {
      DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      solve->solve(*out,*in);
      solverParam.updateInvertParam(*param);
      delete solve;
    } else { // norm_error_solve
      DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      errorQuda("norm_error_solve not supported in multi source solve");
      //cudaColorSpinorField tmp(*out);
      // SolverParam solverParam(*param);
      //Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      //(*solve)(tmp, *in); // y = (M M^\dag) b
      //dirac.Mdag(*out, tmp);  // x = M^dag y
      //solverParam.updateInvertParam(*param,i,i);
      // delete solve;
    }

    if (getVerbosity() >= QUDA_VERBOSE){
      for(int i=0; i < param->num_src; i++) {
        double nx = blas::norm2(x->Component(i));
        printfQuda("Solution %i = %g\n",i, nx);
      }
    }


  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);
  for(int i=0; i< param->num_src; i++){
    dirac.reconstruct(x->Component(i), b->Component(i), param->solution_type);
  }
  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    for(int i=0; i< param->num_src; i++){
      // rescale the solution
      blas::ax(sqrt(nb[i]), x->Component(i));
    }
  }

  // MW -- not sure how to handle that here
  if (!param->make_resident_solution) {
    profileInvert.TPSTART(QUDA_PROFILE_D2H);
    for(int i=0; i< param->num_src; i++){
      *h_x[i] = x->Component(i);
    }
    profileInvert.TPSTOP(QUDA_PROFILE_D2H);
  }

//  if (param->make_resident_solution) {
//    for (unsigned int i=0; i<solutionResident.size(); i++) {
//      if (solutionResident[i]) delete solutionResident[i];
//    }
//    solutionResident.resize(1);
//    solutionResident[0] = static_cast<cudaColorSpinorField*>(x);
//  }

  if (getVerbosity() >= QUDA_VERBOSE){
    for(int i=0; i< param->num_src; i++){
      double nx = blas::norm2(x->Component(i));
      double nh_x = blas::norm2(*h_x[i]);
      printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
    }
  }

  //FIX need to make sure all deletes are correct again
  for(int i=0; i < param->num_src; i++){
    delete h_x[i];
    // delete x[i];
    delete h_b[i];
    // delete b[i];
  }
   delete [] hp_b;
   delete [] hp_x;
//   delete [] b;
//  if (!param->make_resident_solution) delete x;

  delete d;
  delete dSloppy;
  delete dPre;
  delete x;
  delete b;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
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

  profileMulti.TPSTART(QUDA_PROFILE_TOTAL);
  profileMulti.TPSTART(QUDA_PROFILE_INIT);

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");

  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

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
  std::vector<ColorSpinorField*> x;  // Cuda Solutions
  x.resize(param->num_offset);

  // Grab the dimension array of the input gauge field.
  const int *X = ( param->dslash_type == QUDA_ASQTAD_DSLASH ) ?
    gaugeFatPrecise->X() : gaugePrecise->X();

  // This creates a ColorSpinorParam struct, from the host data
  // pointer, the definitions in param, the dimensions X, and whether
  // the solution is on a checkerboard instruction or not. These can
  // then be used as 'instructions' to create the actual
  // ColorSpinorField
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  std::vector<ColorSpinorField*> h_x;
  h_x.resize(param->num_offset);

  cpuParam.location = param->output_location;
  for(int i=0; i < param->num_offset; i++) {
    cpuParam.v = hp_x[i];
    h_x[i] = ColorSpinorField::Create(cpuParam);
  }

  profileMulti.TPSTOP(QUDA_PROFILE_INIT);
  profileMulti.TPSTART(QUDA_PROFILE_H2D);
  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it
  profileMulti.TPSTOP(QUDA_PROFILE_H2D);

  profileMulti.TPSTART(QUDA_PROFILE_INIT);
  // Create the solution fields filled with zero
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int i=0; i < param->num_offset; i++) {
    x[i] = new cudaColorSpinorField(cudaParam);
  }
  profileMulti.TPSTOP(QUDA_PROFILE_INIT);


  profileMulti.TPSTART(QUDA_PROFILE_PREAMBLE);

  // Check source norms
  double nb = blas::norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if(getVerbosity() >= QUDA_VERBOSE ) {
    double nh_b = blas::norm2(*h_b);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
  }

  // rescale the source vector to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    blas::ax(1.0/sqrt(nb), *b);
  }

  massRescale(*b, *param);
  profileMulti.TPSTOP(QUDA_PROFILE_PREAMBLE);

  // use multi-shift CG
  {
    DiracMdagM m(dirac), mSloppy(diracSloppy);
    SolverParam solverParam(*param);
    MultiShiftCG cg_m(m, mSloppy, solverParam, profileMulti);
    cg_m(x, *b);
    solverParam.updateInvertParam(*param);
  }

  // check each shift has the desired tolerance and use sequential CG to refine

  profileMulti.TPSTART(QUDA_PROFILE_INIT);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField r(*b, cudaParam);
  profileMulti.TPSTOP(QUDA_PROFILE_INIT);

#define REFINE_INCREASING_MASS
#ifdef REFINE_INCREASING_MASS
  for(int i=0; i < param->num_offset; i++) {
#else
  for(int i=param->num_offset-1; i >= 0; i--) {
#endif
    double rsd_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->true_res_hq_offset[i] : 0;
    double tol_hq = param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ?
      param->tol_hq_offset[i] : 0;

    /*
      In the case where the shifted systems have zero tolerance
      specified, we refine these systems until either the limit of
      precision is reached (prec_tol) or until the tolerance reaches
      the iterated residual tolerance of the previous multi-shift
      solver (iter_res_offset[i]), which ever is greater.
     */
    const double prec_tol = pow(10.,(-2*(int)param->cuda_prec+2));
    const double iter_tol = (param->iter_res_offset[i] < prec_tol ? prec_tol : (param->iter_res_offset[i] *1.1));
    const double refine_tol = (param->tol_offset[i] == 0.0 ? iter_tol : param->tol_offset[i]);
    // refine if either L2 or heavy quark residual tolerances have not been met, only if desired residual is > 0
    if ((param->true_res_offset[i] > refine_tol || rsd_hq > tol_hq)) {
      if (getVerbosity() >= QUDA_SUMMARIZE)
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

      if (0) { // experimenting with Minimum residual extrapolation
	// only perform MRE using current and previously refined solutions
#ifdef REFINE_INCREASING_MASS
	const int nRefine = i+1;
#else
	const int nRefine = param->num_offset - i + 1;
#endif

	std::vector<ColorSpinorField*> q;
	q.resize(nRefine);
	std::vector<ColorSpinorField*> z;
	z.resize(nRefine);
	cudaParam.create = QUDA_NULL_FIELD_CREATE;
	cudaColorSpinorField tmp(cudaParam);

	for(int j=0; j < nRefine; j++) {
	  q[j] = new cudaColorSpinorField(cudaParam);
	  z[j] = new cudaColorSpinorField(cudaParam);
	}

	*z[0] = *x[0]; // zero solution already solved
#ifdef REFINE_INCREASING_MASS
	for (int j=1; j<nRefine; j++) *z[j] = *x[j];
#else
	for (int j=1; j<nRefine; j++) *z[j] = *x[param->num_offset-j];
#endif

	MinResExt mre(m, profileMulti);
	blas::copy(tmp, *b);
	mre(*x[i], tmp, z, q, nRefine);

	for(int j=0; j < nRefine; j++) {
	  delete q[j];
	  delete z[j];
	}
      }

      SolverParam solverParam(*param);
      solverParam.iter = 0;
      solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
      solverParam.tol = (param->tol_offset[i] > 0.0 ?  param->tol_offset[i] : iter_tol); // set L2 tolerance
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

  profileMulti.TPSTART(QUDA_PROFILE_D2H);
  for(int i=0; i < param->num_offset; i++) {
    if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) { // rescale the solution
      blas::ax(sqrt(nb), *x[i]);
    }

    if (getVerbosity() >= QUDA_VERBOSE){
      double nx = blas::norm2(*x[i]);
      printfQuda("Solution %d = %g\n", i, nx);
    }

    if (!param->make_resident_solution) *h_x[i] = *x[i];
  }
  profileMulti.TPSTOP(QUDA_PROFILE_D2H);

  profileMulti.TPSTART(QUDA_PROFILE_EPILOGUE);
  if (param->make_resident_solution) {
    for (unsigned int i=0; i<solutionResident.size(); i++) {
      if (solutionResident[i]) delete solutionResident[i];
    }

    solutionResident.resize(param->num_offset);
    for (unsigned int i=0; i<solutionResident.size(); i++) {
      solutionResident[i] = static_cast<cudaColorSpinorField*>(x[i]);
    }
  }
  profileMulti.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileMulti.TPSTART(QUDA_PROFILE_FREE);
  for(int i=0; i < param->num_offset; i++){
    delete h_x[i];
    if (!param->make_resident_solution) delete x[i];
  }

  delete h_b;
  delete b;

  delete [] hp_x;

  delete d;
  delete dSloppy;
  delete dPre;
  profileMulti.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileMulti.TPSTOP(QUDA_PROFILE_TOTAL);
}


void incrementalEigQuda(void *_h_x, void *_h_b, QudaInvertParam *param, void *_h_u, double *inv_eigenvals)
{

  openMagma();

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

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
    param->spinorGiB *= ((param->inv_type == QUDA_EIGCG_INVERTER || param->inv_type == QUDA_INC_EIGCG_INVERTER) ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= ((param->inv_type == QUDA_EIGCG_INVERTER || param->inv_type == QUDA_INC_EIGCG_INVERTER) ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  DiracParam diracParam;
  DiracParam diracSloppyParam;
  //DiracParam diracDeflateParam;
//!
  DiracParam diracHalfPrecParam;//sloppy precision for initCG
//!
  setDiracParam(diracParam, param, pc_solve);
  setDiracSloppyParam(diracSloppyParam, param, pc_solve);

  if(param->cuda_prec_precondition != QUDA_HALF_PRECISION)
  {
     errorQuda("\nInitCG requires sloppy gauge field in half precision. It seems that the half precision field is not loaded,\n please check you cuda_prec_precondition parameter.\n");
  }

//!half precision Dirac field (for the initCG)
  setDiracParam(diracHalfPrecParam, param, pc_solve);

  diracHalfPrecParam.gauge = gaugePrecondition;
  diracHalfPrecParam.fatGauge = gaugeFatPrecondition;
  diracHalfPrecParam.longGauge = gaugeLongPrecondition;

  diracHalfPrecParam.clover = cloverPrecondition;

  for (int i=0; i<4; i++) {
      diracHalfPrecParam.commDim[i] = 1; // comms are on.
  }
//!

  Dirac *d        = Dirac::create(diracParam); // create the Dirac operator
  Dirac *dSloppy  = Dirac::create(diracSloppyParam);
  //Dirac *dDeflate = Dirac::create(diracPreParam);
  Dirac *dHalfPrec = Dirac::create(diracHalfPrecParam);

  Dirac &dirac = *d;
  Dirac &diracSloppy   = *dSloppy;
  Dirac &diracHalf     = *dHalfPrec;
  Dirac &diracDeflate  = (param->cuda_prec_ritz == param->cuda_prec) ? *d : *dSloppy;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(_h_b, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = _h_x;
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

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  double nb = blas::norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = blas::norm2(*h_b);
    double nh_x = blas::norm2(*h_x);
    double nx = blas::norm2(*x);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    blas::ax(1.0/sqrt(nb), *b);
    blas::ax(1.0/sqrt(nb), *x);
  }

  massRescale(*static_cast<cudaColorSpinorField*>(b), *param);

  dirac.prepare(in, out, *x, *b, param->solution_type);
//here...
  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = blas::norm2(*in);
    double nout = blas::norm2(*out);
    printfQuda("Prepared source = %g\n", nin);
    printfQuda("Prepared solution = %g\n", nout);
  }

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = blas::norm2(*in);
    printfQuda("Prepared source post mass rescale = %g\n", nin);
  }

  if (param->max_search_dim == 0 || param->nev == 0 || (param->max_search_dim < param->nev))
     errorQuda("\nIncorrect eigenvector space setup...\n");

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }

  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }

  if (mat_solution && !direct_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  }

  if(param->inv_type == QUDA_INC_EIGCG_INVERTER || param->inv_type == QUDA_EIGCG_INVERTER)
  {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mHalf(diracHalf), mDeflate(diracDeflate);
    SolverParam solverParam(*param);

    DeflatedSolver *solve = DeflatedSolver::create(solverParam, &m, &mSloppy, &mHalf, &mDeflate, &profileInvert);

    (*solve)(static_cast<cudaColorSpinorField*>(out), static_cast<cudaColorSpinorField*>(in));//run solver

    solverParam.updateInvertParam(*param);//will update rhs_idx as well...

    delete solve;
  }
  else if (param->inv_type == QUDA_GMRESDR_INVERTER && direct_solve)
  {
    DiracM m(dirac), mSloppy(diracSloppy), mHalf(diracHalf), mDeflate(diracDeflate);
    //DiracMdagM m(dirac), mSloppy(diracSloppy), mHalf(diracHalf), mDeflate(diracDeflate);//use for tests only.
    SolverParam solverParam(*param);

    DeflatedSolver *solve = DeflatedSolver::create(solverParam, &m, &mSloppy, &mHalf, &mHalf, &profileInvert);  //mDeflate - > mHalf

    (*solve)(static_cast<cudaColorSpinorField*>(out), static_cast<cudaColorSpinorField*>(in));//run solver

    solverParam.updateInvertParam(*param);//will update rhs_idx as well...

    delete solve;

  }
  else if (param->inv_type == QUDA_GMRESDR_PROJ_INVERTER && direct_solve)
  {
     DiracM m(dirac), mSloppy(diracSloppy), mHalf(diracHalf), mDeflate(diracDeflate);
     SolverParam solverParam(*param);
     DeflatedSolver *solve = DeflatedSolver::create(solverParam, &m, &mSloppy, &mHalf, &mHalf, &profileInvert);

     (*solve)(static_cast<cudaColorSpinorField*>(out), static_cast<cudaColorSpinorField*>(in));//run solver
     solverParam.updateInvertParam(*param);//will update rhs_idx as well...

     delete solve;
  }
  else
  {
    errorQuda("\nUnknown deflated solver...\n");
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = blas::norm2(*x);
    printfQuda("Solution = %g\n",nx);
  }
  dirac.reconstruct(*x, *b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    blas::ax(sqrt(nb), *x);
  }

  profileInvert.TPSTART(QUDA_PROFILE_D2H);
  *h_x = *x;
  profileInvert.TPSTOP(QUDA_PROFILE_D2H);

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = blas::norm2(*x);
    double nh_x = blas::norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
  }

  delete h_b;
  delete h_x;
  delete b;
  delete x;

  delete d;
  delete dSloppy;
//  delete dDeflate;
  delete dHalfPrec;

  popVerbosity();

  closeMagma();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

void destroyDeflationQuda(QudaInvertParam *param, const int *X,  void *_h_u, double *inv_eigenvals)
{
   SolverParam solverParam(*param);
   DeflatedSolver *solve = DeflatedSolver::create(solverParam, NULL, NULL, NULL, NULL, NULL);

   if(param->inv_type == QUDA_INC_EIGCG_INVERTER || param->inv_type == QUDA_EIGCG_INVERTER){
      if(_h_u) solve->StoreRitzVecs(_h_u, inv_eigenvals, X, param, param->nev);
      printfQuda("\nDelete incremental EigCG solver resources...\n");
      //clean resources:
      solve->CleanResources();
      //
      printfQuda("\n...done.\n");
   }
   else
   {
      solve->CleanResources();
   }

   return;
}


#ifdef GPU_FATLINK
#include <sys/time.h>

void setFatLinkPadding(QudaGaugeParam* param)
{
  int* X    = param->X;
  int Vsh_x = X[1]*X[2]*X[3]/2;
  int Vsh_y = X[0]*X[2]*X[3]/2;
  int Vsh_z = X[0]*X[1]*X[3]/2;
  int Vsh_t = X[0]*X[1]*X[2]/2;

  int E[4];
  for (int i=0; i<4; i++) E[i] = X[i] + 4;

  // fat-link padding
  param->llfat_ga_pad = std::max(Vsh_x, std::max(Vsh_y, std::max(Vsh_z, Vsh_t)));

  // site-link padding
  param->site_ga_pad = (E[0]*E[1]*E[2]/2)*3;
  param->ga_pad = param->site_ga_pad;

  // staple padding
  param->staple_pad = (E[0]*E[1]*E[2]/2)*3;

  return;
}


namespace quda {
  void computeFatLinkCore(cudaGaugeField* cudaSiteLink, double* act_path_coeff,
			  QudaGaugeParam* qudaGaugeParam, cudaGaugeField* cudaFatLink,
			  cudaGaugeField* cudaLongLink, TimeProfile &profile)
  {

    profile.TPSTART(QUDA_PROFILE_INIT);
    GaugeFieldParam gParam(0,*qudaGaugeParam);

    for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam->X[dir] + 4;

    gParam.pad    = qudaGaugeParam->staple_pad;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.geometry = QUDA_SCALAR_GEOMETRY; // only require a scalar matrix field for the staple
    gParam.setPrecision(gParam.precision);
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

    cudaGaugeField *cudaStapleField  = new cudaGaugeField(gParam);
    cudaGaugeField *cudaStapleField1 = new cudaGaugeField(gParam);

    profile.TPSTOP(QUDA_PROFILE_INIT);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    llfat_cuda_ex(cudaFatLink, cudaLongLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, qudaGaugeParam, act_path_coeff);
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    delete cudaStapleField;
    delete cudaStapleField1;
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }
} // namespace quda


namespace quda {
  namespace fatlink {
#include <dslash_init.cuh>
  }
}

#endif // GPU_FATLINK

void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink, double *path_coeff, QudaGaugeParam *param)
{
#ifdef GPU_FATLINK
  profileFatLink.TPSTART(QUDA_PROFILE_TOTAL);
  profileFatLink.TPSTART(QUDA_PROFILE_INIT);

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
  int R[4] = {2, 2, 2, 2};
  for(int dir=0; dir<4; ++dir){ qudaGaugeParam_ex->X[dir] = param->X[dir]+2*R[dir]; }

  // fat-link padding
  setFatLinkPadding(param);
  qudaGaugeParam_ex->llfat_ga_pad = param->llfat_ga_pad;
  qudaGaugeParam_ex->staple_pad   = param->staple_pad;
  qudaGaugeParam_ex->site_ga_pad  = param->site_ga_pad;

  GaugeFieldParam gParam(0, *param);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  // create the host fatlink
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.gauge = fatlink;
  cpuGaugeField cpuFatLink(gParam);
  gParam.gauge = longlink;
  cpuGaugeField cpuLongLink(gParam);
  gParam.gauge = ulink;
  cpuGaugeField cpuUnitarizedLink(gParam);

  // create the host sitelink
  gParam.link_type = param->type;
  gParam.gauge     = inlink;
  cpuGaugeField cpuInLink(gParam);

  // create the device fatlink
  gParam.pad    = param->llfat_ga_pad;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(param->cuda_prec);
  cudaFatLink = new cudaGaugeField(gParam);
  if(ulink) cudaUnitarizedLink = new cudaGaugeField(gParam);
  if(longlink) cudaLongLink = new cudaGaugeField(gParam);

  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(param->cuda_prec);
  gParam.pad         = param->site_ga_pad;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param->type;
  cudaGaugeField* cudaInLink = new cudaGaugeField(gParam);

  for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam_ex->X[dir];
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaInLinkEx = new cudaGaugeField(gParam);

  profileFatLink.TPSTOP(QUDA_PROFILE_INIT);
  fatlink::initLatticeConstants(*cudaFatLink, profileFatLink);
  profileFatLink.TPSTART(QUDA_PROFILE_INIT);
  cudaGaugeField* inlinkPtr;
  llfat_init_cuda_ex(qudaGaugeParam_ex);
  inlinkPtr = cudaInLinkEx;
  profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

  profileFatLink.TPSTART(QUDA_PROFILE_H2D);
  cudaInLink->loadCPUField(cpuInLink);
  profileFatLink.TPSTOP(QUDA_PROFILE_H2D);

  profileFatLink.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaInLinkEx, *cudaInLink, QUDA_CUDA_FIELD_LOCATION);
  cudaInLinkEx->exchangeExtendedGhost(R,true);
  profileFatLink.TPSTOP(QUDA_PROFILE_COMMS);

  quda::computeFatLinkCore(inlinkPtr, const_cast<double*>(path_coeff), param, cudaFatLink, cudaLongLink, profileFatLink);

  if(ulink){
    profileFatLink.TPSTART(QUDA_PROFILE_INIT);
    *num_failures_h = 0;
    profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

    profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
    quda::unitarizeLinks(*cudaUnitarizedLink, *cudaFatLink, num_failures_d); // unitarize on the gpu
    profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);

    if(*num_failures_h>0){
      errorQuda("Error in the unitarization component of the hisq fattening: %d failures\n", *num_failures_h);
    }
    profileFatLink.TPSTART(QUDA_PROFILE_D2H);
    cudaUnitarizedLink->saveCPUField(cpuUnitarizedLink);
    profileFatLink.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileFatLink.TPSTART(QUDA_PROFILE_D2H);
  if(fatlink) cudaFatLink->saveCPUField(cpuFatLink);
  if(longlink) cudaLongLink->saveCPUField(cpuLongLink);
  profileFatLink.TPSTOP(QUDA_PROFILE_D2H);

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  if(longlink) delete cudaLongLink;
  delete cudaFatLink;
  delete cudaInLink;
  delete cudaUnitarizedLink;
  if(cudaInLinkEx) delete cudaInLinkEx;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  profileFatLink.TPSTOP(QUDA_PROFILE_TOTAL);
#else
  errorQuda("Fat-link has not been built");
#endif // GPU_FATLINK

  return;
}

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

int computeGaugeForceQuda(void* mom, void* siteLink,  int*** input_path_buf, int* path_length,
			  double* loop_coeff, int num_paths, int max_length, double eb3,
			  QudaGaugeParam* qudaGaugeParam)
{
#ifdef GPU_GAUGE_FORCE
  profileGaugeForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(0, *qudaGaugeParam);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.pad = 0;

#ifdef MULTI_GPU
  // do extended fill so we can reuse this extended gauge field if needed
  GaugeFieldParam gParamEx(gParam);
  for (int d=0; d<4; d++) gParamEx.x[d] = gParam.x[d] + 2*R[d];
#endif

  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = siteLink;
  cpuGaugeField *cpuSiteLink = (!qudaGaugeParam->use_resident_gauge) ? new cpuGaugeField(gParam) : NULL;

  cudaGaugeField* cudaSiteLink = NULL;

  if (qudaGaugeParam->use_resident_gauge) {
    if (!gaugePrecise) errorQuda("No resident gauge field to use");
    cudaSiteLink = gaugePrecise;
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = qudaGaugeParam->reconstruct;
    gParam.order = (qudaGaugeParam->reconstruct == QUDA_RECONSTRUCT_NO ||
        qudaGaugeParam->cuda_prec == QUDA_DOUBLE_PRECISION) ?
      QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

    cudaSiteLink = new cudaGaugeField(gParam);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);

    profileGaugeForce.TPSTART(QUDA_PROFILE_H2D);
    cudaSiteLink->loadCPUField(*cpuSiteLink);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_H2D);
  }

  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

#ifndef MULTI_GPU
  cudaGaugeField *cudaGauge = cudaSiteLink;
  qudaGaugeParam->site_ga_pad = gParam.pad; //need to set this value
#else

  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.reconstruct = qudaGaugeParam->reconstruct;
  gParamEx.order = (qudaGaugeParam->reconstruct == QUDA_RECONSTRUCT_NO ||
      qudaGaugeParam->cuda_prec == QUDA_DOUBLE_PRECISION) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  qudaGaugeParam->site_ga_pad = gParamEx.pad;//need to set this value

  cudaGaugeField *cudaGauge = new cudaGaugeField(gParamEx);

  copyExtendedGauge(*cudaGauge, *cudaSiteLink, QUDA_CUDA_FIELD_LOCATION);

  profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);

  profileGaugeForce.TPSTART(QUDA_PROFILE_COMMS);
  cudaGauge->exchangeExtendedGhost(R,redundant_comms);
  profileGaugeForce.TPSTOP(QUDA_PROFILE_COMMS);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);
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

  cpuGaugeField* cpuMom = (!qudaGaugeParam->use_resident_mom) ? new cpuGaugeField(gParamMom) : NULL;

  cudaGaugeField* cudaMom = NULL;
  if (qudaGaugeParam->use_resident_mom) {
    if (!gaugePrecise) errorQuda("No resident momentum field to use");
    cudaMom = momResident;
    if (qudaGaugeParam->overwrite_mom) cudaMom->zero();
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    gParamMom.create = qudaGaugeParam->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE;
    gParamMom.order = QUDA_FLOAT2_GAUGE_ORDER;
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
    gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
    gParamMom.precision = qudaGaugeParam->cuda_prec;
    gParamMom.create = QUDA_ZERO_FIELD_CREATE;
    cudaMom = new cudaGaugeField(gParamMom);
    if (!qudaGaugeParam->overwrite_mom) cudaMom->loadCPUField(*cpuMom);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);
  }

  // actually do the computation
  profileGaugeForce.TPSTART(QUDA_PROFILE_COMPUTE);
  gaugeForce(*cudaMom, *cudaGauge, eb3, input_path_buf,  path_length, loop_coeff, num_paths, max_length);
  profileGaugeForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (qudaGaugeParam->return_result_mom) {
    profileGaugeForce.TPSTART(QUDA_PROFILE_D2H);
    cudaMom->saveCPUField(*cpuMom);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileGaugeForce.TPSTART(QUDA_PROFILE_FREE);
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

  if (cpuSiteLink) delete cpuSiteLink;
  if (cpuMom) delete cpuMom;

#ifdef MULTI_GPU
  if (qudaGaugeParam->make_resident_gauge) {
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaGauge;
  } else {
    delete cudaGauge;
  }
#endif
  profileGaugeForce.TPSTOP(QUDA_PROFILE_FREE);

  profileGaugeForce.TPSTOP(QUDA_PROFILE_TOTAL);

  checkCudaError();
#else
  errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  return 0;
}


void createCloverQuda(QudaInvertParam* invertParam)
{
  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  profileClover.TPSTART(QUDA_PROFILE_INIT);
  if (!cloverPrecise) errorQuda("Clover field not allocated");

  int y[4];
  for(int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 2*R[dir];
  int pad = 0;
  // clover creation not supported from 8-reconstruct presently so convert to 12
  QudaReconstructType recon = (gaugePrecise->Reconstruct() == QUDA_RECONSTRUCT_8) ?
    QUDA_RECONSTRUCT_12 : gaugePrecise->Reconstruct();
  GaugeFieldParam gParamEx(y, gaugePrecise->Precision(), recon, pad,
			   QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = gaugePrecise->Order();
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gaugePrecise->TBoundary();
  gParamEx.nFace = 1;
  for (int d=0; d<4; d++) gParamEx.r[d] = R[d];

  cudaGaugeField *cudaGaugeExtended = NULL;
  if (extendedGaugeResident) {
    cudaGaugeExtended = extendedGaugeResident;
    profileClover.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    cudaGaugeExtended = new cudaGaugeField(gParamEx);

    // copy gaugePrecise into the extended device gauge field
    copyExtendedGauge(*cudaGaugeExtended, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);

    profileClover.TPSTOP(QUDA_PROFILE_INIT);
    profileClover.TPSTART(QUDA_PROFILE_COMMS);
    cudaGaugeExtended->exchangeExtendedGhost(R,redundant_comms);
    profileClover.TPSTOP(QUDA_PROFILE_COMMS);
  }

#ifdef MULTI_GPU
  GaugeField *gauge = cudaGaugeExtended;
#else
  GaugeField *gauge = gaugePrecise;
#endif

  profileClover.TPSTART(QUDA_PROFILE_INIT);
  // create the Fmunu field
  GaugeFieldParam tensorParam(gaugePrecise->X(), gauge->Precision(), QUDA_RECONSTRUCT_NO, pad, QUDA_TENSOR_GEOMETRY);
  tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField Fmunu(tensorParam);
  profileClover.TPSTOP(QUDA_PROFILE_INIT);

  profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
  computeFmunu(Fmunu, *gauge, QUDA_CUDA_FIELD_LOCATION);
  computeClover(*cloverPrecise, Fmunu, invertParam->clover_coeff, QUDA_CUDA_FIELD_LOCATION);
  profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileClover.TPSTOP(QUDA_PROFILE_TOTAL);

  // FIXME always preserve the extended gauge
  extendedGaugeResident = cudaGaugeExtended;

  return;
}

void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
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
    cudaGauge->loadCPUField(cpuGauge);
  }
  return cudaGauge;
}


void saveGaugeFieldQuda(void* gauge, void* inGauge, QudaGaugeParam* param){

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
  cudaGauge->saveCPUField(cpuGauge);
}


void destroyGaugeFieldQuda(void* gauge){
  cudaGaugeField* g = reinterpret_cast<cudaGaugeField*>(gauge);
  delete g;
}


void computeKSOprodQuda(void* oprod,
    void* fermion,
    double coeff,
    int X[4],
    QudaPrecision prec)

{
/*
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

*/
  return;
}

void computeStaggeredForceQuda(void* cudaMom, void* qudaQuark, double coeff)
{
  bool use_resident_solution = false;
  if (solutionResident[0]) {
    qudaQuark = solutionResident[0];
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
    delete solutionResident[0];
    solutionResident.clear();
  }

  return;
}


void computeAsqtadForceQuda(void* const milc_momentum,
    long long *flops,
    const double act_path_coeff[6],
    const void* const one_link_src[4],
    const void* const naik_src[4],
    const void* const link,
    const QudaGaugeParam* gParam)
{

#ifdef GPU_HISQ_FORCE
  long long partialFlops;
  using namespace quda::fermion_force;
  profileAsqtadForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileAsqtadForce.TPSTART(QUDA_PROFILE_INIT);

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
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_INIT);

#ifdef MULTI_GPU
  int R[4] = {2, 2, 2, 2};
#endif

  profileAsqtadForce.TPSTART(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(*cpuGauge);
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaInForce_ex->Gauge_p()), 0, cudaInForce_ex->Bytes());
  copyExtendedGauge(*cudaGauge_ex, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGauge_ex->exchangeExtendedGhost(R,true);
#endif

  profileAsqtadForce.TPSTART(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuOneLinkInForce);
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaInForce_ex->Gauge_p()), 0, cudaInForce_ex->Bytes());
  copyExtendedGauge(*cudaInForce_ex, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce_ex->exchangeExtendedGhost(R,true);
#endif

  cudaMemset((void**)(cudaOutForce->Gauge_p()), 0, cudaOutForce->Bytes());
  profileAsqtadForce.TPSTART(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  cudaMemset((void**)(cudaOutForce_ex->Gauge_p()), 0, cudaOutForce_ex->Bytes());
  hisqStaplesForceCuda(act_path_coeff, *gParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex, &partialFlops);
  *flops += partialFlops;
#else
  hisqStaplesForceCuda(act_path_coeff, *gParam, *cudaInForce, *cudaGauge, cudaOutForce, &partialFlops);
  *flops += partialFlops;
#endif
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileAsqtadForce.TPSTART(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuNaikInForce);
#ifdef MULTI_GPU
  copyExtendedGauge(*cudaInForce_ex, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce_ex->exchangeExtendedGhost(R,true);
#endif
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_H2D);

  profileAsqtadForce.TPSTART(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex, &partialFlops);
  *flops += partialFlops;
  completeKSForce(*cudaMom, *cudaOutForce_ex, *cudaGauge_ex, QUDA_CUDA_FIELD_LOCATION, &partialFlops);
  *flops += partialFlops;
#else
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *cudaInForce, *cudaGauge, cudaOutForce, &partialFlops);
  *flops += partialFlops;
  hisqCompleteForceCuda(*gParam, *cudaOutForce, *cudaGauge, cudaMom, &partialFlops);
  *flops += partialFlops;
#endif
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileAsqtadForce.TPSTART(QUDA_PROFILE_D2H);
  cudaMom->saveCPUField(*cpuMom);
  profileAsqtadForce.TPSTOP(QUDA_PROFILE_D2H);

  profileAsqtadForce.TPSTART(QUDA_PROFILE_FREE);
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

  profileAsqtadForce.TPSTOP(QUDA_PROFILE_FREE);

  profileAsqtadForce.TPSTOP(QUDA_PROFILE_TOTAL);
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

/*
  void* oprod[2];

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
    long long *flops,
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

  long long partialFlops;

  using namespace quda::fermion_force;
  profileHISQForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);

  double act_path_coeff[6] = {0,1,level2_coeff[2],level2_coeff[3],level2_coeff[4],level2_coeff[5]};
  // You have to look at the MILC routine to understand the following
  // Basically, I have already absorbed the one-link coefficient

  GaugeFieldParam param(0, *gParam);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.order  = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.gauge = (void*)milc_momentum;
  param.ghostExchange =  QUDA_GHOST_EXCHANGE_NO;
  cpuGaugeField* cpuMom = (!gParam->use_resident_mom) ? new cpuGaugeField(param) : NULL;

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
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(cpuWLink);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  int R[4] = {2, 2, 2, 2};
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#endif

  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuStapleForce);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaInForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuOneLinkForce);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaOutForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaOutForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#else
  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaOutForce->loadCPUField(*cpuOneLinkForce);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
#endif

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForceCuda(act_path_coeff, *gParam, *inForcePtr, *gaugePtr, outForcePtr, &partialFlops);
  *flops += partialFlops;
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Load naik outer product
  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaInForce->loadCPUField(*cpuNaikForce);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaInForceEx, *cudaInForce, QUDA_CUDA_FIELD_LOCATION);
  cudaInForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#endif

  // Compute Naik three-link term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqLongLinkForceCuda(act_path_coeff[1], *gParam, *inForcePtr, *gaugePtr, outForcePtr, &partialFlops);
  *flops += partialFlops;
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  cudaOutForceEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#endif
  // load v-link
  profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
  cudaGauge->loadCPUField(cpuVLink);
  profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);
#ifdef MULTI_GPU
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#endif
  // Done with cudaInForce. It becomes the output force. Oops!

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  *num_failures_h = 0;
  unitarizeForce(*inForcePtr, *outForcePtr, *gaugePtr, num_failures_d, &partialFlops);
  *flops += partialFlops;
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  if(*num_failures_h>0){
    errorQuda("Error in the unitarization component of the hisq fermion force: %d failures\n", *num_failures_h);
  }

  cudaMemset((void**)(outForcePtr->Gauge_p()), 0, outForcePtr->Bytes());
  // read in u-link
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  cudaGauge->loadCPUField(cpuULink);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);
#ifdef MULTI_GPU
  profileHISQForce.TPSTART(QUDA_PROFILE_COMMS);
  copyExtendedGauge(*cudaGaugeEx, *cudaGauge, QUDA_CUDA_FIELD_LOCATION);
  cudaGaugeEx->exchangeExtendedGhost(R,true);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMMS);
#endif

  // Compute Fat7-staple term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForceCuda(fat7_coeff, *gParam, *inForcePtr, *gaugePtr, outForcePtr, &partialFlops);
  *flops += partialFlops;
  hisqCompleteForceCuda(*gParam, *outForcePtr, *gaugePtr, cudaMom, &partialFlops);
  *flops += partialFlops;
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (gParam->use_resident_mom) {
    if (!momResident) errorQuda("No resident momentum field to use");
    updateMomentum(*momResident, 1.0, *cudaMom);
  }

  if (gParam->return_result_mom) {
    profileHISQForce.TPSTART(QUDA_PROFILE_D2H);
    // Close the paths, make anti-hermitian, and store in compressed format
    if (gParam->return_result_mom) cudaMom->saveCPUField(*cpuMom);
    profileHISQForce.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileHISQForce.TPSTART(QUDA_PROFILE_FREE);

  delete cpuStapleForce;
  delete cpuOneLinkForce;
  delete cpuNaikForce;
  if (cpuMom) delete cpuMom;

  delete cudaInForce;
  delete cudaGauge;

  if (!gParam->make_resident_mom) {
    delete momResident;
    momResident = NULL;
  }
  if (cudaMom) delete cudaMom;

#ifdef MULTI_GPU
  delete cudaInForceEx;
  delete cudaOutForceEx;
  delete cudaGaugeEx;
#else
  delete cudaOutForce;
#endif
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);
  profileHISQForce.TPSTOP(QUDA_PROFILE_TOTAL);
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

#ifdef  GPU_STAGGERED_OPROD
#ifndef BUILD_QDP_INTERFACE
#error "Staggered oprod requires BUILD_QDP_INTERFACE";
#endif
  using namespace quda;
  profileStaggeredOprod.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  profileStaggeredOprod.TPSTART(QUDA_PROFILE_INIT);
  GaugeFieldParam oParam(0, *param);

  oParam.nDim = 4;
  oParam.nFace = 0;
  // create the host outer-product field
  oParam.pad = 0;
  oParam.create = QUDA_REFERENCE_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.order = QUDA_QDP_GAUGE_ORDER;
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  oParam.gauge = oprod[0];
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO; // no need for ghost exchange here
  cpuGaugeField cpuOprod0(oParam);

  oParam.gauge = oprod[1];
  cpuGaugeField cpuOprod1(oParam);

  // create the device outer-product field
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField cudaOprod0(oParam);
  cudaGaugeField cudaOprod1(oParam);
  profileStaggeredOprod.TPSTOP(QUDA_PROFILE_INIT);

  //initLatticeConstants(cudaOprod0, profileStaggeredOprod);

  profileStaggeredOprod.TPSTART(QUDA_PROFILE_H2D);
  cudaOprod0.loadCPUField(cpuOprod0);
  cudaOprod1.loadCPUField(cpuOprod1);
  profileStaggeredOprod.TPSTOP(QUDA_PROFILE_H2D);


  profileStaggeredOprod.TPSTART(QUDA_PROFILE_INIT);



  ColorSpinorParam qParam;
  qParam.nColor = 3;
  qParam.nSpin = 1;
  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.precision = oParam.precision;
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];
  qParam.x[0] /= 2;

  // create the device quark field
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  cudaColorSpinorField cudaQuarkEven(qParam);
  cudaColorSpinorField cudaQuarkOdd(qParam);

  // create the host quark field
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;

  const int Ls = 1;
  const int Ninternal = 6;
  FaceBuffer faceBuffer1(cudaOprod0.X(), 4, Ninternal, 3, cudaOprod0.Precision(), Ls);
  FaceBuffer faceBuffer2(cudaOprod0.X(), 4, Ninternal, 3, cudaOprod0.Precision(), Ls);
  profileStaggeredOprod.TPSTOP(QUDA_PROFILE_INIT);

  // loop over different quark fields
  for(int i=0; i<num_terms; ++i){

    // Wrap the even-parity MILC quark field
    profileStaggeredOprod.TPSTART(QUDA_PROFILE_INIT);
    qParam.v = fermion[i];
    cpuColorSpinorField cpuQuarkEven(qParam); // create host quark field
    qParam.v = (char*)fermion[i] + cpuQuarkEven.RealLength()*cpuQuarkEven.Precision();
    cpuColorSpinorField cpuQuarkOdd(qParam); // create host field
    profileStaggeredOprod.TPSTOP(QUDA_PROFILE_INIT);

    profileStaggeredOprod.TPSTART(QUDA_PROFILE_H2D);
    cudaQuarkEven = cpuQuarkEven;
    cudaQuarkOdd = cpuQuarkOdd;
    profileStaggeredOprod.TPSTOP(QUDA_PROFILE_H2D);


    profileStaggeredOprod.TPSTART(QUDA_PROFILE_COMPUTE);
    // Operate on even-parity sites
    computeStaggeredOprod(cudaOprod0, cudaOprod1, cudaQuarkEven, cudaQuarkOdd, faceBuffer1, 0, coeff[i]);

    // Operate on odd-parity sites
    computeStaggeredOprod(cudaOprod0, cudaOprod1, cudaQuarkEven, cudaQuarkOdd, faceBuffer2, 1, coeff[i]);
    profileStaggeredOprod.TPSTOP(QUDA_PROFILE_COMPUTE);
  }


  // copy the outer product field back to the host
  profileStaggeredOprod.TPSTART(QUDA_PROFILE_D2H);
  cudaOprod0.saveCPUField(cpuOprod0);
  cudaOprod1.saveCPUField(cpuOprod1);
  profileStaggeredOprod.TPSTOP(QUDA_PROFILE_D2H);


  profileStaggeredOprod.TPSTOP(QUDA_PROFILE_TOTAL);

  checkCudaError();
  return;
#else
  errorQuda("Staggered oprod has not been built");
#endif
}


/*
   void computeStaggeredOprodQuda(void** oprod,
   void** fermion,
   int num_terms,
   double** coeff,
   QudaGaugeParam* param)
   {
   using namespace quda;
   profileStaggeredOprod.TPSTART(QUDA_PROFILE_TOTAL);

   checkGaugeParam(param);

   profileStaggeredOprod.TPSTART(QUDA_PROFILE_INIT);
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

profileStaggeredOprod.TPSTOP(QUDA_PROFILE_INIT);


profileStaggeredOprod.TPSTART(QUDA_PROFILE_H2D);
cudaOprod0.loadCPUField(cpuOprod0);
cudaOprod1.loadCPUField(cpuOprod1);
profileStaggeredOprod.TPSTOP(QUDA_PROFILE_H2D);



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
profileStaggeredOprod.TPSTART(QUDA_PROFILE_D2H);
cudaOprod0.saveCPUField(cpuOprod0);
cudaOprod1.saveCPUField(cpuOprod1);
profileStaggeredOprod.TPSTOP(QUDA_PROFILE_D2H);


for(int i=0; i<num_terms; ++i){
  delete dQuark[i];
}
delete[] dQuark;
delete[] new_coeff;

profileStaggeredOprod.TPSTOP(QUDA_PROFILE_TOTAL);

checkCudaError();
return;
}
*/


void computeCloverForceQuda(void *h_mom, double dt, void **h_x, void **h_p,
			    double *coeff, double kappa2, double ck,
			    int nvector, double multiplicity, void *gauge,
			    QudaGaugeParam *gauge_param, QudaInvertParam *inv_param) {


  using namespace quda;
  profileCloverForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileCloverForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(gauge_param);
  if (!gaugePrecise) errorQuda("No resident gauge field");

  GaugeFieldParam fParam(0, *gauge_param);
  // create the host momentum field
  fParam.create = QUDA_REFERENCE_FIELD_CREATE;
  fParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.order = gauge_param->gauge_order;
  fParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  fParam.gauge = h_mom;
  cpuGaugeField cpuMom(fParam);

  // create the device momentum field
  fParam.create = QUDA_ZERO_FIELD_CREATE;
  fParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField cudaMom(fParam);

  // create the device force field
  fParam.link_type = QUDA_GENERAL_LINKS;
  fParam.create = QUDA_ZERO_FIELD_CREATE;
  fParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  fParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGaugeField cudaForce(fParam);

  ColorSpinorParam qParam;
  qParam.nColor = 3;
  qParam.nSpin = 4;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.precision = fParam.precision;
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = fParam.x[dir];

  // create the device quark field
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  cudaColorSpinorField **cudaQuarkX = new cudaColorSpinorField*[nvector];
  cudaColorSpinorField **cudaQuarkP = new cudaColorSpinorField*[nvector];
  for (int i=0; i<nvector; i++) {
    cudaQuarkX[i] = new cudaColorSpinorField(qParam);
    cudaQuarkP[i] = new cudaColorSpinorField(qParam);
  }

  // create the host quark field
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // need expose this to interface

  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc_solve);
  Dirac *dirac = Dirac::create(diracParam);

  // for downloading x_e
  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.x[0] /= 2;

  if (inv_param->use_resident_solution) {
    if (solutionResident.size() != (unsigned int)nvector)
      errorQuda("solutionResident.size() %lu does not match number of shifts %d",
		solutionResident.size(), nvector);
  }

  profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);

  // loop over different quark fields
  for(int i=0; i<nvector; ++i){
    cudaColorSpinorField &x = *(cudaQuarkX[i]);
    cudaColorSpinorField &p = *(cudaQuarkP[i]);

    if (!inv_param->use_resident_solution) {
      // Wrap the even-parity MILC quark field
      profileCloverForce.TPSTART(QUDA_PROFILE_INIT);
      qParam.v = h_x[i];
      cpuColorSpinorField cpuQuarkX(qParam); // create host quark field
      profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);

      profileCloverForce.TPSTART(QUDA_PROFILE_H2D);
      x.Even() = cpuQuarkX;
      profileCloverForce.TPSTOP(QUDA_PROFILE_H2D);

      profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);
      gamma5Cuda(static_cast<cudaColorSpinorField*>(&x.Even()), static_cast<cudaColorSpinorField*>(&x.Even()));
      profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);
      x.Even() = *(solutionResident[i]);
      profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);

      profileCloverForce.TPSTART(QUDA_PROFILE_FREE);
      delete solutionResident[i];
      profileCloverForce.TPSTOP(QUDA_PROFILE_FREE);
    }
    profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);
    dirac->Dslash(x.Odd(), x.Even(), QUDA_ODD_PARITY);
    dirac->M(p.Even(), x.Even());
    dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(p.Odd(), p.Even(), QUDA_ODD_PARITY);
    dirac->Dagger(QUDA_DAG_NO);

    gamma5Cuda(static_cast<cudaColorSpinorField*>(&x.Even()), static_cast<cudaColorSpinorField*>(&x.Even()));
    gamma5Cuda(static_cast<cudaColorSpinorField*>(&x.Odd()), static_cast<cudaColorSpinorField*>(&x.Odd()));
    gamma5Cuda(static_cast<cudaColorSpinorField*>(&p.Even()), static_cast<cudaColorSpinorField*>(&p.Even()));
    gamma5Cuda(static_cast<cudaColorSpinorField*>(&p.Odd()), static_cast<cudaColorSpinorField*>(&p.Odd()));

    checkCudaError();

    computeCloverForce(cudaForce, *gaugePrecise, x, p, 2.0*dt*coeff[i]*kappa2);
    profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  profileCloverForce.TPSTART(QUDA_PROFILE_FREE);
  if (inv_param->use_resident_solution) solutionResident.clear();
  delete dirac;
  profileCloverForce.TPSTOP(QUDA_PROFILE_FREE);

  profileCloverForce.TPSTART(QUDA_PROFILE_INIT);

  cudaGaugeField &gaugeEx = *extendedGaugeResident;

  // create oprod and trace fields
  fParam.geometry = QUDA_TENSOR_GEOMETRY;
  cudaGaugeField oprod(fParam);
  cudaGaugeField &trace = oprod;

  // create extended oprod field
  for (int i=0; i<4; i++) fParam.x[i] += 2*R[i];
  fParam.nFace = 1; // breaks with out this - why?

  cudaGaugeField oprodEx(fParam);
  cudaGaugeField &traceEx = oprodEx;

  profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);

  profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);

  computeCloverSigmaTrace(trace, *cloverPrecise, QUDA_CUDA_FIELD_LOCATION);
  copyExtendedGauge(traceEx, trace, QUDA_CUDA_FIELD_LOCATION); // FIXME this is unnecessary if we write directly to traceEx

  profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);
  profileCloverForce.TPSTART(QUDA_PROFILE_COMMS);

  traceEx.exchangeExtendedGhost(R,redundant_comms);

  profileCloverForce.TPSTOP(QUDA_PROFILE_COMMS);
  profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);

  // In double precision the clover derivative is faster with no reconstruct
  cudaGaugeField *u = &gaugeEx;
  if (gaugeEx.Reconstruct() == QUDA_RECONSTRUCT_12 && gaugeEx.Precision() == QUDA_DOUBLE_PRECISION) {
    GaugeFieldParam param(gaugeEx);
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    u = new cudaGaugeField(param);
    u -> copy(gaugeEx);
  }

  cloverDerivative(cudaForce, *u, traceEx, 2.0*ck*multiplicity*dt, QUDA_ODD_PARITY, 0);

  /* Now the U dA/dU terms */
  for(int shift = 0; shift < nvector; shift++){
    double ferm_epsilon = 2.0*dt*coeff[shift];
    computeCloverSigmaOprod(oprod, *(cudaQuarkX[shift]), *(cudaQuarkP[shift]), ferm_epsilon, shift);
  }
  copyExtendedGauge(oprodEx, oprod, QUDA_CUDA_FIELD_LOCATION); // FIXME this is unnecessary if we write directly to oprod

  profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);
  profileCloverForce.TPSTART(QUDA_PROFILE_COMMS);

  oprodEx.exchangeExtendedGhost(R,redundant_comms);

  profileCloverForce.TPSTOP(QUDA_PROFILE_COMMS);
  profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);

  // TODO this first derivative can be combined with the previous one
  // if we sum project oprodEx and sum to traceEx with the appropriate
  // weights.  This will also half the amount of communication
  cloverDerivative(cudaForce, *u, oprodEx, -kappa2*ck, QUDA_ODD_PARITY, 1);
  cloverDerivative(cudaForce, *u, oprodEx, ck, QUDA_EVEN_PARITY, 1);

  if (u != &gaugeEx) delete u;

  updateMomentum(cudaMom, -1.0, cudaForce);
  profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // copy the outer product field back to the host
  profileCloverForce.TPSTART(QUDA_PROFILE_D2H);
  cudaMom.saveCPUField(cpuMom);
  profileCloverForce.TPSTOP(QUDA_PROFILE_D2H);

  profileCloverForce.TPSTART(QUDA_PROFILE_FREE);

  for (int i=0; i<nvector; i++) {
    delete cudaQuarkX[i];
    delete cudaQuarkP[i];
  }
  delete []cudaQuarkX;
  delete []cudaQuarkP;

  checkCudaError();

  profileCloverForce.TPSTOP(QUDA_PROFILE_FREE);
  profileCloverForce.TPSTOP(QUDA_PROFILE_TOTAL);

  return;
}



void updateGaugeFieldQuda(void* gauge,
    void* momentum,
    double dt,
    int conj_mom,
    int exact,
    QudaGaugeParam* param)
{
  profileGaugeUpdate.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  profileGaugeUpdate.TPSTART(QUDA_PROFILE_INIT);
  GaugeFieldParam gParam(0, *param);

  // create the host fields
  gParam.pad = 0;
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.link_type = QUDA_SU3_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.gauge = gauge;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : NULL;

  gParam.reconstruct = gParam.order == QUDA_TIFR_GAUGE_ORDER ?
   QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.gauge = momentum;
  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParam) : NULL;

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

  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_INIT);

  profileGaugeUpdate.TPSTART(QUDA_PROFILE_H2D);

  if (!param->use_resident_gauge) {   // load fields onto the device
    cudaInGauge->loadCPUField(*cpuGauge);
  } else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = NULL;
  }

  if (!param->use_resident_mom) {
    cudaMom->loadCPUField(*cpuMom);
  } else {
    if (!momResident) errorQuda("No resident mom field allocated");
    cudaMom = momResident;
    momResident = NULL;
  }

  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_H2D);

  // perform the update
  profileGaugeUpdate.TPSTART(QUDA_PROFILE_COMPUTE);
  updateGaugeField(*cudaOutGauge, dt, *cudaInGauge, *cudaMom,
      (bool)conj_mom, (bool)exact);
  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (param->return_result_gauge) {
    // copy the gauge field back to the host
    profileGaugeUpdate.TPSTART(QUDA_PROFILE_D2H);
    cudaOutGauge->saveCPUField(*cpuGauge);
    profileGaugeUpdate.TPSTOP(QUDA_PROFILE_D2H);
  }


  profileGaugeUpdate.TPSTART(QUDA_PROFILE_FREE);
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
  if (cpuMom) delete cpuMom;
  if (cpuGauge) delete cpuGauge;
  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_FREE);

  checkCudaError();

  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_TOTAL);
  return;
}

 void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param) {
   profileProject.TPSTART(QUDA_PROFILE_TOTAL);

   profileProject.TPSTART(QUDA_PROFILE_INIT);
   checkGaugeParam(param);

   // create the gauge field
   GaugeFieldParam gParam(0, *param);
   gParam.pad = 0;
   gParam.create = QUDA_REFERENCE_FIELD_CREATE;
   gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
   gParam.reconstruct = QUDA_RECONSTRUCT_NO;
   gParam.link_type = QUDA_GENERAL_LINKS;
   gParam.gauge = gauge_h;
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : NULL;

   // create the device fields
   gParam.create = QUDA_NULL_FIELD_CREATE;
   gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
   gParam.reconstruct = param->reconstruct;
   cudaGaugeField *cudaGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : NULL;
   profileProject.TPSTOP(QUDA_PROFILE_INIT);

   if (param->use_resident_gauge) {
     if (!gaugePrecise) errorQuda("No resident gauge field to use");
     cudaGauge = gaugePrecise;
   } else {
     profileProject.TPSTART(QUDA_PROFILE_H2D);
     cudaGauge->loadCPUField(*cpuGauge);
     profileProject.TPSTOP(QUDA_PROFILE_H2D);
   }

   profileProject.TPSTART(QUDA_PROFILE_COMPUTE);
   *num_failures_h = 0;

   // project onto SU(3)
   projectSU3(*cudaGauge, tol, num_failures_d);

   profileProject.TPSTOP(QUDA_PROFILE_COMPUTE);

   if(*num_failures_h>0)
     errorQuda("Error in the SU(3) unitarization: %d failures\n", *num_failures_h);

   profileProject.TPSTART(QUDA_PROFILE_D2H);
   if (param->return_result_gauge) cudaGauge->saveCPUField(*cpuGauge);
   profileProject.TPSTOP(QUDA_PROFILE_D2H);

   if (param->make_resident_gauge) {
     if (gaugePrecise != NULL && cudaGauge != gaugePrecise) delete gaugePrecise;
     gaugePrecise = cudaGauge;
   } else {
     delete cudaGauge;
   }

   profileProject.TPSTART(QUDA_PROFILE_FREE);
   if (cpuGauge) delete cpuGauge;
   profileProject.TPSTOP(QUDA_PROFILE_FREE);

   profileProject.TPSTOP(QUDA_PROFILE_TOTAL);
 }

 void staggeredPhaseQuda(void *gauge_h, QudaGaugeParam *param) {
   profilePhase.TPSTART(QUDA_PROFILE_TOTAL);

   profilePhase.TPSTART(QUDA_PROFILE_INIT);
   checkGaugeParam(param);

   // create the gauge field
   GaugeFieldParam gParam(0, *param);
   gParam.pad = 0;
   gParam.create = QUDA_REFERENCE_FIELD_CREATE;
   gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
   gParam.reconstruct = QUDA_RECONSTRUCT_NO;
   gParam.link_type = QUDA_GENERAL_LINKS;
   gParam.gauge = gauge_h;
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : NULL;

   // create the device fields
   gParam.create = QUDA_NULL_FIELD_CREATE;
   gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
   gParam.reconstruct = param->reconstruct;
   cudaGaugeField *cudaGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : NULL;
   profilePhase.TPSTOP(QUDA_PROFILE_INIT);

   if (param->use_resident_gauge) {
     if (!gaugePrecise) errorQuda("No resident gauge field to use");
     cudaGauge = gaugePrecise;
   } else {
     profilePhase.TPSTART(QUDA_PROFILE_H2D);
     cudaGauge->loadCPUField(*cpuGauge);
     profilePhase.TPSTOP(QUDA_PROFILE_H2D);
   }

   profilePhase.TPSTART(QUDA_PROFILE_COMPUTE);
   *num_failures_h = 0;

   // apply / remove phase as appropriate
   if (!cudaGauge->StaggeredPhaseApplied()) cudaGauge->applyStaggeredPhase();
   else cudaGauge->removeStaggeredPhase();

   profilePhase.TPSTOP(QUDA_PROFILE_COMPUTE);

   profilePhase.TPSTART(QUDA_PROFILE_D2H);
   if (param->return_result_gauge) cudaGauge->saveCPUField(*cpuGauge);
   profilePhase.TPSTOP(QUDA_PROFILE_D2H);

   if (param->make_resident_gauge) {
     if (gaugePrecise != NULL && cudaGauge != gaugePrecise) delete gaugePrecise;
     gaugePrecise = cudaGauge;
   } else {
     delete cudaGauge;
   }

   profilePhase.TPSTART(QUDA_PROFILE_FREE);
   if (cpuGauge) delete cpuGauge;
   profilePhase.TPSTOP(QUDA_PROFILE_FREE);

   profilePhase.TPSTOP(QUDA_PROFILE_TOTAL);
 }

// evaluate the momentum action
double momActionQuda(void* momentum, QudaGaugeParam* param)
{
  profileMomAction.TPSTART(QUDA_PROFILE_TOTAL);

  profileMomAction.TPSTART(QUDA_PROFILE_INIT);
  checkGaugeParam(param);

  // create the momentum fields
  GaugeFieldParam gParam(0, *param);
  gParam.pad = 0;
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.reconstruct = (gParam.order == QUDA_TIFR_GAUGE_ORDER) ?
    QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.gauge = momentum;

  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParam) : NULL;

  // create the device fields
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;

  cudaGaugeField *cudaMom = !param->use_resident_mom ? new cudaGaugeField(gParam) : NULL;

  profileMomAction.TPSTOP(QUDA_PROFILE_INIT);

  profileMomAction.TPSTART(QUDA_PROFILE_H2D);
  if (!param->use_resident_mom) {
    cudaMom->loadCPUField(*cpuMom);
  } else {
    if (!momResident) errorQuda("No resident mom field allocated");
    cudaMom = momResident;
  }
  profileMomAction.TPSTOP(QUDA_PROFILE_H2D);

  // perform the update
  profileMomAction.TPSTART(QUDA_PROFILE_COMPUTE);
  double action = computeMomAction(*cudaMom);
  profileMomAction.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileMomAction.TPSTART(QUDA_PROFILE_FREE);
  if (param->make_resident_mom) {
    if (momResident != NULL && momResident != cudaMom) delete momResident;
    momResident = cudaMom;
  } else {
    delete cudaMom;
    momResident = NULL;
  }
  if (cpuMom) {
    delete cpuMom;
  }

  profileMomAction.TPSTOP(QUDA_PROFILE_FREE);

  checkCudaError();

  profileMomAction.TPSTOP(QUDA_PROFILE_TOTAL);
  return action;
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
void invert_quda_(void *hp_x, void *hp_b, QudaInvertParam *param) {
  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;
  invertQuda(hp_x, hp_b, param);
}
void invert_multishift_quda_(void *hp_x[QUDA_MAX_MULTI_SHIFT], void *hp_b, QudaInvertParam *param) {
  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;
  invertMultiShiftQuda(hp_x, hp_b, param);
}
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

  // fortran uses multi-dimensional arrays which we have convert into an array of pointers to pointers
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

  computeGaugeForceQuda(mom, gauge, input_path, path_length, loop_coeff, *num_paths, *max_length, *dt, param);

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
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("applying staggered phase\n");
  if (gaugePrecise) {
    gaugePrecise->applyStaggeredPhase();
  } else {
    errorQuda("No persistent gauge field");
  }
}

// remove the staggered phases
void remove_staggered_phase_quda_() {
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("removing staggered phase\n");
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
#ifdef MULTI_GPU
static int bqcd_rank_from_coords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = dims[i] * rank + coords[i];
  }
  return rank;
}
#endif

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

/*
 * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
 */
void plaq_quda_(double plaq[3]) {
  plaqQuda(plaq);
}


void plaqQuda (double plq[3])
{
  profilePlaq.TPSTART(QUDA_PROFILE_TOTAL);

  profilePlaq.TPSTART(QUDA_PROFILE_INIT);
  if (!gaugePrecise)
    errorQuda("Cannot compute plaquette as there is no resident gauge field");

  cudaGaugeField *data = NULL;
#ifndef MULTI_GPU
  data = gaugePrecise;
#else
  if (extendedGaugeResident) {
    data = extendedGaugeResident;
  } else {
    int y[4];
    for(int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 2*R[dir];
    int pad = 0;
    GaugeFieldParam gParamEx(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
			     pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = gaugePrecise->Order();
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = gaugePrecise->TBoundary();
    gParamEx.nFace = 1;
    gParamEx.tadpole = gaugePrecise->Tadpole();
    for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];

    data = new cudaGaugeField(gParamEx);

    copyExtendedGauge(*data, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);
    profilePlaq.TPSTOP(QUDA_PROFILE_INIT);

    profilePlaq.TPSTART(QUDA_PROFILE_COMMS);
    data->exchangeExtendedGhost(R,redundant_comms);
    profilePlaq.TPSTOP(QUDA_PROFILE_COMMS);

    profilePlaq.TPSTART(QUDA_PROFILE_INIT);
    extendedGaugeResident = data;
  }
#endif

  profilePlaq.TPSTOP(QUDA_PROFILE_INIT);

  profilePlaq.TPSTART(QUDA_PROFILE_COMPUTE);
  double3 plaq = quda::plaquette(*data, QUDA_CUDA_FIELD_LOCATION);
  plq[0] = plaq.x;
  plq[1] = plaq.y;
  plq[2] = plaq.z;
  profilePlaq.TPSTOP(QUDA_PROFILE_COMPUTE);

  profilePlaq.TPSTOP(QUDA_PROFILE_TOTAL);
  return;
}

void performAPEnStep(unsigned int nSteps, double alpha)
{
  profileAPE.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == NULL) {
    errorQuda("Gauge field must be loaded");
  }

  int pad = 0;
  int y[4];

#ifdef MULTI_GPU
  for (int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 2 * R[dir];
  GaugeFieldParam gParam(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
                         pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
#else
  for (int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir];
  GaugeFieldParam gParam(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
                         pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_NO);
#endif

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.order = gaugePrecise->Order();
  gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParam.t_boundary = gaugePrecise->TBoundary();
  gParam.nFace = 1;
  gParam.tadpole = gaugePrecise->Tadpole();

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir) gParam.r[dir] = R[dir];
#endif

  if (gaugeSmeared != NULL) {
    delete gaugeSmeared;
  }

  gaugeSmeared = new cudaGaugeField(gParam);

#ifdef MULTI_GPU
  copyExtendedGauge(*gaugeSmeared, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);
  gaugeSmeared->exchangeExtendedGhost(R,redundant_comms);
#else
  gaugeSmeared->copy(*gaugePrecise);
#endif

  cudaGaugeField *cudaGaugeTemp = NULL;
  cudaGaugeTemp = new cudaGaugeField(gParam);

  if (getVerbosity() == QUDA_VERBOSE) {
    double3 plq = plaquette(*gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    printfQuda("Plaquette after 0 APE steps: %le\n", plq.x);
  }

  for (unsigned int i=0; i<nSteps; i++) {
      cudaGaugeTemp->copy(*gaugeSmeared);
#ifdef MULTI_GPU
      cudaGaugeTemp->exchangeExtendedGhost(R,redundant_comms);
#endif
      APEStep(*gaugeSmeared, *cudaGaugeTemp, alpha, QUDA_CUDA_FIELD_LOCATION);
  }

  delete cudaGaugeTemp;

#ifdef MULTI_GPU
  gaugeSmeared->exchangeExtendedGhost(R,redundant_comms);
#endif

  if (getVerbosity() == QUDA_VERBOSE) {
    double3 plq = plaquette(*gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    printfQuda("Plaquette after %d APE steps: %le\n", nSteps, plq.x);
  }

  profileAPE.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performSTOUTnStep(unsigned int nSteps, double rho)
{
  profileSTOUT.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == NULL) {
    errorQuda("Gauge field must be loaded");
  }

  int pad = 0;
  int y[4];

#ifdef MULTI_GPU
  for (int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 2 * R[dir];
  GaugeFieldParam gParam(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
                         pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
#else
  for (int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir];
  GaugeFieldParam gParam(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
                         pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_NO);
#endif

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.order = gaugePrecise->Order();
  gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParam.t_boundary = gaugePrecise->TBoundary();
  gParam.nFace = 1;
  gParam.tadpole = gaugePrecise->Tadpole();

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir) gParam.r[dir] = R[dir];
#endif

  if (gaugeSmeared != NULL) {
    delete gaugeSmeared;
  }

  gaugeSmeared = new cudaGaugeField(gParam);

#ifdef MULTI_GPU
  copyExtendedGauge(*gaugeSmeared, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);
  gaugeSmeared->exchangeExtendedGhost(R,redundant_comms);
#else
  gaugeSmeared->copy(*gaugePrecise);
#endif

  cudaGaugeField *cudaGaugeTemp = NULL;
  cudaGaugeTemp = new cudaGaugeField(gParam);

  if (getVerbosity() == QUDA_VERBOSE) {
    double3 plq = plaquette(*gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    printfQuda("Plaquette after 0 STOUT steps: %le\n", plq.x);
  }

  for (unsigned int i=0; i<nSteps; i++) {
      cudaGaugeTemp->copy(*gaugeSmeared);
#ifdef MULTI_GPU
      cudaGaugeTemp->exchangeExtendedGhost(R,redundant_comms);
#endif
      STOUTStep(*gaugeSmeared, *cudaGaugeTemp, rho, QUDA_CUDA_FIELD_LOCATION);
  }

  delete cudaGaugeTemp;

#ifdef MULTI_GPU
  gaugeSmeared->exchangeExtendedGhost(R,redundant_comms);
#endif

  if (getVerbosity() == QUDA_VERBOSE) {
    double3 plq = plaquette(*gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    printfQuda("Plaquette after %d STOUT steps: %le\n", nSteps, plq.x);
  }

  profileSTOUT.TPSTOP(QUDA_PROFILE_TOTAL);
}


int computeGaugeFixingOVRQuda(void* gauge, const unsigned int gauge_dir,  const unsigned int Nsteps, \
  const unsigned int verbose_interval, const double relax_boost, const double tolerance, const unsigned int reunit_interval, \
  const unsigned int  stopWtheta, QudaGaugeParam* param , double* timeinfo)
{

  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_INIT);
  GaugeFieldParam gParam(gauge, *param);
  cpuGaugeField *cpuGauge = new cpuGaugeField(gParam);

  //gParam.pad = getFatLinkPadding(param->X);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.order       = (gParam.precision == QUDA_DOUBLE_PRECISION || gParam.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *cudaInGauge = new cudaGaugeField(gParam);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_INIT);

  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_H2D);


  ///if (!param->use_resident_gauge) {   // load fields onto the device
  cudaInGauge->loadCPUField(*cpuGauge);
 /* } else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = NULL;
  } */

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_H2D);

  checkCudaError();

#ifdef MULTI_GPU
  if(comm_size() == 1){
    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugefixingOVR(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, \
      reunit_interval, stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);
    checkCudaError();
    // copy the gauge field back to the host
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_D2H);
  }
  else{

    int y[4];
    for(int dir=0; dir<4; ++dir) y[dir] = cudaInGauge->X()[dir] + 2 * R[dir];
    int pad = 0;
    GaugeFieldParam gParamEx(y, cudaInGauge->Precision(), gParam.reconstruct,
        pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = cudaInGauge->Order();
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = cudaInGauge->TBoundary();
    gParamEx.nFace = 1;
    for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
    cudaGaugeField *cudaInGaugeEx = new cudaGaugeField(gParamEx);

    copyExtendedGauge(*cudaInGaugeEx, *cudaInGauge, QUDA_CUDA_FIELD_LOCATION);
    cudaInGaugeEx->exchangeExtendedGhost(R,redundant_comms);
    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugefixingOVR(*cudaInGaugeEx, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, \
      reunit_interval, stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);

    checkCudaError();
    // copy the gauge field back to the host
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_D2H);
    //HOW TO COPY BACK TO CPU: cudaInGaugeEx->cpuGauge
    copyExtendedGauge(*cudaInGauge, *cudaInGaugeEx, QUDA_CUDA_FIELD_LOCATION);
  }
#else
  // perform the update
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
  gaugefixingOVR(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, \
    reunit_interval, stopWtheta);
  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);
  // copy the gauge field back to the host
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_D2H);
#endif

  checkCudaError();
  cudaInGauge->saveCPUField(*cpuGauge);
  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_D2H);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != NULL) delete gaugePrecise;
    gaugePrecise = cudaInGauge;
  } else {
    delete cudaInGauge;
  }

  if(timeinfo){
    timeinfo[0] = GaugeFixOVRQuda.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = GaugeFixOVRQuda.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = GaugeFixOVRQuda.Last(QUDA_PROFILE_D2H);
  }

  checkCudaError();
  return 0;
}




int computeGaugeFixingFFTQuda(void* gauge, const unsigned int gauge_dir,  const unsigned int Nsteps, \
  const unsigned int verbose_interval, const double alpha, const unsigned int autotune, const double tolerance, \
  const unsigned int  stopWtheta, QudaGaugeParam* param , double* timeinfo)
{

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_INIT);

  GaugeFieldParam gParam(gauge, *param);
  cpuGaugeField *cpuGauge = new cpuGaugeField(gParam);

  //gParam.pad = getFatLinkPadding(param->X);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.order       = (gParam.precision == QUDA_DOUBLE_PRECISION || gParam.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

  cudaGaugeField *cudaInGauge = new cudaGaugeField(gParam);


  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_INIT);

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_H2D);

  //if (!param->use_resident_gauge) {   // load fields onto the device
  cudaInGauge->loadCPUField(*cpuGauge);
  /*} else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = NULL;
  } */


  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_H2D);

  // perform the update
  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_COMPUTE);
  checkCudaError();

  gaugefixingFFT(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_COMPUTE);

  checkCudaError();
  // copy the gauge field back to the host
  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_D2H);
  checkCudaError();
  cudaInGauge->saveCPUField(*cpuGauge);
  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_D2H);
  checkCudaError();

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != NULL) delete gaugePrecise;
    gaugePrecise = cudaInGauge;
  } else {
    delete cudaInGauge;
  }

  if(timeinfo){
    timeinfo[0] = GaugeFixFFTQuda.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = GaugeFixFFTQuda.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = GaugeFixFFTQuda.Last(QUDA_PROFILE_D2H);
  }

  checkCudaError();
  return 0;
}

/**
 * Compute a volume or time-slice contraction of two spinors.
 * @param x     Spinor to contract. This is conjugated before contraction.
 * @param y     Spinor to contract.
 * @param ctrn  Contraction output. The size must be Volume*16
 * @param cType Contraction type, allows for volume or time-slice contractions.
 * @param tC    Time-slice to contract in case the contraction is in a single time-slice.
 */
void contract(const cudaColorSpinorField x, const cudaColorSpinorField y, void *ctrn, const QudaContractType cType)
{
  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    contractCuda(x.Even(), y.Even(), ((double2*)ctrn), cType, QUDA_EVEN_PARITY, profileContract);
    contractCuda(x.Odd(),  y.Odd(),  ((double2*)ctrn), cType, QUDA_ODD_PARITY,  profileContract);
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    contractCuda(x.Even(), y.Even(), ((float2*) ctrn), cType, QUDA_EVEN_PARITY, profileContract);
    contractCuda(x.Odd(),  y.Odd(),  ((float2*) ctrn), cType, QUDA_ODD_PARITY,  profileContract);
  } else {
    errorQuda("Precision not supported for contractions\n");
  }
}

void contract(const cudaColorSpinorField x, const cudaColorSpinorField y, void *ctrn, const QudaContractType cType, const int tC)
{
  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    contractCuda(x.Even(), y.Even(), ((double2*)ctrn), cType, tC, QUDA_EVEN_PARITY, profileContract);
    contractCuda(x.Odd(),  y.Odd(),  ((double2*)ctrn), cType, tC, QUDA_ODD_PARITY,  profileContract);
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    contractCuda(x.Even(), y.Even(), ((float2*) ctrn), cType, tC, QUDA_EVEN_PARITY, profileContract);
    contractCuda(x.Odd(),  y.Odd(),  ((float2*) ctrn), cType, tC, QUDA_ODD_PARITY,  profileContract);
  } else {
    errorQuda("Precision not supported for contractions\n");
  }
}

double qChargeCuda ()
{
  cudaGaugeField *data = NULL;

#ifndef MULTI_GPU
  if (!gaugeSmeared)
    data = gaugePrecise;
  else
    data = gaugeSmeared;
#else
  if ((!gaugeSmeared) && (extendedGaugeResident)) {
    data = extendedGaugeResident;
  } else {
    if (!gaugeSmeared) {
      int y[4];
      for(int dir=0; dir<4; ++dir) y[dir] = gaugePrecise->X()[dir] + 2 * R[dir];
      int pad = 0;
      GaugeFieldParam gParamEx(y, gaugePrecise->Precision(), gaugePrecise->Reconstruct(),
        pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
      gParamEx.create = QUDA_ZERO_FIELD_CREATE;
      gParamEx.order = gaugePrecise->Order();
      gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
      gParamEx.t_boundary = gaugePrecise->TBoundary();
      gParamEx.nFace = 1;
      gParamEx.tadpole = gaugePrecise->Tadpole();
      for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];

      data = new cudaGaugeField(gParamEx);

      copyExtendedGauge(*data, *gaugePrecise, QUDA_CUDA_FIELD_LOCATION);
      data->exchangeExtendedGhost(R,redundant_comms);
      extendedGaugeResident = data;
      cudaDeviceSynchronize();
    } else {
      data = gaugeSmeared;
    }
  }
                                 // Do we keep the smeared extended field on memory, or the unsmeared one?
#endif

  GaugeField *gauge = data;
  // create the Fmunu field

  GaugeFieldParam tensorParam(gaugePrecise->X(), gauge->Precision(), QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
  tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField Fmunu(tensorParam);

  computeFmunu(Fmunu, *data, QUDA_CUDA_FIELD_LOCATION);
  return quda::computeQCharge(Fmunu, QUDA_CUDA_FIELD_LOCATION);
}
