#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <quda.h>
//#include <quda_fortran.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <color_spinor_field.h>
#include <algorithm>
#include <staggered_oprod.h>

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

//#include <ks_force_quda.h>

#ifdef GPU_GAUGE_FORCE
//#include <gauge_force_quda.h>
#endif
//#include <gauge_update_quda.h>

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

cudaGaugeField *extendedGaugeResident = NULL;

std::vector<cudaColorSpinorField*> solutionResident;

// vector of spinors used for forecasting solutions in HMC
#define QUDA_MAX_CHRONO 2
// each entry is a pair for both p and Ap storage
std::vector< std::vector< std::pair<ColorSpinorField*,ColorSpinorField*> > > chronoResident(QUDA_MAX_CHRONO);

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

//!< Profiler for invertQuda
static TimeProfile profileInvert("invertQuda");

//!< Profiler for computeFatLinkQuda
static TimeProfile profileFatLink("computeKSLinkQuda");

//!<Profiler for createExtendedGaugeField
static TimeProfile profileExtendedGauge("createExtendedGaugeField");


//!<Profiler for computeStaggeredOprodQuda
static TimeProfile profileStaggeredOprod("computeStaggeredOprodQuda");

//!<Profiler for plaqQuda
static TimeProfile profilePlaq("plaqQuda");

//!< Profiler for projectSU3Quda
static TimeProfile profileProject("projectSU3Quda");

//!< Profiler for contractions
static TimeProfile profileContract("contractQuda");

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");



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

void flushChronoQuda(int i)
{
  if (i >= QUDA_MAX_CHRONO)
    errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

  auto &basis = chronoResident[i];

  for (unsigned int j=0; j<basis.size(); j++) {
    if (basis[j].first)  delete basis[j].first;
    if (basis[j].second) delete basis[j].second;
  }
  basis.clear();
}

void endQuda(void)
{
  profileEnd.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) return;

  freeGaugeQuda();

  for (int i=0; i<QUDA_MAX_CHRONO; i++) flushChronoQuda(i);

  for (unsigned int i=0; i<solutionResident.size(); i++) {
    if(solutionResident[i]) delete solutionResident[i];
  }
  solutionResident.clear();

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
    profileInvert.Print();
    profileFatLink.Print();
    profileExtendedGauge.Print();
    profileStaggeredOprod.Print();
    profilePlaq.Print();
    profileProject.Print();
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
    case QUDA_STAGGERED_DSLASH:
      diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
      break;
    case QUDA_ASQTAD_DSLASH:
      diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
      break;
    default:
      errorQuda("Unsupported dslash_type %d", inv_param->dslash_type);
    }

    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = gaugePrecise;
    diracParam.fatGauge = gaugeFatPrecise;
    diracParam.longGauge = gaugeLongPrecise;
    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on
  }


  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugeSloppy;
    diracParam.fatGauge = gaugeFatSloppy;
    diracParam.longGauge = gaugeLongSloppy;

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

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = comms ? 1 : 0;
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
		    param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) ? kappa5 : param.kappa;

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
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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
  dirac->Dslash(out, in, parity); // apply the operator

  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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
    errorQuda("This type of dslashQuda operator is defined for QUDA_DOMAIN_WALL_4D_DSLASH only");

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), 1, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) setKernelPackT(true);

  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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
      inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), pc, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE){
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

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE){
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

  return cudaGauge;
}


multigrid_solver::multigrid_solver(QudaMultigridParam &mg_param, TimeProfile &profile)
  : profile(profile) {
#if 0
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
#endif
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
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) setKernelPackT(true);

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

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  if ((int)solutionResident.size() >= 1 && cudaParam.siteSubset == solutionResident[0]->SiteSubset()) {
    // perhaps need to make this more robust in case solutionResident[0] is of the correct type for x
    x = solutionResident[0];
  } else {
    x = new cudaColorSpinorField(cudaParam); // solution
  }

  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
        (param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }

    *x = *h_x; // solution
  } else { // zero initial guess
    blas::zero(*x);
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

  if (param->inv_type_precondition == QUDA_MG_INVERTER && (!direct_solve || !mat_solution)) {
    errorQuda("Multigrid preconditioning only supported for direct solves");
  }

  if (param->use_resident_chrono && (direct_solve || norm_error_solve) ){
    errorQuda("Chronological forcasting only presently supported for M^dagger M solver");
  }

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
#if 0
    // chronological forecasting
    if (param->use_resident_chrono && chronoResident[param->chrono_index].size() > 0) {
      auto &basis = chronoResident[param->chrono_index];

      cudaColorSpinorField tmp(*in), tmp2(*in);

      for (unsigned int j=0; j<basis.size(); j++) m(*basis[j].second, *basis[j].first, tmp, tmp2);

      bool orthogonal = true;
      bool apply_mat = false;
      MinResExt mre(m, orthogonal, apply_mat, profileInvert);
      blas::copy(tmp, *in);

      mre(*out, tmp, basis);
    }
#endif
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

  if (param->make_resident_chrono) {
    int i = param->chrono_index;
    if (i >= QUDA_MAX_CHRONO)
      errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

    auto &basis = chronoResident[i];

    // if we have filled the space yet just augment
    if ((int)basis.size() < param->max_chrono_dim) {
      ColorSpinorParam cs_param(*x);
      basis.push_back(std::pair<ColorSpinorField*,ColorSpinorField*>(ColorSpinorField::Create(cs_param),ColorSpinorField::Create(cs_param)));
    }

    // shuffle every entry down one and bring the last to the front
    ColorSpinorField *tmp = basis[basis.size()-1].first;
    for (unsigned int j=basis.size()-1; j>0; j--) basis[j].first = basis[j-1].first;
    basis[0].first = tmp;
    *(basis[0]).first = *x; // set first entry to new solution
  }

  if (param->make_resident_solution) {
#if 0
    for (unsigned int i=0; i<solutionResident.size(); i++) {
      if (solutionResident[i]) delete solutionResident[i];
    }
#endif
    //if (!solutionResident.size()) solutionResident.resize(1);
    solutionResident[0] = static_cast<cudaColorSpinorField*>(x);
  }

  if (param->compute_action) {
    Complex action = blas::cDotProduct(*b, *x);
    param->action[0] = action.real();
    param->action[1] = action.imag();
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
  //if (!param->make_resident_solution) delete x;
  if (!solutionResident.size() || x != solutionResident[0]) {
    delete x;
  }

  delete d;
  delete dSloppy;
  delete dPre;

  profileInvert.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
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


//A.S. : If we want to use external improved staggered configurations , not just randomly generated, we can re-introduce all stuff for the routine below (files / parameters / structures etc.) 
// or maybe just to read pre-computed long links, or compute them in a simple c-code, see construct_fat_long_gauge_field in tests/test_util.cpp? (I did this long time ago)
#if 0
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

#endif
