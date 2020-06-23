#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

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
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <unitarization_links.h>
#include <algorithm>
#include <staggered_oprod.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>
#include <random_quda.h>
#include <mpi_comm_handle.h>

#include <multigrid.h>

#include <deflation.h>

#ifdef NUMA_NVML
#include <numa_affinity.h>
#endif

#ifdef QUDA_NVML
#include <nvml.h>
#endif

#include <cuda.h>

#include <ks_force_quda.h>

#ifdef GPU_GAUGE_FORCE
#include <gauge_force_quda.h>
#endif
#include <gauge_update_quda.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

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
#include <contract_quda.h>

#include <momentum.h>


#include <cuda_profiler_api.h>

using namespace quda;

static int R[4] = {0, 0, 0, 0};
// setting this to false prevents redundant halo exchange but isn't yet compatible with HISQ / ASQTAD kernels
static bool redundant_comms = false;

#include <blas_cublas.h>

//for MAGMA lib:
#include <blas_magma.h>

static bool InitMagma = false;

void openMagma() {

  if (!InitMagma) {
    OpenMagma();
    InitMagma = true;
  } else {
    printfQuda("\nMAGMA library was already initialized..\n");
  }

}

void closeMagma(){

  if (InitMagma) {
    CloseMagma();
    InitMagma = false;
  } else {
    printfQuda("\nMAGMA library was not initialized..\n");
  }

}

cudaGaugeField *gaugePrecise = nullptr;
cudaGaugeField *gaugeSloppy = nullptr;
cudaGaugeField *gaugePrecondition = nullptr;
cudaGaugeField *gaugeRefinement = nullptr;
cudaGaugeField *gaugeExtended = nullptr;

cudaGaugeField *gaugeFatPrecise = nullptr;
cudaGaugeField *gaugeFatSloppy = nullptr;
cudaGaugeField *gaugeFatPrecondition = nullptr;
cudaGaugeField *gaugeFatRefinement = nullptr;
cudaGaugeField *gaugeFatExtended = nullptr;

cudaGaugeField *gaugeLongExtended = nullptr;
cudaGaugeField *gaugeLongPrecise = nullptr;
cudaGaugeField *gaugeLongSloppy = nullptr;
cudaGaugeField *gaugeLongPrecondition = nullptr;
cudaGaugeField *gaugeLongRefinement = nullptr;

cudaGaugeField *gaugeSmeared = nullptr;

cudaCloverField *cloverPrecise = nullptr;
cudaCloverField *cloverSloppy = nullptr;
cudaCloverField *cloverPrecondition = nullptr;
cudaCloverField *cloverRefinement = nullptr;

cudaGaugeField *momResident = nullptr;
cudaGaugeField *extendedGaugeResident = nullptr;

std::vector<cudaColorSpinorField*> solutionResident;

// vector of spinors used for forecasting solutions in HMC
#define QUDA_MAX_CHRONO 12
// each entry is one p
std::vector< std::vector<ColorSpinorField*> > chronoResident(QUDA_MAX_CHRONO);

// Mapped memory buffer used to hold unitarization failures
static int *num_failures_h = nullptr;
static int *num_failures_d = nullptr;

cudaDeviceProp deviceProp;
qudaStream_t *streams;

static bool initialized = false;

//!< Profiler for initQuda
static TimeProfile profileInit("initQuda");

//!< Profile for loadGaugeQuda / saveGaugeQuda
static TimeProfile profileGauge("loadGaugeQuda");

//!< Profile for loadCloverQuda
static TimeProfile profileClover("loadCloverQuda");

//!< Profiler for dslashQuda
static TimeProfile profileDslash("dslashQuda");

//!< Profiler for invertQuda
static TimeProfile profileInvert("invertQuda");

//!< Profiler for invertMultiShiftQuda
static TimeProfile profileMulti("invertMultiShiftQuda");

//!< Profiler for eigensolveQuda
static TimeProfile profileEigensolve("eigensolveQuda");

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

//!<Profiler for computeStaggeredForceQuda
static TimeProfile profileStaggeredForce("computeStaggeredForceQuda");

//!<Profiler for computeHISQForceQuda
static TimeProfile profileHISQForce("computeHISQForceQuda");

//!<Profiler for plaqQuda
static TimeProfile profilePlaq("plaqQuda");

//!< Profiler for wuppertalQuda
static TimeProfile profileWuppertal("wuppertalQuda");

//!<Profiler for gaussQuda
static TimeProfile profileGauss("gaussQuda");

//!< Profiler for gaugeObservableQuda
static TimeProfile profileGaugeObs("gaugeObservablesQuda");

//!< Profiler for APEQuda
static TimeProfile profileAPE("APEQuda");

//!< Profiler for STOUTQuda
static TimeProfile profileSTOUT("STOUTQuda");

//!< Profiler for OvrImpSTOUTQuda
static TimeProfile profileOvrImpSTOUT("OvrImpSTOUTQuda");

//!< Profiler for wFlowQuda
static TimeProfile profileWFlow("wFlowQuda");

//!< Profiler for projectSU3Quda
static TimeProfile profileProject("projectSU3Quda");

//!< Profiler for staggeredPhaseQuda
static TimeProfile profilePhase("staggeredPhaseQuda");

//!< Profiler for contractions
static TimeProfile profileContract("contractQuda");

//!< Profiler for covariant derivative
static TimeProfile profileCovDev("covDevQuda");

//!< Profiler for momentum action
static TimeProfile profileMomAction("momActionQuda");

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");

//!< Profiler for GaugeFixing
static TimeProfile GaugeFixFFTQuda("GaugeFixFFTQuda");
static TimeProfile GaugeFixOVRQuda("GaugeFixOVRQuda");

//!< Profiler for toal time spend between init and end
static TimeProfile profileInit2End("initQuda-endQuda",false);

static bool enable_profiler = false;
static bool do_not_profile_quda = false;

static void profilerStart(const char *f) {



  static std::vector<int> target_list;
  static bool enable = false;
  static bool init = false;
  if (!init) {
    char *profile_target_env = getenv("QUDA_ENABLE_TARGET_PROFILE"); // selectively enable profiling for a given solve

    if ( profile_target_env ) {
      std::stringstream target_stream(profile_target_env);

      int target;
      while(target_stream >> target) {
       target_list.push_back(target);
       if (target_stream.peek() == ',') target_stream.ignore();
     }

     if (target_list.size() > 0) {
       std::sort(target_list.begin(), target_list.end());
       target_list.erase( unique( target_list.begin(), target_list.end() ), target_list.end() );
       warningQuda("Targeted profiling enabled for %lu functions\n", target_list.size());
       enable = true;
     }
   }


    char* donotprofile_env = getenv("QUDA_DO_NOT_PROFILE"); // disable profiling of QUDA parts
    if (donotprofile_env && (!(strcmp(donotprofile_env, "0") == 0)))  {
      do_not_profile_quda=true;
      printfQuda("Disabling profiling in QUDA\n");
    }
    init = true;
  }

  static int target_count = 0;
  static unsigned int i = 0;
  if (do_not_profile_quda){
    cudaProfilerStop();
    printfQuda("Stopping profiling in QUDA\n");
  } else {
    if (enable) {
      if (i < target_list.size() && target_count++ == target_list[i]) {
        enable_profiler = true;
        printfQuda("Starting profiling for %s\n", f);
        cudaProfilerStart();
      i++; // advance to next target
    }
  }
}
}

static void profilerStop(const char *f) {
  if(do_not_profile_quda){
    cudaProfilerStart();
  } else {

    if (enable_profiler) {
      printfQuda("Stopping profiling for %s\n", f);
      cudaProfilerStop();
      enable_profiler = false;
    }
  }
}


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
  auto *md = static_cast<LexMapData *>(fdata);

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

// Provision for user control over MPI comm handle
// Assumes an MPI implementation of QMP

#if defined(QMP_COMMS) || defined(MPI_COMMS)
MPI_Comm MPI_COMM_HANDLE;
static int user_set_comm_handle = 0;
#endif

void setMPICommHandleQuda(void *mycomm)
{
#if defined(QMP_COMMS) || defined(MPI_COMMS)
  MPI_COMM_HANDLE = *((MPI_Comm *)mycomm);
  user_set_comm_handle = 1;
#endif
}

#ifdef QMP_COMMS
static void initQMPComms(void)
{
  // Default comm handle is taken from QMP
  // WARNING: Assumes an MPI implementation of QMP
  if (!user_set_comm_handle) {
    void *mycomm;
    QMP_get_mpi_comm(QMP_comm_get_default(), &mycomm);
    setMPICommHandleQuda(mycomm);
  }
}
#elif defined(MPI_COMMS)
static void initMPIComms(void)
{
  // Default comm handle is MPI_COMM_WORLD
  if (!user_set_comm_handle) {
    static MPI_Comm mycomm;
    MPI_Comm_dup(MPI_COMM_WORLD, &mycomm);
    setMPICommHandleQuda((void *)&mycomm);
  }
}
#endif

static bool comms_initialized = false;

void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)
{
  if (comms_initialized) return;

#if QMP_COMMS
  initQMPComms();
#elif defined(MPI_COMMS)
  initMPIComms();
#endif

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
      fdata = nullptr;
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
    initCommsGridQuda(ndim, dims, nullptr, nullptr);
  } else {
    errorQuda("initQuda() called without prior call to initCommsGridQuda(),"
        " and QMP logical topology has not been declared");
  }
#elif defined(MPI_COMMS)
  errorQuda("When using MPI for communications, initCommsGridQuda() must be called before initQuda()");
#else // single-GPU
  const int dims[4] = {1, 1, 1, 1};
  initCommsGridQuda(4, dims, nullptr, nullptr);
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
  profileInit.TPSTART(QUDA_PROFILE_INIT);

  if (getVerbosity() >= QUDA_SUMMARIZE) {
#ifdef GITVERSION
    printfQuda("QUDA %s (git %s)\n",quda_version.c_str(),gitversion);
#else
    printfQuda("QUDA %s\n",quda_version.c_str());
#endif
  }

  int driver_version;
  cudaDriverGetVersion(&driver_version);
  printfQuda("CUDA Driver version = %d\n", driver_version);

  int runtime_version;
  cudaRuntimeGetVersion(&runtime_version);
  printfQuda("CUDA Runtime version = %d\n", runtime_version);

#ifdef QUDA_NVML
  nvmlReturn_t result = nvmlInit();
  if (NVML_SUCCESS != result) errorQuda("NVML Init failed with error %d", result);
  const int length = 80;
  char graphics_version[length];
  result = nvmlSystemGetDriverVersion(graphics_version, length);
  if (NVML_SUCCESS != result) errorQuda("nvmlSystemGetDriverVersion failed with error %d", result);
  printfQuda("Graphic driver version = %s\n", graphics_version);
  result = nvmlShutdown();
  if (NVML_SUCCESS != result) errorQuda("NVML Shutdown failed with error %d", result);
#endif

#if defined(MULTI_GPU) && (CUDA_VERSION == 4000)
  //check if CUDA_NIC_INTEROP is set to 1 in the enviroment
  // not needed for CUDA >= 4.1
  char* cni_str = getenv("CUDA_NIC_INTEROP");
  if(cni_str == nullptr){
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


// Check GPU and QUDA build compatibiliy
// 4 cases:
// a) QUDA and GPU match: great
// b) QUDA built for higher compute capability: error
// c) QUDA built for lower major compute capability: warn if QUDA_ALLOW_JIT, else error
// d) QUDA built for same major compute capability but lower minor: warn

  const int my_major = __COMPUTE_CAPABILITY__ / 100;
  const int my_minor = (__COMPUTE_CAPABILITY__  - my_major * 100) / 10;
// b) UDA was compiled for a higher compute capability
  if (deviceProp.major * 100 + deviceProp.minor * 10 < __COMPUTE_CAPABILITY__)
    errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. ** \n --- Please set the correct QUDA_GPU_ARCH when running cmake.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);


// c) QUDA was compiled for a lower compute capability
  if (deviceProp.major < my_major) {
    char *allow_jit_env = getenv("QUDA_ALLOW_JIT");
    if (allow_jit_env && strcmp(allow_jit_env, "1") == 0) {
      if (getVerbosity() > QUDA_SILENT) warningQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- Jitting the PTX since QUDA_ALLOW_JIT=1 was set. Note that this will take some time.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);
    } else {
      errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n --- Please set the correct QUDA_GPU_ARCH when running cmake.\n If you want the PTX to be jitted for your current GPU arch please set the enviroment variable QUDA_ALLOW_JIT=1.", deviceProp.major, deviceProp.minor, my_major, my_minor);
    }
  }
// d) QUDA built for same major compute capability but lower minor
  if (deviceProp.major == my_major and deviceProp.minor > my_minor) {
    warningQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- This might result in a lower performance. Please consider adjusting QUDA_GPU_ARCH when running cmake.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);
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
  // cudaGetDeviceProperties(&deviceProp, dev);

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

  profileInit.TPSTOP(QUDA_PROFILE_INIT);
  profileInit.TPSTOP(QUDA_PROFILE_TOTAL);
}

/*
 * Any persistent memory allocations that QUDA uses are done here.
 */
void initQudaMemory()
{
  profileInit.TPSTART(QUDA_PROFILE_TOTAL);
  profileInit.TPSTART(QUDA_PROFILE_INIT);

  if (!comms_initialized) init_default_comms();

  streams = new qudaStream_t[Nstream];

  int greatestPriority;
  int leastPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  for (int i=0; i<Nstream-1; i++) {
    cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, greatestPriority);
  }
  cudaStreamCreateWithPriority(&streams[Nstream-1], cudaStreamDefault, leastPriority);

  checkCudaError();
  createDslashEvents();
  blas::init();
  cublas::init();

  // initalize the memory pool allocators
  pool::init();

  num_failures_h = static_cast<int*>(mapped_malloc(sizeof(int)));
  cudaHostGetDevicePointer(&num_failures_d, num_failures_h, 0);

  loadTuneCache();

  for (int d=0; d<4; d++) R[d] = 2 * (redundant_comms || commDimPartitioned(d));

  profileInit.TPSTOP(QUDA_PROFILE_INIT);
  profileInit.TPSTOP(QUDA_PROFILE_TOTAL);
}

void updateR()
{
  for (int d=0; d<4; d++) R[d] = 2 * (redundant_comms || commDimPartitioned(d));
}

void initQuda(int dev)
{
  // initialize communications topology, if not already done explicitly via initCommsGridQuda()
  if (!comms_initialized) init_default_comms();

  // set the device that QUDA uses
  initQudaDevice(dev);

  // set the persistant memory allocations that QUDA uses (Blas, streams, etc.)
  initQudaMemory();
}

// This is a flag used to signal when we have downloaded new gauge
// field.  Set by loadGaugeQuda and consumed by loadCloverQuda as one
// possible flag to indicate we need to recompute the clover field
static bool invalidate_clover = true;

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  checkGaugeParam(param);

  profileGauge.TPSTART(QUDA_PROFILE_INIT);
  // Set the specific input parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);

  if (gauge_param.order <= 4) gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  GaugeField *in = (param->location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<GaugeField*>(new cpuGaugeField(gauge_param)) :
    static_cast<GaugeField*>(new cudaGaugeField(gauge_param));

  if (in->Order() == QUDA_BQCD_GAUGE_ORDER) {
    static size_t checksum = SIZE_MAX;
    size_t in_checksum = in->checksum(true);
    if (in_checksum == checksum) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Gauge field unchanged - using cached gauge field %lu\n", checksum);
      profileGauge.TPSTOP(QUDA_PROFILE_INIT);
      profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
      delete in;
      invalidate_clover = false;
      return;
    }
    checksum = in_checksum;
    invalidate_clover = true;
  }

  // free any current gauge field before new allocations to reduce memory overhead
  switch (param->type) {
    case QUDA_WILSON_LINKS:
      if (gaugeRefinement != gaugeSloppy && gaugeRefinement) delete gaugeRefinement;
      if (gaugeSloppy != gaugePrecondition && gaugePrecise != gaugePrecondition && gaugePrecondition)
        delete gaugePrecondition;
      if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
      if (gaugePrecise && !param->use_resident_gauge) delete gaugePrecise;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      if (gaugeFatRefinement != gaugeFatSloppy && gaugeFatRefinement) delete gaugeFatRefinement;
      if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecise != gaugeFatPrecondition && gaugeFatPrecondition)
        delete gaugeFatPrecondition;
      if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
      if (gaugeFatPrecise && !param->use_resident_gauge) delete gaugeFatPrecise;
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      if (gaugeLongRefinement != gaugeLongSloppy && gaugeLongRefinement) delete gaugeLongRefinement;
      if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecise != gaugeLongPrecondition && gaugeLongPrecondition)
        delete gaugeLongPrecondition;
      if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
      if (gaugeLongPrecise) delete gaugeLongPrecise;
      break;
    case QUDA_SMEARED_LINKS:
      if (gaugeSmeared) delete gaugeSmeared;
      break;
    default:
      errorQuda("Invalid gauge type %d", param->type);
  }

  // if not preserving then copy the gauge field passed in
  cudaGaugeField *precise = nullptr;

  // switch the parameters for creating the mirror precise cuda gauge field
  gauge_param.create = QUDA_NULL_FIELD_CREATE;
  gauge_param.reconstruct = param->reconstruct;
  gauge_param.setPrecision(param->cuda_prec, true);
  gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  gauge_param.pad = param->ga_pad;

  precise = new cudaGaugeField(gauge_param);

  if (param->use_resident_gauge) {
    if(gaugePrecise == nullptr) errorQuda("No resident gauge field");
    // copy rather than point at to ensure that the padded region is filled in
    precise->copy(*gaugePrecise);
    precise->exchangeGhost();
    delete gaugePrecise;
    gaugePrecise = nullptr;
    profileGauge.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    profileGauge.TPSTOP(QUDA_PROFILE_INIT);
    profileGauge.TPSTART(QUDA_PROFILE_H2D);
    precise->copy(*in);
    profileGauge.TPSTOP(QUDA_PROFILE_H2D);
  }

  // for gaugeSmeared we are interested only in the precise version
  if (param->type == QUDA_SMEARED_LINKS) {
    gaugeSmeared = createExtendedGauge(*precise, R, profileGauge);

    profileGauge.TPSTART(QUDA_PROFILE_FREE);
    delete precise;
    delete in;
    profileGauge.TPSTOP(QUDA_PROFILE_FREE);

    profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
    return;
  }

  // creating sloppy fields isn't really compute, but it is work done on the gpu
  profileGauge.TPSTART(QUDA_PROFILE_COMPUTE);

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.setPrecision(param->cuda_prec_sloppy, true);
  cudaGaugeField *sloppy = nullptr;
  if (param->cuda_prec == param->cuda_prec_sloppy && param->reconstruct == param->reconstruct_sloppy) {
    sloppy = precise;
  } else {
    sloppy = new cudaGaugeField(gauge_param);
    sloppy->copy(*precise);
  }

  // switch the parameters for creating the mirror preconditioner cuda gauge field
  gauge_param.reconstruct = param->reconstruct_precondition;
  gauge_param.setPrecision(param->cuda_prec_precondition, true);
  cudaGaugeField *precondition = nullptr;
  if (param->cuda_prec == param->cuda_prec_precondition && param->reconstruct == param->reconstruct_precondition) {
    precondition = precise;
  } else if (param->cuda_prec_sloppy == param->cuda_prec_precondition
             && param->reconstruct_sloppy == param->reconstruct_precondition) {
    precondition = sloppy;
  } else {
    precondition = new cudaGaugeField(gauge_param);
    precondition->copy(*precise);
  }

  // switch the parameters for creating the refinement cuda gauge field
  gauge_param.reconstruct = param->reconstruct_refinement_sloppy;
  gauge_param.setPrecision(param->cuda_prec_refinement_sloppy, true);
  cudaGaugeField *refinement = nullptr;
  if (param->cuda_prec_sloppy == param->cuda_prec_refinement_sloppy
      && param->reconstruct_sloppy == param->reconstruct_refinement_sloppy) {
    refinement = sloppy;
  } else {
    refinement = new cudaGaugeField(gauge_param);
    refinement->copy(*sloppy);
  }

  profileGauge.TPSTOP(QUDA_PROFILE_COMPUTE);

  // create an extended preconditioning field
  cudaGaugeField* extended = nullptr;
  if (param->overlap){
    int R[4]; // domain-overlap widths in different directions
    for (int i=0; i<4; ++i) R[i] = param->overlap*commDimPartitioned(i);
    extended = createExtendedGauge(*precondition, R, profileGauge);
  }

  switch (param->type) {
    case QUDA_WILSON_LINKS:
      gaugePrecise = precise;
      gaugeSloppy = sloppy;
      gaugePrecondition = precondition;
      gaugeRefinement = refinement;

      if(param->overlap) gaugeExtended = extended;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      gaugeFatPrecise = precise;
      gaugeFatSloppy = sloppy;
      gaugeFatPrecondition = precondition;
      gaugeFatRefinement = refinement;

      if(param->overlap){
        if(gaugeFatExtended) errorQuda("Extended gauge fat field already allocated");
	gaugeFatExtended = extended;
      }
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      gaugeLongPrecise = precise;
      gaugeLongSloppy = sloppy;
      gaugeLongPrecondition = precondition;
      gaugeLongRefinement = refinement;

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

  if (extendedGaugeResident) {
    // updated the resident gauge field if needed
    const int *R_ = extendedGaugeResident->R();
    const int R[] = { R_[0], R_[1], R_[2], R_[3] };
    QudaReconstructType recon = extendedGaugeResident->Reconstruct();
    delete extendedGaugeResident;

    extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGauge, false, recon);
  }

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
  cudaGaugeField *cudaGauge = nullptr;
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
    case QUDA_SMEARED_LINKS:
      gauge_param.create = QUDA_NULL_FIELD_CREATE;
      gauge_param.reconstruct = param->reconstruct;
      gauge_param.setPrecision(param->cuda_prec, true);
      gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      gauge_param.pad = param->ga_pad;
      cudaGauge = new cudaGaugeField(gauge_param);
      copyExtendedGauge(*cudaGauge, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
      break;
    default:
      errorQuda("Invalid gauge type");
  }

  profileGauge.TPSTART(QUDA_PROFILE_D2H);
  cudaGauge->saveCPUField(cpuGauge);
  profileGauge.TPSTOP(QUDA_PROFILE_D2H);

  if (param->type == QUDA_SMEARED_LINKS) { delete cudaGauge; }

  profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
}

void loadSloppyCloverQuda(const QudaPrecision prec[]);
void freeSloppyCloverQuda();

void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  profileClover.TPSTART(QUDA_PROFILE_INIT);

  checkCloverParam(inv_param);
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
  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded before clover");
  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH)
      && (inv_param->dslash_type != QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH)) {
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

  CloverFieldParam clover_param;
  clover_param.nDim = 4;
  clover_param.csw = inv_param->clover_coeff;
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
  clover_param.inverse = (h_clovinv || pc_solve) && !dynamic_clover_inverse() ? true : false;
  CloverField *in = nullptr;
  profileClover.TPSTOP(QUDA_PROFILE_INIT);

  // FIXME do we need to make this more robust to changing other meta data (compare cloverPrecise against clover_param)
  bool clover_update = false;
  double csw_old = cloverPrecise ? cloverPrecise->Csw() : 0.0;
  if (!cloverPrecise || invalidate_clover || inv_param->clover_coeff != csw_old) clover_update = true;

  // compute or download clover field only if gauge field has been updated or clover field doesn't exist
  if (clover_update) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating new clover field\n");
    freeSloppyCloverQuda();
    if (cloverPrecise) delete cloverPrecise;

    profileClover.TPSTART(QUDA_PROFILE_INIT);
    cloverPrecise = new cudaCloverField(clover_param);

    if (!device_calc || inv_param->return_clover || inv_param->return_clover_inverse) {
      // create a param for the cpu clover field
      CloverFieldParam inParam(clover_param);
      inParam.setPrecision(inv_param->clover_cpu_prec);
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
      bool inverse = (h_clovinv && !inv_param->compute_clover_inverse && !dynamic_clover_inverse());
      cloverPrecise->copy(*in, inverse);
      profileClover.TPSTOP(QUDA_PROFILE_H2D);
    } else {
      profileClover.TPSTOP(QUDA_PROFILE_TOTAL);
      createCloverQuda(inv_param);
      profileClover.TPSTART(QUDA_PROFILE_TOTAL);
    }

    // inverted clover term is required when applying preconditioned operator
    if ((!h_clovinv || inv_param->compute_clover_inverse) && pc_solve) {
      profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
      if (!dynamic_clover_inverse()) {
	cloverInvert(*cloverPrecise, inv_param->compute_clover_trlog);
	if (inv_param->compute_clover_trlog) {
	  inv_param->trlogA[0] = cloverPrecise->TrLog()[0];
	  inv_param->trlogA[1] = cloverPrecise->TrLog()[1];
	}
      }
      profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);
    }
  } else {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Gauge field unchanged - using cached clover field\n");
  }

  clover_param.direct = true;
  clover_param.inverse = dynamic_clover_inverse() ? false : true;

  cloverPrecise->setRho(inv_param->clover_rho);

  QudaPrecision prec[] = {inv_param->clover_cuda_prec_sloppy, inv_param->clover_cuda_prec_precondition,
                          inv_param->clover_cuda_prec_refinement_sloppy};
  loadSloppyCloverQuda(prec);

  // if requested, copy back the clover / inverse field
  if (inv_param->return_clover || inv_param->return_clover_inverse) {
    if (!h_clover && !h_clovinv) errorQuda("Requested clover field return but no clover host pointers set");

    // copy the inverted clover term into host application order on the device
    clover_param.setPrecision(inv_param->clover_cpu_prec);
    clover_param.direct = (h_clover && inv_param->return_clover);
    clover_param.inverse = (h_clovinv && inv_param->return_clover_inverse);

    // this isn't really "epilogue" but this label suffices
    profileClover.TPSTART(QUDA_PROFILE_EPILOGUE);
    cudaCloverField *hack = nullptr;
    if (!dynamic_clover_inverse()) {
      clover_param.order = inv_param->clover_order;
      hack = new cudaCloverField(clover_param);
      hack->copy(*cloverPrecise); // FIXME this can lead to an redundant copies if we're not copying back direct + inverse
    } else {
      auto *hackOfTheHack = new cudaCloverField(clover_param);	// Hack of the hack
      hackOfTheHack->copy(*cloverPrecise, false);
      cloverInvert(*hackOfTheHack, inv_param->compute_clover_trlog);
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

void freeSloppyCloverQuda();

void loadSloppyCloverQuda(const QudaPrecision *prec)
{
  freeSloppyCloverQuda();

  if (cloverPrecise) {
    // create the mirror sloppy clover field
    CloverFieldParam clover_param(*cloverPrecise);

    clover_param.setPrecision(prec[0]);

    if (cloverPrecise->V(false) != cloverPrecise->V(true)) {
      clover_param.direct = true;
      clover_param.inverse = true;
    } else {
      clover_param.direct = false;
      clover_param.inverse = true;
    }

    if (clover_param.Precision() != cloverPrecise->Precision()) {
      cloverSloppy = new cudaCloverField(clover_param);
      cloverSloppy->copy(*cloverPrecise, clover_param.inverse);
    } else {
      cloverSloppy = cloverPrecise;
    }

    // switch the parameters for creating the mirror preconditioner clover field
    clover_param.setPrecision(prec[1]);

    // create the mirror preconditioner clover field
    if (clover_param.Precision() == cloverPrecise->Precision()) {
      cloverPrecondition = cloverPrecise;
    } else if (clover_param.Precision() == cloverSloppy->Precision()) {
      cloverPrecondition = cloverSloppy;
    } else {
      cloverPrecondition = new cudaCloverField(clover_param);
      cloverPrecondition->copy(*cloverPrecise, clover_param.inverse);
    }

    // switch the parameters for creating the mirror preconditioner clover field
    clover_param.setPrecision(prec[2]);

    // create the mirror preconditioner clover field
    if (clover_param.Precision() != cloverSloppy->Precision()) {
      cloverRefinement = new cudaCloverField(clover_param);
      cloverRefinement->copy(*cloverSloppy, clover_param.inverse);
    } else {
      cloverRefinement = cloverSloppy;
    }
  }

}

// just free the sloppy fields used in mixed-precision solvers
void freeSloppyGaugeQuda()
{
  if (!initialized) errorQuda("QUDA not initialized");

  if (gaugeSloppy != gaugeRefinement && gaugeRefinement) delete gaugeRefinement;
  if (gaugeSloppy != gaugePrecondition && gaugePrecise != gaugePrecondition && gaugePrecondition)
    delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;

  gaugeRefinement = nullptr;
  gaugePrecondition = nullptr;
  gaugeSloppy = nullptr;

  if (gaugeLongSloppy != gaugeLongRefinement && gaugeLongRefinement) delete gaugeLongRefinement;
  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecise != gaugeLongPrecondition && gaugeLongPrecondition)
    delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;

  gaugeLongRefinement = nullptr;
  gaugeLongPrecondition = nullptr;
  gaugeLongSloppy = nullptr;

  if (gaugeFatSloppy != gaugeFatRefinement && gaugeFatRefinement) delete gaugeFatRefinement;
  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecise != gaugeFatPrecondition && gaugeFatPrecondition)
    delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;

  gaugeFatRefinement = nullptr;
  gaugeFatPrecondition = nullptr;
  gaugeFatSloppy = nullptr;
}

void freeGaugeQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");

  freeSloppyGaugeQuda();

  if (gaugePrecise) delete gaugePrecise;
  if (gaugeExtended) delete gaugeExtended;

  gaugePrecise = nullptr;
  gaugeExtended = nullptr;

  if (gaugeLongPrecise) delete gaugeLongPrecise;
  if (gaugeLongExtended) delete gaugeLongExtended;

  gaugeLongPrecise = nullptr;
  gaugeLongExtended = nullptr;

  if (gaugeFatPrecise) delete gaugeFatPrecise;

  gaugeFatPrecise = nullptr;
  gaugeFatExtended = nullptr;

  if (gaugeSmeared) delete gaugeSmeared;

  gaugeSmeared = nullptr;
  // Need to merge extendedGaugeResident and gaugeFatPrecise/gaugePrecise
  if (extendedGaugeResident) {
    delete extendedGaugeResident;
    extendedGaugeResident = nullptr;
  }
}

void loadSloppyGaugeQuda(const QudaPrecision *prec, const QudaReconstructType *recon)
{
  // first do SU3 links (if they exist)
  if (gaugePrecise) {
    GaugeFieldParam gauge_param(*gaugePrecise);
    // switch the parameters for creating the mirror sloppy cuda gauge field

    gauge_param.reconstruct = recon[0];
    gauge_param.setPrecision(prec[0], true);

    if (gaugeSloppy) errorQuda("gaugeSloppy already exists");

    if (gauge_param.Precision() == gaugePrecise->Precision() && gauge_param.reconstruct == gaugePrecise->Reconstruct()) {
      gaugeSloppy = gaugePrecise;
    } else {
      gaugeSloppy = new cudaGaugeField(gauge_param);
      gaugeSloppy->copy(*gaugePrecise);
    }

    // switch the parameters for creating the mirror preconditioner cuda gauge field
    gauge_param.reconstruct = recon[1];
    gauge_param.setPrecision(prec[1], true);

    if (gaugePrecondition) errorQuda("gaugePrecondition already exists");

    if (gauge_param.Precision() == gaugePrecise->Precision() && gauge_param.reconstruct == gaugePrecise->Reconstruct()) {
      gaugePrecondition = gaugePrecise;
    } else if (gauge_param.Precision() == gaugeSloppy->Precision()
               && gauge_param.reconstruct == gaugeSloppy->Reconstruct()) {
      gaugePrecondition = gaugeSloppy;
    } else {
      gaugePrecondition = new cudaGaugeField(gauge_param);
      gaugePrecondition->copy(*gaugePrecise);
    }

    // switch the parameters for creating the mirror refinement cuda gauge field
    gauge_param.reconstruct = recon[2];
    gauge_param.setPrecision(prec[2], true);

    if (gaugeRefinement) errorQuda("gaugeRefinement already exists");

    if (gauge_param.Precision() == gaugeSloppy->Precision() && gauge_param.reconstruct == gaugeSloppy->Reconstruct()) {
      gaugeRefinement = gaugeSloppy;
    } else {
      gaugeRefinement = new cudaGaugeField(gauge_param);
      gaugeRefinement->copy(*gaugeSloppy);
    }
  }

  // fat links (if they exist)
  if (gaugeFatPrecise) {
    GaugeFieldParam gauge_param(*gaugeFatPrecise);

    gauge_param.setPrecision(prec[0], true);

    if (gaugeFatSloppy) errorQuda("gaugeFatSloppy already exists");

    if (gauge_param.Precision() == gaugeFatPrecise->Precision()
        && gauge_param.reconstruct == gaugeFatPrecise->Reconstruct()) {
      gaugeFatSloppy = gaugeFatPrecise;
    } else {
      gaugeFatSloppy = new cudaGaugeField(gauge_param);
      gaugeFatSloppy->copy(*gaugeFatPrecise);
    }

    // switch the parameters for creating the mirror preconditioner cuda gauge field
    gauge_param.setPrecision(prec[1], true);

    if (gaugeFatPrecondition) errorQuda("gaugeFatPrecondition already exists\n");

    if (gauge_param.Precision() == gaugeFatPrecise->Precision()
        && gauge_param.reconstruct == gaugeFatPrecise->Reconstruct()) {
      gaugeFatPrecondition = gaugeFatPrecise;
    } else if (gauge_param.Precision() == gaugeFatSloppy->Precision()
               && gauge_param.reconstruct == gaugeFatSloppy->Reconstruct()) {
      gaugeFatPrecondition = gaugeFatSloppy;
    } else {
      gaugeFatPrecondition = new cudaGaugeField(gauge_param);
      gaugeFatPrecondition->copy(*gaugeFatPrecise);
    }

    // switch the parameters for creating the mirror refinement cuda gauge field
    gauge_param.setPrecision(prec[2], true);

    if (gaugeFatRefinement) errorQuda("gaugeFatRefinement already exists\n");

    if (gauge_param.Precision() == gaugeFatSloppy->Precision()
        && gauge_param.reconstruct == gaugeFatSloppy->Reconstruct()) {
      gaugeFatRefinement = gaugeFatSloppy;
    } else {
      gaugeFatRefinement = new cudaGaugeField(gauge_param);
      gaugeFatRefinement->copy(*gaugeFatSloppy);
    }
  }

  // long links (if they exist)
  if (gaugeLongPrecise) {
    GaugeFieldParam gauge_param(*gaugeLongPrecise);

    gauge_param.reconstruct = recon[0];
    gauge_param.setPrecision(prec[0], true);

    if (gaugeLongSloppy) errorQuda("gaugeLongSloppy already exists");

    if (gauge_param.Precision() == gaugeLongPrecise->Precision()
        && gauge_param.reconstruct == gaugeLongPrecise->Reconstruct()) {
      gaugeLongSloppy = gaugeLongPrecise;
    } else {
      gaugeLongSloppy = new cudaGaugeField(gauge_param);
      gaugeLongSloppy->copy(*gaugeLongPrecise);
    }

    // switch the parameters for creating the mirror preconditioner cuda gauge field
    gauge_param.reconstruct = recon[1];
    gauge_param.setPrecision(prec[1], true);

    if (gaugeLongPrecondition) errorQuda("gaugeLongPrecondition already exists\n");

    if (gauge_param.Precision() == gaugeLongPrecise->Precision()
        && gauge_param.reconstruct == gaugeLongPrecise->Reconstruct()) {
      gaugeLongPrecondition = gaugeLongPrecise;
    } else if (gauge_param.Precision() == gaugeLongSloppy->Precision()
               && gauge_param.reconstruct == gaugeLongSloppy->Reconstruct()) {
      gaugeLongPrecondition = gaugeLongSloppy;
    } else {
      gaugeLongPrecondition = new cudaGaugeField(gauge_param);
      gaugeLongPrecondition->copy(*gaugeLongPrecise);
    }

    // switch the parameters for creating the mirror refinement cuda gauge field
    gauge_param.reconstruct = recon[2];
    gauge_param.setPrecision(prec[2], true);

    if (gaugeLongRefinement) errorQuda("gaugeLongRefinement already exists\n");

    if (gauge_param.Precision() == gaugeLongSloppy->Precision()
        && gauge_param.reconstruct == gaugeLongSloppy->Reconstruct()) {
      gaugeLongRefinement = gaugeLongSloppy;
    } else {
      gaugeLongRefinement = new cudaGaugeField(gauge_param);
      gaugeLongRefinement->copy(*gaugeLongSloppy);
    }
  }
}

void freeSloppyCloverQuda()
{
  if (!initialized) errorQuda("QUDA not initialized");
  if (cloverRefinement != cloverSloppy && cloverRefinement) delete cloverRefinement;
  if (cloverPrecondition != cloverSloppy && cloverPrecondition != cloverPrecise && cloverPrecondition)
    delete cloverPrecondition;
  if (cloverSloppy != cloverPrecise && cloverSloppy) delete cloverSloppy;

  cloverRefinement = nullptr;
  cloverPrecondition = nullptr;
  cloverSloppy = nullptr;
}

void freeCloverQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");
  freeSloppyCloverQuda();
  if (cloverPrecise) delete cloverPrecise;
  cloverPrecise = nullptr;
}

void flushChronoQuda(int i)
{
  if (i >= QUDA_MAX_CHRONO)
    errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

  auto &basis = chronoResident[i];

  for (auto v : basis) {
    if (v)  delete v;
  }
  basis.clear();
}

void endQuda(void)
{
  profileEnd.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) return;

  freeGaugeQuda();
  freeCloverQuda();

  for (int i=0; i<QUDA_MAX_CHRONO; i++) flushChronoQuda(i);

  for (auto v : solutionResident) if (v) delete v;
  solutionResident.clear();

  if(momResident) delete momResident;

  LatticeField::freeGhostBuffer();
  cpuColorSpinorField::freeGhostBuffer();

  cublas::destroy();
  blas::end();

  pool::flush_pinned();
  pool::flush_device();

  host_free(num_failures_h);
  num_failures_h = nullptr;
  num_failures_d = nullptr;

  if (streams) {
    for (int i=0; i<Nstream; i++) cudaStreamDestroy(streams[i]);
    delete []streams;
    streams = nullptr;
  }
  destroyDslashEvents();

  saveTuneCache();
  saveProfile();

  // flush any outstanding force monitoring (if enabled)
  flushForceMonitor();

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
    profileDslash.Print();
    profileInvert.Print();
    profileMulti.Print();
    profileEigensolve.Print();
    profileFatLink.Print();
    profileGaugeForce.Print();
    profileGaugeUpdate.Print();
    profileExtendedGauge.Print();
    profileCloverForce.Print();
    profileStaggeredForce.Print();
    profileHISQForce.Print();
    profileContract.Print();
    profileCovDev.Print();
    profilePlaq.Print();
    profileGaugeObs.Print();
    profileAPE.Print();
    profileSTOUT.Print();
    profileOvrImpSTOUT.Print();
    profileWFlow.Print();
    profileProject.Print();
    profilePhase.Print();
    profileMomAction.Print();
    profileEnd.Print();

    profileInit2End.Print();
    TimeProfile::PrintGlobal();

    printLaunchTimer();
    printAPIProfile();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }

  assertAllMemFree();

  char *device_reset_env = getenv("QUDA_DEVICE_RESET");
  if (device_reset_env && strcmp(device_reset_env,"1") == 0) {
    // end this CUDA context
    cudaDeviceReset();
  }

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
    case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH:
      diracParam.type = pc ? QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC : QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC;
      break;
    case QUDA_DOMAIN_WALL_DSLASH:
      diracParam.type = pc ? QUDA_DOMAIN_WALLPC_DIRAC : QUDA_DOMAIN_WALL_DIRAC;
      diracParam.Ls = inv_param->Ls;
      break;
    case QUDA_DOMAIN_WALL_4D_DSLASH:
      diracParam.type = pc ? QUDA_DOMAIN_WALL_4DPC_DIRAC : QUDA_DOMAIN_WALL_4D_DIRAC;
      diracParam.Ls = inv_param->Ls;
      break;
    case QUDA_MOBIUS_DWF_EOFA_DSLASH:
      if (inv_param->Ls > QUDA_MAX_DWF_LS) {
        errorQuda("Length of Ls dimension %d greater than QUDA_MAX_DWF_LS %d", inv_param->Ls, QUDA_MAX_DWF_LS);
      }
      diracParam.type = pc ? QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC : QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC;
      diracParam.Ls = inv_param->Ls;
      if (sizeof(Complex) != sizeof(double _Complex)) {
        errorQuda("Irreconcilable difference between interface and internal complex number conventions");
      }
      memcpy(diracParam.b_5, inv_param->b_5, sizeof(Complex) * inv_param->Ls);
      memcpy(diracParam.c_5, inv_param->c_5, sizeof(Complex) * inv_param->Ls);
      diracParam.eofa_shift = inv_param->eofa_shift;
      diracParam.eofa_pm = inv_param->eofa_pm;
      diracParam.mq1 = inv_param->mq1;
      diracParam.mq2 = inv_param->mq2;
      diracParam.mq3 = inv_param->mq3;
      break;
    case QUDA_MOBIUS_DWF_DSLASH:
      if (inv_param->Ls > QUDA_MAX_DWF_LS)
	errorQuda("Length of Ls dimension %d greater than QUDA_MAX_DWF_LS %d", inv_param->Ls, QUDA_MAX_DWF_LS);
      diracParam.type = pc ? QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC : QUDA_MOBIUS_DOMAIN_WALL_DIRAC;
      diracParam.Ls = inv_param->Ls;
      if (sizeof(Complex) != sizeof(double _Complex)) {
        errorQuda("Irreconcilable difference between interface and internal complex number conventions");
      }
      memcpy(diracParam.b_5, inv_param->b_5, sizeof(Complex) * inv_param->Ls);
      memcpy(diracParam.c_5, inv_param->c_5, sizeof(Complex) * inv_param->Ls);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
        printfQuda("Printing b_5 and c_5 values\n");
        for (int i = 0; i < diracParam.Ls; i++) {
          printfQuda("fromQUDA diracParam: b5[%d] = %f + i%f, c5[%d] = %f + i%f\n", i, diracParam.b_5[i].real(),
              diracParam.b_5[i].imag(), i, diracParam.c_5[i].real(), diracParam.c_5[i].imag());
          // printfQuda("fromQUDA inv_param: b5[%d] = %f %f c5[%d] = %f %f\n", i, inv_param->b_5[i], i,
          // inv_param->c_5[i] ); printfQuda("fromQUDA creal: b5[%d] = %f %f c5[%d] = %f %f \n", i,
          // creal(inv_param->b_5[i]), cimag(inv_param->b_5[i]), i, creal(inv_param->c_5[i]), cimag(inv_param->c_5[i]) );
        }
      }
      break;
    case QUDA_STAGGERED_DSLASH:
      diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
      break;
    case QUDA_ASQTAD_DSLASH:
      diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
      break;
    case QUDA_TWISTED_MASS_DSLASH:
      diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
      if (inv_param->twist_flavor == QUDA_TWIST_SINGLET) {
	diracParam.Ls = 1;
	diracParam.epsilon = 0.0;
      } else {
	diracParam.Ls = 2;
	diracParam.epsilon = inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ? inv_param->epsilon : 0.0;
      }
      break;
    case QUDA_TWISTED_CLOVER_DSLASH:
      diracParam.type = pc ? QUDA_TWISTED_CLOVERPC_DIRAC : QUDA_TWISTED_CLOVER_DIRAC;
      if (inv_param->twist_flavor == QUDA_TWIST_SINGLET)  {
	diracParam.Ls = 1;
	diracParam.epsilon = 0.0;
      } else {
	diracParam.Ls = 2;
	diracParam.epsilon = inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ? inv_param->epsilon : 0.0;
      }
      break;
    case QUDA_LAPLACE_DSLASH:
      diracParam.type = pc ? QUDA_GAUGE_LAPLACEPC_DIRAC : QUDA_GAUGE_LAPLACE_DIRAC;
      diracParam.laplace3D = inv_param->laplace3D;
      break;
    case QUDA_COVDEV_DSLASH:
      diracParam.type = QUDA_GAUGE_COVDEV_DIRAC;
      break;
    default:
      errorQuda("Unsupported dslash_type %d", inv_param->dslash_type);
    }

    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatPrecise : gaugePrecise;
    diracParam.fatGauge = gaugeFatPrecise;
    diracParam.longGauge = gaugeLongPrecise;
    diracParam.clover = cloverPrecise;
    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on

    if (diracParam.gauge->Precision() != inv_param->cuda_prec)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec);
  }


  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatSloppy : gaugeSloppy;
    diracParam.fatGauge = gaugeFatSloppy;
    diracParam.longGauge = gaugeLongSloppy;
    diracParam.clover = cloverSloppy;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }

    if (diracParam.gauge->Precision() != inv_param->cuda_prec_sloppy)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec_sloppy);
  }

  void setDiracRefineParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatRefinement : gaugeRefinement;
    diracParam.fatGauge = gaugeFatRefinement;
    diracParam.longGauge = gaugeLongRefinement;
    diracParam.clover = cloverRefinement;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }

    if (diracParam.gauge->Precision() != inv_param->cuda_prec_refinement_sloppy)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec_refinement_sloppy);
  }

  // The preconditioner currently mimicks the sloppy operator with no comms
  void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc, bool comms)
  {
    setDiracParam(diracParam, inv_param, pc);

    if (inv_param->overlap) {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatExtended : gaugeExtended;
      diracParam.fatGauge = gaugeFatExtended;
      diracParam.longGauge = gaugeLongExtended;
    } else {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatPrecondition : gaugePrecondition;
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

    if (diracParam.gauge->Precision() != inv_param->cuda_prec_precondition)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec_precondition);
  }

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    // eigCG and deflation need 2 sloppy precisions and do not use Schwarz
    bool comms_flag = (param.schwarz_type != QUDA_INVALID_SCHWARZ) ? false : true;
    setDiracPreParam(diracPreParam, &param, pc_solve, comms_flag);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
  }

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dRef, QudaInvertParam &param, const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;
    DiracParam diracRefParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    setDiracRefineParam(diracRefParam, &param, pc_solve);
    // eigCG and deflation need 2 sloppy precisions and do not use Schwarz
    bool comms_flag = (param.inv_type == QUDA_INC_EIGCG_INVERTER || param.eig_param) ? true : false;
    setDiracPreParam(diracPreParam, &param, pc_solve, comms_flag);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
    dRef = Dirac::create(diracRefParam);
  }

  static double unscaled_shifts[QUDA_MAX_MULTI_SHIFT];

  void massRescale(cudaColorSpinorField &b, QudaInvertParam &param) {

    double kappa5 = (0.5/(5.0 + param.m5));
    double kappa = (param.dslash_type == QUDA_DOMAIN_WALL_DSLASH || param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
                    || param.dslash_type == QUDA_MOBIUS_DWF_DSLASH || param.dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) ?
      kappa5 :
      param.kappa;

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
	  blas::ax(16.0*std::pow(kappa,4), b);
	  for(int i=0; i<param.num_offset; i++)  param.offset[i] *= 16.0*std::pow(kappa,4);
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
  profileDslash.TPSTART(QUDA_PROFILE_TOTAL);
  profileDslash.TPSTART(QUDA_PROFILE_INIT);

  const auto &gauge = (inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
      || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
    errorQuda("Gauge field not allocated");
  if (cloverPrecise == nullptr && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gauge.X(), true, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField in(*in_h, cudaParam);
  cudaColorSpinorField out(in, cudaParam);

  bool pc = true;
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  profileDslash.TPSTOP(QUDA_PROFILE_INIT);

  profileDslash.TPSTART(QUDA_PROFILE_H2D);
  in = *in_h;
  profileDslash.TPSTOP(QUDA_PROFILE_H2D);

  profileDslash.TPSTART(QUDA_PROFILE_COMPUTE);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  if (inv_param->mass_normalization == QUDA_KAPPA_NORMALIZATION &&
      (inv_param->dslash_type == QUDA_STAGGERED_DSLASH ||
       inv_param->dslash_type == QUDA_ASQTAD_DSLASH) )
    blas::ax(1.0/(2.0*inv_param->mass), in);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    blas::ax(gauge.Anisotropy(), in);
  }

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH && inv_param->dagger) {
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    cudaColorSpinorField tmp1(in, cudaParam);
    ((DiracTwistedCloverPC*) dirac)->TwistCloverInv(tmp1, in, (parity+1)%2); // apply the clover-twist
    dirac->Dslash(out, tmp1, parity); // apply the operator
  } else {
    dirac->Dslash(out, in, parity); // apply the operator
  }
  profileDslash.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileDslash.TPSTART(QUDA_PROFILE_D2H);
  *out_h = out;
  profileDslash.TPSTOP(QUDA_PROFILE_D2H);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  profileDslash.TPSTART(QUDA_PROFILE_FREE);
  delete dirac; // clean up

  delete out_h;
  delete in_h;
  profileDslash.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();
  profileDslash.TPSTOP(QUDA_PROFILE_TOTAL);
}

void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  const auto &gauge = (inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
      || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
    errorQuda("Gauge field not allocated");
  if (cloverPrecise == nullptr && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gauge.X(), pc, inv_param->input_location);
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

  const auto &gauge = (inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
      || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
    errorQuda("Gauge field not allocated");
  if (cloverPrecise == nullptr && ((inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, *inv_param, gauge.X(), pc, inv_param->input_location);
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
      blas::ax(1.0/std::pow(2.0*kappa,4), out);
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

namespace quda
{
  bool canReuseResidentGauge(QudaInvertParam *param)
  {
    if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
      return (gaugePrecise != nullptr) and param->cuda_prec == gaugePrecise->Precision();
    } else {
      return (gaugeFatPrecise != nullptr) and param->cuda_prec == gaugeFatPrecise->Precision();
    }
  }
} // namespace quda

void checkClover(QudaInvertParam *param) {

  if (param->dslash_type != QUDA_CLOVER_WILSON_DSLASH && param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    return;
  }

  if (param->cuda_prec != cloverPrecise->Precision()) {
    errorQuda("Solve precision %d doesn't match clover precision %d", param->cuda_prec, cloverPrecise->Precision());
  }

  if ( (!cloverSloppy || param->cuda_prec_sloppy != cloverSloppy->Precision()) ||
       (!cloverPrecondition || param->cuda_prec_precondition != cloverPrecondition->Precision()) ||
       (!cloverRefinement || param->cuda_prec_refinement_sloppy != cloverRefinement->Precision()) ) {
    freeSloppyCloverQuda();
    QudaPrecision prec[] = {param->cuda_prec_sloppy, param->cuda_prec_precondition, param->cuda_prec_refinement_sloppy};
    loadSloppyCloverQuda(prec);
  }

  if (cloverPrecise == nullptr) errorQuda("Precise clover field doesn't exist");
  if (cloverSloppy == nullptr) errorQuda("Sloppy clover field doesn't exist");
  if (cloverPrecondition == nullptr) errorQuda("Precondition clover field doesn't exist");
  if (cloverRefinement == nullptr) errorQuda("Refinement clover field doesn't exist");
}

quda::cudaGaugeField *checkGauge(QudaInvertParam *param)
{
  quda::cudaGaugeField *cudaGauge = nullptr;
  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    if (gaugePrecise == nullptr) errorQuda("Precise gauge field doesn't exist");

    if (param->cuda_prec != gaugePrecise->Precision()) {
      errorQuda("Solve precision %d doesn't match gauge precision %d", param->cuda_prec, gaugePrecise->Precision());
    }

    if (param->cuda_prec_sloppy != gaugeSloppy->Precision()
        || param->cuda_prec_precondition != gaugePrecondition->Precision()
        || param->cuda_prec_refinement_sloppy != gaugeRefinement->Precision()) {
      QudaPrecision precision[3]
          = {param->cuda_prec_sloppy, param->cuda_prec_precondition, param->cuda_prec_refinement_sloppy};
      QudaReconstructType recon[3]
          = {gaugeSloppy->Reconstruct(), gaugePrecondition->Reconstruct(), gaugeRefinement->Reconstruct()};
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(precision, recon);
    }

    if (gaugeSloppy == nullptr) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == nullptr) errorQuda("Precondition gauge field doesn't exist");
    if (gaugeRefinement == nullptr) errorQuda("Refinement gauge field doesn't exist");
    if (param->overlap) {
      if (gaugeExtended == nullptr) errorQuda("Extended gauge field doesn't exist");
    }
    cudaGauge = gaugePrecise;
  } else {
    if (gaugeFatPrecise == nullptr) errorQuda("Precise gauge fat field doesn't exist");
    if (gaugeLongPrecise == nullptr) errorQuda("Precise gauge long field doesn't exist");

    if (param->cuda_prec != gaugeFatPrecise->Precision()) {
      errorQuda("Solve precision %d doesn't match gauge precision %d", param->cuda_prec, gaugeFatPrecise->Precision());
    }

    if (param->cuda_prec_sloppy != gaugeFatSloppy->Precision()
        || param->cuda_prec_precondition != gaugeFatPrecondition->Precision()
        || param->cuda_prec_refinement_sloppy != gaugeFatRefinement->Precision()
        || param->cuda_prec_sloppy != gaugeLongSloppy->Precision()
        || param->cuda_prec_precondition != gaugeLongPrecondition->Precision()
        || param->cuda_prec_refinement_sloppy != gaugeLongRefinement->Precision()) {

      QudaPrecision precision[3]
        = {param->cuda_prec_sloppy, param->cuda_prec_precondition, param->cuda_prec_refinement_sloppy};
      // recon is always no for fat links, so just use long reconstructs here
      QudaReconstructType recon[3]
        = {gaugeLongSloppy->Reconstruct(), gaugeLongPrecondition->Reconstruct(), gaugeLongRefinement->Reconstruct()};
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(precision, recon);
    }

    if (gaugeFatSloppy == nullptr) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == nullptr) errorQuda("Precondition gauge fat field doesn't exist");
    if (gaugeFatRefinement == nullptr) errorQuda("Refinement gauge fat field doesn't exist");
    if (param->overlap) {
      if (gaugeFatExtended == nullptr) errorQuda("Extended gauge fat field doesn't exist");
    }

    if (gaugeLongSloppy == nullptr) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == nullptr) errorQuda("Precondition gauge long field doesn't exist");
    if (gaugeLongRefinement == nullptr) errorQuda("Refinement gauge long field doesn't exist");
    if (param->overlap) {
      if (gaugeLongExtended == nullptr) errorQuda("Extended gauge long field doesn't exist");
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
  if (gaugePrecise == nullptr) errorQuda("Gauge field not allocated");
  if (cloverPrecise == nullptr) errorQuda("Clover field not allocated");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH))
    errorQuda("Cannot apply the clover term for a non Wilson-clover or Twisted-mass-clover dslash");

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), true);

  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

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
	//FIXME: Do we need this for twisted clover???
  DiracCloverPC dirac(diracParam); // create the Dirac operator
  if (!inverse) dirac.Clover(out, in, parity); // apply the clover operator
  else dirac.CloverInv(out, in, parity);

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
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

void eigensolveQuda(void **host_evecs, double _Complex *host_evals, QudaEigParam *eig_param)
{
  profileEigensolve.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolve.TPSTART(QUDA_PROFILE_INIT);

  // Transfer the inv param structure contained in eig_param
  QudaInvertParam *inv_param = eig_param->invert_param;

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH || inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH)
    setKernelPackT(true);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(inv_param);
    printQudaEigParam(eig_param);
  }

  checkInvertParam(inv_param);
  checkEigParam(eig_param);
  cudaGaugeField *cudaGauge = checkGauge(inv_param);

  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE)
    || (inv_param->solve_type == QUDA_NORMERR_PC_SOLVE);

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  // Define problem matrix
  //------------------------------------------------------
  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);
  Dirac &dirac = *d;

  // Create device side ColorSpinorField vector space and to pass to the
  // compute function.
  const int *X = cudaGauge->X();
  ColorSpinorParam cpuParam(host_evecs[0], *inv_param, X, inv_param->solution_type, inv_param->input_location);

  // create wrappers around application vector set
  std::vector<ColorSpinorField *> host_evecs_;
  for (int i = 0; i < eig_param->nConv; i++) {
    cpuParam.v = host_evecs[i];
    host_evecs_.push_back(ColorSpinorField::Create(cpuParam));
  }

  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(eig_param->cuda_prec_ritz, eig_param->cuda_prec_ritz, true);

  std::vector<Complex> evals(eig_param->nConv, 0.0);
  std::vector<ColorSpinorField *> kSpace;
  for (int i = 0; i < eig_param->nConv; i++) { kSpace.push_back(ColorSpinorField::Create(cudaParam)); }

  // If you use polynomial acceleration on a non-symmetric matrix,
  // the solver will fail.
  if (eig_param->use_poly_acc && !eig_param->use_norm_op && !(inv_param->dslash_type == QUDA_LAPLACE_DSLASH)) {
    // Breaking up the boolean check a little bit. If it's a staggered dslash type and a PC type, we can use poly accel.
    if (!((inv_param->dslash_type == QUDA_STAGGERED_DSLASH || inv_param->dslash_type == QUDA_ASQTAD_DSLASH) && inv_param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Polynomial acceleration with non-symmetric matrices not supported");
    }
  }

  profileEigensolve.TPSTOP(QUDA_PROFILE_INIT);

  if (!eig_param->use_norm_op && !eig_param->use_dagger) {
    DiracM m(dirac);
    if (eig_param->arpack_check) {
      arpack_solve(host_evecs_, evals, m, eig_param, profileEigensolve);
    } else {
      EigenSolver *eig_solve = EigenSolver::create(eig_param, m, profileEigensolve);
      (*eig_solve)(kSpace, evals);
      delete eig_solve;
    }
  } else if (!eig_param->use_norm_op && eig_param->use_dagger) {
    DiracMdag m(dirac);
    if (eig_param->arpack_check) {
      arpack_solve(host_evecs_, evals, m, eig_param, profileEigensolve);
    } else {
      EigenSolver *eig_solve = EigenSolver::create(eig_param, m, profileEigensolve);
      (*eig_solve)(kSpace, evals);
      delete eig_solve;
    }
  } else if (eig_param->use_norm_op && !eig_param->use_dagger) {
    DiracMdagM m(dirac);
    if (eig_param->arpack_check) {
      arpack_solve(host_evecs_, evals, m, eig_param, profileEigensolve);
    } else {
      EigenSolver *eig_solve = EigenSolver::create(eig_param, m, profileEigensolve);
      (*eig_solve)(kSpace, evals);
      delete eig_solve;
    }
  } else if (eig_param->use_norm_op && eig_param->use_dagger) {
    DiracMMdag m(dirac);
    if (eig_param->arpack_check) {
      arpack_solve(host_evecs_, evals, m, eig_param, profileEigensolve);
    } else {
      EigenSolver *eig_solve = EigenSolver::create(eig_param, m, profileEigensolve);
      (*eig_solve)(kSpace, evals);
      delete eig_solve;
    }
  } else {
    errorQuda("Invalid use_norm_op and dagger combination");
  }

  // Copy eigen values back
  for (int i = 0; i < eig_param->nConv; i++) { host_evals[i] = real(evals[i]) + imag(evals[i]) * _Complex_I; }

  // Transfer Eigenpairs back to host if using GPU eigensolver
  if (!(eig_param->arpack_check)) {
    profileEigensolve.TPSTART(QUDA_PROFILE_D2H);
    for (int i = 0; i < eig_param->nConv; i++) *host_evecs_[i] = *kSpace[i];
    profileEigensolve.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileEigensolve.TPSTART(QUDA_PROFILE_FREE);
  for (int i = 0; i < eig_param->nConv; i++) delete host_evecs_[i];
  delete d;
  delete dSloppy;
  delete dPre;
  for (int i = 0; i < eig_param->nConv; i++) delete kSpace[i];
  profileEigensolve.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileEigensolve.TPSTOP(QUDA_PROFILE_TOTAL);
}

multigrid_solver::multigrid_solver(QudaMultigridParam &mg_param, TimeProfile &profile)
  : profile(profile) {
  profile.TPSTART(QUDA_PROFILE_INIT);
  QudaInvertParam *param = mg_param.invert_param;

  checkMultigridParam(&mg_param);
  cudaGaugeField *cudaGauge = checkGauge(param);

  // check MG params (needs to go somewhere else)
  if (mg_param.n_level > QUDA_MAX_MG_LEVEL)
    errorQuda("Requested MG levels %d greater than allowed maximum %d", mg_param.n_level, QUDA_MAX_MG_LEVEL);
  for (int i=0; i<mg_param.n_level; i++) {
    if (mg_param.smoother_solve_type[i] != QUDA_DIRECT_SOLVE && mg_param.smoother_solve_type[i] != QUDA_DIRECT_PC_SOLVE)
      errorQuda("Unsupported smoother solve type %d on level %d", mg_param.smoother_solve_type[i], i);
  }
  if (param->solve_type != QUDA_DIRECT_SOLVE)
    errorQuda("Outer MG solver can only use QUDA_DIRECT_SOLVE at present");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaMultigridParam(&mg_param);
  mg_param.secs = 0;
  mg_param.gflops = 0;

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool outer_pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);

  // create the dirac operators for the fine grid

  // this is the Dirac operator we use for inter-grid residual computation
  DiracParam diracParam;
  setDiracSloppyParam(diracParam, param, outer_pc_solve);
  d = Dirac::create(diracParam);
  m = new DiracM(*d);

  // this is the Dirac operator we use for smoothing
  DiracParam diracSmoothParam;
  bool fine_grid_pc_solve = (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE) ||
    (mg_param.smoother_solve_type[0] == QUDA_NORMOP_PC_SOLVE);
  setDiracSloppyParam(diracSmoothParam, param, fine_grid_pc_solve);
  diracSmoothParam.halo_precision = mg_param.smoother_halo_precision[0];
  dSmooth = Dirac::create(diracSmoothParam);
  mSmooth = new DiracM(*dSmooth);

  // this is the Dirac operator we use for sloppy smoothing (we use the preconditioner fields for this)
  DiracParam diracSmoothSloppyParam;
  setDiracPreParam(diracSmoothSloppyParam, param, fine_grid_pc_solve,
		   mg_param.smoother_schwarz_type[0] == QUDA_INVALID_SCHWARZ ? true : false);
  diracSmoothSloppyParam.halo_precision = mg_param.smoother_halo_precision[0];

  dSmoothSloppy = Dirac::create(diracSmoothSloppyParam);
  mSmoothSloppy = new DiracM(*dSmoothSloppy);

  ColorSpinorParam csParam(nullptr, *param, cudaGauge->X(), pc_solution, mg_param.setup_location[0]);
  csParam.create = QUDA_NULL_FIELD_CREATE;
  QudaPrecision Bprec = mg_param.precision_null[0];
  Bprec = (mg_param.setup_location[0] == QUDA_CPU_FIELD_LOCATION && Bprec < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : Bprec);
  csParam.setPrecision(Bprec);
  csParam.fieldOrder = mg_param.setup_location[0] == QUDA_CUDA_FIELD_LOCATION ? QUDA_FLOAT2_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.mem_type = mg_param.setup_minimize_memory == QUDA_BOOLEAN_TRUE ? QUDA_MEMORY_MAPPED : QUDA_MEMORY_DEVICE;
  B.resize(mg_param.n_vec[0]);

  if (mg_param.is_staggered == QUDA_BOOLEAN_TRUE) {
    // Create the ColorSpinorField as a "container" for metadata.
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;

    // These never get accessed, `nullptr` on its own leads to an error in texture binding
    csParam.v = (void *)std::numeric_limits<uint64_t>::max();
    csParam.norm = (void *)std::numeric_limits<uint64_t>::max();
  }

  for (int i = 0; i < mg_param.n_vec[0]; i++) { B[i] = ColorSpinorField::Create(csParam); }

  // fill out the MG parameters for the fine level
  mgParam = new MGParam(mg_param, B, m, mSmooth, mSmoothSloppy);

  mg = new MG(*mgParam, profile);
  mgParam->updateInvertParam(*param);

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();
  profile.TPSTOP(QUDA_PROFILE_INIT);
}

void* newMultigridQuda(QudaMultigridParam *mg_param) {
  profilerStart(__func__);

  pushVerbosity(mg_param->invert_param->verbosity);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  auto *mg = new multigrid_solver(*mg_param, profileInvert);
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  saveTuneCache();

  popVerbosity();

  profilerStop(__func__);
  return static_cast<void*>(mg);
}

void destroyMultigridQuda(void *mg) {
  delete static_cast<multigrid_solver*>(mg);
}

void updateMultigridQuda(void *mg_, QudaMultigridParam *mg_param)
{
  profilerStart(__func__);

  pushVerbosity(mg_param->invert_param->verbosity);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  profileInvert.TPSTART(QUDA_PROFILE_PREAMBLE);

  auto *mg = static_cast<multigrid_solver*>(mg_);
  checkMultigridParam(mg_param);

  QudaInvertParam *param = mg_param->invert_param;
  // check the gauge fields have been created and set the precision as needed
  checkGauge(param);

  // for reporting level 1 is the fine level but internally use level 0 for indexing
  // sprintf(mg->prefix,"MG level 1 (%s): ", param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
  // setOutputPrefix(prefix);
  setOutputPrefix("MG level 1 (GPU): "); //fix me

  bool outer_pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);

  // free the previous dirac operators
  if (mg->m) delete mg->m;
  if (mg->mSmooth) delete mg->mSmooth;
  if (mg->mSmoothSloppy) delete mg->mSmoothSloppy;

  if (mg->d) delete mg->d;
  if (mg->dSmooth) delete mg->dSmooth;
  if (mg->dSmoothSloppy && mg->dSmoothSloppy != mg->dSmooth) delete mg->dSmoothSloppy;

  // create new fine dirac operators

  // this is the Dirac operator we use for inter-grid residual computation
  DiracParam diracParam;
  setDiracSloppyParam(diracParam, param, outer_pc_solve);
  mg->d = Dirac::create(diracParam);
  mg->m = new DiracM(*(mg->d));

  // this is the Dirac operator we use for smoothing
  DiracParam diracSmoothParam;
  bool fine_grid_pc_solve = (mg_param->smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE) ||
    (mg_param->smoother_solve_type[0] == QUDA_NORMOP_PC_SOLVE);
  setDiracSloppyParam(diracSmoothParam, param, fine_grid_pc_solve);
  mg->dSmooth = Dirac::create(diracSmoothParam);
  mg->mSmooth = new DiracM(*(mg->dSmooth));

  // this is the Dirac operator we use for sloppy smoothing (we use the preconditioner fields for this)
  DiracParam diracSmoothSloppyParam;
  setDiracPreParam(diracSmoothSloppyParam, param, fine_grid_pc_solve, true);
  mg->dSmoothSloppy = Dirac::create(diracSmoothSloppyParam);;
  mg->mSmoothSloppy = new DiracM(*(mg->dSmoothSloppy));

  mg->mgParam->matResidual = mg->m;
  mg->mgParam->matSmooth = mg->mSmooth;
  mg->mgParam->matSmoothSloppy = mg->mSmoothSloppy;

  mg->mgParam->updateInvertParam(*param);
  if(mg->mgParam->mg_global.invert_param != param)
    mg->mgParam->mg_global.invert_param = param;

  bool refresh = true;
  mg->mg->reset(refresh);

  setOutputPrefix("");

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  popVerbosity();

  profilerStop(__func__);
}

void dumpMultigridQuda(void *mg_, QudaMultigridParam *mg_param)
{
  profilerStart(__func__);
  pushVerbosity(mg_param->invert_param->verbosity);
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  auto *mg = static_cast<multigrid_solver*>(mg_);
  checkMultigridParam(mg_param);
  checkGauge(mg_param->invert_param);

  mg->mg->dumpNullVectors();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
  popVerbosity();
  profilerStop(__func__);
}

deflated_solver::deflated_solver(QudaEigParam &eig_param, TimeProfile &profile)
  : d(nullptr), m(nullptr), RV(nullptr), deflParam(nullptr), defl(nullptr),  profile(profile) {

  QudaInvertParam *param = eig_param.invert_param;

  if (param->inv_type != QUDA_EIGCG_INVERTER && param->inv_type != QUDA_INC_EIGCG_INVERTER) return;

  profile.TPSTART(QUDA_PROFILE_INIT);

  cudaGaugeField *cudaGauge = checkGauge(param);
  eig_param.secs   = 0;
  eig_param.gflops = 0;

  DiracParam diracParam;
  if(eig_param.cuda_prec_ritz == param->cuda_prec)
  {
    setDiracParam(diracParam, param, (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE));
  } else {
    setDiracSloppyParam(diracParam, param, (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE));
  }

  const bool pc_solve = (param->solve_type == QUDA_NORMOP_PC_SOLVE);

  d = Dirac::create(diracParam);
  m = pc_solve ? static_cast<DiracMatrix*>( new DiracMdagM(*d) ) : static_cast<DiracMatrix*>( new DiracM(*d));

  ColorSpinorParam ritzParam(nullptr, *param, cudaGauge->X(), pc_solve, eig_param.location);

  ritzParam.create        = QUDA_ZERO_FIELD_CREATE;
  ritzParam.is_composite  = true;
  ritzParam.is_component  = false;
  ritzParam.composite_dim = param->nev*param->deflation_grid;
  ritzParam.setPrecision(param->cuda_prec_ritz);

  if (ritzParam.location==QUDA_CUDA_FIELD_LOCATION) {
    ritzParam.setPrecision(param->cuda_prec_ritz, param->cuda_prec_ritz, true); // set native field order
    if (ritzParam.nSpin != 1) ritzParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    //select memory location here, by default ritz vectors will be allocated on the device
    //but if not sufficient device memory, then the user may choose mapped type of memory
    ritzParam.mem_type = eig_param.mem_type_ritz;
  } else { //host location
    ritzParam.mem_type = QUDA_MEMORY_PINNED;
  }

  int ritzVolume = 1;
  for(int d = 0; d < ritzParam.nDim; d++) ritzVolume *= ritzParam.x[d];

  if (getVerbosity() == QUDA_DEBUG_VERBOSE) {

    size_t byte_estimate = (size_t)ritzParam.composite_dim*(size_t)ritzVolume*(ritzParam.nColor*ritzParam.nSpin*ritzParam.Precision());
    printfQuda("allocating bytes: %lu (lattice volume %d, prec %d)", byte_estimate, ritzVolume, ritzParam.Precision());
    if(ritzParam.mem_type == QUDA_MEMORY_DEVICE) printfQuda("Using device memory type.\n");
    else if (ritzParam.mem_type == QUDA_MEMORY_MAPPED)
      printfQuda("Using mapped memory type.\n");
  }

  RV = ColorSpinorField::Create(ritzParam);

  deflParam = new DeflationParam(eig_param, RV, *m);

  defl = new Deflation(*deflParam, profile);

  profile.TPSTOP(QUDA_PROFILE_INIT);
}

void* newDeflationQuda(QudaEigParam *eig_param) {
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
#ifdef MAGMA_LIB
  openMagma();
#endif
  auto *defl = new deflated_solver(*eig_param, profileInvert);

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  saveProfile(__func__);
  flushProfile();
  return static_cast<void*>(defl);
}

void destroyDeflationQuda(void *df) {
#ifdef MAGMA_LIB
  closeMagma();
#endif
  delete static_cast<deflated_solver*>(df);
}

void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  profilerStart(__func__);

  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH || param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || param->dslash_type == QUDA_MOBIUS_DWF_DSLASH || param->dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH)
    setKernelPackT(true);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param, hp_x, hp_b);

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

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = nullptr;
  ColorSpinorField *x = nullptr;
  ColorSpinorField *in = nullptr;
  ColorSpinorField *out = nullptr;

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

  // now check if we need to invalidate the solutionResident vectors
  bool invalidate = false;
  if (param->use_resident_solution == 1) {
    for (auto v : solutionResident)
      if (b->Precision() != v->Precision() || b->SiteSubset() != v->SiteSubset()) { invalidate = true; break; }

    if (invalidate) {
      for (auto v : solutionResident) if (v) delete v;
      solutionResident.clear();
    }

    if (!solutionResident.size()) {
      cudaParam.create = QUDA_NULL_FIELD_CREATE;
      solutionResident.push_back(new cudaColorSpinorField(cudaParam)); // solution
    }
    x = solutionResident[0];
  } else {
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam);
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

  // if we're doing a managed memory MG solve and prefetching is
  // enabled, prefetch all the Dirac matrices. There's probably
  // a better place to put this...
  if (param->inv_type_precondition == QUDA_MG_INVERTER) {
    dirac.prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracSloppy.prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracPre.prefetch(QUDA_CUDA_FIELD_LOCATION);
  }

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  profileInvert.TPSTART(QUDA_PROFILE_PREAMBLE);

  double nb = blas::norm2(*b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = blas::norm2(*h_b);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      double nh_x = blas::norm2(*h_x);
      double nx = blas::norm2(*x);
      printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
    }
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

  if (param->chrono_use_resident && ( norm_error_solve) ){
    errorQuda("Chronological forcasting only presently supported for M^dagger M solver");
  }

  profileInvert.TPSTOP(QUDA_PROFILE_PREAMBLE);

  if (mat_solution && !direct_solve && !norm_error_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    blas::copy(*in, *out);
    delete solve;
    solverParam.updateInvertParam(*param);
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    // chronological forecasting
    if (param->chrono_use_resident && chronoResident[param->chrono_index].size() > 0) {
      profileInvert.TPSTART(QUDA_PROFILE_CHRONO);

      auto &basis = chronoResident[param->chrono_index];

      ColorSpinorParam cs_param(*basis[0]);
      ColorSpinorField *tmp = ColorSpinorField::Create(cs_param);
      ColorSpinorField *tmp2 = (param->chrono_precision == out->Precision()) ? out : ColorSpinorField::Create(cs_param);
      std::vector<ColorSpinorField*> Ap;
      for (unsigned int k=0; k < basis.size(); k++) {
        Ap.emplace_back((ColorSpinorField::Create(cs_param)));
      }

      if (param->chrono_precision == param->cuda_prec) {
        for (unsigned int j=0; j<basis.size(); j++) m(*Ap[j], *basis[j], *tmp, *tmp2);
      } else if (param->chrono_precision == param->cuda_prec_sloppy) {
        for (unsigned int j=0; j<basis.size(); j++) mSloppy(*Ap[j], *basis[j], *tmp, *tmp2);
      } else {
        errorQuda("Unexpected precision %d for chrono vectors (doesn't match outer %d or sloppy precision %d)",
                  param->chrono_precision, param->cuda_prec, param->cuda_prec_sloppy);
      }

      bool orthogonal = true;
      bool apply_mat = false;
      bool hermitian = false;
      MinResExt mre(m, orthogonal, apply_mat, hermitian, profileInvert);

      blas::copy(*tmp, *in);
      mre(*out, *tmp, basis, Ap);

      for (auto ap: Ap) {
        if (ap) delete (ap);
      }
      delete tmp;
      if (tmp2 != out) delete tmp2;

      profileInvert.TPSTOP(QUDA_PROFILE_CHRONO);
    }

    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    delete solve;
    solverParam.updateInvertParam(*param);
  } else if (!norm_error_solve) {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);

    // chronological forecasting
    if (param->chrono_use_resident && chronoResident[param->chrono_index].size() > 0) {
      profileInvert.TPSTART(QUDA_PROFILE_CHRONO);

      auto &basis = chronoResident[param->chrono_index];

      ColorSpinorParam cs_param(*basis[0]);
      std::vector<ColorSpinorField*> Ap;
      ColorSpinorField *tmp = ColorSpinorField::Create(cs_param);
      ColorSpinorField *tmp2 = (param->chrono_precision == out->Precision()) ? out : ColorSpinorField::Create(cs_param);
      for (unsigned int k=0; k < basis.size(); k++) {
        Ap.emplace_back((ColorSpinorField::Create(cs_param)));
      }

      if (param->chrono_precision == param->cuda_prec) {
        for (unsigned int j=0; j<basis.size(); j++) m(*Ap[j], *basis[j], *tmp, *tmp2);
      } else if (param->chrono_precision == param->cuda_prec_sloppy) {
        for (unsigned int j=0; j<basis.size(); j++) mSloppy(*Ap[j], *basis[j], *tmp, *tmp2);
      } else {
        errorQuda("Unexpected precision %d for chrono vectors (doesn't match outer %d or sloppy precision %d)",
                  param->chrono_precision, param->cuda_prec, param->cuda_prec_sloppy);
      }

      bool orthogonal = true;
      bool apply_mat = false;
      bool hermitian = true;
      MinResExt mre(m, orthogonal, apply_mat, hermitian, profileInvert);

      blas::copy(*tmp, *in);
      mre(*out, *tmp, basis, Ap);

      for (auto ap: Ap) {
        if (ap) delete(ap);
      }
      delete tmp;
      if (tmp2 != out) delete tmp2;

      profileInvert.TPSTOP(QUDA_PROFILE_CHRONO);
    }

    // if using a Schwarz preconditioner with a normal operator then we must use the DiracMdagMLocal operator
    if (param->inv_type_precondition != QUDA_INVALID_INVERTER && param->schwarz_type != QUDA_INVALID_SCHWARZ) {
      DiracMdagMLocal mPreLocal(diracPre);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPreLocal, profileInvert);
      (*solve)(*out, *in);
      solverParam.updateInvertParam(*param);
      delete solve;
    } else {
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      (*solve)(*out, *in);
      solverParam.updateInvertParam(*param);
      delete solve;
    }
  } else { // norm_error_solve
    DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    cudaColorSpinorField tmp(*out);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(tmp, *in); // y = (M M^\dag) b
    dirac.Mdag(*out, tmp);  // x = M^dag y
    delete solve;
    solverParam.updateInvertParam(*param);
  }

  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = blas::norm2(*x);
    printfQuda("Solution = %g\n",nx);
  }

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);
  if (param->chrono_make_resident) {
    if(param->chrono_max_dim < 1){
      errorQuda("Cannot chrono_make_resident with chrono_max_dim %i",param->chrono_max_dim);
    }

    const int i = param->chrono_index;
    if (i >= QUDA_MAX_CHRONO)
      errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

    auto &basis = chronoResident[i];

    if(param->chrono_max_dim < (int)basis.size()){
      errorQuda("Requested chrono_max_dim %i is smaller than already existing chroology %i",param->chrono_max_dim,(int)basis.size());
    }

    if(not param->chrono_replace_last){
      // if we have not filled the space yet just augment
      if ((int)basis.size() < param->chrono_max_dim) {
        ColorSpinorParam cs_param(*out);
        cs_param.setPrecision(param->chrono_precision);
        basis.emplace_back(ColorSpinorField::Create(cs_param));
      }

      // shuffle every entry down one and bring the last to the front
      ColorSpinorField *tmp = basis[basis.size()-1];
      for (unsigned int j=basis.size()-1; j>0; j--) basis[j] = basis[j-1];
        basis[0] = tmp;
    }
    *(basis[0]) = *out; // set first entry to new solution
  }
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

  if (param->use_resident_solution && !param->make_resident_solution) {
    for (auto v: solutionResident) if (v) delete v;
    solutionResident.clear();
  } else if (!param->make_resident_solution) {
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

  profilerStop(__func__);
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
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param, _hp_x[0], _hp_b[0]);

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

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

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
  ColorSpinorField* in;  // = nullptr;
  //in.resize(param->num_src);
  ColorSpinorField* out;  // = nullptr;
  //out.resize(param->num_src);

  // for(int i=0;i < param->num_src;i++){
  //   in[i] = nullptr;
  //   out[i] = nullptr;
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

  auto * nb = new double[param->num_src];
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
      solve->blocksolve(*out,*in);
      for(int i=0; i < param->num_src; i++) {
        blas::copy(in->Component(i), out->Component(i));
      }
      delete solve;
      solverParam.updateInvertParam(*param);
    }

    if (direct_solve) {
      DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      solve->blocksolve(*out,*in);
      delete solve;
      solverParam.updateInvertParam(*param);
    } else if (!norm_error_solve) {
      DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      solve->blocksolve(*out,*in);
      delete solve;
      solverParam.updateInvertParam(*param);
    } else { // norm_error_solve
      DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      errorQuda("norm_error_solve not supported in multi source solve");
      // cudaColorSpinorField tmp(*out);
      // SolverParam solverParam(*param);
      // Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
      //(*solve)(tmp, *in); // y = (M M^\dag) b
      // dirac.Mdag(*out, tmp);  // x = M^dag y
      // delete solve;
      // solverParam.updateInvertParam(*param,i,i);
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
//  if (!param->make_resident_solution) delete x; // FIXME make this cleaner

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
 * For Wilson-type fermions, the solution_type must be MATDAG_MAT or MATPCDAG_MATPC,
 * and solve_type must be NORMOP or NORMOP_PC. The solution and solve
 * preconditioning have to match.
 *
 * For Staggered-type fermions, the solution_type must be MATPC, and the
 * solve type must be DIRECT_PC. This difference in convention is because
 * preconditioned staggered operator is normal, unlike with Wilson-type fermions.
 */
void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param)
{
  profilerStart(__func__);

  profileMulti.TPSTART(QUDA_PROFILE_TOTAL);
  profileMulti.TPSTART(QUDA_PROFILE_INIT);

  if (!initialized) errorQuda("QUDA not initialized");

  checkInvertParam(param, _hp_x[0], _hp_b);

  // check the gauge fields have been created
  checkGauge(param);

  if (param->num_offset > QUDA_MAX_MULTI_SHIFT)
    errorQuda("Number of shifts %d requested greater than QUDA_MAX_MULTI_SHIFT %d", param->num_offset,
              QUDA_MAX_MULTI_SHIFT);

  pushVerbosity(param->verbosity);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
      param->dslash_type == QUDA_STAGGERED_DSLASH) {

    if (param->solution_type != QUDA_MATPC_SOLUTION) {
      errorQuda("For Staggered-type fermions, multi-shift solver only suports MATPC solution type");
    }

    if (param->solve_type != QUDA_DIRECT_PC_SOLVE) {
      errorQuda("For Staggered-type fermions, multi-shift solver only supports DIRECT_PC solve types");
    }

  } else { // Wilson type

    if (mat_solution) {
      errorQuda("For Wilson-type fermions, multi-shift solver does not support MAT or MATPC solution types");
    }
    if (direct_solve) {
      errorQuda("For Wilson-type fermions, multi-shift solver does not support DIRECT or DIRECT_PC solve types");
    }
    if (pc_solution & !pc_solve) {
      errorQuda("For Wilson-type fermions, preconditioned (PC) solution_type requires a PC solve_type");
    }
    if (!pc_solution & pc_solve) {
      errorQuda("For Wilson-type fermions, in multi-shift solver, a preconditioned (PC) solve_type requires a PC solution_type");
    }
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

  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;
  Dirac *dRefine = nullptr;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, dRefine, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;


  cudaColorSpinorField *b = nullptr;   // Cuda RHS
  std::vector<ColorSpinorField*> x;  // Cuda Solutions
  x.resize(param->num_offset);
  std::vector<ColorSpinorField*> p;
  std::unique_ptr<double[]> r2_old(new double[param->num_offset]);

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
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it
  profileMulti.TPSTOP(QUDA_PROFILE_H2D);

  profileMulti.TPSTART(QUDA_PROFILE_INIT);
  // Create the solution fields filled with zero
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;

  // now check if we need to invalidate the solutionResident vectors
  bool invalidate = false;
  for (auto v : solutionResident) {
    if (cudaParam.Precision() != v->Precision()) {
      invalidate = true;
      break;
    }
  }

  if (invalidate) {
    for (auto v : solutionResident) delete v;
    solutionResident.clear();
  }

  // grow resident solutions to be big enough
  for (int i=solutionResident.size(); i < param->num_offset; i++) {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Adding vector %d to solutionsResident\n", i);
    solutionResident.push_back(new cudaColorSpinorField(cudaParam));
  }
  for (int i=0; i < param->num_offset; i++) x[i] = solutionResident[i];
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

  DiracMatrix *m, *mSloppy;

  if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
      param->dslash_type == QUDA_STAGGERED_DSLASH) {
    m = new DiracM(dirac);
    mSloppy = new DiracM(diracSloppy);
  } else {
    m = new DiracMdagM(dirac);
    mSloppy = new DiracMdagM(diracSloppy);
  }

  SolverParam solverParam(*param);
  {
    MultiShiftCG cg_m(*m, *mSloppy, solverParam, profileMulti);
    cg_m(x, *b, p, r2_old.get());
  }
  solverParam.updateInvertParam(*param);

  delete m;
  delete mSloppy;

  if (param->compute_true_res) {
    // check each shift has the desired tolerance and use sequential CG to refine
    profileMulti.TPSTART(QUDA_PROFILE_INIT);
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField r(*b, cudaParam);
    profileMulti.TPSTOP(QUDA_PROFILE_INIT);
    QudaInvertParam refineparam = *param;
    refineparam.cuda_prec_sloppy = param->cuda_prec_refinement_sloppy;
    Dirac &dirac = *d;
    Dirac &diracSloppy = *dRefine;

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
      const double prec_tol = std::pow(10.,(-2*(int)param->cuda_prec+4)); // implicit refinment limit of 1e-12
      const double iter_tol = (param->iter_res_offset[i] < prec_tol ? prec_tol : (param->iter_res_offset[i] *1.1));
      const double refine_tol = (param->tol_offset[i] == 0.0 ? iter_tol : param->tol_offset[i]);
      // refine if either L2 or heavy quark residual tolerances have not been met, only if desired residual is > 0
      if (param->true_res_offset[i] > refine_tol || rsd_hq > tol_hq) {
	if (getVerbosity() >= QUDA_SUMMARIZE)
	  printfQuda("Refining shift %d: L2 residual %e / %e, heavy quark %e / %e (actual / requested)\n",
		     i, param->true_res_offset[i], param->tol_offset[i], rsd_hq, tol_hq);

        // for staggered the shift is just a change in mass term (FIXME: for twisted mass also)
        if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
            param->dslash_type == QUDA_STAGGERED_DSLASH) {
          dirac.setMass(sqrt(param->offset[i]/4));
          diracSloppy.setMass(sqrt(param->offset[i]/4));
        }

        DiracMatrix *m, *mSloppy;

        if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
            param->dslash_type == QUDA_STAGGERED_DSLASH) {
          m = new DiracM(dirac);
          mSloppy = new DiracM(diracSloppy);
        } else {
          m = new DiracMdagM(dirac);
          mSloppy = new DiracMdagM(diracSloppy);
        }

        // need to curry in the shift if we are not doing staggered
        if (param->dslash_type != QUDA_ASQTAD_DSLASH && param->dslash_type != QUDA_STAGGERED_DSLASH) {
          m->shift = param->offset[i];
          mSloppy->shift = param->offset[i];
        }

        if (false) { // experimenting with Minimum residual extrapolation
                     // only perform MRE using current and previously refined solutions
#ifdef REFINE_INCREASING_MASS
	  const int nRefine = i+1;
#else
	  const int nRefine = param->num_offset - i + 1;
#endif

          std::vector<ColorSpinorField *> q;
          q.resize(nRefine);
          std::vector<ColorSpinorField *> z;
          z.resize(nRefine);
          cudaParam.create = QUDA_NULL_FIELD_CREATE;
          cudaColorSpinorField tmp(cudaParam);

          for (int j = 0; j < nRefine; j++) {
            q[j] = new cudaColorSpinorField(cudaParam);
            z[j] = new cudaColorSpinorField(cudaParam);
          }

          *z[0] = *x[0]; // zero solution already solved
#ifdef REFINE_INCREASING_MASS
	  for (int j=1; j<nRefine; j++) *z[j] = *x[j];
#else
	  for (int j=1; j<nRefine; j++) *z[j] = *x[param->num_offset-j];
#endif

          bool orthogonal = true;
          bool apply_mat = true;
          bool hermitian = true;
	  MinResExt mre(*m, orthogonal, apply_mat, hermitian, profileMulti);
	  blas::copy(tmp, *b);
	  mre(*x[i], tmp, z, q);

	  for(int j=0; j < nRefine; j++) {
	    delete q[j];
	    delete z[j];
	  }
        }

        SolverParam solverParam(refineparam);
        solverParam.iter = 0;
        solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
        solverParam.tol = (param->tol_offset[i] > 0.0 ? param->tol_offset[i] : iter_tol); // set L2 tolerance
        solverParam.tol_hq = param->tol_hq_offset[i];                                     // set heavy quark tolerance
        solverParam.delta = param->reliable_delta_refinement;

        {
          CG cg(*m, *mSloppy, *mSloppy, solverParam, profileMulti);
          if (i==0)
            cg(*x[i], *b, p[i], r2_old[i]);
          else
            cg(*x[i], *b);
        }

        solverParam.true_res_offset[i] = solverParam.true_res;
        solverParam.true_res_hq_offset[i] = solverParam.true_res_hq;
        solverParam.updateInvertParam(*param,i);

        if (param->dslash_type == QUDA_ASQTAD_DSLASH ||
            param->dslash_type == QUDA_STAGGERED_DSLASH) {
          dirac.setMass(sqrt(param->offset[0]/4)); // restore just in case
          diracSloppy.setMass(sqrt(param->offset[0]/4)); // restore just in case
        }

        delete m;
        delete mSloppy;
      }
    }
  }

  // restore shifts -- avoid side effects
  for(int i=0; i < param->num_offset; i++) {
    param->offset[i] = unscaled_shifts[i];
  }

  profileMulti.TPSTART(QUDA_PROFILE_D2H);

  if (param->compute_action) {
    Complex action(0);
    for (int i=0; i<param->num_offset; i++) action += param->residue[i] * blas::cDotProduct(*b, *x[i]);
    param->action[0] = action.real();
    param->action[1] = action.imag();
  }

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

  if (!param->make_resident_solution) {
    for (auto v: solutionResident) if (v) delete v;
    solutionResident.clear();
  }

  profileMulti.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileMulti.TPSTART(QUDA_PROFILE_FREE);
  for(int i=0; i < param->num_offset; i++){
    delete h_x[i];
    //if (!param->make_resident_solution) delete x[i];
  }

  delete h_b;
  delete b;

  delete [] hp_x;

  delete d;
  delete dSloppy;
  delete dPre;
  delete dRefine;
  for (auto& pp : p) delete pp;

  profileMulti.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileMulti.TPSTOP(QUDA_PROFILE_TOTAL);

  profilerStop(__func__);
}

void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink, double *path_coeff, QudaGaugeParam *param) {

#ifdef GPU_FATLINK
  profileFatLink.TPSTART(QUDA_PROFILE_TOTAL);
  profileFatLink.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(param);

  if (ulink) {
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    quda::setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only,
				     svd_rel_error, svd_abs_error);
  }

  GaugeFieldParam gParam(fatlink, *param, QUDA_GENERAL_LINKS);
  cpuGaugeField cpuFatLink(gParam);   // create the host fatlink
  gParam.gauge = longlink;
  cpuGaugeField cpuLongLink(gParam);  // create the host longlink
  gParam.gauge = ulink;
  cpuGaugeField cpuUnitarizedLink(gParam);
  gParam.link_type = param->type;
  gParam.gauge     = inlink;
  cpuGaugeField cpuInLink(gParam);    // create the host sitelink

  // create the device fields
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  cudaGaugeField *cudaInLink = new cudaGaugeField(gParam);

  profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

  profileFatLink.TPSTART(QUDA_PROFILE_H2D);
  cudaInLink->loadCPUField(cpuInLink);
  profileFatLink.TPSTOP(QUDA_PROFILE_H2D);

  cudaGaugeField *cudaInLinkEx = createExtendedGauge(*cudaInLink, R, profileFatLink);

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  delete cudaInLink;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField *cudaFatLink = new cudaGaugeField(gParam);
  cudaGaugeField *cudaUnitarizedLink = ulink ? new cudaGaugeField(gParam) : nullptr;
  cudaGaugeField *cudaLongLink = longlink ? new cudaGaugeField(gParam) : nullptr;

  profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
  fatLongKSLink(cudaFatLink, cudaLongLink, *cudaInLinkEx, path_coeff);
  profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (ulink) {
    profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
    *num_failures_h = 0;
    quda::unitarizeLinks(*cudaUnitarizedLink, *cudaFatLink, num_failures_d); // unitarize on the gpu
    if (*num_failures_h>0) errorQuda("Error in unitarization component of the hisq fattening: %d failures\n", *num_failures_h);
    profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  profileFatLink.TPSTART(QUDA_PROFILE_D2H);
  if (ulink) cudaUnitarizedLink->saveCPUField(cpuUnitarizedLink);
  if (fatlink) cudaFatLink->saveCPUField(cpuFatLink);
  if (longlink) cudaLongLink->saveCPUField(cpuLongLink);
  profileFatLink.TPSTOP(QUDA_PROFILE_D2H);

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  delete cudaFatLink;
  if (longlink) delete cudaLongLink;
  if (ulink) delete cudaUnitarizedLink;
  delete cudaInLinkEx;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  profileFatLink.TPSTOP(QUDA_PROFILE_TOTAL);
#else
  errorQuda("Fat-link has not been built");
#endif // GPU_FATLINK
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
			  double* loop_coeff, int num_paths, int max_length, double eb3, QudaGaugeParam* qudaGaugeParam)
{
#ifdef GPU_GAUGE_FORCE
  profileGaugeForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(siteLink, *qudaGaugeParam);
  gParam.site_offset = qudaGaugeParam->gauge_offset;
  gParam.site_size = qudaGaugeParam->site_size;
  cpuGaugeField *cpuSiteLink = (!qudaGaugeParam->use_resident_gauge) ? new cpuGaugeField(gParam) : nullptr;

  cudaGaugeField* cudaSiteLink = nullptr;

  if (qudaGaugeParam->use_resident_gauge) {
    if (!gaugePrecise) errorQuda("No resident gauge field to use");
    cudaSiteLink = gaugePrecise;
  } else {
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = qudaGaugeParam->reconstruct;
    gParam.setPrecision(qudaGaugeParam->cuda_prec, true);

    cudaSiteLink = new cudaGaugeField(gParam);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);

    profileGaugeForce.TPSTART(QUDA_PROFILE_H2D);
    cudaSiteLink->loadCPUField(*cpuSiteLink);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_H2D);

    profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);
  }

  GaugeFieldParam gParamMom(mom, *qudaGaugeParam, QUDA_ASQTAD_MOM_LINKS);
  if (gParamMom.order == QUDA_TIFR_GAUGE_ORDER || gParamMom.order == QUDA_TIFR_PADDED_GAUGE_ORDER)
    gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  else
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;

  gParamMom.site_offset = qudaGaugeParam->mom_offset;
  gParamMom.site_size = qudaGaugeParam->site_size;
  cpuGaugeField* cpuMom = (!qudaGaugeParam->use_resident_mom) ? new cpuGaugeField(gParamMom) : nullptr;

  cudaGaugeField* cudaMom = nullptr;
  if (qudaGaugeParam->use_resident_mom) {
    if (!momResident) errorQuda("No resident momentum field to use");
    cudaMom = momResident;
    if (qudaGaugeParam->overwrite_mom) cudaMom->zero();
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);
  } else {
    gParamMom.create = qudaGaugeParam->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE;
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
    gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
    gParamMom.setPrecision(qudaGaugeParam->cuda_prec, true);
    gParamMom.create = QUDA_ZERO_FIELD_CREATE;
    cudaMom = new cudaGaugeField(gParamMom);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);
    if (!qudaGaugeParam->overwrite_mom) {
      profileGaugeForce.TPSTART(QUDA_PROFILE_H2D);
      cudaMom->loadCPUField(*cpuMom);
      profileGaugeForce.TPSTOP(QUDA_PROFILE_H2D);
    }
  }

  cudaGaugeField *cudaGauge = createExtendedGauge(*cudaSiteLink, R, profileGaugeForce);

  // actually do the computation
  profileGaugeForce.TPSTART(QUDA_PROFILE_COMPUTE);
  if (!forceMonitor()) {
    gaugeForce(*cudaMom, *cudaGauge, eb3, input_path_buf,  path_length, loop_coeff, num_paths, max_length);
  } else {
    // if we are monitoring the force, separate the force computation from the momentum update
    GaugeFieldParam gParam(*cudaMom);
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    GaugeField *force = GaugeField::Create(gParam);
    gaugeForce(*force, *cudaGauge, 1.0, input_path_buf,  path_length, loop_coeff, num_paths, max_length);
    updateMomentum(*cudaMom, eb3, *force, "gauge");
    delete force;
  }
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

  if (qudaGaugeParam->make_resident_gauge) {
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaGauge;
  } else {
    delete cudaGauge;
  }
  profileGaugeForce.TPSTOP(QUDA_PROFILE_FREE);

  profileGaugeForce.TPSTOP(QUDA_PROFILE_TOTAL);

  checkCudaError();
#else
  errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  return 0;
}

void momResidentQuda(void *mom, QudaGaugeParam *param)
{
  profileGaugeForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(param);

  GaugeFieldParam gParamMom(mom, *param, QUDA_ASQTAD_MOM_LINKS);
  if (gParamMom.order == QUDA_TIFR_GAUGE_ORDER || gParamMom.order == QUDA_TIFR_PADDED_GAUGE_ORDER)
    gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  else
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.site_offset = param->mom_offset;
  gParamMom.site_size = param->site_size;

  cpuGaugeField cpuMom(gParamMom);

  if (param->make_resident_mom && !param->return_result_mom) {
    if (momResident) delete momResident;

    gParamMom.create = QUDA_NULL_FIELD_CREATE;
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
    gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
    gParamMom.setPrecision(param->cuda_prec, true);
    gParamMom.create = QUDA_ZERO_FIELD_CREATE;
    momResident = new cudaGaugeField(gParamMom);
  } else if (param->return_result_mom && !param->make_resident_mom) {
    if (!momResident) errorQuda("No resident momentum to return");
  } else {
    errorQuda("Unexpected combination make_resident_mom = %d return_result_mom = %d", param->make_resident_mom,
              param->return_result_mom);
  }

  profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);

  if (param->make_resident_mom) {
    // we are downloading the momentum from the host
    profileGaugeForce.TPSTART(QUDA_PROFILE_H2D);
    momResident->loadCPUField(cpuMom);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_H2D);
  } else if (param->return_result_mom) {
    // we are uploading the momentum to the host
    profileGaugeForce.TPSTART(QUDA_PROFILE_D2H);
    momResident->saveCPUField(cpuMom);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_D2H);

    profileGaugeForce.TPSTART(QUDA_PROFILE_FREE);
    delete momResident;
    momResident = nullptr;
    profileGaugeForce.TPSTOP(QUDA_PROFILE_FREE);
  }

  profileGaugeForce.TPSTOP(QUDA_PROFILE_TOTAL);

  checkCudaError();
}

void createCloverQuda(QudaInvertParam* invertParam)
{
  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  if (!cloverPrecise) errorQuda("Clover field not allocated");

  QudaReconstructType recon = (gaugePrecise->Reconstruct() == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_12 : gaugePrecise->Reconstruct();
  // for clover we optimize to only send depth 1 halos in y/z/t (FIXME - make work for x, make robust in general)
  int R[4];
  for (int d=0; d<4; d++) R[d] = (d==0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  cudaGaugeField *gauge = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profileClover, false, recon);

  profileClover.TPSTART(QUDA_PROFILE_INIT);
  // create the Fmunu field
  GaugeFieldParam tensorParam(gaugePrecise->X(), gauge->Precision(), QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
  tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField Fmunu(tensorParam);
  profileClover.TPSTOP(QUDA_PROFILE_INIT);
  profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
  computeFmunu(Fmunu, *gauge);
  computeClover(*cloverPrecise, Fmunu, invertParam->clover_coeff, QUDA_CUDA_FIELD_LOCATION);
  profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);
  profileClover.TPSTOP(QUDA_PROFILE_TOTAL);

  // FIXME always preserve the extended gauge
  extendedGaugeResident = gauge;
}

void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
{
  GaugeFieldParam gParam(gauge, *param, QUDA_GENERAL_LINKS);
  gParam.geometry = static_cast<QudaFieldGeometry>(geometry);
  if (geometry != QUDA_SCALAR_GEOMETRY && geometry != QUDA_VECTOR_GEOMETRY)
    errorQuda("Only scalar and vector geometries are supported\n");

  cpuGaugeField *cpuGauge = nullptr;
  if (gauge) cpuGauge = new cpuGaugeField(gParam);

  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  auto* cudaGauge = new cudaGaugeField(gParam);

  if (gauge) {
    cudaGauge->loadCPUField(*cpuGauge);
    delete cpuGauge;
  }

  return cudaGauge;
}


void saveGaugeFieldQuda(void* gauge, void* inGauge, QudaGaugeParam* param){

  auto* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);

  GaugeFieldParam gParam(gauge, *param, QUDA_GENERAL_LINKS);
  gParam.geometry = cudaGauge->Geometry();

  cpuGaugeField cpuGauge(gParam);
  cudaGauge->saveCPUField(cpuGauge);

}


void destroyGaugeFieldQuda(void* gauge){
  auto* g = reinterpret_cast<cudaGaugeField*>(gauge);
  delete g;
}


void computeStaggeredForceQuda(void* h_mom, double dt, double delta, void *h_force, void **x,
			       QudaGaugeParam *gauge_param, QudaInvertParam *inv_param)
{
  profileStaggeredForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_INIT);

  GaugeFieldParam gParam(h_mom, *gauge_param, QUDA_ASQTAD_MOM_LINKS);

  // create the host momentum field
  gParam.reconstruct = gauge_param->reconstruct;
  gParam.t_boundary = QUDA_PERIODIC_T;
  cpuGaugeField cpuMom(gParam);

  // create the host momentum field
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.gauge = h_force;
  cpuGaugeField cpuForce(gParam);

  // create the device momentum field
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.create = QUDA_ZERO_FIELD_CREATE; // FIXME
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  cudaGaugeField *cudaMom = !gauge_param->use_resident_mom ? new cudaGaugeField(gParam) : nullptr;

  // create temporary field for quark-field outer product
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaGaugeField cudaForce(gParam);
  GaugeField *cudaForce_[2] = {&cudaForce};

  ColorSpinorParam qParam;
  qParam.location = QUDA_CUDA_FIELD_LOCATION;
  qParam.nColor = 3;
  qParam.nSpin = 1;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 5; // 5 since staggered mrhs
  qParam.setPrecision(gParam.Precision());
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = gParam.x[dir];
  qParam.x[4] = 1;
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_INIT);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_H2D);

  if (gauge_param->use_resident_mom) {
    if (!momResident) errorQuda("Cannot use resident momentum field since none appears resident");
    cudaMom = momResident;
  } else {
    // download the initial momentum (FIXME make an option just to return?)
    cudaMom->loadCPUField(cpuMom);
  }

  // resident gauge field is required
  if (!gauge_param->use_resident_gauge || !gaugePrecise)
    errorQuda("Resident gauge field is required");

  if (!gaugePrecise->StaggeredPhaseApplied()) {
    errorQuda("Gauge field requires the staggered phase factors to be applied");
  }

  // check if staggered phase is the desired one
  if (gauge_param->staggered_phase_type != gaugePrecise->StaggeredPhase()) {
    errorQuda("Requested staggered phase %d, but found %d\n",
              gauge_param->staggered_phase_type, gaugePrecise->StaggeredPhase());
  }

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_H2D);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_INIT);

  const int nvector = inv_param->num_offset;
  std::vector<ColorSpinorField*> X(nvector);
  for ( int i=0; i<nvector; i++) X[i] = ColorSpinorField::Create(qParam);

  if (inv_param->use_resident_solution) {
    if (solutionResident.size() < (unsigned int)nvector)
      errorQuda("solutionResident.size() %lu does not match number of shifts %d",
		solutionResident.size(), nvector);
  }

  // create the staggered operator
  DiracParam diracParam;
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  if (!pc_solve)
    errorQuda("Preconditioned solve type required not %d\n", inv_param->solve_type);
  setDiracParam(diracParam, inv_param, pc_solve);
  Dirac *dirac = Dirac::create(diracParam);

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_INIT);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_PREAMBLE);

  for (int i=0; i<nvector; i++) {
    ColorSpinorField &x = *(X[i]);

    if (inv_param->use_resident_solution) x.Even() = *(solutionResident[i]);
    else errorQuda("%s requires resident solution", __func__);

    // set the odd solution component
    dirac->Dslash(x.Odd(), x.Even(), QUDA_ODD_PARITY);
  }

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_FREE);

#if 0
  if (inv_param->use_resident_solution) {
    for (auto v : solutionResident) if (v) delete solutionResident[i];
    solutionResident.clear();
  }
#endif
  delete dirac;

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_FREE);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_COMPUTE);

  // compute quark-field outer product
  for (int i=0; i<nvector; i++) {
    ColorSpinorField &x = *(X[i]);
    // second component is zero since we have no three hop term
    double coeff[2] = {inv_param->residue[i], 0.0};

    // Operate on even-parity sites
    computeStaggeredOprod(cudaForce_, x, coeff, 1);
  }

  // mom += delta * [U * force]TA
  applyU(cudaForce, *gaugePrecise);
  updateMomentum(*cudaMom, dt * delta, cudaForce, "staggered");
  qudaDeviceSynchronize();

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_COMPUTE);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_D2H);

  if (gauge_param->return_result_mom) {
    // copy the momentum field back to the host
    cudaMom->saveCPUField(cpuMom);
  }

  if (gauge_param->make_resident_mom) {
    // make the momentum field resident
    momResident = cudaMom;
  } else {
    delete cudaMom;
  }

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_D2H);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_FREE);

  for (int i=0; i<nvector; i++) delete X[i];

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_FREE);
  profileStaggeredForce.TPSTOP(QUDA_PROFILE_TOTAL);

  checkCudaError();
}

void computeHISQForceQuda(void* const milc_momentum,
                          double dt,
                          const double level2_coeff[6],
                          const double fat7_coeff[6],
                          const void* const w_link,
                          const void* const v_link,
                          const void* const u_link,
                          void **fermion,
                          int num_terms,
                          int num_naik_terms,
                          double **coeff,
                          QudaGaugeParam* gParam)
{
#ifdef  GPU_STAGGERED_OPROD
  using namespace quda;
  using namespace quda::fermion_force;
  profileHISQForce.TPSTART(QUDA_PROFILE_TOTAL);
  if (gParam->gauge_order != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported input field order %d", gParam->gauge_order);

  checkGaugeParam(gParam);

  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);

  // create the device outer-product field
  GaugeFieldParam oParam(0, *gParam, QUDA_GENERAL_LINKS);
  oParam.nFace = 0;
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaGaugeField *stapleOprod = new cudaGaugeField(oParam);
  cudaGaugeField *oneLinkOprod = new cudaGaugeField(oParam);
  cudaGaugeField *naikOprod = new cudaGaugeField(oParam);

  {
    // default settings for the unitarization
    const double unitarize_eps = 1e-14;
    const double hisq_force_filter = 5e-5;
    const double max_det_error = 1e-10;
    const bool   allow_svd = true;
    const bool   svd_only = false;
    const double svd_rel_err = 1e-8;
    const double svd_abs_err = 1e-8;

    setUnitarizeForceConstants(unitarize_eps, hisq_force_filter, max_det_error, allow_svd, svd_only, svd_rel_err, svd_abs_err);
  }

  double act_path_coeff[6] = {0,1,level2_coeff[2],level2_coeff[3],level2_coeff[4],level2_coeff[5]};
  // You have to look at the MILC routine to understand the following
  // Basically, I have already absorbed the one-link coefficient

  GaugeFieldParam param(milc_momentum, *gParam, QUDA_ASQTAD_MOM_LINKS);
  //param.nFace = 0;
  param.order  = QUDA_MILC_GAUGE_ORDER;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.ghostExchange =  QUDA_GHOST_EXCHANGE_NO;
  cpuGaugeField* cpuMom = (!gParam->use_resident_mom) ? new cpuGaugeField(param) : nullptr;

  param.link_type = QUDA_GENERAL_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.gauge = (void*)w_link;
  cpuGaugeField cpuWLink(param);
  param.gauge = (void*)v_link;
  cpuGaugeField cpuVLink(param);
  param.gauge = (void*)u_link;
  cpuGaugeField cpuULink(param);

  param.create = QUDA_ZERO_FIELD_CREATE;
  param.order  = QUDA_FLOAT2_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  GaugeFieldParam momParam(param);

  param.create = QUDA_ZERO_FIELD_CREATE;
  param.link_type = QUDA_GENERAL_LINKS;
  param.setPrecision(gParam->cpu_prec, true);

  int R[4] = { 2*comm_dim_partitioned(0), 2*comm_dim_partitioned(1), 2*comm_dim_partitioned(2), 2*comm_dim_partitioned(3) };
  for (int dir=0; dir<4; ++dir) {
    param.x[dir] += 2*R[dir];
    param.r[dir] = R[dir];
  }

  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.setPrecision(gParam->cpu_prec);
  param.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;

  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  { // do outer-product computation
    ColorSpinorParam qParam;
    qParam.nColor = 3;
    qParam.nSpin = 1;
    qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qParam.nDim = 4;
    qParam.setPrecision(oParam.Precision());
    qParam.pad = 0;
    for (int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];

    // create the device quark field
    qParam.create = QUDA_NULL_FIELD_CREATE;
    qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    cudaColorSpinorField cudaQuark(qParam);

    // create the host quark field
    qParam.create = QUDA_REFERENCE_FIELD_CREATE;
    qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
    qParam.v = fermion[0];

    { // regular terms
      GaugeField *oprod[2] = {stapleOprod, naikOprod};

      // loop over different quark fields
      for(int i=0; i<num_terms; ++i){

        // Wrap the MILC quark field
        profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
        qParam.v = fermion[i];
        cpuColorSpinorField cpuQuark(qParam); // create host quark field
        profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

        profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
        cudaQuark = cpuQuark;
        profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);

        profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
        computeStaggeredOprod(oprod, cudaQuark, coeff[i], 3);
        profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    }

    { // naik terms
      oneLinkOprod->copy(*stapleOprod);
      ax(level2_coeff[0], *oneLinkOprod);
      GaugeField *oprod[2] = {oneLinkOprod, naikOprod};

      // loop over different quark fields
      for(int i=0; i<num_naik_terms; ++i){

        // Wrap the MILC quark field
        profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
        qParam.v = fermion[i + num_terms - num_naik_terms];
        cpuColorSpinorField cpuQuark(qParam); // create host quark field
        profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

        profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
        cudaQuark = cpuQuark;
        profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);

        profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
        computeStaggeredOprod(oprod, cudaQuark, coeff[i + num_terms], 3);
        profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    }
  }

  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
  cudaGaugeField* cudaInForce = new cudaGaugeField(param);
  copyExtendedGauge(*cudaInForce, *stapleOprod, QUDA_CUDA_FIELD_LOCATION);
  delete stapleOprod;

  cudaGaugeField* cudaOutForce = new cudaGaugeField(param);
  copyExtendedGauge(*cudaOutForce, *oneLinkOprod, QUDA_CUDA_FIELD_LOCATION);
  delete oneLinkOprod;

  cudaGaugeField* cudaGauge = new cudaGaugeField(param);
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  cudaGauge->loadCPUField(cpuWLink, profileHISQForce);

  cudaInForce->exchangeExtendedGhost(R,profileHISQForce,true);
  cudaGauge->exchangeExtendedGhost(R,profileHISQForce,true);
  cudaOutForce->exchangeExtendedGhost(R,profileHISQForce,true);

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForce(*cudaOutForce, *cudaInForce, *cudaGauge, act_path_coeff);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Load naik outer product
  copyExtendedGauge(*cudaInForce, *naikOprod, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce->exchangeExtendedGhost(R,profileHISQForce,true);
  delete naikOprod;

  // Compute Naik three-link term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqLongLinkForce(*cudaOutForce, *cudaInForce, *cudaGauge, act_path_coeff[1]);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  cudaOutForce->exchangeExtendedGhost(R,profileHISQForce,true);

  // load v-link
  cudaGauge->loadCPUField(cpuVLink, profileHISQForce);
  cudaGauge->exchangeExtendedGhost(R,profileHISQForce,true);

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  *num_failures_h = 0;
  unitarizeForce(*cudaInForce, *cudaOutForce, *cudaGauge, num_failures_d);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (*num_failures_h>0) errorQuda("Error in the unitarization component of the hisq fermion force: %d failures\n", *num_failures_h);

  cudaMemset((void**)(cudaOutForce->Gauge_p()), 0, cudaOutForce->Bytes());

  // read in u-link
  cudaGauge->loadCPUField(cpuULink, profileHISQForce);
  cudaGauge->exchangeExtendedGhost(R,profileHISQForce,true);

  // Compute Fat7-staple term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForce(*cudaOutForce, *cudaInForce, *cudaGauge, fat7_coeff);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  delete cudaInForce;
  cudaGaugeField* cudaMom = new cudaGaugeField(momParam);

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqCompleteForce(*cudaOutForce, *cudaGauge);
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (gParam->use_resident_mom) {
    if (!momResident) errorQuda("No resident momentum field to use");
    updateMomentum(*momResident, dt, *cudaOutForce, "hisq");
  } else {
    updateMomentum(*cudaMom, dt, *cudaOutForce, "hisq");
  }

  if (gParam->return_result_mom) {
    // Close the paths, make anti-hermitian, and store in compressed format
    if (gParam->return_result_mom) cudaMom->saveCPUField(*cpuMom, profileHISQForce);
  }

  profileHISQForce.TPSTART(QUDA_PROFILE_FREE);

  if (cpuMom) delete cpuMom;

  if (!gParam->make_resident_mom) {
    delete momResident;
    momResident = nullptr;
  }
  if (cudaMom) delete cudaMom;
  delete cudaOutForce;
  delete cudaGauge;
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);

  profileHISQForce.TPSTOP(QUDA_PROFILE_TOTAL);

#else
  errorQuda("HISQ force has not been built");
#endif
}

void computeCloverForceQuda(void *h_mom, double dt, void **h_x, void **h_p,
			    double *coeff, double kappa2, double ck,
			    int nvector, double multiplicity, void *gauge,
			    QudaGaugeParam *gauge_param, QudaInvertParam *inv_param) {

  using namespace quda;
  profileCloverForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileCloverForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(gauge_param);
  if (!gaugePrecise) errorQuda("No resident gauge field");

  GaugeFieldParam fParam(h_mom, *gauge_param, QUDA_ASQTAD_MOM_LINKS);
  // create the host momentum field
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.order = gauge_param->gauge_order;
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
  qParam.location = QUDA_CUDA_FIELD_LOCATION;
  qParam.nColor = 3;
  qParam.nSpin = 4;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.setPrecision(fParam.Precision());
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = fParam.x[dir];

  // create the device quark field
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField*> quarkX, quarkP;
  for (int i=0; i<nvector; i++) {
    quarkX.push_back(ColorSpinorField::Create(qParam));
    quarkP.push_back(ColorSpinorField::Create(qParam));
  }

  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.x[0] /= 2;
  cudaColorSpinorField tmp(qParam);

  // create the host quark field
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // need expose this to interface

  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc_solve);
  diracParam.tmp1 = &tmp; // use as temporary for dirac->M
  Dirac *dirac = Dirac::create(diracParam);

  if (inv_param->use_resident_solution) {
    if (solutionResident.size() < (unsigned int)nvector)
      errorQuda("solutionResident.size() %lu does not match number of shifts %d",
		solutionResident.size(), nvector);
  }

  cudaGaugeField &gaugeEx = *extendedGaugeResident;

  // create oprod and trace fields
  fParam.geometry = QUDA_TENSOR_GEOMETRY;
  cudaGaugeField oprod(fParam);

  profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);
  profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);

  std::vector<double> force_coeff(nvector);
  // loop over different quark fields
  for(int i=0; i<nvector; i++){
    ColorSpinorField &x = *(quarkX[i]);
    ColorSpinorField &p = *(quarkP[i]);

    if (!inv_param->use_resident_solution) {
      // for downloading x_e
      qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      qParam.x[0] /= 2;

      // Wrap the even-parity MILC quark field
      profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);
      profileCloverForce.TPSTART(QUDA_PROFILE_INIT);
      qParam.v = h_x[i];
      cpuColorSpinorField cpuQuarkX(qParam); // create host quark field
      profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);

      profileCloverForce.TPSTART(QUDA_PROFILE_H2D);
      x.Even() = cpuQuarkX;
      profileCloverForce.TPSTOP(QUDA_PROFILE_H2D);

      profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);
      gamma5(x.Even(), x.Even());
    } else {
      x.Even() = *(solutionResident[i]);
    }

    dirac->Dslash(x.Odd(), x.Even(), QUDA_ODD_PARITY);
    dirac->M(p.Even(), x.Even());
    dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(p.Odd(), p.Even(), QUDA_ODD_PARITY);
    dirac->Dagger(QUDA_DAG_NO);

    gamma5(x, x);
    gamma5(p, p);

    force_coeff[i] = 2.0*dt*coeff[i]*kappa2;
  }

  computeCloverForce(cudaForce, *gaugePrecise, quarkX, quarkP, force_coeff);

  // In double precision the clover derivative is faster with no reconstruct
  cudaGaugeField *u = &gaugeEx;
  if (gaugeEx.Reconstruct() == QUDA_RECONSTRUCT_12 && gaugeEx.Precision() == QUDA_DOUBLE_PRECISION) {
    GaugeFieldParam param(gaugeEx);
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    u = new cudaGaugeField(param);
    u -> copy(gaugeEx);
  }

  computeCloverSigmaTrace(oprod, *cloverPrecise, 2.0*ck*multiplicity*dt);

  /* Now the U dA/dU terms */
  std::vector< std::vector<double> > ferm_epsilon(nvector);
  for (int shift = 0; shift < nvector; shift++) {
    ferm_epsilon[shift].reserve(2);
    ferm_epsilon[shift][0] = 2.0*ck*coeff[shift]*dt;
    ferm_epsilon[shift][1] = -kappa2 * 2.0*ck*coeff[shift]*dt;
  }

  computeCloverSigmaOprod(oprod, quarkX, quarkP, ferm_epsilon);

  cudaGaugeField *oprodEx = createExtendedGauge(oprod, R, profileCloverForce);

  profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);

  cloverDerivative(cudaForce, *u, *oprodEx, 1.0, QUDA_ODD_PARITY);
  cloverDerivative(cudaForce, *u, *oprodEx, 1.0, QUDA_EVEN_PARITY);

  if (u != &gaugeEx) delete u;

  updateMomentum(cudaMom, -1.0, cudaForce, "clover");
  profileCloverForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // copy the outer product field back to the host
  profileCloverForce.TPSTART(QUDA_PROFILE_D2H);
  cudaMom.saveCPUField(cpuMom);
  profileCloverForce.TPSTOP(QUDA_PROFILE_D2H);

  profileCloverForce.TPSTART(QUDA_PROFILE_FREE);

  for (int i=0; i<nvector; i++) {
    delete quarkX[i];
    delete quarkP[i];
  }

#if 0
  if (inv_param->use_resident_solution) {
    for (auto v : solutionResident) if (v) delete v;
    solutionResident.clear();
  }
#endif
  delete dirac;
  profileCloverForce.TPSTOP(QUDA_PROFILE_FREE);

  checkCudaError();
  profileCloverForce.TPSTOP(QUDA_PROFILE_TOTAL);
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

  // create the host fields
  GaugeFieldParam gParam(gauge, *param, QUDA_SU3_LINKS);
  gParam.site_offset = param->gauge_offset;
  gParam.site_size = param->site_size;
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

  GaugeFieldParam gParamMom(momentum, *param);
  gParamMom.reconstruct = (gParamMom.order == QUDA_TIFR_GAUGE_ORDER || gParamMom.order == QUDA_TIFR_PADDED_GAUGE_ORDER) ?
   QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParamMom.site_offset = param->mom_offset;
  gParamMom.site_size = param->site_size;
  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParamMom) : nullptr;

  // create the device fields
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.pad = 0;
  cudaGaugeField *cudaMom = !param->use_resident_mom ? new cudaGaugeField(gParam) : nullptr;

  gParam.link_type = QUDA_SU3_LINKS;
  gParam.reconstruct = param->reconstruct;
  cudaGaugeField *cudaInGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : nullptr;
  auto *cudaOutGauge = new cudaGaugeField(gParam);

  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_INIT);

  profileGaugeUpdate.TPSTART(QUDA_PROFILE_H2D);

  if (!param->use_resident_gauge) {   // load fields onto the device
    cudaInGauge->loadCPUField(*cpuGauge);
  } else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = nullptr;
  }

  if (!param->use_resident_mom) {
    cudaMom->loadCPUField(*cpuMom);
  } else {
    if (!momResident) errorQuda("No resident mom field allocated");
    cudaMom = momResident;
    momResident = nullptr;
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
    if (gaugePrecise != nullptr) delete gaugePrecise;
    gaugePrecise = cudaOutGauge;
  } else {
    delete cudaOutGauge;
  }

  if (param->make_resident_mom) {
    if (momResident != nullptr && momResident != cudaMom) delete momResident;
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
}

 void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param) {
   profileProject.TPSTART(QUDA_PROFILE_TOTAL);

   profileProject.TPSTART(QUDA_PROFILE_INIT);
   checkGaugeParam(param);

   // create the gauge field
   GaugeFieldParam gParam(gauge_h, *param, QUDA_GENERAL_LINKS);
   gParam.site_offset = param->gauge_offset;
   gParam.site_size = param->site_size;
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

   // create the device fields
   gParam.create = QUDA_NULL_FIELD_CREATE;
   gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
   gParam.reconstruct = param->reconstruct;
   cudaGaugeField *cudaGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : nullptr;
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
     if (gaugePrecise != nullptr && cudaGauge != gaugePrecise) delete gaugePrecise;
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
   GaugeFieldParam gParam(gauge_h, *param, QUDA_GENERAL_LINKS);
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

   // create the device fields
   gParam.create = QUDA_NULL_FIELD_CREATE;
   gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
   gParam.reconstruct = param->reconstruct;
   cudaGaugeField *cudaGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : nullptr;
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
     if (gaugePrecise != nullptr && cudaGauge != gaugePrecise) delete gaugePrecise;
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
  GaugeFieldParam gParam(momentum, *param, QUDA_ASQTAD_MOM_LINKS);
  gParam.reconstruct = (gParam.order == QUDA_TIFR_GAUGE_ORDER || gParam.order == QUDA_TIFR_PADDED_GAUGE_ORDER) ?
    QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParam.site_offset = param->mom_offset;
  gParam.site_size = param->site_size;

  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParam) : nullptr;

  // create the device fields
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;

  cudaGaugeField *cudaMom = !param->use_resident_mom ? new cudaGaugeField(gParam) : nullptr;

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
    if (momResident != nullptr && momResident != cudaMom) delete momResident;
    momResident = cudaMom;
  } else {
    delete cudaMom;
    momResident = nullptr;
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
  fflush(stdout);
  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;
  invertQuda(hp_x, hp_b, param);
  fflush(stdout);
}

void invert_multishift_quda_(void *h_x, void *hp_b, QudaInvertParam *param) {
  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;

  if (!gaugePrecise) errorQuda("Resident gauge field not allocated");

  // get data into array of pointers
  int nSpin = (param->dslash_type == QUDA_STAGGERED_DSLASH || param->dslash_type == QUDA_ASQTAD_DSLASH) ? 1 : 4;

  // compute offset assuming TIFR padded ordering (FIXME)
  if (param->dirac_order != QUDA_TIFR_PADDED_DIRAC_ORDER)
    errorQuda("Fortran multi-shift solver presently only supports QUDA_TIFR_PADDED_DIRAC_ORDER");

  const int *X = gaugePrecise->X();
  size_t cb_offset = (X[0]/2) * X[1] * (X[2] + 4) * X[3] * gaugePrecise->Ncolor() * nSpin * 2 * param->cpu_prec;
  void *hp_x[QUDA_MAX_MULTI_SHIFT];
  for (int i=0; i<param->num_offset; i++) hp_x[i] = static_cast<char*>(h_x) + i*cb_offset;

  invertMultiShiftQuda(hp_x, hp_b, param);
}

void flush_chrono_quda_(int *index) { flushChronoQuda(*index); }

void register_pinned_quda_(void *ptr, size_t *bytes) {
  cudaHostRegister(ptr, *bytes, cudaHostRegisterDefault);
  checkCudaError();
}

void unregister_pinned_quda_(void *ptr) {
  cudaHostUnregister(ptr);
  checkCudaError();
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

static inline int opp(int dir) { return 7-dir; }

static void createGaugeForcePaths(int **paths, int dir, int num_loop_types){

  int index=0;
  // Plaquette paths
  if (num_loop_types >= 1)
    for(int i=0; i<4; ++i){
      if(i==dir) continue;
      paths[index][0] = i;        paths[index][1] = opp(dir);   paths[index++][2] = opp(i);
      paths[index][0] = opp(i);   paths[index][1] = opp(dir);   paths[index++][2] = i;
    }

  // Rectangle Paths
  if (num_loop_types >= 2)
    for(int i=0; i<4; ++i){
      if(i==dir) continue;
      paths[index][0] = paths[index][1] = i;       paths[index][2] = opp(dir); paths[index][3] = paths[index][4] = opp(i);
      index++;
      paths[index][0] = paths[index][1] = opp(i);  paths[index][2] = opp(dir); paths[index][3] = paths[index][4] = i;
      index++;
      paths[index][0] = dir; paths[index][1] = i; paths[index][2] = paths[index][3] = opp(dir); paths[index][4] = opp(i);
      index++;
      paths[index][0] = dir; paths[index][1] = opp(i); paths[index][2] = paths[index][3] = opp(dir); paths[index][4] = i;
      index++;
      paths[index][0] = i;  paths[index][1] = paths[index][2] = opp(dir); paths[index][3] = opp(i); paths[index][4] = dir;
      index++;
      paths[index][0] = opp(i);  paths[index][1] = paths[index][2] = opp(dir); paths[index][3] = i; paths[index][4] = dir;
      index++;
    }

  if (num_loop_types >= 3) {
    // Staple paths
    for(int i=0; i<4; ++i){
      for(int j=0; j<4; ++j){
	if(i==dir || j==dir || i==j) continue;
	paths[index][0] = i; paths[index][1] = j; paths[index][2] = opp(dir); paths[index][3] = opp(i), paths[index][4] = opp(j);
	index++;
	paths[index][0] = i; paths[index][1] = opp(j); paths[index][2] = opp(dir); paths[index][3] = opp(i), paths[index][4] = j;
	index++;
	paths[index][0] = opp(i); paths[index][1] = j; paths[index][2] = opp(dir); paths[index][3] = i, paths[index][4] = opp(j);
	index++;
	paths[index][0] = opp(i); paths[index][1] = opp(j); paths[index][2] = opp(dir); paths[index][3] = i, paths[index][4] = j;
	index++;
     }
    }
  }

}

void compute_gauge_force_quda_(void *mom, void *gauge, int *num_loop_types, double *coeff, double *dt,
			       QudaGaugeParam *param) {

  int numPaths = 0;
  switch (*num_loop_types) {
  case 1:
    numPaths = 6;
    break;
  case 2:
    numPaths = 24;
    break;
  case 3:
    numPaths = 48;
    break;
  default:
    errorQuda("Invalid num_loop_types = %d\n", *num_loop_types);
  }

  auto *loop_coeff = static_cast<double*>(safe_malloc(numPaths*sizeof(double)));
  int *path_length = static_cast<int*>(safe_malloc(numPaths*sizeof(int)));

  if (*num_loop_types >= 1) for(int i= 0; i< 6; ++i) {
      loop_coeff[i] = coeff[0];
      path_length[i] = 3;
    }
  if (*num_loop_types >= 2) for(int i= 6; i<24; ++i) {
      loop_coeff[i] = coeff[1];
      path_length[i] = 5;
    }
  if (*num_loop_types >= 3) for(int i=24; i<48; ++i) {
      loop_coeff[i] = coeff[2];
      path_length[i] = 5;
    }

  int** input_path_buf[4];
  for(int dir=0; dir<4; ++dir){
    input_path_buf[dir] = static_cast<int**>(safe_malloc(numPaths*sizeof(int*)));
    for(int i=0; i<numPaths; ++i){
      input_path_buf[dir][i] = static_cast<int*>(safe_malloc(path_length[i]*sizeof(int)));
    }
    createGaugeForcePaths(input_path_buf[dir], dir, *num_loop_types);
  }

  int max_length = 6;

  computeGaugeForceQuda(mom, gauge, input_path_buf, path_length, loop_coeff, numPaths, max_length, *dt, param);

  for(auto & dir : input_path_buf){
    for(int i=0; i<numPaths; ++i) host_free(dir[i]);
    host_free(dir);
  }

  host_free(path_length);
  host_free(loop_coeff);
}

void compute_staggered_force_quda_(void* h_mom, double *dt, double *delta, void *gauge, void *x, QudaGaugeParam *gauge_param, QudaInvertParam *inv_param) {
  computeStaggeredForceQuda(h_mom, *dt, *delta, gauge, (void**)x, gauge_param, inv_param);
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
  qudaDeviceSynchronize();
}

// evaluate the kinetic term
void kinetic_quda_(double *kin, void* momentum, QudaGaugeParam* param) {
  *kin = momActionQuda(momentum, param);
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

void gaussGaugeQuda(unsigned long long seed, double sigma)
{
  profileGauss.TPSTART(QUDA_PROFILE_TOTAL);

  if (!gaugePrecise) errorQuda("Cannot generate Gauss GaugeField as there is no resident gauge field");

  cudaGaugeField *data = gaugePrecise;

  GaugeFieldParam param(*data);
  param.reconstruct = QUDA_RECONSTRUCT_12;
  param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField u(param);

  profileGauss.TPSTART(QUDA_PROFILE_COMPUTE);
  quda::gaugeGauss(*data, seed, sigma);
  profileGauss.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (extendedGaugeResident) {
    *extendedGaugeResident = *gaugePrecise;
    extendedGaugeResident->exchangeExtendedGhost(R, profileGauss, redundant_comms);
  }

  profileGauss.TPSTOP(QUDA_PROFILE_TOTAL);
}


/*
 * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
 */
void plaq_quda_(double plaq[3]) {
  plaqQuda(plaq);
}

void plaqQuda(double plaq[3])
{
  profilePlaq.TPSTART(QUDA_PROFILE_TOTAL);

  if (!gaugePrecise) errorQuda("Cannot compute plaquette as there is no resident gauge field");

  cudaGaugeField *data = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profilePlaq);
  extendedGaugeResident = data;

  profilePlaq.TPSTART(QUDA_PROFILE_COMPUTE);
  double3 plaq3 = quda::plaquette(*data);
  plaq[0] = plaq3.x;
  plaq[1] = plaq3.y;
  plaq[2] = plaq3.z;
  profilePlaq.TPSTOP(QUDA_PROFILE_COMPUTE);

  profilePlaq.TPSTOP(QUDA_PROFILE_TOTAL);
}

/*
 * Performs a deep copy from the internal extendedGaugeResident field.
 */
void copyExtendedResidentGaugeQuda(void* resident_gauge, QudaFieldLocation loc)
{
  //profilePlaq.TPSTART(QUDA_PROFILE_TOTAL);

  if (!gaugePrecise) errorQuda("Cannot perform deep copy of resident gauge field as there is no resident gauge field");

  cudaGaugeField *data = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profilePlaq);
  extendedGaugeResident = data;

  auto* io_gauge = (cudaGaugeField*)resident_gauge;

  copyExtendedGauge(*io_gauge, *extendedGaugeResident, loc);

  //profilePlaq.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *inv_param, unsigned int n_steps, double alpha)
{
  profileWuppertal.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  cudaGaugeField *precise = nullptr;

  if (gaugeSmeared != nullptr) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Wuppertal smearing done with gaugeSmeared\n");
    GaugeFieldParam gParam(*gaugePrecise);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    precise = new cudaGaugeField(gParam);
    copyExtendedGauge(*precise, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    precise->exchangeGhost();
  } else {
    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("Wuppertal smearing done with gaugePrecise\n");
    precise = gaugePrecise;
  }

  ColorSpinorParam cpuParam(h_in, *inv_param, precise->X(), false, inv_param->input_location);
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
  int parity = 0;

  // Computes out(x) = 1/(1+6*alpha)*(in(x) + alpha*\sum_mu (U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)))
  double a = alpha/(1.+6.*alpha);
  double b = 1./(1.+6.*alpha);

  for (unsigned int i = 0; i < n_steps; i++) {
    if (i) in = out;
    ApplyLaplace(out, in, *precise, 3, a, b, in, parity, false, nullptr, profileWuppertal);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      double norm = blas::norm2(out);
      printfQuda("Step %d, vector norm %e\n", i, norm);
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

  if (gaugeSmeared != nullptr)
    delete precise;

  delete out_h;
  delete in_h;

  popVerbosity();

  profileWuppertal.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performAPEnStep(unsigned int n_steps, double alpha, int meas_interval)
{
  profileAPE.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (gaugeSmeared != nullptr) delete gaugeSmeared;
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileAPE);

  GaugeFieldParam gParam(*gaugeSmeared);
  auto *cudaGaugeTemp = new cudaGaugeField(gParam);

  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    gaugeObservablesQuda(&param);
    printfQuda("Q charge at step %03d = %+.16e\n", 0, param.qcharge);
  }

  for (unsigned int i = 0; i < n_steps; i++) {
    profileAPE.TPSTART(QUDA_PROFILE_COMPUTE);
    APEStep(*gaugeSmeared, *cudaGaugeTemp, alpha);
    profileAPE.TPSTOP(QUDA_PROFILE_COMPUTE);
    if ((i + 1) % meas_interval == 0 && getVerbosity() >= QUDA_VERBOSE) {
      gaugeObservablesQuda(&param);
      printfQuda("Q charge at step %03d = %+.16e\n", i + 1, param.qcharge);
    }
  }

  delete cudaGaugeTemp;
  profileAPE.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performSTOUTnStep(unsigned int n_steps, double rho, int meas_interval)
{
  profileSTOUT.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (gaugeSmeared != nullptr) delete gaugeSmeared;
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileSTOUT);

  GaugeFieldParam gParam(*gaugeSmeared);
  auto *cudaGaugeTemp = new cudaGaugeField(gParam);

  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    gaugeObservablesQuda(&param);
    printfQuda("Q charge at step %03d = %+.16e\n", 0, param.qcharge);
  }

  for (unsigned int i = 0; i < n_steps; i++) {
    profileSTOUT.TPSTART(QUDA_PROFILE_COMPUTE);
    STOUTStep(*gaugeSmeared, *cudaGaugeTemp, rho);
    profileSTOUT.TPSTOP(QUDA_PROFILE_COMPUTE);
    if ((i + 1) % meas_interval == 0 && getVerbosity() >= QUDA_VERBOSE) {
      gaugeObservablesQuda(&param);
      printfQuda("Q charge at step %03d = %+.16e\n", i + 1, param.qcharge);
    }
  }

  delete cudaGaugeTemp;
  profileSTOUT.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performOvrImpSTOUTnStep(unsigned int n_steps, double rho, double epsilon, int meas_interval)
{
  profileOvrImpSTOUT.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (gaugeSmeared != nullptr) delete gaugeSmeared;
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileOvrImpSTOUT);

  GaugeFieldParam gParam(*gaugeSmeared);
  auto *cudaGaugeTemp = new cudaGaugeField(gParam);

  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    gaugeObservablesQuda(&param);
    printfQuda("Q charge at step %03d = %+.16e\n", 0, param.qcharge);
  }

  for (unsigned int i = 0; i < n_steps; i++) {
    profileOvrImpSTOUT.TPSTART(QUDA_PROFILE_COMPUTE);
    OvrImpSTOUTStep(*gaugeSmeared, *cudaGaugeTemp, rho, epsilon);
    profileOvrImpSTOUT.TPSTOP(QUDA_PROFILE_COMPUTE);
    if ((i + 1) % meas_interval == 0 && getVerbosity() >= QUDA_VERBOSE) {
      gaugeObservablesQuda(&param);
      printfQuda("Q charge at step %03d = %+.16e\n", i + 1, param.qcharge);
    }
  }

  delete cudaGaugeTemp;
  profileOvrImpSTOUT.TPSTOP(QUDA_PROFILE_TOTAL);
}

void performWFlownStep(unsigned int n_steps, double step_size, int meas_interval, QudaWFlowType wflow_type)
{
  pushOutputPrefix("performWFlownStep: ");
  profileWFlow.TPSTART(QUDA_PROFILE_TOTAL);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (gaugeSmeared != nullptr) delete gaugeSmeared;
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileWFlow);

  GaugeFieldParam gParamEx(*gaugeSmeared);
  auto *gaugeAux = GaugeField::Create(gParamEx);

  GaugeFieldParam gParam(*gaugePrecise);
  gParam.reconstruct = QUDA_RECONSTRUCT_NO; // temporary field is not on manifold so cannot use reconstruct
  auto *gaugeTemp = GaugeField::Create(gParam);

  GaugeField *in = gaugeSmeared;
  GaugeField *out = gaugeAux;

  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_plaquette = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    gaugeObservables(*in, param, profileWFlow);
    printfQuda("flow t, plaquette, E_tot, E_spatial, E_temporal, Q charge\n");
    printfQuda("%le %.16e %+.16e %+.16e %+.16e %+.16e\n", 0.0, param.plaquette[0], param.energy[0], param.energy[1],
               param.energy[2], param.qcharge);
  }

  for (unsigned int i = 0; i < n_steps; i++) {
    // Perform W1, W2, and Vt Wilson Flow steps as defined in
    // https://arxiv.org/abs/1006.4518v3
    profileWFlow.TPSTART(QUDA_PROFILE_COMPUTE);
    if (i > 0) std::swap(in, out); // output from prior step becomes input for next step

    WFlowStep(*out, *gaugeTemp, *in, step_size, wflow_type);
    profileWFlow.TPSTOP(QUDA_PROFILE_COMPUTE);

    if ((i + 1) % meas_interval == 0 && getVerbosity() >= QUDA_SUMMARIZE) {
      gaugeObservables(*out, param, profileWFlow);
      printfQuda("%le %.16e %+.16e %+.16e %+.16e %+.16e\n", step_size * (i + 1), param.plaquette[0], param.energy[0],
                 param.energy[1], param.energy[2], param.qcharge);
    }
  }

  delete gaugeTemp;
  delete gaugeAux;
  profileWFlow.TPSTOP(QUDA_PROFILE_TOTAL);
  popOutputPrefix();
}

int computeGaugeFixingOVRQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                              const unsigned int verbose_interval, const double relax_boost, const double tolerance,
                              const unsigned int reunit_interval, const unsigned int stopWtheta, QudaGaugeParam *param,
                              double *timeinfo)
{
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_INIT);
  GaugeFieldParam gParam(gauge, *param);
  auto *cpuGauge = new cpuGaugeField(gParam);

  // gParam.pad = getFatLinkPadding(param->X);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  auto *cudaInGauge = new cudaGaugeField(gParam);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_INIT);
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_H2D);

  ///if (!param->use_resident_gauge) {   // load fields onto the device
  cudaInGauge->loadCPUField(*cpuGauge);
 /* } else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = nullptr;
  } */

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_H2D);

  checkCudaError();

  if (comm_size() == 1) {
    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugefixingOVR(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, \
      reunit_interval, stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);
  } else {
    cudaGaugeField *cudaInGaugeEx = createExtendedGauge(*cudaInGauge, R, GaugeFixOVRQuda);

    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugefixingOVR(*cudaInGaugeEx, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, \
      reunit_interval, stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);

    //HOW TO COPY BACK TO CPU: cudaInGaugeEx->cpuGauge
    copyExtendedGauge(*cudaInGauge, *cudaInGaugeEx, QUDA_CUDA_FIELD_LOCATION);
  }

  checkCudaError();
  // copy the gauge field back to the host
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_D2H);
  cudaInGauge->saveCPUField(*cpuGauge);
  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_D2H);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != nullptr) delete gaugePrecise;
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
  auto *cpuGauge = new cpuGaugeField(gParam);

  //gParam.pad = getFatLinkPadding(param->X);
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  auto *cudaInGauge = new cudaGaugeField(gParam);


  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_INIT);

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_H2D);

  //if (!param->use_resident_gauge) {   // load fields onto the device
  cudaInGauge->loadCPUField(*cpuGauge);
  /*} else { // or use resident fields already present
    if (!gaugePrecise) errorQuda("No resident gauge field allocated");
    cudaInGauge = gaugePrecise;
    gaugePrecise = nullptr;
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
    if (gaugePrecise != nullptr) delete gaugePrecise;
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

void contractQuda(const void *hp_x, const void *hp_y, void *h_result, const QudaContractType cType,
                  QudaInvertParam *param, const int *X)
{
  // DMH: Easiest way to construct ColorSpinorField? Do we require the user
  //     to declare and fill and invert_param, or can it just be hacked?.

  profileContract.TPSTART(QUDA_PROFILE_TOTAL);
  profileContract.TPSTART(QUDA_PROFILE_INIT);
  // wrap CPU host side pointers
  ColorSpinorParam cpuParam((void *)hp_x, *param, X, false, param->input_location);
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  cpuParam.v = (void *)hp_y;
  ColorSpinorField *h_y = ColorSpinorField::Create(cpuParam);

  // Create device parameter
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  // Quda uses Degrand-Rossi gamma basis for contractions and will
  // automatically reorder data if necessary.
  cudaParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  cudaParam.setPrecision(cpuParam.Precision(), cpuParam.Precision(), true);

  std::vector<ColorSpinorField *> x, y;
  x.push_back(ColorSpinorField::Create(cudaParam));
  y.push_back(ColorSpinorField::Create(cudaParam));

  size_t data_bytes = x[0]->Volume() * x[0]->Nspin() * x[0]->Nspin() * 2 * x[0]->Precision();
  void *d_result = pool_device_malloc(data_bytes);
  profileContract.TPSTOP(QUDA_PROFILE_INIT);

  profileContract.TPSTART(QUDA_PROFILE_H2D);
  *x[0] = *h_x;
  *y[0] = *h_y;
  profileContract.TPSTOP(QUDA_PROFILE_H2D);

  profileContract.TPSTART(QUDA_PROFILE_COMPUTE);
  contractQuda(*x[0], *y[0], d_result, cType);
  profileContract.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileContract.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(h_result, d_result, data_bytes, cudaMemcpyDeviceToHost);
  profileContract.TPSTOP(QUDA_PROFILE_D2H);

  profileContract.TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_result);
  delete x[0];
  delete y[0];
  delete h_y;
  delete h_x;
  profileContract.TPSTOP(QUDA_PROFILE_FREE);

  profileContract.TPSTOP(QUDA_PROFILE_TOTAL);
}

void gaugeObservablesQuda(QudaGaugeObservableParam *param)
{
  profileGaugeObs.TPSTART(QUDA_PROFILE_TOTAL);
  checkGaugeObservableParam(param);

  cudaGaugeField *gauge = nullptr;
  if (!gaugeSmeared) {
    if (!extendedGaugeResident) extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGaugeObs);
    gauge = extendedGaugeResident;
  } else {
    gauge = gaugeSmeared;
  }

  gaugeObservables(*gauge, *param, profileGaugeObs);
  profileGaugeObs.TPSTOP(QUDA_PROFILE_TOTAL);
}
