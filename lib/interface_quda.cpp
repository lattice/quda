#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

#include <quda.h>
#include <quda_internal.h>
#include <device.h>
#include <timer.h>
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

#include <split_grid.h>

#include <ks_force_quda.h>
#include <ks_qsmear.h>

#include <gauge_path_quda.h>
#include <gauge_update_quda.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM
void checkBLASParam(QudaBLASParam &param) { checkBLASParam(&param); }

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM

#include <gauge_tools.h>
#include <contract_quda.h>
#include <momentum.h>

using namespace quda;

static lat_dim_t R = {};
// setting this to false prevents redundant halo exchange but isn't yet compatible with HISQ / ASQTAD kernels
static bool redundant_comms = false;

#include <blas_lapack.h>


cudaGaugeField *gaugePrecise = nullptr;
cudaGaugeField *gaugeSloppy = nullptr;
cudaGaugeField *gaugePrecondition = nullptr;
cudaGaugeField *gaugeRefinement = nullptr;
cudaGaugeField *gaugeEigensolver = nullptr;
cudaGaugeField *gaugeExtended = nullptr;

cudaGaugeField *gaugeFatPrecise = nullptr;
cudaGaugeField *gaugeFatSloppy = nullptr;
cudaGaugeField *gaugeFatPrecondition = nullptr;
cudaGaugeField *gaugeFatRefinement = nullptr;
cudaGaugeField *gaugeFatEigensolver = nullptr;
cudaGaugeField *gaugeFatExtended = nullptr;

cudaGaugeField *gaugeLongPrecise = nullptr;
cudaGaugeField *gaugeLongSloppy = nullptr;
cudaGaugeField *gaugeLongPrecondition = nullptr;
cudaGaugeField *gaugeLongRefinement = nullptr;
cudaGaugeField *gaugeLongEigensolver = nullptr;
cudaGaugeField *gaugeLongExtended = nullptr;

cudaGaugeField *gaugeSmeared = nullptr;

CloverField *cloverPrecise = nullptr;
CloverField *cloverSloppy = nullptr;
CloverField *cloverPrecondition = nullptr;
CloverField *cloverRefinement = nullptr;
CloverField *cloverEigensolver = nullptr;

cudaGaugeField *momResident = nullptr;
cudaGaugeField *extendedGaugeResident = nullptr;

std::vector<ColorSpinorField> solutionResident;

// vector of spinors used for forecasting solutions in HMC
#define QUDA_MAX_CHRONO 12
// each entry is one p
std::vector<std::vector<ColorSpinorField>> chronoResident(QUDA_MAX_CHRONO);

// Mapped memory buffer used to hold unitarization failures
static int *num_failures_h = nullptr;
static int *num_failures_d = nullptr;

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

//!< Profiler for invertMultiSrcQuda
static TimeProfile profileInvertMultiSrc("invertMultiSrcQuda");

//!< Profiler for invertMultiShiftQuda
static TimeProfile profileMulti("invertMultiShiftQuda");

//!< Profiler for eigensolveQuda
static TimeProfile profileEigensolve("eigensolveQuda");

//!< Profiler for computeFatLinkQuda
static TimeProfile profileFatLink("computeKSLinkQuda");

//!< Profiler for computeGaugeForceQuda
static TimeProfile profileGaugeForce("computeGaugeForceQuda");

//!< Profiler for computeGaugePathQuda
static TimeProfile profileGaugePath("computeGaugePathQuda");

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

//!< Profiler for gaussianSmearQuda
static TimeProfile profileGaussianSmear("gaussianSmearQuda");

//!<Profiler for gaussQuda
static TimeProfile profileGauss("gaussQuda");

//!< Profiler for gaugeObservableQuda
static TimeProfile profileGaugeObs("gaugeObservablesQuda");

//!< Profiler for gaugeSmearQuda
static TimeProfile profileGaugeSmear("gaugeSmearQuda");

//!< Profiler for wFlowQuda
static TimeProfile profileWFlow("wFlowQuda");

//!< Profiler for projectSU3Quda
static TimeProfile profileProject("projectSU3Quda");

//!< Profiler for staggeredPhaseQuda
static TimeProfile profilePhase("staggeredPhaseQuda");

//!< Profiler for contractions
static TimeProfile profileContract("contractQuda");

//!< Profiler for GEMM and other BLAS
static TimeProfile profileBLAS("blasQuda");
TimeProfile &getProfileBLAS() { return profileBLAS; }

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

static void profilerStart(const char *f)
{
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
    device::profile::stop();
    printfQuda("Stopping profiling in QUDA\n");
  } else {
    if (enable) {
      if (i < target_list.size() && target_count++ == target_list[i]) {
        enable_profiler = true;
        printfQuda("Starting profiling for %s\n", f);
        device::profile::start();
        i++; // advance to next target
    }
  }
}
}

static void profilerStop(const char *f) {
  if (do_not_profile_quda) {
    device::profile::start();
  } else {

    if (enable_profiler) {
      printfQuda("Stopping profiling for %s\n", f);
      device::profile::stop();
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
static int qmp_rank_from_coords(const int *coords, void *) { return QMP_get_node_number_from(coords); }
#endif

// Provision for user control over MPI comm handle
// Assumes an MPI implementation of QMP

#if defined(QMP_COMMS) || defined(MPI_COMMS)
MPI_Comm MPI_COMM_HANDLE_USER;
static bool user_set_comm_handle = false;
#endif

#if defined(QMP_COMMS) || defined(MPI_COMMS)
void setMPICommHandleQuda(void *mycomm)
{
  MPI_COMM_HANDLE_USER = *((MPI_Comm *)mycomm);
  user_set_comm_handle = true;
}
#else
void setMPICommHandleQuda(void *) { }
#endif

static bool comms_initialized = false;

void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)
{
  if (comms_initialized) return;

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

#if defined(QMP_COMMS) || defined(MPI_COMMS)
  comm_init(nDim, dims, func, fdata, user_set_comm_handle, (void *)&MPI_COMM_HANDLE_USER);
#else
  comm_init(nDim, dims, func, fdata);
#endif

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
void initQudaDevice(int dev)
{
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

  device::init(dev);

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

  loadTuneCache();

  device::create_context();

  loadTuneCache();

  // initalize the memory pool allocators
  pool::init();

  createDslashEvents();

  blas_lapack::native::init();

  num_failures_h = static_cast<int *>(mapped_malloc(sizeof(int)));
  num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));

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

// These utility functions are defined by the other "free" functions, but they
// are declared here so they can be used in the initial cleanup phase of loadGaugeQuda

/**
 * Abstraction utility that cleans up a set of sloppy fields, typically one of Wilson,
 * HISQ fat, or HISQ long. The utility safely frees the fields as appropriate and sets
 * all of the pointers to nullptr.
 * @param precise[in] Reference to the pointer of a given "precise" field, used for aliasing checks.
 * @param sloppy[in/out] Reference to the pointer of a given "sloppy" field.
 * @param precondition[in/out] Reference the to pointer of a given "precondition" field.
 * @param refinement[in/out] Reference the to pointer of a given "refinement" field.
 * @param eigensolver[in/out] Reference then to pointer of a given "eigensolver" field.
 */
void freeUniqueSloppyGaugeUtility(cudaGaugeField *&precise, cudaGaugeField *&sloppy, cudaGaugeField *&precondition,
                                  cudaGaugeField *&refinement, cudaGaugeField *&eigensolver);

/**
 * Abstraction utility that cleans up the full set of sloppy fields, as well as
 * precise (unless requested otherwise) and extended fields. The set can correspond
 * to the internal Wilson, HISQ fat, or HISQ long fields. This utility safely frees the
 * fields as appropriate and sets all of the pointers to nullptr.
 * @param precise[in/out] Reference to the pointer of a given "precise" field.
 * @param sloppy[in/out] Reference to the pointer of a given "sloppy" field.
 * @param precondition[in/out] Reference to the pointer of a given "precondition" field.
 * @param refinement[in/out] Reference to the pointer of a given "refinement" field.
 * @param eigensolver[in/out] Reference to the pointer of a given "eigensolver" field.
 * @param extended[in/out] Reference to the pointer of a given "extended" field.
 * @param preserve_precise[in] Whether (true) or not (false) to preserve the precise field.
 */
void freeUniqueGaugeUtility(cudaGaugeField *&precise, cudaGaugeField *&sloppy, cudaGaugeField *&precondition,
                            cudaGaugeField *&refinement, cudaGaugeField *&eigensolver, cudaGaugeField *&extended,
                            bool preserve_precise);

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  checkGaugeParam(param);

  profileGauge.TPSTART(QUDA_PROFILE_INIT);
  // Set the specific input parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(*param, h_gauge);

  if (gauge_param.order <= 4) gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  GaugeField *in = (param->location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<GaugeField*>(new cpuGaugeField(gauge_param)) :
    static_cast<GaugeField*>(new cudaGaugeField(gauge_param));

  if (in->Order() == QUDA_BQCD_GAUGE_ORDER) {
    static size_t checksum = SIZE_MAX;
    size_t in_checksum = in->checksum(true);
    if (in_checksum == checksum) {
      if (getVerbosity() >= QUDA_VERBOSE)
        printfQuda("Gauge field unchanged - using cached gauge field %lu\n", checksum);
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
      freeUniqueGaugeUtility(gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver,
                             gaugeExtended, param->use_resident_gauge);
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      freeUniqueGaugeUtility(gaugeFatPrecise, gaugeFatSloppy, gaugeFatPrecondition, gaugeFatRefinement,
                             gaugeFatEigensolver, gaugeFatExtended, param->use_resident_gauge);
      break;
    case QUDA_ASQTAD_LONG_LINKS:
      freeUniqueGaugeUtility(gaugeLongPrecise, gaugeLongSloppy, gaugeLongPrecondition, gaugeLongRefinement,
                             gaugeLongEigensolver, gaugeLongExtended, param->use_resident_gauge);
      break;
    case QUDA_SMEARED_LINKS: freeUniqueGaugeQuda(QUDA_SMEARED_LINKS); break;
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
  gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

  precise = new cudaGaugeField(gauge_param);

  if (param->use_resident_gauge) {
    if(gaugePrecise == nullptr) errorQuda("No resident gauge field");
    // copy rather than point at to ensure that the padded region is filled in
    precise->copy(*gaugePrecise);
    precise->exchangeGhost();
    freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
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

  // switch the parameters for creating the eigensolver cuda gauge field
  gauge_param.reconstruct = param->reconstruct_eigensolver;
  gauge_param.setPrecision(param->cuda_prec_eigensolver, true);
  cudaGaugeField *eigensolver = nullptr;
  if (param->cuda_prec == param->cuda_prec_eigensolver && param->reconstruct == param->reconstruct_eigensolver) {
    eigensolver = precise;
  } else if (param->cuda_prec_precondition == param->cuda_prec_eigensolver
             && param->reconstruct_precondition == param->reconstruct_eigensolver) {
    eigensolver = precondition;
  } else if (param->cuda_prec_sloppy == param->cuda_prec_eigensolver
             && param->reconstruct_sloppy == param->reconstruct_eigensolver) {
    eigensolver = sloppy;
  } else {
    eigensolver = new cudaGaugeField(gauge_param);
    eigensolver->copy(*precise);
  }

  profileGauge.TPSTOP(QUDA_PROFILE_COMPUTE);

  // create an extended preconditioning field
  cudaGaugeField* extended = nullptr;
  if (param->overlap){
    lat_dim_t R; // domain-overlap widths in different directions
    for (int i=0; i<4; ++i) R[i] = param->overlap*commDimPartitioned(i);
    extended = createExtendedGauge(*precondition, R, profileGauge);
  }

  switch (param->type) {
    case QUDA_WILSON_LINKS:
      gaugePrecise = precise;
      gaugeSloppy = sloppy;
      gaugePrecondition = precondition;
      gaugeRefinement = refinement;
      gaugeEigensolver = eigensolver;

      if(param->overlap) gaugeExtended = extended;
      break;
    case QUDA_ASQTAD_FAT_LINKS:
      gaugeFatPrecise = precise;
      gaugeFatSloppy = sloppy;
      gaugeFatPrecondition = precondition;
      gaugeFatRefinement = refinement;
      gaugeFatEigensolver = eigensolver;

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
      gaugeLongEigensolver = eigensolver;

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
    QudaReconstructType recon = extendedGaugeResident->Reconstruct();
    delete extendedGaugeResident;
    // Use the static R (which is defined at the very beginning of lib/interface_quda.cpp) here
    extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGauge, false, recon);
  }

  profileGauge.TPSTOP(QUDA_PROFILE_TOTAL);
}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge.TPSTART(QUDA_PROFILE_TOTAL);

  if (param->location != QUDA_CPU_FIELD_LOCATION) errorQuda("Non-cpu output location not yet supported");

  if (!initialized) errorQuda("QUDA not initialized");
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(*param, h_gauge);
  cpuGaugeField cpuGauge(gauge_param);
  cudaGaugeField *cudaGauge = nullptr;
  switch (param->type) {
  case QUDA_WILSON_LINKS: cudaGauge = gaugePrecise; break;
  case QUDA_ASQTAD_FAT_LINKS: cudaGauge = gaugeFatPrecise; break;
  case QUDA_ASQTAD_LONG_LINKS: cudaGauge = gaugeLongPrecise; break;
  case QUDA_SMEARED_LINKS:
    gauge_param.create = QUDA_NULL_FIELD_CREATE;
    gauge_param.reconstruct = param->reconstruct;
    gauge_param.setPrecision(param->cuda_prec, true);
    gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gauge_param.pad = param->ga_pad;
    cudaGauge = new cudaGaugeField(gauge_param);
    copyExtendedGauge(*cudaGauge, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    break;
  default: errorQuda("Invalid gauge type");
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
  pushVerbosity(inv_param->verbosity);
  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  profileClover.TPSTART(QUDA_PROFILE_INIT);

  checkCloverParam(inv_param);
  bool device_calc = false; // calculate clover and inverse on the device?

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if (!initialized) errorQuda("QUDA not initialized");

  if (!h_clover || inv_param->compute_clover) {
    device_calc = true;
    if (inv_param->clover_coeff == 0.0 && inv_param->clover_csw == 0.0)
      errorQuda("neither clover coefficient nor Csw set");
    if (gaugePrecise->Anisotropy() != 1.0) errorQuda("cannot compute anisotropic clover field");
  }
  if (!h_clover && !device_calc) errorQuda("Uninverted clover term not loaded");

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded before clover");
  if ((inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) && (inv_param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH)
      && (inv_param->dslash_type != QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH)) {
    errorQuda("Wrong dslash_type %d in loadCloverQuda()", inv_param->dslash_type);
  }

  CloverFieldParam clover_param(*inv_param, gaugePrecise->X());
  clover_param.create = QUDA_NULL_FIELD_CREATE;
  // do initial creation and download in same precision as caller, and demote after if needed
  clover_param.setPrecision(inv_param->clover_cpu_prec, true);
  clover_param.inverse = !clover::dynamic_inverse();
  clover_param.location = QUDA_CUDA_FIELD_LOCATION;

  // Adjust inv_param->clover_coeff: if a user has set kappa and Csw,
  // populate inv_param->clover_coeff for them as the computeClover
  // routines uses that value
  inv_param->clover_coeff
    = (inv_param->clover_coeff == 0.0 ? inv_param->kappa * inv_param->clover_csw : inv_param->clover_coeff);

  CloverField *in = nullptr;

  profileClover.TPSTOP(QUDA_PROFILE_INIT);

  bool clover_update = false;
  // If either of the clover params have changed, trigger a recompute
  double csw_old = cloverPrecise ? cloverPrecise->Csw() : 0.0;
  double coeff_old = cloverPrecise ? cloverPrecise->Coeff() : 0.0;
  double rho_old = cloverPrecise ? cloverPrecise->Rho() : 0.0;
  double mu2_old = cloverPrecise ? cloverPrecise->Mu2() : 0.0;
  if (!cloverPrecise || invalidate_clover || inv_param->clover_coeff != coeff_old || inv_param->clover_csw != csw_old
      || inv_param->clover_csw != csw_old || inv_param->clover_rho != rho_old
      || 4 * inv_param->kappa * inv_param->kappa * inv_param->mu * inv_param->mu != mu2_old)
    clover_update = true;

  // compute or download clover field only if gauge field has been updated or clover field doesn't exist
  if (clover_update) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating new clover field\n");
    freeSloppyCloverQuda();
    if (cloverPrecise) delete cloverPrecise;

    profileClover.TPSTART(QUDA_PROFILE_INIT);
    cloverPrecise = new CloverField(clover_param);

    if (!device_calc || inv_param->return_clover || inv_param->return_clover_inverse) {
      // create a param for the cpu clover field
      CloverFieldParam inParam(clover_param);
      inParam.order = inv_param->clover_order;
      inParam.setPrecision(inv_param->clover_cpu_prec);
      inParam.inverse = h_clovinv ? true : false;
      inParam.clover = h_clover;
      inParam.cloverInv = h_clovinv;
      inParam.create = QUDA_REFERENCE_FIELD_CREATE;
      inParam.location = inv_param->clover_location;
      inParam.reconstruct = false;
      in = new CloverField(inParam);
    }
    profileClover.TPSTOP(QUDA_PROFILE_INIT);

    if (!device_calc) {
      profileClover.TPSTART(QUDA_PROFILE_H2D);
      cloverPrecise->copy(*in, false);
      if ((h_clovinv && !inv_param->compute_clover_inverse) && !clover::dynamic_inverse())
        cloverPrecise->copy(*in, true);
      profileClover.TPSTOP(QUDA_PROFILE_H2D);
    } else {
      profileClover.TPSTOP(QUDA_PROFILE_TOTAL);
      createCloverQuda(inv_param);
      profileClover.TPSTART(QUDA_PROFILE_TOTAL);
    }

    if ((!h_clovinv || inv_param->compute_clover_inverse) && !clover::dynamic_inverse()) {
      profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
      cloverInvert(*cloverPrecise, inv_param->compute_clover_trlog);
      if (inv_param->compute_clover_trlog) {
        inv_param->trlogA[0] = cloverPrecise->TrLog()[0];
        inv_param->trlogA[1] = cloverPrecise->TrLog()[1];
      }
      profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);
    }
  } else {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Gauge field unchanged - using cached clover field\n");
  }

  // if requested, copy back the clover / inverse field
  if (inv_param->return_clover || inv_param->return_clover_inverse) {
    if (inv_param->return_clover) {
      if (!h_clover) errorQuda("Requested clover field return but no clover host pointer set");
      profileClover.TPSTART(QUDA_PROFILE_D2H);
      in->copy(*cloverPrecise, false);
      profileClover.TPSTOP(QUDA_PROFILE_D2H);
    }

    if (inv_param->return_clover_inverse) {
      if (!h_clovinv) errorQuda("Requested clover field inverse return but no clover host pointer set");
      profileClover.TPSTART(QUDA_PROFILE_D2H);
      in->copy(*cloverPrecise, true);
      profileClover.TPSTOP(QUDA_PROFILE_D2H);
    }
  }

  if (cloverPrecise->Precision() != inv_param->clover_cuda_prec) {
    // we created the clover field in caller precision, and now need to demote to the desired precision
    CloverFieldParam param(*cloverPrecise);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.setPrecision(inv_param->clover_cuda_prec, true);
    CloverField *tmp = new CloverField(param);
    tmp->copy(*cloverPrecise);
    std::swap(tmp, cloverPrecise);
    delete tmp;
  }

  profileClover.TPSTART(QUDA_PROFILE_FREE);
  if (in) delete in; // delete object referencing input field
  profileClover.TPSTOP(QUDA_PROFILE_FREE);

  QudaPrecision prec[] = {inv_param->clover_cuda_prec_sloppy, inv_param->clover_cuda_prec_precondition,
                          inv_param->clover_cuda_prec_refinement_sloppy, inv_param->clover_cuda_prec_eigensolver};
  loadSloppyCloverQuda(prec);

  profileClover.TPSTOP(QUDA_PROFILE_TOTAL);
  popVerbosity();
}

void freeSloppyCloverQuda();

void loadSloppyCloverQuda(const QudaPrecision *prec)
{
  freeSloppyCloverQuda();

  if (cloverPrecise) {
    // create the mirror sloppy clover field
    CloverFieldParam clover_param(*cloverPrecise);
    clover_param.setPrecision(prec[0], true);

    if (clover_param.Precision() != cloverPrecise->Precision()) {
      cloverSloppy = new CloverField(clover_param);
      cloverSloppy->copy(*cloverPrecise);
    } else {
      cloverSloppy = cloverPrecise;
    }

    // switch the parameters for creating the mirror preconditioner clover field
    clover_param.setPrecision(prec[1], true);

    // create the mirror preconditioner clover field
    if (clover_param.Precision() == cloverPrecise->Precision()) {
      cloverPrecondition = cloverPrecise;
    } else if (clover_param.Precision() == cloverSloppy->Precision()) {
      cloverPrecondition = cloverSloppy;
    } else {
      cloverPrecondition = new CloverField(clover_param);
      cloverPrecondition->copy(*cloverPrecise);
    }

    // switch the parameters for creating the mirror refinement clover field
    clover_param.setPrecision(prec[2], true);

    // create the mirror refinement clover field
    if (clover_param.Precision() != cloverSloppy->Precision()) {
      cloverRefinement = new CloverField(clover_param);
      cloverRefinement->copy(*cloverSloppy);
    } else {
      cloverRefinement = cloverSloppy;
    }
    // switch the parameters for creating the mirror eigensolver clover field
    clover_param.setPrecision(prec[3]);

    // create the mirror eigensolver clover field
    if (clover_param.Precision() == cloverPrecise->Precision()) {
      cloverEigensolver = cloverPrecise;
    } else if (clover_param.Precision() == cloverSloppy->Precision()) {
      cloverEigensolver = cloverSloppy;
    } else if (clover_param.Precision() == cloverPrecondition->Precision()) {
      cloverEigensolver = cloverPrecondition;
    } else {
      cloverEigensolver = new CloverField(clover_param);
      cloverEigensolver->copy(*cloverPrecise);
    }
  }

}

// just free the sloppy fields used in mixed-precision solvers
void freeSloppyGaugeQuda()
{
  if (!initialized) errorQuda("QUDA not initialized");

  // Wilson gauges
  freeUniqueSloppyGaugeUtility(gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver);

  // Long gauges
  freeUniqueSloppyGaugeUtility(gaugeLongPrecise, gaugeLongSloppy, gaugeLongPrecondition, gaugeLongRefinement,
                               gaugeLongEigensolver);

  // Fat gauges
  freeUniqueSloppyGaugeUtility(gaugeFatPrecise, gaugeFatSloppy, gaugeFatPrecondition, gaugeFatRefinement,
                               gaugeFatEigensolver);
}

void freeGaugeQuda(void)
{
  if (!initialized) errorQuda("QUDA not initialized");

  freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
  freeUniqueGaugeQuda(QUDA_ASQTAD_FAT_LINKS);
  freeUniqueGaugeQuda(QUDA_ASQTAD_LONG_LINKS);
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);

  // Need to merge extendedGaugeResident and gaugeFatPrecise/gaugePrecise
  if (extendedGaugeResident) {
    delete extendedGaugeResident;
    extendedGaugeResident = nullptr;
  }
}

// These utility functions are declared w/doxygen above
void freeUniqueSloppyGaugeUtility(cudaGaugeField *&precise, cudaGaugeField *&sloppy, cudaGaugeField *&precondition,
                                  cudaGaugeField *&refinement, cudaGaugeField *&eigensolver)
{
  // In theory, we're checking for aliasing and freeing fields in the opposite order
  // from which they were allocated... but in any case, we're doing an all-to-all
  // checking of aliasing, so it doesn't really matter if the order matches.

  // The last field to get allocated is the eigensolver
  if (eigensolver != refinement && eigensolver != precondition && eigensolver != sloppy && eigensolver != precise
      && eigensolver)
    delete eigensolver;
  eigensolver = nullptr;

  // Second to last: refinement
  if (refinement != precondition && refinement != sloppy && refinement != precise && refinement) delete refinement;
  refinement = nullptr;

  // Third to last: precondition
  if (precondition != sloppy && precondition != precise && precondition) delete precondition;
  precondition = nullptr;

  // Fourth to last: sloppy
  if (sloppy != precise && sloppy) delete sloppy;
  sloppy = nullptr;
}

void freeUniqueGaugeUtility(cudaGaugeField *&precise, cudaGaugeField *&sloppy, cudaGaugeField *&precondition,
                            cudaGaugeField *&refinement, cudaGaugeField *&eigensolver, cudaGaugeField *&extended,
                            bool preserve_precise)
{
  freeUniqueSloppyGaugeUtility(precise, sloppy, precondition, refinement, eigensolver);

  if (precise && !preserve_precise) {
    delete precise;
    precise = nullptr;
  }

  if (extended) delete extended;
  extended = nullptr;
}

void freeUniqueGaugeQuda(QudaLinkType link_type)
{
  if (!initialized) errorQuda("QUDA not initialized");

  // Narrowly free a single type of links
  switch (link_type) {
  case QUDA_WILSON_LINKS:
    freeUniqueGaugeUtility(gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver,
                           gaugeExtended, false);
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    freeUniqueGaugeUtility(gaugeFatPrecise, gaugeFatSloppy, gaugeFatPrecondition, gaugeFatRefinement,
                           gaugeFatEigensolver, gaugeFatExtended, false);
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    freeUniqueGaugeUtility(gaugeLongPrecise, gaugeLongSloppy, gaugeLongPrecondition, gaugeLongRefinement,
                           gaugeLongEigensolver, gaugeLongExtended, false);
    break;
  case QUDA_SMEARED_LINKS:
    if (gaugeSmeared) delete gaugeSmeared;
    gaugeSmeared = nullptr;
    break;
  default: errorQuda("Invalid gauge type %d", link_type);
  }
}

void freeGaugeSmearedQuda()
{
  // thin wrapper
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
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

    // switch the parameters for creating the mirror eigensolver cuda gauge field
    gauge_param.reconstruct = recon[3];
    gauge_param.setPrecision(prec[3], true);

    if (gaugeEigensolver) errorQuda("gaugeEigensolver already exists");

    if (gauge_param.Precision() == gaugePrecise->Precision() && gauge_param.reconstruct == gaugePrecise->Reconstruct()) {
      gaugeEigensolver = gaugePrecise;
    } else if (gauge_param.Precision() == gaugeSloppy->Precision()
               && gauge_param.reconstruct == gaugeSloppy->Reconstruct()) {
      gaugeEigensolver = gaugeSloppy;
    } else if (gauge_param.Precision() == gaugePrecondition->Precision()
               && gauge_param.reconstruct == gaugePrecondition->Reconstruct()) {
      gaugeEigensolver = gaugePrecondition;
    } else {
      gaugeEigensolver = new cudaGaugeField(gauge_param);
      gaugeEigensolver->copy(*gaugePrecise);
    }
  }

  // fat links (if they exist)
  if (gaugeFatPrecise) {
    GaugeFieldParam gauge_param(*gaugeFatPrecise);
    // switch the parameters for creating the mirror sloppy cuda gauge field

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

    // switch the parameters for creating the mirror eigensolver cuda gauge field
    gauge_param.setPrecision(prec[3], true);

    if (gaugeFatEigensolver) errorQuda("gaugeFatEigensolver already exists");

    if (gauge_param.Precision() == gaugeFatPrecise->Precision()
        && gauge_param.reconstruct == gaugeFatPrecise->Reconstruct()) {
      gaugeFatEigensolver = gaugeFatPrecise;
    } else if (gauge_param.Precision() == gaugeFatSloppy->Precision()
               && gauge_param.reconstruct == gaugeFatSloppy->Reconstruct()) {
      gaugeFatEigensolver = gaugeFatSloppy;
    } else if (gauge_param.Precision() == gaugeFatPrecondition->Precision()
               && gauge_param.reconstruct == gaugeFatPrecondition->Reconstruct()) {
      gaugeFatEigensolver = gaugeFatPrecondition;
    } else {
      gaugeFatEigensolver = new cudaGaugeField(gauge_param);
      gaugeFatEigensolver->copy(*gaugeFatPrecise);
    }
  }

  // long links (if they exist)
  if (gaugeLongPrecise) {
    GaugeFieldParam gauge_param(*gaugeLongPrecise);
    // switch the parameters for creating the mirror sloppy cuda gauge field

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

    // switch the parameters for creating the mirror eigensolver cuda gauge field
    gauge_param.reconstruct = recon[3];
    gauge_param.setPrecision(prec[3], true);

    if (gaugeLongEigensolver) errorQuda("gaugePrecondition already exists");

    if (gauge_param.Precision() == gaugeLongPrecise->Precision()
        && gauge_param.reconstruct == gaugeLongPrecise->Reconstruct()) {
      gaugeLongEigensolver = gaugeLongPrecise;
    } else if (gauge_param.Precision() == gaugeLongSloppy->Precision()
               && gauge_param.reconstruct == gaugeLongSloppy->Reconstruct()) {
      gaugeLongEigensolver = gaugeLongSloppy;
    } else if (gauge_param.Precision() == gaugeLongPrecondition->Precision()
               && gauge_param.reconstruct == gaugeLongPrecondition->Reconstruct()) {
      gaugeLongEigensolver = gaugeLongPrecondition;
    } else {
      gaugeLongEigensolver = new cudaGaugeField(gauge_param);
      gaugeLongEigensolver->copy(*gaugeLongPrecise);
    }
  }
}

void freeSloppyCloverQuda()
{
  if (!initialized) errorQuda("QUDA not initialized");

  // Delete cloverRefinement if it does not alias gaugeSloppy.
  if (cloverRefinement != cloverSloppy && cloverRefinement) delete cloverRefinement;

  // Delete cloverPrecondition if it does not alias cloverPrecise, cloverSloppy, or cloverEigensolver.
  if (cloverPrecondition != cloverSloppy && cloverPrecondition != cloverPrecise
      && cloverPrecondition != cloverEigensolver && cloverPrecondition)
    delete cloverPrecondition;

  // Delete cloverEigensolver if it does not alias cloverPrecise or cloverSloppy.
  if (cloverEigensolver != cloverSloppy && cloverEigensolver != cloverPrecise && cloverEigensolver)
    delete cloverEigensolver;

  // Delete cloverSloppy if it does not alias cloverPrecise.
  if (cloverSloppy != cloverPrecise && cloverSloppy) delete cloverSloppy;

  cloverEigensolver = nullptr;
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

  chronoResident[i].clear();
}

void endQuda(void)
{
  profileEnd.TPSTART(QUDA_PROFILE_TOTAL);

  if (!initialized) return;

  freeGaugeQuda();
  freeCloverQuda();

  for (int i = 0; i < QUDA_MAX_CHRONO; i++) flushChronoQuda(i);

  solutionResident.clear();

  if(momResident) delete momResident;

  LatticeField::freeGhostBuffer();
  ColorSpinorField::freeGhostBuffer();
  FieldTmp<ColorSpinorField>::destroy();

  blas_lapack::generic::destroy();
  blas_lapack::native::destroy();
  reducer::destroy();

  pool::flush_pinned();
  pool::flush_device();

  host_free(num_failures_h);
  num_failures_h = nullptr;
  num_failures_d = nullptr;

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
    profileInvertMultiSrc.Print();
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
    profileBLAS.Print();
    profileCovDev.Print();
    profilePlaq.Print();
    profileGaugeObs.Print();
    profileGaussianSmear.Print();
    profileGaugeSmear.Print();
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

  device::destroy();
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
    diracParam.tm_rho = inv_param->tm_rho;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on

    if (diracParam.gauge->Precision() != inv_param->cuda_prec)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec);

    diracParam.use_mobius_fused_kernel = inv_param->use_mobius_fused_kernel;
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

  // The deflation preconditioner currently mimicks the sloppy operator with no comms
  void setDiracEigParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc, bool comms)
  {
    setDiracParam(diracParam, inv_param, pc);

    if (inv_param->overlap) {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatExtended : gaugeExtended;
      diracParam.fatGauge = gaugeFatExtended;
      diracParam.longGauge = gaugeLongExtended;
    } else {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatEigensolver : gaugeEigensolver;
      diracParam.fatGauge = gaugeFatEigensolver;
      diracParam.longGauge = gaugeLongEigensolver;
    }
    diracParam.clover = cloverEigensolver;

    for (int i = 0; i < 4; i++) { diracParam.commDim[i] = comms ? 1 : 0; }

    // In the deflated staggered CG allow a different dslash type
    if (inv_param->inv_type == QUDA_PCG_INVERTER && inv_param->dslash_type == QUDA_ASQTAD_DSLASH
        && inv_param->dslash_type_precondition == QUDA_STAGGERED_DSLASH) {
      diracParam.type = pc ? QUDA_STAGGEREDPC_DIRAC : QUDA_STAGGERED_DIRAC;
      diracParam.gauge = gaugeFatEigensolver;
    }

    if (diracParam.gauge->Precision() != inv_param->cuda_prec_eigensolver)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec_eigensolver);
  }

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    // eigCG and deflation need 2 sloppy precisions and do not use Schwarz
    bool pre_comms_flag = (param.schwarz_type != QUDA_INVALID_SCHWARZ) ? false : true;
    setDiracPreParam(diracPreParam, &param, pc_solve, pre_comms_flag);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
  }

  void createDiracWithRefine(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dRef, QudaInvertParam &param,
                             const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;
    DiracParam diracRefParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    setDiracRefineParam(diracRefParam, &param, pc_solve);
    // eigCG and deflation need 2 sloppy precisions and do not use Schwarz
    bool pre_comms_flag = (param.schwarz_type != QUDA_INVALID_SCHWARZ) ? false : true;
    setDiracPreParam(diracPreParam, &param, pc_solve, pre_comms_flag);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
    dRef = Dirac::create(diracRefParam);
  }

  void createDiracWithEig(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dEig, QudaInvertParam &param,
                          const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;
    DiracParam diracEigParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    // eigCG and deflation need 2 sloppy precisions and do not use Schwarz
    bool pre_comms_flag = (param.schwarz_type != QUDA_INVALID_SCHWARZ) ? false : true;
    setDiracPreParam(diracPreParam, &param, pc_solve, pre_comms_flag);
    bool eig_comms_flag = (param.inv_type == QUDA_INC_EIGCG_INVERTER || param.eig_param) ? true : false;
    setDiracEigParam(diracEigParam, &param, pc_solve, eig_comms_flag);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
    dEig = Dirac::create(diracEigParam);
  }

  void massRescale(ColorSpinorField &b, QudaInvertParam &param, bool for_multishift)
  {
    double kappa5 = (0.5/(5.0 + param.m5));
    double kappa = (param.dslash_type == QUDA_DOMAIN_WALL_DSLASH || param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
                    || param.dslash_type == QUDA_MOBIUS_DWF_DSLASH || param.dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) ?
      kappa5 :
      param.kappa;

    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: Kappa is: %g\n", kappa);
    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: mass normalization: %d\n", param.mass_normalization);
    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: norm of source in = %g\n", blas::norm2(b));

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

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    // you are responsible for restoring what's in param.offset
    switch (param.solution_type) {
      case QUDA_MAT_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION ||
            param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(2.0*kappa, b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 2.0 * kappa;
        }
        break;
      case QUDA_MATDAG_MAT_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION ||
            param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
        }
        break;
      case QUDA_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
        } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(2.0*kappa, b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 2.0 * kappa;
        }
        break;
      case QUDA_MATPCDAG_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  blas::ax(16.0*std::pow(kappa,4), b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 16.0 * std::pow(kappa, 4);
        } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  blas::ax(4.0*kappa*kappa, b);
          if (for_multishift)
            for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
        }
        break;
      default:
        errorQuda("Solution type %d not supported", param.solution_type);
    }

    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: norm of source out = %g\n", blas::norm2(b));
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
  ColorSpinorField in_h(cpuParam);
  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField out_h(cpuParam);

  ColorSpinorField in(cudaParam);
  ColorSpinorField out(cudaParam);

  bool pc = true;
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  profileDslash.TPSTOP(QUDA_PROFILE_INIT);

  profileDslash.TPSTART(QUDA_PROFILE_H2D);
  in = in_h;
  profileDslash.TPSTOP(QUDA_PROFILE_H2D);

  profileDslash.TPSTART(QUDA_PROFILE_COMPUTE);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

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
    ColorSpinorField tmp1(cudaParam);
    ((DiracTwistedCloverPC*) dirac)->TwistCloverInv(tmp1, in, (parity+1)%2); // apply the clover-twist
    dirac->Dslash(out, tmp1, parity); // apply the operator
  } else if (inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH
             || inv_param->dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dirac->Dslash4(out, in, parity);
  } else {
    dirac->Dslash(out, in, parity); // apply the operator
  }
  profileDslash.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileDslash.TPSTART(QUDA_PROFILE_D2H);
  out_h = out;
  profileDslash.TPSTOP(QUDA_PROFILE_D2H);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

  profileDslash.TPSTART(QUDA_PROFILE_FREE);
  delete dirac; // clean up

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
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);
  in = in_h;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

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
  ColorSpinorField out_h(cpuParam);
  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

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
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);
  in = in_h;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField out(cudaParam);

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
  ColorSpinorField out_h(cpuParam);
  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

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

  GaugeField *getResidentGauge() { return gaugePrecise; }

} // namespace quda

void checkClover(QudaInvertParam *param) {

  if (param->dslash_type != QUDA_CLOVER_WILSON_DSLASH && param->dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    return;
  }

  if (param->cuda_prec != cloverPrecise->Precision()) {
    errorQuda("Solve precision %d doesn't match clover precision %d", param->cuda_prec, cloverPrecise->Precision());
  }

  if ((!cloverSloppy || param->cuda_prec_sloppy != cloverSloppy->Precision())
      || (!cloverPrecondition || param->cuda_prec_precondition != cloverPrecondition->Precision())
      || (!cloverRefinement || param->cuda_prec_refinement_sloppy != cloverRefinement->Precision())
      || (!cloverEigensolver || param->cuda_prec_eigensolver != cloverEigensolver->Precision())) {
    freeSloppyCloverQuda();
    QudaPrecision prec[4] = {param->cuda_prec_sloppy, param->cuda_prec_precondition, param->cuda_prec_refinement_sloppy,
                             param->cuda_prec_eigensolver};
    loadSloppyCloverQuda(prec);
  }

  if (cloverPrecise == nullptr) errorQuda("Precise clover field doesn't exist");
  if (cloverSloppy == nullptr) errorQuda("Sloppy clover field doesn't exist");
  if (cloverPrecondition == nullptr) errorQuda("Precondition clover field doesn't exist");
  if (cloverRefinement == nullptr) errorQuda("Refinement clover field doesn't exist");
  if (cloverEigensolver == nullptr) errorQuda("Eigensolver clover field doesn't exist");
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
        || param->cuda_prec_refinement_sloppy != gaugeRefinement->Precision()
        || param->cuda_prec_eigensolver != gaugeEigensolver->Precision()) {
      QudaPrecision precision[4] = {param->cuda_prec_sloppy, param->cuda_prec_precondition,
                                    param->cuda_prec_refinement_sloppy, param->cuda_prec_eigensolver};
      QudaReconstructType recon[4] = {gaugeSloppy->Reconstruct(), gaugePrecondition->Reconstruct(),
                                      gaugeRefinement->Reconstruct(), gaugeEigensolver->Reconstruct()};
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(precision, recon);
    }

    if (gaugeSloppy == nullptr) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == nullptr) errorQuda("Precondition gauge field doesn't exist");
    if (gaugeRefinement == nullptr) errorQuda("Refinement gauge field doesn't exist");
    if (gaugeEigensolver == nullptr) errorQuda("Refinement gauge field doesn't exist");
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
        || param->cuda_prec_eigensolver != gaugeFatEigensolver->Precision()
        || param->cuda_prec_sloppy != gaugeLongSloppy->Precision()
        || param->cuda_prec_precondition != gaugeLongPrecondition->Precision()
        || param->cuda_prec_refinement_sloppy != gaugeLongRefinement->Precision()
        || param->cuda_prec_eigensolver != gaugeLongEigensolver->Precision()) {

      QudaPrecision precision[4] = {param->cuda_prec_sloppy, param->cuda_prec_precondition,
                                    param->cuda_prec_refinement_sloppy, param->cuda_prec_eigensolver};
      // recon is always no for fat links, so just use long reconstructs here
      QudaReconstructType recon[4] = {gaugeLongSloppy->Reconstruct(), gaugeLongPrecondition->Reconstruct(),
                                      gaugeLongRefinement->Reconstruct(), gaugeLongEigensolver->Reconstruct()};
      freeSloppyGaugeQuda();
      loadSloppyGaugeQuda(precision, recon);
    }

    if (gaugeFatSloppy == nullptr) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == nullptr) errorQuda("Precondition gauge fat field doesn't exist");
    if (gaugeFatRefinement == nullptr) errorQuda("Refinement gauge fat field doesn't exist");
    if (gaugeFatEigensolver == nullptr) errorQuda("Eigensolver gauge fat field doesn't exist");
    if (param->overlap) {
      if (gaugeFatExtended == nullptr) errorQuda("Extended gauge fat field doesn't exist");
    }

    if (gaugeLongSloppy == nullptr) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == nullptr) errorQuda("Precondition gauge long field doesn't exist");
    if (gaugeLongRefinement == nullptr) errorQuda("Refinement gauge long field doesn't exist");
    if (gaugeLongEigensolver == nullptr) errorQuda("Eigensolver gauge long field doesn't exist");
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

  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);
  in = in_h;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField out(cudaParam);

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
  ColorSpinorField out_h(cpuParam);
  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

  popVerbosity();
}

void eigensolveQuda(void **host_evecs, double _Complex *host_evals, QudaEigParam *eig_param)
{
  if (!initialized) errorQuda("QUDA not initialized");

  profileEigensolve.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolve.TPSTART(QUDA_PROFILE_INIT);

  // Transfer the inv param structure contained in eig_param.
  // This will define the operator to be eigensolved.
  QudaInvertParam *inv_param = eig_param->invert_param;

  // QUDA can employ even-odd preconditioning to an operator.
  // For the eigensolver the solution type must match
  // the solve type, i.e., there is no full solution reconstruction
  // for an even-odd preconditioned solve. In the eigensolver we allow
  // for M, Mdag, MdagM, and MMdag type operators, chosen via
  // eig_use_dagger and eig_use_norm_op booleans,
  // each combination of which may be preconditioned via eig_use_pc_op. We select
  // the correct QudaInvertParam values for the solve_type and
  // solution_type based on those three booleans

  if (eig_param->use_pc) {
    if (eig_param->use_norm_op)
      inv_param->solve_type = QUDA_NORMOP_PC_SOLVE;
    else
      inv_param->solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param->solution_type = QUDA_MATPC_SOLUTION;
  } else {
    if (eig_param->use_norm_op)
      inv_param->solve_type = QUDA_NORMOP_SOLVE;
    else
      inv_param->solve_type = QUDA_DIRECT_SOLVE;
    inv_param->solution_type = QUDA_MAT_SOLUTION;
  }
  //------------------------------------------------------------------

  // Ensure that the parameter structures are sound.
  checkInvertParam(inv_param);
  checkEigParam(eig_param);

  // Check that the gauge field is valid
  cudaGaugeField *cudaGauge = checkGauge(inv_param);

  // Set all timing statistics to zero
  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  // Dump all eigensolver and invert param variables to stdout if requested.
  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(inv_param);
    printQudaEigParam(eig_param);
  }

  // Define problem matrix
  //------------------------------------------------------
  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

  // Create the dirac operator with a sloppy and a precon.
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);
  Dirac &dirac = *d;
  //------------------------------------------------------

  // Construct vectors
  //------------------------------------------------------
  // Create host wrappers around application vector set
  ColorSpinorParam cpuParam(nullptr, *inv_param, cudaGauge->X(), inv_param->solution_type, inv_param->input_location);

  int n_eig = eig_param->n_conv;
  if (eig_param->compute_svd) n_eig *= 2;
  std::vector<ColorSpinorField> host_evecs_(n_eig);

  if (host_evecs) {
    cpuParam.create = QUDA_REFERENCE_FIELD_CREATE;
    for (int i = 0; i < n_eig; i++) {
      cpuParam.v = host_evecs[i];
      host_evecs_[i] = ColorSpinorField(cpuParam);
    }
  } else {
    cpuParam.create = QUDA_ZERO_FIELD_CREATE;
    for (int i = 0; i < n_eig; i++) { host_evecs_[i] = ColorSpinorField(cpuParam); }
  }

  // Create device side ColorSpinorField vector space to pass to the
  // compute function. Download any user supplied data as an initial guess.
  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(inv_param->cuda_prec_eigensolver, inv_param->cuda_prec_eigensolver, true);
  // Ensure device vectors qre in UKQCD basis for Wilson type fermions
  if (cudaParam.nSpin != 1) cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> kSpace(n_eig);
  for (int i = 0; i < n_eig; i++) {
    kSpace[i] = ColorSpinorField(cudaParam);
    if (i < eig_param->block_size) kSpace[i] = host_evecs_[i];
  }

  // Simple vector for eigenvalues.
  std::vector<Complex> evals(eig_param->n_conv, 0.0);
  //------------------------------------------------------

  // Sanity checks for operator/eigensolver compatibility.
  //------------------------------------------------------
  // If you attempt to compute part of the imaginary spectrum of a hermitian matrix,
  // the solver will fail.
  // Is the spectrum pure imaginary?
  if (eig_param->spectrum == QUDA_SPECTRUM_LI_EIG || eig_param->spectrum == QUDA_SPECTRUM_SI_EIG) {
    // Is the operator hermitian?
    if ((eig_param->use_norm_op || (inv_param->dslash_type == QUDA_LAPLACE_DSLASH))
        || ((inv_param->dslash_type == QUDA_STAGGERED_DSLASH || inv_param->dslash_type == QUDA_ASQTAD_DSLASH)
            && inv_param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Cannot compute the pure imaginary spectrum of a hermitian operator");
    }
  }

  // Gamma5 pre-multiplication is only supported for the M type operator
  if (eig_param->compute_gamma5) {
    if (eig_param->use_norm_op || eig_param->use_dagger) {
      errorQuda("gamma5 premultiplication is only supported for M type operators: dag = %s, normop = %s",
                eig_param->use_dagger ? "true" : "false", eig_param->use_norm_op ? "true" : "false");
    }
  }
  //------------------------------------------------------
  profileEigensolve.TPSTOP(QUDA_PROFILE_INIT);

  // We must construct the correct Dirac operator type based on the three
  // options: The normal operator, the daggered operator, and if we pre
  // multiply by gamma5. Each combination requires a unique Dirac operator
  // object.
  DiracMatrix *m = nullptr;
  if (!eig_param->use_norm_op && !eig_param->use_dagger && eig_param->compute_gamma5) {
    m = new DiracG5M(dirac);
  } else if (!eig_param->use_norm_op && !eig_param->use_dagger && !eig_param->compute_gamma5) {
    m = new DiracM(dirac);
  } else if (!eig_param->use_norm_op && eig_param->use_dagger) {
    m = new DiracMdag(dirac);
  } else if (eig_param->use_norm_op && !eig_param->use_dagger) {
    m = new DiracMdagM(dirac);
  } else if (eig_param->use_norm_op && eig_param->use_dagger) {
    m = new DiracMMdag(dirac);
  } else {
    errorQuda("Invalid use_norm_op, dagger, gamma_5 combination");
  }

  // Perfrom the eigensolve
  if (eig_param->arpack_check) {
    arpack_solve(host_evecs_, evals, *m, eig_param, profileEigensolve);
  } else {
    auto *eig_solve = quda::EigenSolver::create(eig_param, *m, profileEigensolve);
    (*eig_solve)(kSpace, evals);
    delete eig_solve;
  }

  delete m;

  // Transfer Eigenpairs back to host if using GPU eigensolver. The copy
  // will automatically rotate from device UKQCD gamma basis to the
  // host side gamma basis.
  for (int i = 0; i < eig_param->n_conv; i++) { memcpy(host_evals + i, &evals[i], sizeof(Complex)); }
  if (!(eig_param->arpack_check)) {
    profileEigensolve.TPSTART(QUDA_PROFILE_D2H);
    for (int i = 0; i < n_eig; i++) host_evecs_[i] = kSpace[i];
    profileEigensolve.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileEigensolve.TPSTART(QUDA_PROFILE_FREE);
  delete d;
  delete dSloppy;
  delete dPre;
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
  // set whether we are going use native or generic blas
  blas_lapack::set_native(param->native_blas_lapack);

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
  csParam.setPrecision(Bprec, Bprec, true);
  if (mg_param.setup_location[0] == QUDA_CPU_FIELD_LOCATION) csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.mem_type = mg_param.setup_minimize_memory == QUDA_BOOLEAN_TRUE ? QUDA_MEMORY_MAPPED : QUDA_MEMORY_DEVICE;
  B.resize(mg_param.n_vec[0]);

  if (mg_param.transfer_type[0] == QUDA_TRANSFER_COARSE_KD || mg_param.transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD
      || mg_param.transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
    // Create the ColorSpinorField as a "container" for metadata.
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
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

  // Check if we're doing a thin update only
  if (mg_param->thin_update_only) {
    // FIXME: add support for updating kappa, mu as appropriate

    // FIXME: assumes gauge parameters haven't changed.
    // These routines will set gauge = gaugeFat for DiracImprovedStaggered
    mg->d->updateFields(gaugeSloppy, gaugeFatSloppy, gaugeLongSloppy, cloverSloppy);
    mg->d->setMass(param->mass);

    mg->dSmooth->updateFields(gaugeSloppy, gaugeFatSloppy, gaugeLongSloppy, cloverSloppy);
    mg->dSmooth->setMass(param->mass);

    if (mg->dSmoothSloppy != mg->dSmooth) {
      if (param->overlap) {
        mg->dSmoothSloppy->updateFields(gaugeExtended, gaugeFatExtended, gaugeLongExtended, cloverPrecondition);
      } else {
        mg->dSmoothSloppy->updateFields(gaugePrecondition, gaugeFatPrecondition, gaugeLongPrecondition,
                                        cloverPrecondition);
      }
      mg->dSmoothSloppy->setMass(param->mass);
    }
    // The above changes are propagated internally by use of references, pointers, etc, so
    // no further updates are needed.

    // If we're doing a staggered or asqtad KD op, a thin update needs to update the
    // fields for the KD op as well.
    if (mg_param->transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD
        || mg_param->transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      if (param->overlap) errorQuda("Updating the staggered/asqtad KD field with param->overlap set is not supported");

      mg->mg->resetStaggeredKD(gaugeSloppy, gaugeFatSloppy, gaugeLongSloppy, gaugePrecondition, gaugeFatPrecondition,
                               gaugeLongPrecondition, param->mass);
    }

  } else {

    bool outer_pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE);

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
    bool fine_grid_pc_solve = (mg_param->smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE)
      || (mg_param->smoother_solve_type[0] == QUDA_NORMOP_PC_SOLVE);
    setDiracSloppyParam(diracSmoothParam, param, fine_grid_pc_solve);
    mg->dSmooth = Dirac::create(diracSmoothParam);
    mg->mSmooth = new DiracM(*(mg->dSmooth));

    // this is the Dirac operator we use for sloppy smoothing (we use the preconditioner fields for this)
    DiracParam diracSmoothSloppyParam;
    setDiracPreParam(diracSmoothSloppyParam, param, fine_grid_pc_solve, true);
    mg->dSmoothSloppy = Dirac::create(diracSmoothSloppyParam);
    ;
    mg->mSmoothSloppy = new DiracM(*(mg->dSmoothSloppy));

    mg->mgParam->matResidual = mg->m;
    mg->mgParam->matSmooth = mg->mSmooth;
    mg->mgParam->matSmoothSloppy = mg->mSmoothSloppy;

    mg->mgParam->updateInvertParam(*param);
    if (mg->mgParam->mg_global.invert_param != param) mg->mgParam->mg_global.invert_param = param;

    bool refresh = true;
    mg->mg->reset(refresh);
  }

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
  ritzParam.composite_dim = param->n_ev * param->deflation_grid;
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
  auto *defl = new deflated_solver(*eig_param, profileInvert);

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  saveProfile(__func__);
  flushProfile();
  return static_cast<void*>(defl);
}

void destroyDeflationQuda(void *df) {
  delete static_cast<deflated_solver*>(df);
}

void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  profilerStart(__func__);

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
  Dirac *dEig = nullptr;

  // Create the dirac operator and operators for sloppy, precondition,
  // and an eigensolver
  createDiracWithEig(d, dSloppy, dPre, dEig, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  Dirac &diracEig = *dEig;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *in = nullptr;
  ColorSpinorField *out = nullptr;

  const auto X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution, param->input_location);
  ColorSpinorField h_b(cpuParam);

  cpuParam.v = hp_x;
  cpuParam.location = param->output_location;
  ColorSpinorField h_x(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param, QUDA_CUDA_FIELD_LOCATION);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  cudaParam.field = &h_b;
  ColorSpinorField b(cudaParam);

  // now check if we need to invalidate the solutionResident vectors
  ColorSpinorField x;
  if (param->use_resident_solution == 1) {
    for (auto &v : solutionResident) {
      if (b.Precision() != v.Precision() || b.SiteSubset() != v.SiteSubset()) {
        solutionResident.clear();
        break;
      }
    }

    if (!solutionResident.size()) {
      cudaParam.create = QUDA_NULL_FIELD_CREATE;
      solutionResident = std::vector<ColorSpinorField>(1, cudaParam);
    }
    x = solutionResident[0].create_alias(cudaParam);
  } else {
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    x = ColorSpinorField(cudaParam);
  }

  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES && !param->chrono_use_resident) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
        (param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }

    x = h_x; // solution
  } else { // zero initial guess
    blas::zero(x);
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

  double nb = blas::norm2(b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", blas::norm2(h_b), nb);
    if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      printfQuda("Initial guess: CPU = %g, CUDA copy = %g\n", blas::norm2(h_x), blas::norm2(x));
    }
  } else if (getVerbosity() >= QUDA_VERBOSE) {
    printfQuda("Source: %g\n", nb);
    if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { printfQuda("Initial guess: %g\n", blas::norm2(x)); }
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    blas::ax(1.0 / sqrt(nb), b);
    blas::ax(1.0 / sqrt(nb), x);
  }

  massRescale(b, *param, false);

  dirac.prepare(in, out, x, b, param->solution_type);

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
    ColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig, profileInvert);
    (*solve)(*out, *in);
    blas::copy(*in, *out);
    delete solve;
    solverParam.updateInvertParam(*param);
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
    SolverParam solverParam(*param);
    // chronological forecasting
    if (param->chrono_use_resident && chronoResident[param->chrono_index].size() > 0) {
      profileInvert.TPSTART(QUDA_PROFILE_CHRONO);

      auto &basis = chronoResident[param->chrono_index];

      ColorSpinorParam cs_param(basis[0]);
      std::vector<ColorSpinorField> Ap(basis.size(), cs_param);

      if (param->chrono_precision == param->cuda_prec) {
        for (unsigned int j = 0; j < basis.size(); j++) m(Ap[j], basis[j]);
      } else if (param->chrono_precision == param->cuda_prec_sloppy) {
        for (unsigned int j = 0; j < basis.size(); j++) mSloppy(Ap[j], basis[j]);
      } else {
        errorQuda("Unexpected precision %d for chrono vectors (doesn't match outer %d or sloppy precision %d)",
                  param->chrono_precision, param->cuda_prec, param->cuda_prec_sloppy);
      }

      bool orthogonal = true;
      bool apply_mat = false;
      bool hermitian = false;
      MinResExt mre(m, orthogonal, apply_mat, hermitian, profileInvert);
      mre(*out, *in, basis, Ap);

      profileInvert.TPSTOP(QUDA_PROFILE_CHRONO);
    }

    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig, profileInvert);
    (*solve)(*out, *in);
    delete solve;
    solverParam.updateInvertParam(*param);
  } else if (!norm_error_solve) {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
    SolverParam solverParam(*param);

    // chronological forecasting
    if (param->chrono_use_resident && chronoResident[param->chrono_index].size() > 0) {
      profileInvert.TPSTART(QUDA_PROFILE_CHRONO);

      auto &basis = chronoResident[param->chrono_index];

      ColorSpinorParam cs_param(basis[0]);
      std::vector<ColorSpinorField> Ap(basis.size(), cs_param);

      if (param->chrono_precision == param->cuda_prec) {
        for (unsigned int j = 0; j < basis.size(); j++) m(Ap[j], basis[j]);
      } else if (param->chrono_precision == param->cuda_prec_sloppy) {
        for (unsigned int j = 0; j < basis.size(); j++) mSloppy(Ap[j], basis[j]);
      } else {
        errorQuda("Unexpected precision %d for chrono vectors (doesn't match outer %d or sloppy precision %d)",
                  param->chrono_precision, param->cuda_prec, param->cuda_prec_sloppy);
      }

      bool orthogonal = true;
      bool apply_mat = false;
      bool hermitian = true;
      MinResExt mre(m, orthogonal, apply_mat, hermitian, profileInvert);
      mre(*out, *in, basis, Ap);

      profileInvert.TPSTOP(QUDA_PROFILE_CHRONO);
    }

    // if using a Schwarz preconditioner with a normal operator then we must use the DiracMdagMLocal operator
    if (param->inv_type_precondition != QUDA_INVALID_INVERTER && param->schwarz_type != QUDA_INVALID_SCHWARZ) {
      DiracMdagMLocal mPreLocal(diracPre);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPreLocal, mEig, profileInvert);
      (*solve)(*out, *in);
      delete solve;
      solverParam.updateInvertParam(*param);
    } else {
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig, profileInvert);
      (*solve)(*out, *in);
      delete solve;
      solverParam.updateInvertParam(*param);
    }
  } else { // norm_error_solve
    DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
    ColorSpinorField tmp(*out);
    SolverParam solverParam(*param);
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig, profileInvert);
    (*solve)(tmp, *in); // y = (M M^\dag) b
    dirac.Mdag(*out, tmp);  // x = M^dag y
    delete solve;
    solverParam.updateInvertParam(*param);
  }

  if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("Solution = %g\n", blas::norm2(x)); }

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);
  if (param->chrono_make_resident) {
    if(param->chrono_max_dim < 1){
      errorQuda("Cannot chrono_make_resident with chrono_max_dim %i", param->chrono_max_dim);
    }

    const int i = param->chrono_index;
    if (i >= QUDA_MAX_CHRONO)
      errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

    auto &basis = chronoResident[i];

    if (param->chrono_max_dim < (int)basis.size()) {
      errorQuda("Requested chrono_max_dim %i is smaller than already existing chronology %lu", param->chrono_max_dim, basis.size());
    }

    if(not param->chrono_replace_last){
      // if we have not filled the space yet just augment
      if ((int)basis.size() < param->chrono_max_dim) {
        ColorSpinorParam cs_param(*out);
        cs_param.setPrecision(param->chrono_precision);
        basis.emplace_back(cs_param);
      }

      // shuffle every entry down one and bring the last to the front
      std::rotate(basis.begin(), basis.end() - 1, basis.end());
    }
    basis[0] = *out; // set first entry to new solution
  }
  dirac.reconstruct(x, b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    blas::ax(sqrt(nb), x);
  }
  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  if (!param->make_resident_solution) {
    profileInvert.TPSTART(QUDA_PROFILE_D2H);
    h_x = x;
    profileInvert.TPSTOP(QUDA_PROFILE_D2H);
  }

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);

  if (param->compute_action) {
    Complex action = blas::cDotProduct(b, x);
    param->action[0] = action.real();
    param->action[1] = action.imag();
  }

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printfQuda("Reconstructed solution: CUDA = %g, CPU copy = %g\n", blas::norm2(x), blas::norm2(h_x));
  } else if (getVerbosity() >= QUDA_VERBOSE) {
    printfQuda("Reconstructed solution: %g\n", blas::norm2(x));
  }
  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileInvert.TPSTART(QUDA_PROFILE_FREE);

  if (param->use_resident_solution && !param->make_resident_solution) solutionResident.clear();

  delete d;
  delete dSloppy;
  delete dPre;
  delete dEig;

  profileInvert.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

  profilerStop(__func__);
}

void loadFatLongGaugeQuda(QudaInvertParam *inv_param, QudaGaugeParam *gauge_param, void *milc_fatlinks,
                          void *milc_longlinks)
{
  auto link_recon = gauge_param->reconstruct;
  auto link_recon_sloppy = gauge_param->reconstruct_sloppy;
  auto link_recon_precondition = gauge_param->reconstruct_precondition;

  // Specific gauge parameters for MILC
  int pad_size = 0;
#ifdef MULTI_GPU
  int x_face_size = gauge_param->X[1] * gauge_param->X[2] * gauge_param->X[3] / 2;
  int y_face_size = gauge_param->X[0] * gauge_param->X[2] * gauge_param->X[3] / 2;
  int z_face_size = gauge_param->X[0] * gauge_param->X[1] * gauge_param->X[3] / 2;
  int t_face_size = gauge_param->X[0] * gauge_param->X[1] * gauge_param->X[2] / 2;
  pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
#endif

  int fat_pad = pad_size;
  int link_pad = 3 * pad_size;

  gauge_param->type = (inv_param->dslash_type == QUDA_STAGGERED_DSLASH || inv_param->dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;

  gauge_param->ga_pad = fat_pad;
  if (inv_param->dslash_type == QUDA_STAGGERED_DSLASH || inv_param->dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param->reconstruct = link_recon;
    gauge_param->reconstruct_sloppy = link_recon_sloppy;
    gauge_param->reconstruct_refinement_sloppy = link_recon_sloppy;
  } else {
    gauge_param->reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param->reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gauge_param->reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  loadGaugeQuda(milc_fatlinks, gauge_param);

  if (inv_param->dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param->type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param->ga_pad = link_pad;
    gauge_param->staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param->reconstruct = link_recon;
    gauge_param->reconstruct_sloppy = link_recon_sloppy;
    gauge_param->reconstruct_refinement_sloppy = link_recon_sloppy;
    gauge_param->reconstruct_precondition = link_recon_precondition;
    loadGaugeQuda(milc_longlinks, gauge_param);
  }
}

template <class Interface, class... Args>
void callMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, // color spinor field pointers, and inv_param
                      void *h_gauge, void *milc_fatlinks, void *milc_longlinks,
                      QudaGaugeParam *gauge_param,     // gauge field pointers
                      void *h_clover, void *h_clovinv, // clover field pointers
                      Interface op, Args... args)
{
  /**
    Here we first re-distribute gauge, color spinor, and clover field to sub-partitions, then call either invertQuda or dslashQuda.
    - For clover and gauge field, we re-distribute the host clover side fields, restore them after.
    - For color spinor field, we re-distribute the host side source fields, and re-collect the host side solution fields.
  */

  profilerStart(__func__);

  CommKey split_key = {param->split_grid[0], param->split_grid[1], param->split_grid[2], param->split_grid[3]};
  int num_sub_partition = quda::product(split_key);

  if (!split_key.is_valid()) {
    errorQuda("split_key = [%d,%d,%d,%d] is not valid.\n", split_key[0], split_key[1], split_key[2], split_key[3]);
  }

  if (num_sub_partition == 1) { // In this case we don't split the grid.

    for (int n = 0; n < param->num_src; n++) { op(_hp_x[n], _hp_b[n], param, args...); }

  } else {

    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_TOTAL);
    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_INIT);

    if (gauge_param == nullptr) { errorQuda("gauge_param == nullptr.\n"); }

    // Doing the sub-partition arithmatics
    if (param->num_src_per_sub_partition * num_sub_partition != param->num_src) {
      errorQuda("We need to have split_grid[0](=%d) * split_grid[1](=%d) * split_grid[2](=%d) * split_grid[3](=%d) * "
                "num_src_per_sub_partition(=%d) == num_src(=%d).",
                split_key[0], split_key[1], split_key[2], split_key[3], param->num_src_per_sub_partition, param->num_src);
    }

    // Determine if the color spinor field is using a 5d e/o preconditioning
    QudaPCType pc_type = QUDA_4D_PC;
    if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) { pc_type = QUDA_5D_PC; }

    // Doesn't work for MG yet.
    if (param->inv_type_precondition == QUDA_MG_INVERTER) { errorQuda("Split Grid does NOT work with MG yet."); }

    checkInvertParam(param, _hp_x[0], _hp_b[0]);

    bool is_staggered;
    if (h_gauge) {
      is_staggered = false;
    } else if (milc_fatlinks) {
      is_staggered = true;
    } else {
      errorQuda("Both h_gauge and milc_fatlinks are null.");
      is_staggered = true; // to suppress compiler warning/error.
    }

    // Gauge fields/params
    GaugeFieldParam *gf_param = nullptr;
    GaugeField *in = nullptr;
    // Staggered gauge fields/params
    GaugeFieldParam *milc_fatlink_param = nullptr;
    GaugeFieldParam *milc_longlink_param = nullptr;
    GaugeField *milc_fatlink_field = nullptr;
    GaugeField *milc_longlink_field = nullptr;

    // set up the gauge field params.
    if (!is_staggered) { // not staggered
      gf_param = new GaugeFieldParam(*gauge_param, h_gauge);
      if (gf_param->order <= 4) { gf_param->ghostExchange = QUDA_GHOST_EXCHANGE_NO; }
      in = GaugeField::Create(*gf_param);
    } else { // staggered
      milc_fatlink_param = new GaugeFieldParam(*gauge_param, milc_fatlinks);
      if (milc_fatlink_param->order <= 4) { milc_fatlink_param->ghostExchange = QUDA_GHOST_EXCHANGE_NO; }
      milc_fatlink_field = GaugeField::Create(*milc_fatlink_param);
      milc_longlink_param = new GaugeFieldParam(*gauge_param, milc_longlinks);
      if (milc_longlink_param->order <= 4) { milc_longlink_param->ghostExchange = QUDA_GHOST_EXCHANGE_NO; }
      milc_longlink_field = GaugeField::Create(*milc_longlink_param);
    }

    // Create the temp host side helper fields, which are just wrappers of the input pointers.
    bool pc_solution
      = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

    lat_dim_t X = {gauge_param->X[0], gauge_param->X[1], gauge_param->X[2], gauge_param->X[3]};
    ColorSpinorParam cpuParam(_hp_b[0], *param, X, pc_solution, param->input_location);
    std::vector<ColorSpinorField *> _h_b(param->num_src);
    for (int i = 0; i < param->num_src; i++) {
      cpuParam.v = _hp_b[i];
      _h_b[i] = ColorSpinorField::Create(cpuParam);
    }

    cpuParam.location = param->output_location;
    std::vector<ColorSpinorField *> _h_x(param->num_src);
    for (int i = 0; i < param->num_src; i++) {
      cpuParam.v = _hp_x[i];
      _h_x[i] = ColorSpinorField::Create(cpuParam);
    }

    // Make the gauge param dimensions larger
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Spliting the grid into sub-partitions: (%2d,%2d,%2d,%2d) / (%2d,%2d,%2d,%2d).\n", comm_dim(0),
                 comm_dim(1), comm_dim(2), comm_dim(3), split_key[0], split_key[1], split_key[2], split_key[3]);
    }
    for (int d = 0; d < CommKey::n_dim; d++) {
      if (comm_dim(d) % split_key[d] != 0) {
        errorQuda("Split not possible: %2d %% %2d != 0.", comm_dim(d), split_key[d]);
      }
      if (!is_staggered) {
        gf_param->x[d] *= split_key[d];
        gf_param->pad *= split_key[d];
      } else {
        milc_fatlink_param->x[d] *= split_key[d];
        milc_fatlink_param->pad *= split_key[d];
        milc_longlink_param->x[d] *= split_key[d];
        milc_longlink_param->pad *= split_key[d];
      }
      gauge_param->X[d] *= split_key[d];
      gauge_param->ga_pad *= split_key[d];
    }

    // Deal with clover field. For Multi source computatons, clover field construction is done
    // exclusively on the GPU.
    quda::CloverField *input_clover = nullptr;
    quda::CloverField *collected_clover = nullptr;
    bool is_clover = param->dslash_type == QUDA_CLOVER_WILSON_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || param->dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH;

    if (is_clover) {
      if (param->clover_coeff == 0.0 && param->clover_csw == 0.0)
        errorQuda("called with neither clover term nor inverse and clover coefficient nor Csw not set");
      if (gaugePrecise->Anisotropy() != 1.0) errorQuda("cannot compute anisotropic clover field");

      if (h_clover || h_clovinv) {
        CloverFieldParam clover_param(*param, X);
        clover_param.create = QUDA_REFERENCE_FIELD_CREATE;
        clover_param.setPrecision(param->clover_cpu_prec);
        clover_param.inverse = h_clovinv ? true : false;
        clover_param.clover = h_clover;
        clover_param.cloverInv = h_clovinv;
        clover_param.order = param->clover_order;
        clover_param.location = param->clover_location;
        clover_param.reconstruct = false;

        // Adjust inv_param->clover_coeff: if a user has set kappa and Csw,
        // populate inv_param->clover_coeff for them as the computeClover
        // routines uses that value
        param->clover_coeff = (param->clover_coeff == 0.0 ? param->kappa * param->clover_csw : param->clover_coeff);

        input_clover = new CloverField(clover_param);

        for (int d = 0; d < CommKey::n_dim; d++) { clover_param.x[d] *= split_key[d]; }
        clover_param.create = QUDA_NULL_FIELD_CREATE;
        collected_clover = new CloverField(clover_param);

        std::vector<quda::CloverField *> v_c(1);
        v_c[0] = input_clover;
        quda::split_field(*collected_clover, v_c, split_key); // Clover uses 4d even-odd preconditioning.
      }
    }

    quda::GaugeField *collected_gauge = nullptr;
    quda::GaugeField *collected_milc_fatlink_field = nullptr;
    quda::GaugeField *collected_milc_longlink_field = nullptr;

    if (!is_staggered) {
      gf_param->create = QUDA_NULL_FIELD_CREATE;
      collected_gauge = new quda::cpuGaugeField(*gf_param);
      std::vector<quda::GaugeField *> v_g(1);
      v_g[0] = in;
      quda::split_field(*collected_gauge, v_g, split_key);
    } else {
      milc_fatlink_param->create = QUDA_NULL_FIELD_CREATE;
      milc_longlink_param->create = QUDA_NULL_FIELD_CREATE;
      collected_milc_fatlink_field = new quda::cpuGaugeField(*milc_fatlink_param);
      collected_milc_longlink_field = new quda::cpuGaugeField(*milc_longlink_param);
      std::vector<quda::GaugeField *> v_g(1);
      v_g[0] = milc_fatlink_field;
      quda::split_field(*collected_milc_fatlink_field, v_g, split_key);
      v_g[0] = milc_longlink_field;
      quda::split_field(*collected_milc_longlink_field, v_g, split_key);
    }

    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_INIT);
    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_PREAMBLE);

    comm_barrier();

    // Split input fermion field
    quda::ColorSpinorParam cpu_cs_param_split(*_h_x[0]);
    cpu_cs_param_split.location = QUDA_CPU_FIELD_LOCATION;
    for (int d = 0; d < CommKey::n_dim; d++) { cpu_cs_param_split.x[d] *= split_key[d]; }
    std::vector<quda::ColorSpinorField *> _collect_b(param->num_src_per_sub_partition, nullptr);
    std::vector<quda::ColorSpinorField *> _collect_x(param->num_src_per_sub_partition, nullptr);
    for (int n = 0; n < param->num_src_per_sub_partition; n++) {
      _collect_b[n] = new quda::ColorSpinorField(cpu_cs_param_split);
      _collect_x[n] = new quda::ColorSpinorField(cpu_cs_param_split);
      auto first = _h_b.begin() + n * num_sub_partition;
      auto last = _h_b.begin() + (n + 1) * num_sub_partition;
      std::vector<ColorSpinorField *> _v_b(first, last);
      split_field(*_collect_b[n], _v_b, split_key, pc_type);
    }
    comm_barrier();

    push_communicator(split_key);
    updateR();
    comm_barrier();

    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_TOTAL);

    // Load gauge field after pushing the split communicator so the comm buffers, etc are setup according to
    // the split topology.
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("Split grid loading gauge field...\n"); }
    if (!is_staggered) {
      loadGaugeQuda(collected_gauge->Gauge_p(), gauge_param);
    } else {
      // freeGaugeQuda();
      loadFatLongGaugeQuda(param, gauge_param, collected_milc_fatlink_field->Gauge_p(),
                           collected_milc_longlink_field->Gauge_p());
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("Split grid loaded gauge field...\n"); }

    if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || param->dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("Split grid loading clover field...\n"); }
      if (collected_clover) {
        loadCloverQuda(collected_clover->V(false), collected_clover->V(true), param);
      } else {
        loadCloverQuda(nullptr, nullptr, param);
      }
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("Split grid loaded clover field...\n"); }
    }

    for (int n = 0; n < param->num_src_per_sub_partition; n++) {
      op(_collect_x[n]->V(), _collect_b[n]->V(), param, args...);
    }

    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_TOTAL);
    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_EPILOGUE);
    push_communicator(default_comm_key);
    updateR();
    comm_barrier();

    for (int d = 0; d < CommKey::n_dim; d++) {
      gauge_param->X[d] /= split_key[d];
      gauge_param->ga_pad /= split_key[d];
    }

    for (int n = 0; n < param->num_src_per_sub_partition; n++) {
      auto first = _h_x.begin() + n * num_sub_partition;
      auto last = _h_x.begin() + (n + 1) * num_sub_partition;
      std::vector<ColorSpinorField *> _v_x(first, last);
      join_field(_v_x, *_collect_x[n], split_key, pc_type);
    }

    for (auto p : _collect_b) { delete p; }
    for (auto p : _collect_x) { delete p; }

    for (auto p : _h_x) { delete p; }
    for (auto p : _h_b) { delete p; }

    if (!is_staggered) {
      delete in;
      delete collected_gauge;
    } else {
      delete milc_fatlink_field;
      delete milc_longlink_field;
      delete collected_milc_fatlink_field;
      delete collected_milc_longlink_field;
    }

    if (input_clover) { delete input_clover; }
    if (collected_clover) { delete collected_clover; }

    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_TOTAL);

    // Restore the gauge field
    if (!is_staggered) {
      loadGaugeQuda(h_gauge, gauge_param);
    } else {
      freeGaugeQuda();
      loadFatLongGaugeQuda(param, gauge_param, milc_fatlinks, milc_longlinks);
    }

    if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      loadCloverQuda(h_clover, h_clovinv, param);
    }
  }

  profilerStop(__func__);
}

void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *h_gauge, QudaGaugeParam *gauge_param)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param) { invertQuda(_x, _b, param); };
  callMultiSrcQuda(_hp_x, _hp_b, param, h_gauge, nullptr, nullptr, gauge_param, nullptr, nullptr, op);
}

void invertMultiSrcStaggeredQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *milc_fatlinks,
                                 void *milc_longlinks, QudaGaugeParam *gauge_param)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param) { invertQuda(_x, _b, param); };
  callMultiSrcQuda(_hp_x, _hp_b, param, nullptr, milc_fatlinks, milc_longlinks, gauge_param, nullptr, nullptr, op);
}

void invertMultiSrcCloverQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *h_gauge,
                              QudaGaugeParam *gauge_param, void *h_clover, void *h_clovinv)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param) { invertQuda(_x, _b, param); };
  callMultiSrcQuda(_hp_x, _hp_b, param, h_gauge, nullptr, nullptr, gauge_param, h_clover, h_clovinv, op);
}

void dslashMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity, void *h_gauge,
                        QudaGaugeParam *gauge_param)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param, QudaParity parity) { dslashQuda(_x, _b, param, parity); };
  callMultiSrcQuda(_hp_x, _hp_b, param, h_gauge, nullptr, nullptr, gauge_param, nullptr, nullptr, op, parity);
}

void dslashMultiSrcStaggeredQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity,
                                 void *milc_fatlinks, void *milc_longlinks, QudaGaugeParam *gauge_param)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param, QudaParity parity) { dslashQuda(_x, _b, param, parity); };
  callMultiSrcQuda(_hp_x, _hp_b, param, nullptr, milc_fatlinks, milc_longlinks, gauge_param, nullptr, nullptr, op,
                   parity);
}

void dslashMultiSrcCloverQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity, void *h_gauge,
                              QudaGaugeParam *gauge_param, void *h_clover, void *h_clovinv)
{
  auto op = [](void *_x, void *_b, QudaInvertParam *param, QudaParity parity) { dslashQuda(_x, _b, param, parity); };
  callMultiSrcQuda(_hp_x, _hp_b, param, h_gauge, nullptr, nullptr, gauge_param, h_clover, h_clovinv, op, parity);
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
void invertMultiShiftQuda(void **hp_x, void *hp_b, QudaInvertParam *param)
{
  profilerStart(__func__);

  profileMulti.TPSTART(QUDA_PROFILE_TOTAL);
  profileMulti.TPSTART(QUDA_PROFILE_INIT);

  if (!initialized) errorQuda("QUDA not initialized");

  checkInvertParam(param, hp_x[0], hp_b);

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

  // Create the dirac operator and a sloppy, precon, and refine.
  createDiracWithRefine(d, dSloppy, dPre, dRefine, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  dirac.prefetch(QUDA_CUDA_FIELD_LOCATION);
  diracSloppy.prefetch(QUDA_CUDA_FIELD_LOCATION);

  std::vector<double> r2_old(param->num_offset);

  // Grab the dimension array of the input gauge field.
  const auto X = (param->dslash_type == QUDA_ASQTAD_DSLASH) ? gaugeFatPrecise->X() : gaugePrecise->X();

  // This creates a ColorSpinorParam struct, from the host data
  // pointer, the definitions in param, the dimensions X, and whether
  // the solution is on a checkerboard instruction or not. These can
  // then be used as 'instructions' to create the actual
  // ColorSpinorField
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution, param->input_location);
  ColorSpinorField h_b(cpuParam);

  std::vector<std::unique_ptr<ColorSpinorField>> h_x;
  h_x.resize(param->num_offset);

  cpuParam.location = param->output_location;
  for(int i=0; i < param->num_offset; i++) {
    cpuParam.v = hp_x[i];
    h_x[i] = std::make_unique<ColorSpinorField>(cpuParam);
  }

  profileMulti.TPSTOP(QUDA_PROFILE_INIT);
  profileMulti.TPSTART(QUDA_PROFILE_H2D);
  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param, QUDA_CUDA_FIELD_LOCATION);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  cudaParam.field = &h_b;
  ColorSpinorField b(cudaParam); // Creates b and downloads h_b to it

  profileMulti.TPSTOP(QUDA_PROFILE_H2D);

  profileMulti.TPSTART(QUDA_PROFILE_INIT);
  // Create the solution fields filled with zero
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;

  // now check if we need to invalidate the solution vectors
  for (auto &v : solutionResident) {
    if (cudaParam.Precision() != v.Precision()) {
      solutionResident.clear();
      break;
    }
  }

  // grow/shrink resident solutions to be correct size
  auto old_size = solutionResident.size();
  solutionResident.resize(param->num_offset);
  for (auto i = old_size; i < solutionResident.size(); i++) solutionResident[i] = ColorSpinorField(cudaParam);

  std::vector<ColorSpinorField> &x = solutionResident;
  std::vector<ColorSpinorField> p;

  profileMulti.TPSTOP(QUDA_PROFILE_INIT);

  profileMulti.TPSTART(QUDA_PROFILE_PREAMBLE);

  // Check source norms
  double nb = blas::norm2(b);
  if (nb==0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", blas::norm2(h_b), nb);
  } else if (getVerbosity() >= QUDA_VERBOSE) {
    printfQuda("Source: %g\n", nb);
  }

  // rescale the source vector to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) { blas::ax(1.0 / sqrt(nb), b); }

  // backup shifts
  double unscaled_shifts[QUDA_MAX_MULTI_SHIFT];
  for (int i = 0; i < param->num_offset; i++) { unscaled_shifts[i] = param->offset[i]; }

  // rescale
  massRescale(b, *param, true);
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
    cg_m(x, b, p, r2_old);
  }
  solverParam.updateInvertParam(*param);

  delete m;
  delete mSloppy;

  if (param->compute_true_res) {
    // check each shift has the desired tolerance and use sequential CG to refine
    profileMulti.TPSTART(QUDA_PROFILE_INIT);
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField r(cudaParam);
    profileMulti.TPSTOP(QUDA_PROFILE_INIT);
    QudaInvertParam refineparam = *param;
    refineparam.cuda_prec_sloppy = param->cuda_prec_refinement_sloppy;
    Dirac &dirac = *d;
    Dirac &diracSloppy = *dRefine;
    diracSloppy.prefetch(QUDA_CUDA_FIELD_LOCATION);

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

          cudaParam.create = QUDA_NULL_FIELD_CREATE;
          std::vector<ColorSpinorField> q(nRefine, cudaParam);
          std::vector<ColorSpinorField> z(nRefine, cudaParam);

          z[0] = x[0]; // zero solution already solved
#ifdef REFINE_INCREASING_MASS
          for (int j = 1; j < nRefine; j++) z[j] = x[j];
#else
          for (int j = 1; j < nRefine; j++) z[j] = x[param->num_offset - j];
#endif

          bool orthogonal = false;
          bool apply_mat = true;
          bool hermitian = true;
	  MinResExt mre(*m, orthogonal, apply_mat, hermitian, profileMulti);
          mre(x[i], b, z, q);
        }

        SolverParam solverParam(refineparam);
        solverParam.iter = 0;
        solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
        solverParam.tol = (param->tol_offset[i] > 0.0 ? param->tol_offset[i] : iter_tol); // set L2 tolerance
        solverParam.tol_hq = param->tol_hq_offset[i];                                     // set heavy quark tolerance
        solverParam.delta = param->reliable_delta_refinement;

        {
          CG cg(*m, *mSloppy, *mSloppy, *mSloppy, solverParam, profileMulti);
          if (i==0)
            cg(x[i], b, &p[i], r2_old[i]);
          else
            cg(x[i], b);
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

  // restore shifts
  for (int i = 0; i < param->num_offset; i++) param->offset[i] = unscaled_shifts[i];

  profileMulti.TPSTART(QUDA_PROFILE_D2H);

  if (param->compute_action) {
    Complex action(0);
    for (int i = 0; i < param->num_offset; i++) action += param->residue[i] * blas::cDotProduct(b, x[i]);
    param->action[0] = action.real();
    param->action[1] = action.imag();
  }

  for(int i=0; i < param->num_offset; i++) {
    if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) { // rescale the solution
      blas::ax(sqrt(nb), x[i]);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Solution %d = %g\n", i, blas::norm2(x[i]));

    if (!param->make_resident_solution) *h_x[i] = x[i];
  }
  profileMulti.TPSTOP(QUDA_PROFILE_D2H);

  profileMulti.TPSTART(QUDA_PROFILE_EPILOGUE);

  if (!param->make_resident_solution) solutionResident.clear();

  profileMulti.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileMulti.TPSTART(QUDA_PROFILE_FREE);
  delete d;
  delete dSloppy;
  delete dPre;
  delete dRefine;
  profileMulti.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileMulti.TPSTOP(QUDA_PROFILE_TOTAL);

  profilerStop(__func__);
}

void computeKSLinkQuda(void *fatlink, void *longlink, void *ulink, void *inlink, double *path_coeff, QudaGaugeParam *param)
{
  profileFatLink.TPSTART(QUDA_PROFILE_TOTAL);
  profileFatLink.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, fatlink, QUDA_GENERAL_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  cpuGaugeField cpuFatLink(gParam);   // create the host fatlink
  gParam.gauge = longlink;
  cpuGaugeField cpuLongLink(gParam);  // create the host longlink
  gParam.gauge = ulink;
  cpuGaugeField cpuUnitarizedLink(gParam);
  gParam.link_type = param->type;
  gParam.gauge = inlink;
  cpuGaugeField cpuInLink(gParam);    // create the host sitelink

  // create the device fields
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  cudaGaugeField *cudaInLink = new cudaGaugeField(gParam);
  profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

  cudaInLink->loadCPUField(cpuInLink, profileFatLink);
  cudaGaugeField *cudaInLinkEx = createExtendedGauge(*cudaInLink, R, profileFatLink);

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  delete cudaInLink;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  if (longlink) {
    profileFatLink.TPSTART(QUDA_PROFILE_INIT);
    cudaGaugeField *cudaLongLink = new cudaGaugeField(gParam);
    profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

    profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
    longKSLink(cudaLongLink, *cudaInLinkEx, path_coeff);
    profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);

    cudaLongLink->saveCPUField(cpuLongLink, profileFatLink);

    profileFatLink.TPSTART(QUDA_PROFILE_FREE);
    delete cudaLongLink;
    profileFatLink.TPSTOP(QUDA_PROFILE_FREE);
  }

  profileFatLink.TPSTART(QUDA_PROFILE_INIT);
  cudaGaugeField *cudaFatLink = new cudaGaugeField(gParam);
  profileFatLink.TPSTOP(QUDA_PROFILE_INIT);

  profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
  fatKSLink(cudaFatLink, *cudaInLinkEx, path_coeff);
  profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (fatlink) cudaFatLink->saveCPUField(cpuFatLink, profileFatLink);

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  delete cudaInLinkEx;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  if (ulink) {
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    quda::setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                                     svd_abs_error);

    cudaGaugeField *cudaUnitarizedLink = new cudaGaugeField(gParam);

    profileFatLink.TPSTART(QUDA_PROFILE_COMPUTE);
    *num_failures_h = 0;
    quda::unitarizeLinks(*cudaUnitarizedLink, *cudaFatLink, num_failures_d); // unitarize on the gpu
    if (*num_failures_h > 0)
      errorQuda("Error in unitarization component of the hisq fattening: %d failures", *num_failures_h);
    profileFatLink.TPSTOP(QUDA_PROFILE_COMPUTE);

    cudaUnitarizedLink->saveCPUField(cpuUnitarizedLink, profileFatLink);

    profileFatLink.TPSTART(QUDA_PROFILE_FREE);
    delete cudaUnitarizedLink;
    profileFatLink.TPSTOP(QUDA_PROFILE_FREE);
  }

  profileFatLink.TPSTART(QUDA_PROFILE_FREE);
  delete cudaFatLink;
  profileFatLink.TPSTOP(QUDA_PROFILE_FREE);

  profileFatLink.TPSTOP(QUDA_PROFILE_TOTAL);
}

void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param)
{
  profileGaussianSmear.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaussianSmear.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, inlink, QUDA_GENERAL_LINKS);
  gParam.gauge     = twolink;
  cpuGaugeField cpuTwoLink(gParam);  // create the host twolink
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_INIT);

  cudaGaugeField *cudaInLinkEx = nullptr;

  if(inlink) {
    gParam.link_type = param->type;
    gParam.gauge     = inlink;
    cpuGaugeField cpuInLink(gParam);    // create the host sitelink

    // create the device fields
    gParam.reconstruct = param->reconstruct;
    gParam.setPrecision(param->cuda_prec, true);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    cudaGaugeField *cudaInLink = new cudaGaugeField(gParam);
    profileGaussianSmear.TPSTOP(QUDA_PROFILE_INIT);

    cudaInLink->loadCPUField(cpuInLink, profileGaussianSmear);
    //
    cudaInLinkEx = createExtendedGauge(*cudaInLink, R, profileGaussianSmear);
    //
    profileGaussianSmear.TPSTART(QUDA_PROFILE_FREE);
    delete cudaInLink;
    profileGaussianSmear.TPSTOP(QUDA_PROFILE_FREE);

  } else {
    cudaInLinkEx = createExtendedGauge(*gaugePrecise, R, profileGaussianSmear);
  }

  GaugeFieldParam gsParam(*gaugePrecise);

  gsParam.create        = QUDA_NULL_FIELD_CREATE;
  gsParam.link_type     = QUDA_ASQTAD_LONG_LINKS;
  gsParam.reconstruct   = QUDA_RECONSTRUCT_NO;
  gsParam.setPrecision(param->cuda_prec, true);
  gsParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  gsParam.nFace         = 3;
  gsParam.pad           = gsParam.pad*gsParam.nFace;

  profileGaussianSmear.TPSTART(QUDA_PROFILE_INIT);

  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = new cudaGaugeField(gsParam);

  
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_INIT);

  profileGaussianSmear.TPSTART(QUDA_PROFILE_COMPUTE);

  computeTwoLink(*gaugeSmeared, *cudaInLinkEx);
  gaugeSmeared->exchangeGhost();

  profileGaussianSmear.TPSTOP(QUDA_PROFILE_COMPUTE);
  //
  gaugeSmeared->saveCPUField(cpuTwoLink, profileGaussianSmear);

  profileGaussianSmear.TPSTART(QUDA_PROFILE_FREE);

  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  delete cudaInLinkEx;

  profileGaussianSmear.TPSTOP(QUDA_PROFILE_FREE);
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_TOTAL);
}

int computeGaugeForceQuda(void* mom, void* siteLink,  int*** input_path_buf, int* path_length,
			  double* loop_coeff, int num_paths, int max_length, double eb3, QudaGaugeParam* qudaGaugeParam)
{
  profileGaugeForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(*qudaGaugeParam, siteLink);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
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
    gParam.location = QUDA_CUDA_FIELD_LOCATION;

    cudaSiteLink = new cudaGaugeField(gParam);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_INIT);

    profileGaugeForce.TPSTART(QUDA_PROFILE_H2D);
    cudaSiteLink->loadCPUField(*cpuSiteLink);
    profileGaugeForce.TPSTOP(QUDA_PROFILE_H2D);

    profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);
  }

  GaugeFieldParam gParamMom(*qudaGaugeParam, mom, QUDA_ASQTAD_MOM_LINKS);
  gParamMom.location = QUDA_CPU_FIELD_LOCATION;
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
    gParamMom.location = QUDA_CUDA_FIELD_LOCATION;
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
  // apply / remove phase as appropriate
  if (cudaGauge->StaggeredPhaseApplied()) cudaGauge->removeStaggeredPhase();

  // wrap 1-d arrays in std::vector
  std::vector<int> path_length_v(num_paths);
  std::vector<double> loop_coeff_v(num_paths);
  for (int i = 0; i < num_paths; i++) {
    path_length_v[i] = path_length[i];
    loop_coeff_v[i] = loop_coeff[i];
  }

  // input_path should encode exactly 4 directions
  std::vector<int **> input_path_v(4);
  for (int d = 0; d < 4; d++) { input_path_v[d] = input_path_buf[d]; }

  // actually do the computation
  profileGaugeForce.TPSTART(QUDA_PROFILE_COMPUTE);
  if (!forceMonitor()) {
    gaugeForce(*cudaMom, *cudaGauge, eb3, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);
  } else {
    // if we are monitoring the force, separate the force computation from the momentum update
    GaugeFieldParam gParam(*cudaMom);
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    GaugeField *force = GaugeField::Create(gParam);
    gaugeForce(*force, *cudaGauge, 1.0, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);
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
    if (gaugePrecise && gaugePrecise != cudaSiteLink) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
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
  return 0;
}

int computeGaugePathQuda(void *out, void *siteLink, int ***input_path_buf, int *path_length, double *loop_coeff,
                         int num_paths, int max_length, double eb3, QudaGaugeParam *qudaGaugeParam)
{
  profileGaugePath.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugePath.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(*qudaGaugeParam, siteLink);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.site_offset = qudaGaugeParam->gauge_offset;
  gParam.site_size = qudaGaugeParam->site_size;
  cpuGaugeField *cpuSiteLink = (!qudaGaugeParam->use_resident_gauge) ? new cpuGaugeField(gParam) : nullptr;

  cudaGaugeField *cudaSiteLink = nullptr;

  if (qudaGaugeParam->use_resident_gauge) {
    if (!gaugePrecise) errorQuda("No resident gauge field to use");
    cudaSiteLink = gaugePrecise;
  } else {
    gParam.location = QUDA_CUDA_FIELD_LOCATION;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = qudaGaugeParam->reconstruct;
    gParam.setPrecision(qudaGaugeParam->cuda_prec, true);

    cudaSiteLink = new cudaGaugeField(gParam);
    profileGaugePath.TPSTOP(QUDA_PROFILE_INIT);

    profileGaugePath.TPSTART(QUDA_PROFILE_H2D);
    cudaSiteLink->loadCPUField(*cpuSiteLink);
    profileGaugePath.TPSTOP(QUDA_PROFILE_H2D);

    profileGaugePath.TPSTART(QUDA_PROFILE_INIT);
  }

  GaugeFieldParam gParamOut(*qudaGaugeParam, out);
  gParamOut.location = QUDA_CPU_FIELD_LOCATION;
  gParamOut.site_offset = qudaGaugeParam->gauge_offset;
  gParamOut.site_size = qudaGaugeParam->site_size;
  cpuGaugeField *cpuOut = new cpuGaugeField(gParamOut);
  gParamOut.location = QUDA_CUDA_FIELD_LOCATION;
  gParamOut.create = qudaGaugeParam->overwrite_gauge ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE;
  gParamOut.reconstruct = QUDA_RECONSTRUCT_NO;
  gParamOut.setPrecision(qudaGaugeParam->cuda_prec, true);
  cudaGaugeField *cudaOut = new cudaGaugeField(gParamOut);
  profileGaugePath.TPSTOP(QUDA_PROFILE_INIT);
  if (!qudaGaugeParam->overwrite_gauge) {
    profileGaugePath.TPSTART(QUDA_PROFILE_H2D);
    cudaOut->loadCPUField(*cpuOut);
    profileGaugePath.TPSTOP(QUDA_PROFILE_H2D);
  }

  cudaGaugeField *cudaGauge = createExtendedGauge(*cudaSiteLink, R, profileGaugePath);
  // apply / remove phase as appropriate
  if (cudaGauge->StaggeredPhaseApplied()) cudaGauge->removeStaggeredPhase();

  // wrap 1-d arrays in a std::vector
  std::vector<int> path_length_v(num_paths);
  std::vector<double> loop_coeff_v(num_paths);
  for (int i = 0; i < num_paths; i++) {
    path_length_v[i] = path_length[i];
    loop_coeff_v[i] = loop_coeff[i];
  }

  // input_path should encode exactly 4 directions
  std::vector<int **> input_path_v(4);
  for (int d = 0; d < 4; d++) { input_path_v[d] = input_path_buf[d]; }

  // actually do the computation
  profileGaugePath.TPSTART(QUDA_PROFILE_COMPUTE);
  gaugePath(*cudaOut, *cudaGauge, eb3, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);
  profileGaugePath.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileGaugePath.TPSTART(QUDA_PROFILE_D2H);
  cudaOut->saveCPUField(*cpuOut);
  profileGaugePath.TPSTOP(QUDA_PROFILE_D2H);

  profileGaugePath.TPSTART(QUDA_PROFILE_FREE);
  if (qudaGaugeParam->make_resident_gauge) {
    if (gaugePrecise && gaugePrecise != cudaSiteLink) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = cudaSiteLink;
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaGauge;
  } else {
    delete cudaSiteLink;
    delete cudaGauge;
  }

  delete cudaOut;

  if (cpuSiteLink) delete cpuSiteLink;
  if (cpuOut) delete cpuOut;
  profileGaugePath.TPSTOP(QUDA_PROFILE_FREE);

  profileGaugePath.TPSTOP(QUDA_PROFILE_TOTAL);
  return 0;
}

void momResidentQuda(void *mom, QudaGaugeParam *param)
{
  profileGaugeForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaugeForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(param);

  GaugeFieldParam gParamMom(*param, mom, QUDA_ASQTAD_MOM_LINKS);
  gParamMom.location = QUDA_CPU_FIELD_LOCATION;
  if (gParamMom.order == QUDA_TIFR_GAUGE_ORDER || gParamMom.order == QUDA_TIFR_PADDED_GAUGE_ORDER)
    gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  else
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.site_offset = param->mom_offset;
  gParamMom.site_size = param->site_size;

  cpuGaugeField cpuMom(gParamMom);

  if (param->make_resident_mom && !param->return_result_mom) {
    if (momResident) delete momResident;
    gParamMom.location = QUDA_CUDA_FIELD_LOCATION;
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
}

void createCloverQuda(QudaInvertParam* invertParam)
{
  profileClover.TPSTART(QUDA_PROFILE_TOTAL);
  if (!cloverPrecise) errorQuda("Clover field not allocated");

  QudaReconstructType recon = (gaugePrecise->Reconstruct() == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_12 : gaugePrecise->Reconstruct();
  // for clover we optimize to only send depth 1 halos in y/z/t (FIXME - make work for x, make robust in general)
  lat_dim_t R;
  for (int d=0; d<4; d++) R[d] = (d==0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  cudaGaugeField *gauge = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profileClover, false, recon);

  profileClover.TPSTART(QUDA_PROFILE_INIT);

  GaugeField *ex = gauge;
  if (gauge->Precision() < cloverPrecise->Precision()) {
    GaugeFieldParam param(*gauge);
    param.setPrecision(cloverPrecise->Precision(), true);
    param.create = QUDA_NULL_FIELD_CREATE;
    ex = GaugeField::Create(param);
    ex->copy(*gauge);
  }

  // create the Fmunu field
  GaugeFieldParam tensorParam(gaugePrecise->X(), ex->Precision(), QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
  tensorParam.location = QUDA_CUDA_FIELD_LOCATION;
  tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  cudaGaugeField Fmunu(tensorParam);
  profileClover.TPSTOP(QUDA_PROFILE_INIT);
  profileClover.TPSTART(QUDA_PROFILE_COMPUTE);
  computeFmunu(Fmunu, *ex);
  computeClover(*cloverPrecise, Fmunu, invertParam->clover_coeff);
  profileClover.TPSTOP(QUDA_PROFILE_COMPUTE);
  profileClover.TPSTOP(QUDA_PROFILE_TOTAL);

  if (ex != gauge) delete ex;

  // FIXME always preserve the extended gauge
  extendedGaugeResident = gauge;
}

void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
{
  GaugeFieldParam gParam(*param, gauge, QUDA_GENERAL_LINKS);
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

void saveGaugeFieldQuda(void *gauge, void *inGauge, QudaGaugeParam *param)
{
  auto* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);

  GaugeFieldParam gParam(*param, gauge, QUDA_GENERAL_LINKS);
  gParam.geometry = cudaGauge->Geometry();

  cpuGaugeField cpuGauge(gParam);
  cudaGauge->saveCPUField(cpuGauge);
}

void destroyGaugeFieldQuda(void *gauge)
{
  auto* g = reinterpret_cast<cudaGaugeField*>(gauge);
  delete g;
}

void computeStaggeredForceQuda(void *h_mom, double dt, double delta, void *, void **, QudaGaugeParam *gauge_param,
                               QudaInvertParam *inv_param)
{
  profileStaggeredForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_INIT);

  GaugeFieldParam gParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);

  // create the host momentum field
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.reconstruct = gauge_param->reconstruct;
  gParam.t_boundary = QUDA_PERIODIC_T;
  cpuGaugeField cpuMom(gParam);

  // create the device momentum field
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
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
  qParam.nDim = 4;
  qParam.pc_type = QUDA_4D_PC;
  qParam.setPrecision(gParam.Precision(), gParam.Precision(), true);
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = gParam.x[dir];
  qParam.x[4] = 1;
  qParam.create = QUDA_NULL_FIELD_CREATE;
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

    if (inv_param->use_resident_solution)
      x.Even() = solutionResident[i];
    else errorQuda("%s requires resident solution", __func__);

    // set the odd solution component
    dirac->Dslash(x.Odd(), x.Even(), QUDA_ODD_PARITY);
  }

  profileStaggeredForce.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profileStaggeredForce.TPSTART(QUDA_PROFILE_FREE);

#if 0
  if (inv_param->use_resident_solution) solutionResident.clear();
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
  using namespace quda;
  using namespace quda::fermion_force;
  profileHISQForce.TPSTART(QUDA_PROFILE_TOTAL);
  if (gParam->gauge_order != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported input field order %d", gParam->gauge_order);

  checkGaugeParam(gParam);

  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);

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

  // Save input reconstruct type (applied to W and U fields) and set
  // the reconstruct type to QUDA_RECONSTRUCT_NO
  QudaReconstructType cuda_link_recon = gParam->reconstruct;
  gParam->reconstruct = QUDA_RECONSTRUCT_NO;

  // Create a copy of the setup for the gauge links
  QudaGaugeParam gParam_field;
  memcpy(&gParam_field, gParam, sizeof(QudaGaugeParam));

  // Check reconstruct
  if (cuda_link_recon == QUDA_RECONSTRUCT_9) {
    warningQuda("Attempting to use recon 9 for HISQ force. Resetting to 13...");
    cuda_link_recon = QUDA_RECONSTRUCT_13;
  }

  if (cuda_link_recon != QUDA_RECONSTRUCT_NO && cuda_link_recon != QUDA_RECONSTRUCT_13)
    errorQuda("Invalid reconstruct %d", cuda_link_recon);

  logQuda(QUDA_VERBOSE, "Reconstruct type for HISQ force: %d\n", cuda_link_recon);

  // create the device outer-product field
  GaugeFieldParam oParam(*gParam);
  oParam.location = QUDA_CUDA_FIELD_LOCATION;
  oParam.nFace = 0;
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.setPrecision(gParam->cpu_prec, true);
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  cudaGaugeField *stapleOprod = new cudaGaugeField(oParam);
  cudaGaugeField *oneLinkOprod = new cudaGaugeField(oParam);
  cudaGaugeField *naikOprod = new cudaGaugeField(oParam);

  double act_path_coeff[6] = {0, 1, level2_coeff[2], level2_coeff[3], level2_coeff[4], level2_coeff[5]};
  // You have to look at the MILC routine to understand the following
  // Basically, I have already absorbed the one-link coefficient

  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  { // do outer-product computation
    ColorSpinorParam qParam;
    qParam.nColor = 3;
    qParam.nSpin = 1;
    qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qParam.nDim = 4;
    qParam.pc_type = QUDA_4D_PC;
    qParam.setPrecision(oParam.Precision(), oParam.Precision(), true);
    qParam.pad = 0;
    for (int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];

    // create the device quark field
    qParam.create = QUDA_NULL_FIELD_CREATE;
    qParam.location = QUDA_CUDA_FIELD_LOCATION;
    ColorSpinorField cudaQuark(qParam);

    // create the host quark field
    qParam.location = QUDA_CPU_FIELD_LOCATION;
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
        ColorSpinorField cpuQuark(qParam); // create host quark field
        profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

        profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
        cudaQuark = cpuQuark;
        profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);

        profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
        computeStaggeredOprod(oprod, cudaQuark, coeff[i], 3);
        qudaDeviceSynchronize();
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
        ColorSpinorField cpuQuark(qParam); // create host quark field
        profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

        profileHISQForce.TPSTART(QUDA_PROFILE_H2D);
        cudaQuark = cpuQuark;
        profileHISQForce.TPSTOP(QUDA_PROFILE_H2D);

        profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
        computeStaggeredOprod(oprod, cudaQuark, coeff[i + num_terms], 3);
        qudaDeviceSynchronize();
        profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    }
  }

  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);

  // Compute the pad size
  int pad_size = 0;
#ifdef MULTI_GPU
  int x_face_size = gParam->X[1] * gParam->X[2] * gParam->X[3] / 2;
  int y_face_size = gParam->X[0] * gParam->X[2] * gParam->X[3] / 2;
  int z_face_size = gParam->X[0] * gParam->X[1] * gParam->X[3] / 2;
  int t_face_size = gParam->X[0] * gParam->X[1] * gParam->X[2] / 2;
  pad_size = std::max({x_face_size, y_face_size, z_face_size, t_face_size});
#endif

  // Copy outer product fields into input force fields
  oParam.create = QUDA_NULL_FIELD_CREATE;
  oParam.nFace = 1;
  oParam.pad = pad_size;
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  lat_dim_t R = {2 * comm_dim_partitioned(0), 2 * comm_dim_partitioned(1), 2 * comm_dim_partitioned(2),
                 2 * comm_dim_partitioned(3)};
  for (int dir = 0; dir < 4; ++dir) {
    oParam.x[dir] += 2 * R[dir];
    oParam.r[dir] = R[dir];
  }

  cudaGaugeField *cudaInForce = new cudaGaugeField(oParam);
  copyExtendedGauge(*cudaInForce, *stapleOprod, QUDA_CUDA_FIELD_LOCATION);
  delete stapleOprod;

  cudaGaugeField *cudaOutForce = new cudaGaugeField(oParam);
  copyExtendedGauge(*cudaOutForce, *oneLinkOprod, QUDA_CUDA_FIELD_LOCATION);
  delete oneLinkOprod;

  // Create CPU momentum fields, prepare GPU momentum param
  GaugeFieldParam param(*gParam);
  param.location = QUDA_CPU_FIELD_LOCATION;
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.order = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  param.gauge = milc_momentum;
  cpuGaugeField *cpuMom = (!gParam->use_resident_mom) ? new cpuGaugeField(param) : nullptr;

  param.location = QUDA_CUDA_FIELD_LOCATION;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.order = QUDA_FLOAT2_GAUGE_ORDER;
  GaugeFieldParam momParam(param);

  // Create CPU W, V, and U fields
  gParam_field.type = QUDA_GENERAL_LINKS;
  gParam_field.t_boundary = QUDA_ANTI_PERIODIC_T;
  gParam_field.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  gParam_field.staggered_phase_applied = true;
  gParam_field.gauge_fix = QUDA_GAUGE_FIXED_NO;

  GaugeFieldParam wParam(gParam_field);
  wParam.location = QUDA_CPU_FIELD_LOCATION;
  wParam.create = QUDA_REFERENCE_FIELD_CREATE;
  wParam.order = QUDA_MILC_GAUGE_ORDER;
  wParam.link_type = QUDA_GENERAL_LINKS;
  wParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  wParam.gauge = (void *)w_link;
  cpuGaugeField cpuWLink(wParam);

  GaugeFieldParam vParam(wParam);
  vParam.gauge = (void *)v_link;
  cpuGaugeField cpuVLink(vParam);

  GaugeFieldParam uParam(vParam);
  uParam.gauge = (void *)u_link;
  cpuGaugeField cpuULink(uParam);

  // Load the W field, which contains U(3) matrices, to the device
  gParam_field.ga_pad = 3 * pad_size;
  wParam = GaugeFieldParam(gParam_field);
  for (int dir = 0; dir < 4; dir++) {
    wParam.x[dir] += 2 * R[dir];
    wParam.r[dir] = R[dir];
  }
  wParam.location = QUDA_CUDA_FIELD_LOCATION;
  wParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  wParam.reconstruct = cuda_link_recon;
  wParam.create = QUDA_NULL_FIELD_CREATE;
  wParam.setPrecision(gParam->cpu_prec, true);

  cudaGaugeField *cudaWLink = new cudaGaugeField(wParam);
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  cudaWLink->loadCPUField(cpuWLink, profileHISQForce);
  cudaWLink->exchangeExtendedGhost(cudaWLink->R(), profileHISQForce);

  cudaInForce->exchangeExtendedGhost(R, profileHISQForce);
  cudaWLink->exchangeExtendedGhost(cudaWLink->R(), profileHISQForce);
  cudaOutForce->exchangeExtendedGhost(R, profileHISQForce);

  // Compute level two term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForce(*cudaOutForce, *cudaInForce, *cudaWLink, act_path_coeff);
  qudaDeviceSynchronize();
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Load naik outer product
  copyExtendedGauge(*cudaInForce, *naikOprod, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce->exchangeExtendedGhost(cudaWLink->R(), profileHISQForce);
  delete naikOprod;

  // Compute Naik three-link term contribution
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqLongLinkForce(*cudaOutForce, *cudaInForce, *cudaWLink, act_path_coeff[1]);
  qudaDeviceSynchronize();
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  cudaOutForce->exchangeExtendedGhost(R, profileHISQForce);

  // Load the V field, which contains general matrices, to the device
  profileHISQForce.TPSTART(QUDA_PROFILE_FREE);
  delete cudaWLink;
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);
  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
  for (int dir = 0; dir < 4; ++dir) {
    vParam.x[dir] += 2 * R[dir];
    vParam.r[dir] = R[dir];
  }
  vParam.location = QUDA_CUDA_FIELD_LOCATION;
  vParam.link_type = QUDA_GENERAL_LINKS;
  vParam.reconstruct = QUDA_RECONSTRUCT_NO;
  vParam.create = QUDA_NULL_FIELD_CREATE;
  vParam.setPrecision(gParam->cpu_prec, true);
  vParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  vParam.pad = 3 * pad_size;
  cudaGaugeField *cudaVLink = new cudaGaugeField(vParam);
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  cudaVLink->loadCPUField(cpuVLink, profileHISQForce);
  cudaVLink->exchangeExtendedGhost(cudaVLink->R(), profileHISQForce);

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  *num_failures_h = 0;
  unitarizeForce(*cudaInForce, *cudaOutForce, *cudaVLink, num_failures_d);

  if (*num_failures_h>0) errorQuda("Error in the unitarization component of the hisq fermion force: %d failures\n", *num_failures_h);

  cudaOutForce->zero();
  qudaDeviceSynchronize();
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Load the U field, which contains U(3) matrices, to the device
  // TODO: in theory these should just be SU(3) matrices with MILC phases?
  profileHISQForce.TPSTART(QUDA_PROFILE_FREE);
  delete cudaVLink;
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);
  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
  for (int dir = 0; dir < 4; ++dir) {
    uParam.x[dir] += 2 * R[dir];
    uParam.r[dir] = R[dir];
  }
  uParam.location = QUDA_CUDA_FIELD_LOCATION;
  uParam.link_type = QUDA_GENERAL_LINKS;
  uParam.reconstruct = cuda_link_recon;
  uParam.create = QUDA_NULL_FIELD_CREATE;
  uParam.setPrecision(gParam->cpu_prec, true);
  uParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  uParam.pad = 3 * pad_size;
  cudaGaugeField *cudaULink = new cudaGaugeField(uParam);
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  cudaULink->loadCPUField(cpuULink, profileHISQForce);
  cudaULink->exchangeExtendedGhost(cudaULink->R(), profileHISQForce);

  // Compute Fat7-staple term
  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqStaplesForce(*cudaOutForce, *cudaInForce, *cudaULink, fat7_coeff);
  qudaDeviceSynchronize();
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileHISQForce.TPSTART(QUDA_PROFILE_FREE);
  delete cudaInForce;
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);
  profileHISQForce.TPSTART(QUDA_PROFILE_INIT);
  cudaGaugeField* cudaMom = new cudaGaugeField(momParam);
  profileHISQForce.TPSTOP(QUDA_PROFILE_INIT);

  profileHISQForce.TPSTART(QUDA_PROFILE_COMPUTE);
  hisqCompleteForce(*cudaOutForce, *cudaULink);

  if (gParam->use_resident_mom) {
    if (!momResident) errorQuda("No resident momentum field to use");
    updateMomentum(*momResident, dt, *cudaOutForce, "hisq");
  } else {
    updateMomentum(*cudaMom, dt, *cudaOutForce, "hisq");
  }
  qudaDeviceSynchronize();
  profileHISQForce.TPSTOP(QUDA_PROFILE_COMPUTE);

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
  delete cudaULink;
  profileHISQForce.TPSTOP(QUDA_PROFILE_FREE);

  profileHISQForce.TPSTOP(QUDA_PROFILE_TOTAL);
}

void computeCloverForceQuda(void *h_mom, double dt, void **h_x, void **, double *coeff, double kappa2, double ck,
                            int nvector, double multiplicity, void *, QudaGaugeParam *gauge_param,
                            QudaInvertParam *inv_param)
{
  using namespace quda;
  profileCloverForce.TPSTART(QUDA_PROFILE_TOTAL);
  profileCloverForce.TPSTART(QUDA_PROFILE_INIT);

  checkGaugeParam(gauge_param);
  if (!gaugePrecise) errorQuda("No resident gauge field");

  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  // create the host momentum field
  fParam.location = QUDA_CPU_FIELD_LOCATION;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.order = gauge_param->gauge_order;
  cpuGaugeField cpuMom(fParam);

  // create the device momentum field
  fParam.location = QUDA_CUDA_FIELD_LOCATION;
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
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.pad = 0;
  for(int dir=0; dir<4; ++dir) qParam.x[dir] = fParam.x[dir];

  // create the device quark field
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField*> quarkX, quarkP;
  for (int i=0; i<nvector; i++) {
    quarkX.push_back(ColorSpinorField::Create(qParam));
    quarkP.push_back(ColorSpinorField::Create(qParam));
  }

  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.x[0] /= 2;
  ColorSpinorField tmp(qParam);

  // create the host quark field
  qParam.location = QUDA_CPU_FIELD_LOCATION;
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // need expose this to interface

  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc_solve);
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
      ColorSpinorField cpuQuarkX(qParam); // create host quark field
      profileCloverForce.TPSTOP(QUDA_PROFILE_INIT);

      profileCloverForce.TPSTART(QUDA_PROFILE_H2D);
      x.Even() = cpuQuarkX;
      profileCloverForce.TPSTOP(QUDA_PROFILE_H2D);

      profileCloverForce.TPSTART(QUDA_PROFILE_COMPUTE);
      gamma5(x.Even(), x.Even());
    } else {
      x.Even() = solutionResident[i];
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
  if (inv_param->use_resident_solution) solutionResident.clear();
#endif
  delete dirac;
  profileCloverForce.TPSTOP(QUDA_PROFILE_FREE);

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
  GaugeFieldParam gParam(*param, gauge, QUDA_SU3_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.site_offset = param->gauge_offset;
  gParam.site_size = param->site_size;
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

  GaugeFieldParam gParamMom(*param, momentum);
  gParamMom.reconstruct = (gParamMom.order == QUDA_TIFR_GAUGE_ORDER || gParamMom.order == QUDA_TIFR_PADDED_GAUGE_ORDER) ?
   QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParamMom.site_offset = param->mom_offset;
  gParamMom.site_size = param->site_size;
  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParamMom) : nullptr;

  // create the device fields
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
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
    if (gaugePrecise != nullptr) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
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
  profileGaugeUpdate.TPSTOP(QUDA_PROFILE_TOTAL);
}

 void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param) {
   profileProject.TPSTART(QUDA_PROFILE_TOTAL);

   profileProject.TPSTART(QUDA_PROFILE_INIT);
   checkGaugeParam(param);

   // create the gauge field
   GaugeFieldParam gParam(*param, gauge_h, QUDA_GENERAL_LINKS);
   gParam.location = QUDA_CPU_FIELD_LOCATION;
   gParam.site_offset = param->gauge_offset;
   gParam.site_size = param->site_size;
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

   // create the device fields
   gParam.location = QUDA_CUDA_FIELD_LOCATION;
   gParam.create = QUDA_NULL_FIELD_CREATE;
   gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
   gParam.reconstruct = param->reconstruct;
   cudaGaugeField *cudaGauge = !param->use_resident_gauge ? new cudaGaugeField(gParam) : nullptr;
   profileProject.TPSTOP(QUDA_PROFILE_INIT);

   if (param->use_resident_gauge) {
     if (!gaugePrecise) errorQuda("No resident gauge field to use");
     cudaGauge = gaugePrecise;
     gaugePrecise = nullptr;
   } else {
     profileProject.TPSTART(QUDA_PROFILE_H2D);
     cudaGauge->loadCPUField(*cpuGauge);
     profileProject.TPSTOP(QUDA_PROFILE_H2D);
   }

   profileProject.TPSTART(QUDA_PROFILE_COMPUTE);
   *num_failures_h = 0;

   // project onto SU(3)
   if (cudaGauge->StaggeredPhaseApplied()) cudaGauge->removeStaggeredPhase();
   projectSU3(*cudaGauge, tol, num_failures_d);
   if (!cudaGauge->StaggeredPhaseApplied() && param->staggered_phase_applied) cudaGauge->applyStaggeredPhase();

   profileProject.TPSTOP(QUDA_PROFILE_COMPUTE);

   if(*num_failures_h>0)
     errorQuda("Error in the SU(3) unitarization: %d failures\n", *num_failures_h);

   profileProject.TPSTART(QUDA_PROFILE_D2H);
   if (param->return_result_gauge) cudaGauge->saveCPUField(*cpuGauge);
   profileProject.TPSTOP(QUDA_PROFILE_D2H);

   if (param->make_resident_gauge) {
     if (gaugePrecise != nullptr && cudaGauge != gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
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
   GaugeFieldParam gParam(*param, gauge_h, QUDA_GENERAL_LINKS);
   bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
   gParam.location = QUDA_CPU_FIELD_LOCATION;
   cpuGaugeField *cpuGauge = need_cpu ? new cpuGaugeField(gParam) : nullptr;

   // create the device fields
   gParam.location = QUDA_CUDA_FIELD_LOCATION;
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
     if (gaugePrecise != nullptr && cudaGauge != gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
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
  GaugeFieldParam gParam(*param, momentum, QUDA_ASQTAD_MOM_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.reconstruct = (gParam.order == QUDA_TIFR_GAUGE_ORDER || gParam.order == QUDA_TIFR_PADDED_GAUGE_ORDER) ?
    QUDA_RECONSTRUCT_NO : QUDA_RECONSTRUCT_10;
  gParam.site_offset = param->mom_offset;
  gParam.site_size = param->site_size;

  cpuGaugeField *cpuMom = !param->use_resident_mom ? new cpuGaugeField(gParam) : nullptr;

  // create the device fields
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.setPrecision(param->cuda_prec, true);

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
  profileMomAction.TPSTOP(QUDA_PROFILE_TOTAL);

  return action;
}

void gaussGaugeQuda(unsigned long long seed, double sigma)
{
  profileGauss.TPSTART(QUDA_PROFILE_TOTAL);

  if (!gaugePrecise) errorQuda("Cannot generate Gauss GaugeField as there is no resident gauge field");

  cudaGaugeField *data = gaugePrecise;

  profileGauss.TPSTART(QUDA_PROFILE_COMPUTE);
  quda::gaugeGauss(*data, seed, sigma);
  profileGauss.TPSTOP(QUDA_PROFILE_COMPUTE);

  if (extendedGaugeResident) {
    extendedGaugeResident->copy(*gaugePrecise);
    extendedGaugeResident->exchangeExtendedGhost(R, profileGauss, redundant_comms);
  }

  profileGauss.TPSTOP(QUDA_PROFILE_TOTAL);
}

void gaussMomQuda(unsigned long long seed, double sigma)
{
  profileGauss.TPSTART(QUDA_PROFILE_TOTAL);

  if (!momResident) errorQuda("Cannot generate Gauss GaugeField as there is no resident momentum field");

  cudaGaugeField *data = momResident;

  profileGauss.TPSTART(QUDA_PROFILE_COMPUTE);
  quda::gaugeGauss(*data, seed, sigma);
  profileGauss.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileGauss.TPSTOP(QUDA_PROFILE_TOTAL);
}

/*
 * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
 */
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
 * Computes the trace of the Polyakov loop in direction dir from the resident gauge field
 */
void polyakovLoopQuda(double ploop[2], int dir)
{
  if (!gaugePrecise) errorQuda("Cannot compute Polyakov loop as there is no resident gauge field");
  if (dir != 3) errorQuda("The Polyakov loop can only be computed in the t == 3 direction, invalid direction %d", dir);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
  gaugeObservablesQuda(&obsParam);
  ploop[0] = obsParam.ploop[0];
  ploop[1] = obsParam.ploop[1];
}

void computeGaugeLoopTraceQuda(double _Complex *traces, int **input_path_buf, int *path_length, double *loop_coeff,
                               int num_paths, int max_length, double factor)
{
  if (!gaugePrecise) errorQuda("Cannot compute gauge loop traces as there is no resident gauge field");

  if (extendedGaugeResident) delete extendedGaugeResident;
  extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGaugeObs);

  // informed by gauge path code; apply / remove gauge as appropriate
  if (extendedGaugeResident->StaggeredPhaseApplied()) extendedGaugeResident->removeStaggeredPhase();

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_gauge_loop_trace = QUDA_BOOLEAN_TRUE;
  obsParam.traces = traces;
  obsParam.input_path_buff = input_path_buf;
  obsParam.path_length = path_length;
  obsParam.loop_coeff = loop_coeff;
  obsParam.num_paths = num_paths;
  obsParam.max_length = max_length;
  obsParam.factor = factor;
  gaugeObservablesQuda(&obsParam);
}

/*
 * Performs a deep copy from the internal extendedGaugeResident field.
 */
void copyExtendedResidentGaugeQuda(void *resident_gauge)
{
  if (!gaugePrecise) errorQuda("Cannot perform deep copy of resident gauge field as there is no resident gauge field");
  extendedGaugeResident
    = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profilePlaq);
  static_cast<GaugeField *>(resident_gauge)->copy(*extendedGaugeResident);
}

void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *inv_param, unsigned int n_steps, double alpha)
{
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
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);
  in = in_h;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField out(cudaParam);
  int parity = 0;

  // Computes out(x) = 1/(1+6*alpha)*(in(x) + alpha*\sum_mu (U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)))
  double a = alpha / (1. + 6. * alpha);
  double b = 1. / (1. + 6. * alpha);

  int comm_dim[4] = {};
  // only switch on comms needed for directions with a derivative
  for (int i = 0; i < 4; i++) {
    comm_dim[i] = comm_dim_partitioned(i);
    if (i == 3) comm_dim[i] = 0;
  }

  for (unsigned int i = 0; i < n_steps; i++) {
    if (i) in = out;
    ApplyLaplace(out, in, *precise, 3, a, b, in, parity, false, comm_dim, profileWuppertal);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      double norm = blas::norm2(out);
      printfQuda("Step %d, vector norm %e\n", i, norm);
    }
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField out_h(cpuParam);
  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  if (gaugeSmeared != nullptr)
    delete precise;

  popVerbosity();
}
 

void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param)
{
  if(smear_param->n_steps == 0) return;
  
  QudaInvertParam *inv_param = smear_param->inv_param;
  
  profileGaussianSmear.TPSTART(QUDA_PROFILE_TOTAL);
  profileGaussianSmear.TPSTART(QUDA_PROFILE_INIT);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");
    
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if ( gaugeSmeared == nullptr || smear_param->compute_2link != 0 ) {
  
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Gaussian smearing done with gaugeSmeared\n");
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);

    GaugeFieldParam gParam(*gaugePrecise);
    //
    gParam.create        = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct   = QUDA_RECONSTRUCT_NO;
    gParam.setPrecision(inv_param->cuda_prec, true);
    gParam.link_type     = QUDA_ASQTAD_LONG_LINKS;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 3; // FIXME: need a QudaLinkType with nFace=2.
    gParam.pad = gParam.pad*gParam.nFace;
    //
    gaugeSmeared = new cudaGaugeField(gParam);
    
    cudaGaugeField *two_link_ext = createExtendedGauge(*gaugePrecise, R, profileGauge);//aux field
    
    computeTwoLink(*gaugeSmeared, *two_link_ext);
    
    gaugeSmeared->exchangeGhost();
    
    delete two_link_ext;   
  }

  if (!initialized) errorQuda("QUDA not initialized");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printQudaInvertParam(inv_param); }

  checkInvertParam(inv_param);
  
  // Create device side ColorSpinorField vectors and to pass to the
  // compute function.
  const lat_dim_t X = gaugeSmeared->X();
  
  inv_param->dslash_type = QUDA_ASQTAD_DSLASH;
  
  ColorSpinorParam cpuParam(h_in, *inv_param, X, QUDA_MAT_SOLUTION, QUDA_CPU_FIELD_LOCATION);
  cpuParam.nSpin = 1;
  // QUDA style pointer for host data.
  ColorSpinorField in_h(cpuParam);

  // Device side data.
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create   = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(inv_param->cuda_prec, inv_param->cuda_prec, true);
  ColorSpinorField in(cudaParam);
  ColorSpinorField out(cudaParam);
  ColorSpinorField temp1(cudaParam);
 

  // Create the smearing operator
  //------------------------------------------------------
  Dirac *d       = nullptr;
  DiracParam diracParam;
  //
  diracParam.type      = QUDA_ASQTAD_DIRAC;
  diracParam.matpcType = inv_param->matpc_type;
  diracParam.dagger    = inv_param->dagger;
  diracParam.gauge     = gaugeSmeared;
  diracParam.fatGauge  = gaugeFatPrecise;
  diracParam.longGauge = gaugeLongPrecise;
  diracParam.clover = cloverPrecise;
  diracParam.kappa  = inv_param->kappa;
  diracParam.mass   = inv_param->mass;
  diracParam.m5     = inv_param->m5;
  diracParam.mu     = inv_param->mu;
  diracParam.laplace3D = inv_param->laplace3D;

  for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on

  if (diracParam.gauge->Precision() != inv_param->cuda_prec)
    errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(), inv_param->cuda_prec);
  //
  d = Dirac::create(diracParam); // create the Dirac operator
  
  Dirac &dirac = *d;
  DiracM qsmear_op(dirac);
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_INIT);

  // Copy host data to device
  profileGaussianSmear.TPSTART(QUDA_PROFILE_H2D);
  in = in_h;
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_H2D);

  const double ftmp    = -(smear_param->width*smear_param->width)/(4.0*smear_param->n_steps*4.0);  /* Extra 4 to compensate for stride 2 */
  // Scale up the source to prevent underflow
  profileGaussianSmear.TPSTART(QUDA_PROFILE_COMPUTE);
  
  const double msq     = 1. / ftmp;  
  const double a       = inv_param->laplace3D * 2.0 + msq;
  const QudaParity  parity   = QUDA_INVALID_PARITY;
  for (int i = 0; i < smear_param->n_steps; i++) {
    if (i > 0) std::swap(in, out);
    blas::ax(ftmp, in);
    blas::axpy(a, in, temp1);
    
    qsmear_op.Expose()->SmearOp(out, in, a, 0.0, smear_param->t0, parity);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      double norm = blas::norm2(out);
      printfQuda("Step %d, vector norm %e\n", i, norm);
    }
    blas::xpay(temp1, -1.0, out);
    blas::zero(temp1);
  }

  profileGaussianSmear.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Copy device data to host.
  profileGaussianSmear.TPSTART(QUDA_PROFILE_D2H);
  in_h = out;
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_D2H);

  profileGaussianSmear.TPSTART(QUDA_PROFILE_FREE);

  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished 2link Gaussian smearing.\n");

  delete d;

  smear_param->gflops = dirac.Flops();

  if (smear_param->delete_2link != 0) { freeUniqueGaugeQuda(QUDA_SMEARED_LINKS); }

  profileGaussianSmear.TPSTOP(QUDA_PROFILE_FREE);
  profileGaussianSmear.TPSTOP(QUDA_PROFILE_TOTAL);
  saveTuneCache();
}


void performGaugeSmearQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)
{
  pushOutputPrefix("performGaugeSmearQuda: ");
  profileGaugeSmear.TPSTART(QUDA_PROFILE_TOTAL);
  checkGaugeSmearParam(smear_param);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileGaugeSmear);

  GaugeFieldParam gParam(*gaugeSmeared);
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  auto *cudaGaugeTemp = new cudaGaugeField(gParam);

  int measurement_n = 0; // The nth measurement to take
  gaugeObservablesQuda(&obs_param[measurement_n]);
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    printfQuda("Q charge at step %03d = %+.16e\n", 0, obs_param[measurement_n].qcharge);
  }

  for (unsigned int i = 0; i < smear_param->n_steps; i++) {
    profileGaugeSmear.TPSTART(QUDA_PROFILE_COMPUTE);

    switch (smear_param->smear_type) {
    case QUDA_GAUGE_SMEAR_APE: APEStep(*gaugeSmeared, *cudaGaugeTemp, smear_param->alpha); break;
    case QUDA_GAUGE_SMEAR_STOUT: STOUTStep(*gaugeSmeared, *cudaGaugeTemp, smear_param->rho); break;
    case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
      OvrImpSTOUTStep(*gaugeSmeared, *cudaGaugeTemp, smear_param->rho, smear_param->epsilon);
      break;
    default: errorQuda("Unkown gauge smear type %d", smear_param->smear_type);
    }

    profileGaugeSmear.TPSTOP(QUDA_PROFILE_COMPUTE);
    if ((i + 1) % smear_param->meas_interval == 0) {
      measurement_n++;
      gaugeObservablesQuda(&obs_param[measurement_n]);
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Q charge at step %03d = %+.16e\n", i + 1, obs_param[measurement_n].qcharge);
      }
    }
  }

  delete cudaGaugeTemp;
  profileGaugeSmear.TPSTOP(QUDA_PROFILE_TOTAL);
  popOutputPrefix();
}

void performWFlowQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)
{
  pushOutputPrefix("performWFlowQuda: ");
  profileWFlow.TPSTART(QUDA_PROFILE_TOTAL);
  checkGaugeSmearParam(smear_param);

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileWFlow);

  GaugeFieldParam gParamEx(*gaugeSmeared);
  auto *gaugeAux = GaugeField::Create(gParamEx);

  GaugeFieldParam gParam(*gaugePrecise);
  gParam.reconstruct = QUDA_RECONSTRUCT_NO; // temporary field is not on manifold so cannot use reconstruct
  auto *gaugeTemp = GaugeField::Create(gParam);

  GaugeField *in = gaugeSmeared;
  GaugeField *out = gaugeAux;

  int measurement_n = 0; // The nth measurement to take

  gaugeObservables(*in, obs_param[measurement_n], profileWFlow);

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    printfQuda("flow t, plaquette, E_tot, E_spatial, E_temporal, Q charge\n");
    printfQuda("%le %.16e %+.16e %+.16e %+.16e %+.16e\n", 0.0, obs_param[0].plaquette[0], obs_param[0].energy[0],
               obs_param[0].energy[1], obs_param[0].energy[2], obs_param[0].qcharge);
  }

  for (unsigned int i = 0; i < smear_param->n_steps; i++) {
    // Perform W1, W2, and Vt Wilson Flow steps as defined in
    // https://arxiv.org/abs/1006.4518v3
    profileWFlow.TPSTART(QUDA_PROFILE_COMPUTE);
    if (i > 0) std::swap(in, out); // output from prior step becomes input for next step
    WFlowStep(*out, *gaugeTemp, *in, smear_param->epsilon, smear_param->smear_type);
    profileWFlow.TPSTOP(QUDA_PROFILE_COMPUTE);

    if ((i + 1) % smear_param->meas_interval == 0) {
      measurement_n++; // increment measurements.
      gaugeObservables(*out, obs_param[measurement_n], profileWFlow);
      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("%le %.16e %+.16e %+.16e %+.16e %+.16e\n", smear_param->epsilon * (i + 1),
                   obs_param[measurement_n].plaquette[0], obs_param[measurement_n].energy[0],
                   obs_param[measurement_n].energy[1], obs_param[measurement_n].energy[2],
                   obs_param[measurement_n].qcharge);
      }
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

  GaugeFieldParam gParam(*param, gauge);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.site_offset = param->gauge_offset;
  gParam.site_size = param->site_size;
  auto *cpuGauge = new cpuGaugeField(gParam);

  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.link_type = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  auto *cudaInGauge = new cudaGaugeField(gParam);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_INIT);
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_H2D);

  cudaInGauge->loadCPUField(*cpuGauge);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_H2D);

  cudaGaugeField *cudaInGaugeEx = nullptr;

  if (comm_size() == 1) {
    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugeFixingOVR(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval,
                   stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);
  } else {
    cudaInGaugeEx = createExtendedGauge(*cudaInGauge, R, GaugeFixOVRQuda);

    // perform the update
    GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_COMPUTE);
    gaugeFixingOVR(*cudaInGaugeEx, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval,
                   stopWtheta);
    GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_COMPUTE);

    copyExtendedGauge(*cudaInGauge, *cudaInGaugeEx, QUDA_CUDA_FIELD_LOCATION);
  }

  // copy the gauge field back to the host
  GaugeFixOVRQuda.TPSTART(QUDA_PROFILE_D2H);
  cudaInGauge->saveCPUField(*cpuGauge);
  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_D2H);

  GaugeFixOVRQuda.TPSTOP(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != nullptr) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = cudaInGauge;
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaInGaugeEx;
  } else {
    delete cudaInGauge;
    if (cudaInGaugeEx) delete cudaInGaugeEx;
  }

  delete cpuGauge;

  if(timeinfo){
    timeinfo[0] = GaugeFixOVRQuda.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = GaugeFixOVRQuda.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = GaugeFixOVRQuda.Last(QUDA_PROFILE_D2H);
  }

  return 0;
}

int computeGaugeFixingFFTQuda(void* gauge, const unsigned int gauge_dir,  const unsigned int Nsteps, \
  const unsigned int verbose_interval, const double alpha, const unsigned int autotune, const double tolerance, \
  const unsigned int  stopWtheta, QudaGaugeParam* param , double* timeinfo)
{
  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_TOTAL);

  checkGaugeParam(param);

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_INIT);

  GaugeFieldParam gParam(*param, gauge);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.site_offset = param->gauge_offset;
  gParam.site_size = param->site_size;
  auto *cpuGauge = new cpuGaugeField(gParam);

  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.link_type = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  auto *cudaInGauge = new cudaGaugeField(gParam);

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_INIT);

  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_H2D);

  cudaInGauge->loadCPUField(*cpuGauge);

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_H2D);

  // perform the update
  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_COMPUTE);

  gaugeFixingFFT(*cudaInGauge, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_COMPUTE);

  // copy the gauge field back to the host
  GaugeFixFFTQuda.TPSTART(QUDA_PROFILE_D2H);
  cudaInGauge->saveCPUField(*cpuGauge);
  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_D2H);

  GaugeFixFFTQuda.TPSTOP(QUDA_PROFILE_TOTAL);

  if (param->make_resident_gauge) {
    if (gaugePrecise != nullptr) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = cudaInGauge;
  } else {
    delete cudaInGauge;
  }

  if (timeinfo) {
    timeinfo[0] = GaugeFixFFTQuda.Last(QUDA_PROFILE_H2D);
    timeinfo[1] = GaugeFixFFTQuda.Last(QUDA_PROFILE_COMPUTE);
    timeinfo[2] = GaugeFixFFTQuda.Last(QUDA_PROFILE_D2H);
  }

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
  lat_dim_t X_ = {X[0], X[1], X[2], X[3]};
  ColorSpinorParam cpuParam((void *)hp_x, *param, X_, false, param->input_location);
  ColorSpinorField h_x(cpuParam);

  cpuParam.v = (void *)hp_y;
  ColorSpinorField h_y(cpuParam);

  // Create device parameter
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  // Quda uses Degrand-Rossi gamma basis for contractions and will
  // automatically reorder data if necessary.
  cudaParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  cudaParam.setPrecision(cpuParam.Precision(), cpuParam.Precision(), true);

  std::vector<ColorSpinorField> x = {ColorSpinorField(cudaParam)};
  std::vector<ColorSpinorField> y = {ColorSpinorField(cudaParam)};

  size_t data_bytes = x[0].Volume() * x[0].Nspin() * x[0].Nspin() * 2 * x[0].Precision();
  void *d_result = pool_device_malloc(data_bytes);
  profileContract.TPSTOP(QUDA_PROFILE_INIT);

  profileContract.TPSTART(QUDA_PROFILE_H2D);
  x[0] = h_x;
  y[0] = h_y;
  profileContract.TPSTOP(QUDA_PROFILE_H2D);

  profileContract.TPSTART(QUDA_PROFILE_COMPUTE);
  contractQuda(x[0], y[0], d_result, cType);
  profileContract.TPSTOP(QUDA_PROFILE_COMPUTE);

  profileContract.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(h_result, d_result, data_bytes, qudaMemcpyDeviceToHost);
  profileContract.TPSTOP(QUDA_PROFILE_D2H);

  pool_device_free(d_result);
  profileContract.TPSTOP(QUDA_PROFILE_TOTAL);
}

void gaugeObservablesQuda(QudaGaugeObservableParam *param)
{
  profileGaugeObs.TPSTART(QUDA_PROFILE_TOTAL);
  checkGaugeObservableParam(param);

  if (!gaugePrecise) errorQuda("Cannot compute Polyakov loop as there is no resident gauge field");

  cudaGaugeField *gauge = nullptr;
  if (!gaugeSmeared) {
    if (!extendedGaugeResident) extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGaugeObs);
    gauge = extendedGaugeResident;
  } else {
    gauge = gaugeSmeared;
  }

  // Apply / remove gauge as appropriate
  if (param->remove_staggered_phase == QUDA_BOOLEAN_TRUE) {
    if (gauge->StaggeredPhaseApplied())
      gauge->removeStaggeredPhase();
    else
      errorQuda("Removing staggered phases was requested, however staggered phases aren't already applied");
  }

  gaugeObservables(*gauge, *param, profileGaugeObs);
  profileGaugeObs.TPSTOP(QUDA_PROFILE_TOTAL);
}
