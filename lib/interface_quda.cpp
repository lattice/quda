#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>

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
#include <spin_taste.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>
#include <random_quda.h>
#include <mpi_comm_handle.h>

#include <multigrid.h>
#include <deflation.h>

#include <gauge_backup.h>
#include <clover_backup.h>
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

GaugeField *gaugePrecise = nullptr;
GaugeField *gaugeSloppy = nullptr;
GaugeField *gaugePrecondition = nullptr;
GaugeField *gaugeRefinement = nullptr;
GaugeField *gaugeEigensolver = nullptr;
GaugeField *gaugeExtended = nullptr;

GaugeField *gaugeFatPrecise = nullptr;
GaugeField *gaugeFatSloppy = nullptr;
GaugeField *gaugeFatPrecondition = nullptr;
GaugeField *gaugeFatRefinement = nullptr;
GaugeField *gaugeFatEigensolver = nullptr;
GaugeField *gaugeFatExtended = nullptr;

GaugeField *gaugeLongPrecise = nullptr;
GaugeField *gaugeLongSloppy = nullptr;
GaugeField *gaugeLongPrecondition = nullptr;
GaugeField *gaugeLongRefinement = nullptr;
GaugeField *gaugeLongEigensolver = nullptr;
GaugeField *gaugeLongExtended = nullptr;

GaugeField *gaugeSmeared = nullptr;

// Holds the Two Link gauge
GaugeField *gaugeTwoLink = nullptr;

CloverField *cloverPrecise = nullptr;
CloverField *cloverSloppy = nullptr;
CloverField *cloverPrecondition = nullptr;
CloverField *cloverRefinement = nullptr;
CloverField *cloverEigensolver = nullptr;

GaugeField momResident;
GaugeField *extendedGaugeResident = nullptr;

namespace quda
{

  std::vector<ColorSpinorField> solutionResident;

}

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

//!< Profiles for computeTMCloverForceQuda
static TimeProfile profileTMCloverForce("computeTMCloverForceQuda");

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

//!< Profiler for gFlowQuda
static TimeProfile profileGFlow("gFlowQuda");

//!< Profiler for projectSU3Quda
static TimeProfile profileProject("projectSU3Quda");

//!< Profiler for staggeredPhaseQuda
static TimeProfile profilePhase("staggeredPhaseQuda");

//!< Profiler for contractions
static TimeProfile profileContract("contractQuda");

//!< Profiler for FT contractions
static TimeProfile profileContractFT("contractFTQuda");

//!< Profiler for GEMM and other BLAS
static TimeProfile profileBLAS("blasQuda");
TimeProfile &getProfileBLAS() { return profileBLAS; }

//!< Profiler for covariant derivative
static TimeProfile profileCovDev("covDevQuda");

//!< Profiler for momentum action
static TimeProfile profileMomAction("momActionQuda");

//!< Profiler for sink projection
static TimeProfile profileSinkProject("sinkProjectQuda");

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

  void flushChrono(int i = -1);

  void massRescale(cvector_ref<ColorSpinorField> &b, QudaInvertParam &param, bool for_multishift);

  void distanceReweight(cvector_ref<ColorSpinorField> &b, QudaInvertParam &param, bool inverse);

  void solve(const std::vector<void *> &hp_x, const std::vector<void *> &hp_b, QudaInvertParam &param,
             const GaugeField &u);
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
  auto profile = pushProfile(profileInit);
  profileInit.TPSTART(QUDA_PROFILE_INIT);

#ifdef GITVERSION
  logQuda(QUDA_SUMMARIZE, "QUDA %s (git %s)\n", get_quda_version().c_str(), gitversion);
#else
  logQuda(QUDA_SUMMARIZE, "QUDA %s\n", get_quda_version().c_str());
#endif

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
}

/*
 * Any persistent memory allocations that QUDA uses are done here.
 */
void initQudaMemory()
{
  auto profile = pushProfile(profileInit);
  profileInit.TPSTART(QUDA_PROFILE_INIT);

  if (!comms_initialized) init_default_comms();

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
void freeUniqueSloppyGaugeUtility(GaugeField *&precise, GaugeField *&sloppy, GaugeField *&precondition,
                                  GaugeField *&refinement, GaugeField *&eigensolver);

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
void freeUniqueGaugeUtility(GaugeField *&precise, GaugeField *&sloppy, GaugeField *&precondition, GaugeField *&refinement,
                            GaugeField *&eigensolver, GaugeField *&extended, bool preserve_precise);

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileGauge);
  checkGaugeParam(param);

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  // Set the specific input parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(*param, h_gauge);

  if (gauge_param.order <= 4) gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  GaugeField *in = GaugeField::Create(gauge_param);

  if (in->Order() == QUDA_BQCD_GAUGE_ORDER) {
    static size_t checksum = SIZE_MAX;
    size_t in_checksum = in->checksum(true);
    if (in_checksum == checksum) {
      logQuda(QUDA_VERBOSE, "Gauge field unchanged - using cached gauge field %lu\n", checksum);
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
  GaugeField *precise = nullptr;

  // switch the parameters for creating the mirror precise cuda gauge field
  gauge_param.create = QUDA_NULL_FIELD_CREATE;
  gauge_param.reconstruct = param->reconstruct;
  gauge_param.setPrecision(param->cuda_prec, true);
  gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  gauge_param.pad = param->ga_pad;
  gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

  precise = new GaugeField(gauge_param);

  if (param->use_resident_gauge) {
    if(gaugePrecise == nullptr) errorQuda("No resident gauge field");
    // copy rather than point at to ensure that the padded region is filled in
    precise->copy(*gaugePrecise);
    precise->exchangeGhost();
    freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
  } else {
    precise->copy(*in);
  }

  // for gaugeSmeared we are interested only in the precise version
  if (param->type == QUDA_SMEARED_LINKS) {
    gaugeSmeared = createExtendedGauge(*precise, R, profileGauge);

    delete precise;
    delete in;

    return;
  }

  // creating sloppy fields isn't really compute, but it is work done on the gpu
  profileGauge.TPSTART(QUDA_PROFILE_COMPUTE);

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.setPrecision(param->cuda_prec_sloppy, true);
  GaugeField *sloppy = nullptr;
  if (param->cuda_prec == param->cuda_prec_sloppy && param->reconstruct == param->reconstruct_sloppy) {
    sloppy = precise;
  } else {
    sloppy = new GaugeField(gauge_param);
    sloppy->copy(*precise);
  }

  // switch the parameters for creating the mirror preconditioner cuda gauge field
  gauge_param.reconstruct = param->reconstruct_precondition;
  gauge_param.setPrecision(param->cuda_prec_precondition, true);
  GaugeField *precondition = nullptr;
  if (param->cuda_prec == param->cuda_prec_precondition && param->reconstruct == param->reconstruct_precondition) {
    precondition = precise;
  } else if (param->cuda_prec_sloppy == param->cuda_prec_precondition
             && param->reconstruct_sloppy == param->reconstruct_precondition) {
    precondition = sloppy;
  } else {
    precondition = new GaugeField(gauge_param);
    precondition->copy(*precise);
  }

  // switch the parameters for creating the refinement cuda gauge field
  gauge_param.reconstruct = param->reconstruct_refinement_sloppy;
  gauge_param.setPrecision(param->cuda_prec_refinement_sloppy, true);
  GaugeField *refinement = nullptr;
  if (param->cuda_prec_sloppy == param->cuda_prec_refinement_sloppy
      && param->reconstruct_sloppy == param->reconstruct_refinement_sloppy) {
    refinement = sloppy;
  } else {
    refinement = new GaugeField(gauge_param);
    refinement->copy(*sloppy);
  }

  // switch the parameters for creating the eigensolver cuda gauge field
  gauge_param.reconstruct = param->reconstruct_eigensolver;
  gauge_param.setPrecision(param->cuda_prec_eigensolver, true);
  GaugeField *eigensolver = nullptr;
  if (param->cuda_prec == param->cuda_prec_eigensolver && param->reconstruct == param->reconstruct_eigensolver) {
    eigensolver = precise;
  } else if (param->cuda_prec_precondition == param->cuda_prec_eigensolver
             && param->reconstruct_precondition == param->reconstruct_eigensolver) {
    eigensolver = precondition;
  } else if (param->cuda_prec_sloppy == param->cuda_prec_eigensolver
             && param->reconstruct_sloppy == param->reconstruct_eigensolver) {
    eigensolver = sloppy;
  } else {
    eigensolver = new GaugeField(gauge_param);
    eigensolver->copy(*precise);
  }

  profileGauge.TPSTOP(QUDA_PROFILE_COMPUTE);

  // create an extended preconditioning field
  GaugeField *extended = nullptr;
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

  delete in;

  if (extendedGaugeResident) {
    // updated the resident gauge field if needed
    QudaReconstructType recon = extendedGaugeResident->Reconstruct();
    delete extendedGaugeResident;
    // Use the static R (which is defined at the very beginning of lib/interface_quda.cpp) here
    extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileGauge, false, recon);
  }
}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileGauge);

  if (param->location != QUDA_CPU_FIELD_LOCATION) errorQuda("Non-cpu output location not yet supported");

  if (!initialized) errorQuda("QUDA not initialized");
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(*param, h_gauge);
  GaugeField cpuGauge(gauge_param);
  GaugeField *cudaGauge = nullptr;
  switch (param->type) {
  case QUDA_WILSON_LINKS: cudaGauge = gaugePrecise; break;
  case QUDA_ASQTAD_FAT_LINKS: cudaGauge = gaugeFatPrecise; break;
  case QUDA_ASQTAD_LONG_LINKS: cudaGauge = gaugeLongPrecise; break;
  case QUDA_SMEARED_LINKS:
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;
    gauge_param.create = QUDA_NULL_FIELD_CREATE;
    gauge_param.reconstruct = param->reconstruct;
    gauge_param.setPrecision(param->cuda_prec, true);
    gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gauge_param.pad = param->ga_pad;
    cudaGauge = new GaugeField(gauge_param);
    copyExtendedGauge(*cudaGauge, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    break;
  default: errorQuda("Invalid gauge type");
  }

  cpuGauge.copy(*cudaGauge);

  if (param->type == QUDA_SMEARED_LINKS) { delete cudaGauge; }
}

void loadSloppyCloverQuda(const QudaPrecision prec[]);
void freeSloppyCloverQuda();

void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  auto profile = pushProfile(profileClover);
  pushVerbosity(inv_param->verbosity);

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

  CloverField in;

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
    logQuda(QUDA_VERBOSE, "Creating new clover field\n");
    freeSloppyCloverQuda();
    if (cloverPrecise) delete cloverPrecise;

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
      in = CloverField(inParam);
    }

    if (!device_calc) {
      cloverPrecise->copy(in, false);
      if (!clover::dynamic_inverse()) {
        if (h_clovinv && !inv_param->compute_clover_inverse)
          cloverPrecise->copy(in, true);
        else
          cloverInvert(*cloverPrecise, false);
      }
    } else {
      createCloverQuda(inv_param);
    }

    for (auto i = 0; i < 2; i++) inv_param->trlogA[i] = cloverPrecise->TrLog()[i];
  } else {
    logQuda(QUDA_VERBOSE, "Gauge field unchanged - using cached clover field\n");
  }

  // if requested, copy back the clover / inverse field
  if (inv_param->return_clover || inv_param->return_clover_inverse) {
    if (inv_param->return_clover) {
      if (!h_clover) errorQuda("Requested clover field return but no clover host pointer set");
      in.copy(*cloverPrecise, false);
    }

    if (inv_param->return_clover_inverse) {
      if (!h_clovinv) errorQuda("Requested clover field inverse return but no clover host pointer set");
      in.copy(*cloverPrecise, true);
    }
  }

  if (cloverPrecise->Precision() != inv_param->clover_cuda_prec) {
    // we created the clover field in caller precision, and now need to demote to the desired precision
    CloverFieldParam param(*cloverPrecise);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.setPrecision(inv_param->clover_cuda_prec, true);
    CloverField tmp(param);
    tmp.copy(*cloverPrecise);
    std::exchange(*cloverPrecise, tmp);
  }

  QudaPrecision prec[] = {inv_param->clover_cuda_prec_sloppy, inv_param->clover_cuda_prec_precondition,
                          inv_param->clover_cuda_prec_refinement_sloppy, inv_param->clover_cuda_prec_eigensolver};
  loadSloppyCloverQuda(prec);

  popVerbosity();
}

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
  freeUniqueGaugeQuda(QUDA_TWOLINK_LINKS);

  // Need to merge extendedGaugeResident and gaugeFatPrecise/gaugePrecise
  if (extendedGaugeResident) {
    delete extendedGaugeResident;
    extendedGaugeResident = nullptr;
  }
}

// These utility functions are declared w/doxygen above
void freeUniqueSloppyGaugeUtility(GaugeField *&precise, GaugeField *&sloppy, GaugeField *&precondition,
                                  GaugeField *&refinement, GaugeField *&eigensolver)
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

void freeUniqueGaugeUtility(GaugeField *&precise, GaugeField *&sloppy, GaugeField *&precondition, GaugeField *&refinement,
                            GaugeField *&eigensolver, GaugeField *&extended, bool preserve_precise)
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
  case QUDA_TWOLINK_LINKS:
    if (gaugeTwoLink) delete gaugeTwoLink;
    gaugeTwoLink = nullptr;
    break;
  default: errorQuda("Invalid gauge type %d", link_type);
  }
}

void freeGaugeSmearedQuda()
{
  // thin wrapper
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
}

void freeGaugeTwoLinkQuda()
{
  // thin wrapper
  freeUniqueGaugeQuda(QUDA_TWOLINK_LINKS);
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
      gaugeSloppy = new GaugeField(gauge_param);
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
      gaugePrecondition = new GaugeField(gauge_param);
      gaugePrecondition->copy(*gaugePrecise);
    }

    // switch the parameters for creating the mirror refinement cuda gauge field
    gauge_param.reconstruct = recon[2];
    gauge_param.setPrecision(prec[2], true);

    if (gaugeRefinement) errorQuda("gaugeRefinement already exists");

    if (gauge_param.Precision() == gaugeSloppy->Precision() && gauge_param.reconstruct == gaugeSloppy->Reconstruct()) {
      gaugeRefinement = gaugeSloppy;
    } else {
      gaugeRefinement = new GaugeField(gauge_param);
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
      gaugeEigensolver = new GaugeField(gauge_param);
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
      gaugeFatSloppy = new GaugeField(gauge_param);
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
      gaugeFatPrecondition = new GaugeField(gauge_param);
      gaugeFatPrecondition->copy(*gaugeFatPrecise);
    }

    // switch the parameters for creating the mirror refinement cuda gauge field
    gauge_param.setPrecision(prec[2], true);

    if (gaugeFatRefinement) errorQuda("gaugeFatRefinement already exists\n");

    if (gauge_param.Precision() == gaugeFatSloppy->Precision()
        && gauge_param.reconstruct == gaugeFatSloppy->Reconstruct()) {
      gaugeFatRefinement = gaugeFatSloppy;
    } else {
      gaugeFatRefinement = new GaugeField(gauge_param);
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
      gaugeFatEigensolver = new GaugeField(gauge_param);
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
      gaugeLongSloppy = new GaugeField(gauge_param);
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
      gaugeLongPrecondition = new GaugeField(gauge_param);
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
      gaugeLongRefinement = new GaugeField(gauge_param);
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
      gaugeLongEigensolver = new GaugeField(gauge_param);
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

void flushChronoQuda(int i) { flushChrono(i); }

void endQuda(void)
{
  if (!initialized) return;

  {
    auto profile = pushProfile(profileEnd);

    freeGaugeQuda();
    freeCloverQuda();

    flushChrono();

    solutionResident.clear();
    momResident = GaugeField();

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

    assertAllMemFree();
    device::destroy();
  }

  comm_finalize();
  comms_initialized = false;

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
    profileTMCloverForce.Print();
    profileStaggeredForce.Print();
    profileHISQForce.Print();
    profileContract.Print();
    profileContractFT.Print();
    profileBLAS.Print();
    profileCovDev.Print();
    profilePlaq.Print();
    profileGaugeObs.Print();
    profileGaussianSmear.Print();
    profileGaugeSmear.Print();
    profileWFlow.Print();
    profileGFlow.Print();
    profileProject.Print();
    profilePhase.Print();
    profileMomAction.Print();
    profileSinkProject.Print();
    profileEnd.Print();

    profileInit2End.Print();
    TimeProfile::PrintGlobal();

    printLaunchTimer();
    printAPIProfile();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }
}


namespace quda {

  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc)
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
      // check we are safe to cast into a Complex (= std::complex<double>)
      static_assert(sizeof(Complex) == sizeof(double _Complex),
                    "Irreconcilable difference between interface and internal complex number conventions");

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
      diracParam.covdev_mu = inv_param->covdev_mu;
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
    diracParam.distance_pc_alpha0 = inv_param->distance_pc_alpha0;
    diracParam.distance_pc_t0 = inv_param->distance_pc_t0;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on

    if (diracParam.gauge->Precision() != inv_param->cuda_prec)
      errorQuda("Gauge precision %d does not match requested precision %d\n", diracParam.gauge->Precision(),
                inv_param->cuda_prec);

    diracParam.use_mobius_fused_kernel = inv_param->use_mobius_fused_kernel;
  }

  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc)
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

  void setDiracRefineParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc)
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
  void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc, bool comms)
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

  void setDiracEigParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc, bool use_smeared_gauge)
  {
    setDiracParam(diracParam, inv_param, pc);

    if (inv_param->overlap) {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatExtended : gaugeExtended;
      diracParam.fatGauge = gaugeFatExtended;
      diracParam.longGauge = gaugeLongExtended;
    } else if (use_smeared_gauge) {
      if (!gaugeSmeared) errorQuda("No smeared gauge field present");
      if (inv_param->dslash_type == QUDA_LAPLACE_DSLASH) {
        if (gaugeSmeared->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) {
          GaugeFieldParam gauge_param(*gaugePrecise);
          GaugeField gaugeEig(gauge_param);
          copyExtendedGauge(gaugeEig, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
          gaugeEig.exchangeGhost();
          std::swap(gaugeEig, *gaugeSmeared);
        }
        diracParam.gauge = gaugeSmeared;
      } else {
        errorQuda("Smeared gauge field not supported for operator %d", inv_param->dslash_type);
      }
    } else {
      diracParam.gauge = inv_param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatEigensolver : gaugeEigensolver;
      diracParam.fatGauge = gaugeFatEigensolver;
      diracParam.longGauge = gaugeLongEigensolver;
    }
    diracParam.clover = cloverEigensolver;

    for (int i = 0; i < 4; i++) { diracParam.commDim[i] = 1; }

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

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, bool pc_solve)
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

  void createDiracWithRefine(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dRef, QudaInvertParam &param, bool pc_solve)
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

  void createDiracWithEig(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dEig, QudaInvertParam &param, bool pc_solve,
                          bool use_smeared_gauge)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;
    DiracParam diracEigParam;

    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    bool pre_comms_flag = (param.schwarz_type != QUDA_INVALID_SCHWARZ) ? false : true;
    setDiracPreParam(diracPreParam, &param, pc_solve, pre_comms_flag);
    setDiracEigParam(diracEigParam, &param, pc_solve, use_smeared_gauge);

    d = Dirac::create(diracParam); // create the Dirac operator
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
    dEig = Dirac::create(diracEigParam);
  }

}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{
  auto profile = pushProfile(profileDslash, inv_param);
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

  in = in_h;

  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

  logQuda(QUDA_DEBUG_VERBOSE, "In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

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

  distanceReweight(in, *inv_param, true);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH && inv_param->dagger) {
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    ColorSpinorField tmp1(cudaParam);
    ((DiracTwistedCloverPC *)dirac)->TwistCloverInv(tmp1, in, (QudaParity)(1 - parity)); // apply the clover-twist
    dirac->Dslash(out, tmp1, parity); // apply the operator
  } else if (inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH
             || inv_param->dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dirac->Dslash4(out, in, parity);
  } else {
    dirac->Dslash(out, in, parity); // apply the operator
  }
  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

  distanceReweight(out, *inv_param, false);

  out_h = out;

  logQuda(QUDA_DEBUG_VERBOSE, "Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

  delete dirac; // clean up

  popVerbosity();
}

void shiftQuda(void *h_out, void *h_in, int dir, int sym, QudaInvertParam *param)
{
  auto profile = pushProfile(profileCovDev, param);
  const auto &gauge = *gaugePrecise; //(inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  QudaInvertParam &inv_param = *param;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (!gaugePrecise) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param.verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(&inv_param);

  ColorSpinorParam cpuParam(h_in, inv_param, gauge.X(), false, inv_param.input_location);
  ColorSpinorField in_h(cpuParam);
  ColorSpinorParam cudaParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);

  cpuParam.v = h_out;
  cpuParam.location = inv_param.output_location;
  ColorSpinorField out_h(cpuParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField in(cudaParam);
  in = in_h;
  ColorSpinorField out(cudaParam);
  out = in;
  ColorSpinorField tmp(cudaParam);
  tmp = in;

  profileCovDev.TPSTART(QUDA_PROFILE_COMPUTE);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  inv_param.dslash_type = QUDA_COVDEV_DSLASH; // ensure we use the correct dslash
  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, false);

  GaugeCovDev myCovDev(diracParam); // create the Dirac operator

  if (sym & 1) {
    myCovDev.MCD(out, in, dir); // apply the operator
  }
  if (sym & 2) {
    myCovDev.MCD(tmp, in, dir + 4); // apply the operator
  }

  quda::blas::xpy(tmp, out);

  if (sym == 3) quda::blas::ax(0.5, out);

  profileCovDev.TPSTOP(QUDA_PROFILE_COMPUTE);

  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  popVerbosity();
}

void spinTasteQuda(void *h_out, void *h_in, int spin_, int taste, QudaInvertParam *param)
{
  auto profile = pushProfile(profileCovDev, param);
  const auto &gauge = *gaugePrecise; //(inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  QudaInvertParam &inv_param = *param;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (!gaugePrecise) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param.verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(&inv_param);

  ColorSpinorParam cpuParam(h_in, inv_param, gauge.X(), false, inv_param.input_location);
  ColorSpinorField in_h(cpuParam);
  ColorSpinorParam cudaParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);

  cpuParam.v = h_out;
  cpuParam.location = inv_param.output_location;
  ColorSpinorField out_h(cpuParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField in(cudaParam); // cudaColorSpinorField
  in = in_h;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE; // create new field and zero it
  ColorSpinorField out(cudaParam);           // cudaColorSpinorField = 0
  ColorSpinorField tmp(cudaParam);           // cudaColorSpinorField = 0

  profileCovDev.TPSTART(QUDA_PROFILE_COMPUTE);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  inv_param.dslash_type = QUDA_COVDEV_DSLASH; // ensure we use the correct dslash
  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, false);

  GaugeCovDev myCovDev(diracParam); // create the Dirac operator

  int offset = spin_ ^ taste;
  QudaSpinTasteGamma spin = (QudaSpinTasteGamma)spin_;

  constexpr QudaSpinTasteGamma gDirs[4]
    = {QUDA_SPIN_TASTE_GX, QUDA_SPIN_TASTE_GY, QUDA_SPIN_TASTE_GZ, QUDA_SPIN_TASTE_GT};

  switch (offset) {

  case 0: // local
  {
    applySpinTaste(tmp, in, spin);
    applySpinTaste(out, tmp, QUDA_SPIN_TASTE_G5); // antiquark
    break;
  }

  case 1: // one-link X
  case 2: // one-link Y
  case 4: // one-link Z
  case 8: // one-link T
  {
    int cDir = 0;

    if (offset == 1) {
      cDir = 0;
    } else if (offset == 2) {
      cDir = 1;
    } else if (offset == 4) {
      cDir = 2;
    } else if (offset == 8) {
      cDir = 3;
    }

    ColorSpinorField pr1(cudaParam); // cudaColorSpinorField = 0
    applySpinTaste(out, in, spin);
    myCovDev.MCD(tmp, out, cDir);
    myCovDev.MCD(pr1, out, cDir + 4);
    quda::blas::xpy(pr1, tmp);
    applySpinTaste(pr1, tmp, gDirs[cDir]);
    applySpinTaste(out, pr1, QUDA_SPIN_TASTE_G5);
    quda::blas::ax(0.5, out);
    break;
  }

  case 3:  // two-link XY
  case 6:  // two-link YZ
  case 5:  // two-link ZX
  case 9:  // two-link XT
  case 10: // two-link YT
  case 12: // two-link ZT
  {
    int dirs[2];

    {
      if (offset == 3) {
        dirs[0] = 0;
        dirs[1] = 1;
      }
      if (offset == 6) {
        dirs[0] = 1;
        dirs[1] = 2;
      }
      if (offset == 5) {
        dirs[0] = 2;
        dirs[1] = 0;
      }
      if (offset == 9) {
        dirs[0] = 0;
        dirs[1] = 3;
      }
      if (offset == 10) {
        dirs[0] = 1;
        dirs[1] = 3;
      }
      if (offset == 12) {
        dirs[0] = 2;
        dirs[1] = 3;
      }
    }

    ColorSpinorField pr1(cudaParam); // cudaColorSpinorField = 0
    ColorSpinorField acc(cudaParam); // cudaColorSpinorField = 0

    applySpinTaste(out, in, spin);
    // YX result in acc
    myCovDev.MCD(tmp, out, dirs[1]);
    myCovDev.MCD(pr1, out, dirs[1] + 4);
    quda::blas::xpy(pr1, tmp);
    applySpinTaste(pr1, tmp, gDirs[dirs[1]]);
    myCovDev.MCD(tmp, pr1, dirs[0]);
    myCovDev.MCD(acc, pr1, dirs[0] + 4);
    quda::blas::xpy(acc, tmp);
    applySpinTaste(acc, tmp, gDirs[dirs[0]]);
    // XY result in tmp
    myCovDev.MCD(tmp, out, dirs[0]);
    myCovDev.MCD(pr1, out, dirs[0] + 4);
    quda::blas::xpy(pr1, tmp);
    applySpinTaste(pr1, tmp, gDirs[dirs[0]]);
    myCovDev.MCD(tmp, pr1, dirs[1]);
    myCovDev.MCD(out, pr1, dirs[1] + 4);
    quda::blas::xpy(tmp, out);
    applySpinTaste(tmp, out, gDirs[dirs[1]]);

    quda::blas::mxpy(tmp, acc);
    applySpinTaste(out, acc, QUDA_SPIN_TASTE_G5);
    quda::blas::ax(0.125, out);
    break;
  }

  case 14: // three-link 5X
  case 13: // three-link 5Y
  case 11: // three-link 5Z
  case 7:  // three-link 5T
  {
    ColorSpinorField pr1(cudaParam); // cudaColorSpinorField = 0
    ColorSpinorField pr2(cudaParam); // cudaColorSpinorField = 0
    ColorSpinorField acc(cudaParam); // cudaColorSpinorField = 0

    applySpinTaste(out, in, spin);

    int noDir = 0;
    int dirs[3];

    // quda::blas::ax(0.0, acc);
    if (offset == 14) {
      noDir = 0;
    } else if (offset == 13) {
      noDir = 1;
    } else if (offset == 11) {
      noDir = 2;
    } else if (offset == 7) {
      noDir = 3;
    }
    {
      int j = 0;
      for (int i = 0; i < 4; i++) {
        if (i == noDir) continue;
        dirs[j++] = i;
      }
    }

    for (int i = 0; i < 3; i++) {

      const int d1 = dirs[(i + 0) % 3];
      const int d2 = dirs[(i + 1) % 3];
      const int d3 = dirs[(i + 2) % 3];

      // Accumulate result in acc
      myCovDev.MCD(tmp, out, d1);
      myCovDev.MCD(pr1, out, d1 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d1]);
      myCovDev.MCD(tmp, pr1, d2);
      myCovDev.MCD(pr2, pr1, d2 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[d2]);
      myCovDev.MCD(tmp, pr2, d3);
      myCovDev.MCD(pr1, pr2, d3 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d3]);
      quda::blas::xpy(pr1, acc);

      // Accumulate result in acc
      myCovDev.MCD(tmp, out, d3);
      myCovDev.MCD(pr1, out, d3 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d3]);
      myCovDev.MCD(tmp, pr1, d2);
      myCovDev.MCD(pr2, pr1, d2 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[d2]);
      myCovDev.MCD(tmp, pr2, d1);
      myCovDev.MCD(pr1, pr2, d1 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d1]);
      quda::blas::mxpy(pr1, acc);
    }

    applySpinTaste(out, acc, QUDA_SPIN_TASTE_G5);
    quda::blas::ax(0.125 / 6., out);
    break;
  }

  case 15: // four-link 5
  {
    const int dPlus[12][4] = {{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}, {0, 3, 1, 2}, {1, 3, 2, 0}, {2, 3, 0, 1},
                              {3, 2, 1, 0}, {3, 0, 2, 1}, {3, 1, 0, 2}, {2, 1, 3, 0}, {0, 2, 3, 1}, {1, 0, 3, 2}};
    const int dMnus[12][4] = {{0, 2, 1, 3}, {1, 0, 2, 3}, {2, 1, 0, 3}, {0, 3, 2, 1}, {1, 3, 0, 2}, {2, 3, 1, 0},
                              {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 0, 1, 2}, {1, 2, 3, 0}, {2, 0, 3, 1}, {0, 1, 3, 2}};

    ColorSpinorField pr1(cudaParam); // cudaColorSpinorField = 0
    ColorSpinorField pr2(cudaParam); // cudaColorSpinorField = 0
    ColorSpinorField acc(cudaParam); // cudaColorSpinorField = 0

    applySpinTaste(out, in, spin);

    for (int i = 0; i < 12; i++) {

      const int d1 = dPlus[i][0];
      const int d2 = dPlus[i][1];
      const int d3 = dPlus[i][2];
      const int d4 = dPlus[i][3];

      // Accumulate result in acc
      myCovDev.MCD(tmp, out, d1);
      myCovDev.MCD(pr1, out, d1 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d1]);
      myCovDev.MCD(tmp, pr1, d2);
      myCovDev.MCD(pr2, pr1, d2 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[d2]);
      myCovDev.MCD(tmp, pr2, d3);
      myCovDev.MCD(pr1, pr2, d3 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[d3]);
      myCovDev.MCD(tmp, pr1, d4);
      myCovDev.MCD(pr2, pr1, d4 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[d4]);
      quda::blas::xpy(pr2, acc);

      const int m1 = dMnus[i][0];
      const int m2 = dMnus[i][1];
      const int m3 = dMnus[i][2];
      const int m4 = dMnus[i][3];

      // Accumulate result in acc
      myCovDev.MCD(tmp, out, m1);
      myCovDev.MCD(pr1, out, m1 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[m1]);
      myCovDev.MCD(tmp, pr1, m2);
      myCovDev.MCD(pr2, pr1, m2 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[m2]);
      myCovDev.MCD(tmp, pr2, m3);
      myCovDev.MCD(pr1, pr2, m3 + 4);
      quda::blas::xpy(pr1, tmp);
      applySpinTaste(pr1, tmp, gDirs[m3]);
      myCovDev.MCD(tmp, pr1, m4);
      myCovDev.MCD(pr2, pr1, m4 + 4);
      quda::blas::xpy(pr2, tmp);
      applySpinTaste(pr2, tmp, gDirs[m4]);
      quda::blas::mxpy(pr2, acc);
    }

    applySpinTaste(out, acc, QUDA_SPIN_TASTE_G5);
    quda::blas::ax(0.0625 / 24., out);
    break;
  }
  }

  // FIXME: This is not exactly all covDev
  profileCovDev.TPSTOP(QUDA_PROFILE_COMPUTE);

  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  popVerbosity();
}

void covDevQuda(void *h_out, void *h_in, int dir, QudaInvertParam *param)
{
  auto profile = pushProfile(profileCovDev, param);

  QudaInvertParam &inv_param = *param;
  const auto &gauge = *gaugePrecise; //(inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  // if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
  //    || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
  if (!gaugePrecise) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param.verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(&inv_param);

  ColorSpinorParam cpuParam(h_in, inv_param, gauge.X(), false, inv_param.input_location);
  ColorSpinorField in_h(cpuParam);
  ColorSpinorParam cudaParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);

  cpuParam.v = h_out;
  cpuParam.location = inv_param.output_location;
  ColorSpinorField out_h(cpuParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField in(cudaParam); // cudaColorSpinorField
  in = in_h;
  ColorSpinorField out(cudaParam); // cudaColorSpinorField
  out = in;

  profileCovDev.TPSTART(QUDA_PROFILE_COMPUTE);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  inv_param.dslash_type = QUDA_COVDEV_DSLASH; // ensure we use the correct dslash
  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, false);

  GaugeCovDev myCovDev(diracParam); // create the Dirac operator
  myCovDev.MCD(out, in, dir);       // apply the operator
  profileCovDev.TPSTOP(QUDA_PROFILE_COMPUTE);

  out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  popVerbosity();
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

  logQuda(QUDA_DEBUG_VERBOSE, "In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  distanceReweight(in, *inv_param, true);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in); // apply the operator
  delete dirac; // clean up

  distanceReweight(out, *inv_param, false);

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

  logQuda(QUDA_DEBUG_VERBOSE, "Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));
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

  logQuda(QUDA_DEBUG_VERBOSE, "In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField out(cudaParam);

  //  double kappa = inv_param->kappa;
  //  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= gaugePrecise->anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  distanceReweight(in, *inv_param, true);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  distanceReweight(out, *inv_param, false);

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

  logQuda(QUDA_DEBUG_VERBOSE, "Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));
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

quda::GaugeField *checkGauge(QudaInvertParam *param)
{
  quda::GaugeField *U = param->dslash_type == QUDA_ASQTAD_DSLASH ? gaugeFatPrecise :
                                                                   gaugePrecise;

  if (U == nullptr)
    errorQuda("Precise gauge %sfield doesn't exist", param->dslash_type == QUDA_ASQTAD_DSLASH ? "fat " : "");

  if (param->cuda_prec != U->Precision()) {
    errorQuda("Solve precision %d doesn't match gauge precision %d", param->cuda_prec, U->Precision());
  }

  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
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
    if (param->overlap && gaugeExtended == nullptr) errorQuda("Extended gauge field doesn't exist");
  } else {
    if (gaugeLongPrecise == nullptr) errorQuda("Precise gauge long field doesn't exist");

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
    if (param->overlap && gaugeFatExtended == nullptr) errorQuda("Extended gauge fat field doesn't exist");

    if (gaugeLongSloppy == nullptr) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == nullptr) errorQuda("Precondition gauge long field doesn't exist");
    if (gaugeLongRefinement == nullptr) errorQuda("Refinement gauge long field doesn't exist");
    if (gaugeLongEigensolver == nullptr) errorQuda("Eigensolver gauge long field doesn't exist");
    if (param->overlap && gaugeLongExtended == nullptr) errorQuda("Extended gauge long field doesn't exist");
  }

  checkClover(param);

  return U;
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

  logQuda(QUDA_DEBUG_VERBOSE, "In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

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

  logQuda(QUDA_DEBUG_VERBOSE, "Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));
  popVerbosity();
}

void eigensolveQuda(void **host_evecs, double _Complex *host_evals, QudaEigParam *eig_param)
{
  if (!initialized) errorQuda("QUDA not initialized");

  // Transfer the inv param structure contained in eig_param.
  // This will define the operator to be eigensolved.
  QudaInvertParam *inv_param = eig_param->invert_param;

  auto profile = pushProfile(profileEigensolve, inv_param);

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
  GaugeField *cudaGauge = checkGauge(inv_param);

  // Set iter statistics to zero
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
  Dirac *dEig = nullptr;

  // Create the dirac operator with a sloppy and a precon.
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  createDiracWithEig(d, dSloppy, dPre, dEig, *inv_param, pc_solve, eig_param->use_smeared_gauge);
  Dirac &dirac = *dEig;
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

  // Perform the eigensolve
  if (eig_param->arpack_check) {
    arpack_solve(host_evecs_, evals, *m, eig_param);
  } else {
    auto *eig_solve = quda::EigenSolver::create(eig_param, *m);
    (*eig_solve)(kSpace, evals);
    delete eig_solve;
  }

  delete m;

  // Transfer Eigenpairs back to host if using GPU eigensolver. The copy
  // will automatically rotate from device UKQCD gamma basis to the
  // host side gamma basis.
  memcpy(host_evals, evals.data(), sizeof(Complex) * evals.size());

  if (!(eig_param->arpack_check)) {
    for (int i = 0; i < n_eig; i++) host_evecs_[i] = kSpace[i];
  }

  delete d;
  delete dSloppy;
  delete dPre;
  delete dEig;

  popVerbosity();
}

multigrid_solver::multigrid_solver(QudaMultigridParam &mg_param)
{
  QudaInvertParam *param = mg_param.invert_param;
  // set whether we are going to use native or generic blas
  blas_lapack::set_native(param->native_blas_lapack);

  checkMultigridParam(&mg_param);
  GaugeField *cudaGauge = checkGauge(param);

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

  for (int i = 0; i < mg_param.n_vec[0]; i++) { B[i] = ColorSpinorField(csParam); }

  // fill out the MG parameters for the fine level
  mgParam = new MGParam(mg_param, B, m, mSmooth, mSmoothSloppy);

  mg = new MG(*mgParam);
  mgParam->updateInvertParam(*param);
}

void *newMultigridQuda(QudaMultigridParam *mg_param)
{
  profilerStart(__func__);
  auto profile = pushProfile(profileInvert, mg_param->invert_param);
  pushVerbosity(mg_param->invert_param->verbosity);

  auto *mg = new multigrid_solver(*mg_param);

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
  auto profile = pushProfile(profileInvert, mg_param->invert_param);
  pushVerbosity(mg_param->invert_param->verbosity);

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

  profileInvert.TPSTOP(QUDA_PROFILE_PREAMBLE);

  popVerbosity();
  profilerStop(__func__);
}

void dumpMultigridQuda(void *mg_, QudaMultigridParam *mg_param)
{
  profilerStart(__func__);
  auto profile = pushProfile(profileInvert, mg_param->invert_param);
  pushVerbosity(mg_param->invert_param->verbosity);

  auto *mg = static_cast<multigrid_solver*>(mg_);
  checkMultigridParam(mg_param);
  checkGauge(mg_param->invert_param);

  mg->mg->dumpNullVectors();

  popVerbosity();
  profilerStop(__func__);
}

deflated_solver::deflated_solver(QudaEigParam &eig_param, TimeProfile &profile)
  : d(nullptr), m(nullptr), RV(nullptr), deflParam(nullptr), defl(nullptr),  profile(profile) {

  QudaInvertParam *param = eig_param.invert_param;

  if (param->inv_type != QUDA_EIGCG_INVERTER && param->inv_type != QUDA_INC_EIGCG_INVERTER) return;

  GaugeField *cudaGauge = checkGauge(param);

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
    ritzParam.mem_type = QUDA_MEMORY_HOST_PINNED;
  }

  int ritzVolume = 1;
  for(int d = 0; d < ritzParam.nDim; d++) ritzVolume *= ritzParam.x[d];

  if (getVerbosity() == QUDA_DEBUG_VERBOSE) {

    size_t byte_estimate = (size_t)ritzParam.composite_dim*(size_t)ritzVolume*(ritzParam.nColor*ritzParam.nSpin*ritzParam.Precision());
    printfQuda("allocating bytes: %lu (lattice volume %d, prec %d)", byte_estimate, ritzVolume, ritzParam.Precision());
    if (ritzParam.mem_type == QUDA_MEMORY_DEVICE)
      printfQuda("Using device memory type.\n");
    else if (ritzParam.mem_type == QUDA_MEMORY_MAPPED)
      printfQuda("Using mapped memory type.\n");
  }

  RV = ColorSpinorField::Create(ritzParam);

  deflParam = new DeflationParam(eig_param, RV, *m);

  defl = new Deflation(*deflParam, profile);
}

void* newDeflationQuda(QudaEigParam *eig_param) {
  auto profile = pushProfile(profileInvert, eig_param->invert_param);
  auto *defl = new deflated_solver(*eig_param, profileInvert);
  saveProfile(__func__);
  flushProfile();
  return static_cast<void*>(defl);
}

void destroyDeflationQuda(void *df) {
  delete static_cast<deflated_solver*>(df);
}

void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  auto profile = pushProfile(profileInvert, param);
  profilerStart(__func__);

  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  checkInvertParam(param, hp_x, hp_b);

  // check the gauge fields have been created
  GaugeField *cudaGauge = checkGauge(param);

  solve({hp_x}, {hp_b}, *param, *cudaGauge);

  if (param->use_resident_solution && !param->make_resident_solution) solutionResident.clear();

  profilerStop(__func__);
  popVerbosity();
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
                      Interface op, Args... args)
{
  /**
    Here we first re-distribute gauge, color spinor, and clover field to sub-partitions, then call either invertQuda or dslashQuda.
    - For clover and gauge field, we re-distribute the host clover side fields, restore them after.
    - For color spinor field, we re-distribute the host side source fields, and re-collect the host side solution fields.
  */

  profilerStart(__func__);
  auto profile = pushProfile(profileInvertMultiSrc, param);

  CommKey split_key = {param->split_grid[0], param->split_grid[1], param->split_grid[2], param->split_grid[3]};
  int num_sub_partition = quda::product(split_key);

  if (!split_key.is_valid()) {
    errorQuda("split_key = [%d,%d,%d,%d] is not valid", split_key[0], split_key[1], split_key[2], split_key[3]);
  }

  checkInvertParam(param, _hp_x[0], _hp_b[0]);

  if (num_sub_partition == 1) { // In this case we don't split the grid.

    std::vector<void *> x(param->num_src), b(param->num_src);
    for (auto i = 0u; i < x.size(); i++) x[i] = _hp_x[i];
    for (auto i = 0u; i < b.size(); i++) b[i] = _hp_b[i];
    op(x, b, *param, args...);

  } else {

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
    if (param->inv_type_precondition == QUDA_MG_INVERTER) errorQuda("Split Grid does NOT work with MG yet");

    checkInvertParam(param, _hp_x[0], _hp_b[0]);

    // Asqtad loads fat and long links; all others (including naive staggered) load thin links
    bool is_asqtad = Dirac::is_asqtad(param->dslash_type);

    GaugeBundleBackup thin_links_bkup;
    GaugeBundleBackup fat_links_bkup;
    GaugeBundleBackup long_links_bkup;

    if (is_asqtad) {
      if (!gaugeFatPrecise || !gaugeLongPrecise)
        errorQuda("Both milc_fatlinks and milc_longlinks need to be non-null for asqtad-type dslash");

      fat_links_bkup.backup(gaugeFatPrecise, gaugeFatSloppy, gaugeFatPrecondition, gaugeFatRefinement,
                            gaugeFatEigensolver, gaugeFatExtended);
      long_links_bkup.backup(gaugeLongPrecise, gaugeLongSloppy, gaugeLongPrecondition, gaugeLongRefinement,
                             gaugeLongEigensolver, gaugeLongExtended);
    } else {
      if (!gaugePrecise) errorQuda("h_gauge is null for a Wilson-type or naive staggered dslash");
      thin_links_bkup.backup(gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver,
                             gaugeExtended);
    }

    // Deal with Spinors
    bool pc_solution
      = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

    lat_dim_t X = is_asqtad ? gaugeFatPrecise->X() : gaugePrecise->X();

    ColorSpinorParam spinorParam(_hp_b[0], *param, X, pc_solution, param->input_location);
    std::vector<ColorSpinorField> _h_b(param->num_src);
    std::vector<ColorSpinorField> _h_x(param->num_src); // wrappers -- for output

    // Create Aliases
    for (int i = 0; i < param->num_src; i++) {
      spinorParam.v = _hp_b[i];
      _h_b[i] = ColorSpinorField(spinorParam);
    }

    for (int i = 0; i < param->num_src; i++) {
      spinorParam.v = _hp_x[i];
      _h_x[i] = ColorSpinorField(spinorParam);
    }

    // Gauge fields/params
    GaugeFieldParam gf_param;
    GaugeFieldParam milc_fatlink_param;
    GaugeFieldParam milc_longlink_param;

    logQuda(QUDA_DEBUG_VERBOSE, "Spliting the grid into sub-partitions: (%2d,%2d,%2d,%2d) / (%2d,%2d,%2d,%2d)\n",
            comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3), split_key[0], split_key[1], split_key[2], split_key[3]);

    if (!is_asqtad)
      gf_param = GaugeFieldParam(*(thin_links_bkup.precise));
    else {
      milc_fatlink_param = GaugeFieldParam(*(fat_links_bkup.precise));
      milc_longlink_param = GaugeFieldParam(*(long_links_bkup.precise));
    }

    for (int d = 0; d < CommKey::n_dim; d++) {
      if (comm_dim(d) % split_key[d] != 0) {
        errorQuda("Split not possible: %2d %% %2d != 0", comm_dim(d), split_key[d]);
      }
      if (!is_asqtad) {
        gf_param.x[d] *= split_key[d];
        gf_param.pad *= split_key[d];
      } else {
        milc_fatlink_param.x[d] *= split_key[d];
        milc_longlink_param.x[d] *= split_key[d];
      }
    }

    quda::GaugeField *collected_gauge = nullptr;
    quda::GaugeField *collected_milc_fatlink_field = nullptr;
    quda::GaugeField *collected_milc_longlink_field = nullptr;

    if (!is_asqtad) {
      gf_param.create = QUDA_NULL_FIELD_CREATE;
      collected_gauge = new quda::GaugeField(gf_param);
      quda::split_field(*collected_gauge, {*(thin_links_bkup.precise)}, split_key);
    } else {
      std::vector<quda::GaugeField *> v_g(1);

      milc_fatlink_param.create = QUDA_NULL_FIELD_CREATE;
      collected_milc_fatlink_field = new GaugeField(milc_fatlink_param);
      quda::split_field(*collected_milc_fatlink_field, {*(fat_links_bkup.precise)}, split_key);

      milc_longlink_param.create = QUDA_NULL_FIELD_CREATE;
      collected_milc_longlink_field = new GaugeField(milc_longlink_param);
      quda::split_field(*collected_milc_longlink_field, {*(long_links_bkup.precise)}, split_key);
    }

    //  ------ Clover field
    //
    quda::CloverField *collected_clover = nullptr;

    CloverBundleBackup clov_bkup;
    bool is_clover = param->dslash_type == QUDA_CLOVER_WILSON_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || param->dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH;

    if (is_clover) {
      if (param->clover_coeff == 0.0 && param->clover_csw == 0.0)
        errorQuda("called with neither clover term nor inverse and clover coefficient nor Csw not set");
      if (gaugePrecise->Anisotropy() != 1.0) errorQuda("cannot compute anisotropic clover field");

      clov_bkup.backup(cloverPrecise, cloverSloppy, cloverPrecondition, cloverRefinement, cloverEigensolver);

      CloverFieldParam clover_param(*clov_bkup.precise);

      for (int d = 0; d < CommKey::n_dim; d++) { clover_param.x[d] *= split_key[d]; }
      clover_param.create = QUDA_NULL_FIELD_CREATE;
      collected_clover = new CloverField(clover_param);
      quda::split_field(*collected_clover, {*clov_bkup.precise}, split_key); // Clover uses 4d even-odd preconditioning.
    }

    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_PREAMBLE);

    comm_barrier();

    // Split input fermion field
    // we will convert the host fields with external layout to device fields with
    // native layout for the split. We will do the splitting with the collected fields
    // on the device already
    quda::ColorSpinorParam cs_param_split(_h_b[0]);
    cs_param_split.setPrecision(param->cuda_prec, param->cuda_prec, true); // Native format
    cs_param_split.location = QUDA_CUDA_FIELD_LOCATION;                    // Device side

    // Expand the geometry for the collected fields
    for (int d = 0; d < CommKey::n_dim; d++) { cs_param_split.x[d] *= split_key[d]; }
    std::vector<quda::ColorSpinorField> _collect_b(param->num_src_per_sub_partition, cs_param_split);
    std::vector<quda::ColorSpinorField> _collect_x(param->num_src_per_sub_partition, cs_param_split);

    // We will use these dev_buf fields to download (if needed) and convert
    // external fields into internal foramt
    quda::ColorSpinorParam devbuf_param(_h_b[0]);
    devbuf_param.location = cs_param_split.location;                     // same location as collected (for copyOffset)
    devbuf_param.setPrecision(param->cuda_prec, param->cuda_prec, true); // Native format
    std::vector<quda::ColorSpinorField> dev_buf(num_sub_partition, devbuf_param);

    for (int n = 0; n < param->num_src_per_sub_partition; n++) {
      // Download and change to Native Order and split
      for (int j = 0; j < num_sub_partition; j++) dev_buf[j].copy(_h_b[n * num_sub_partition + j]);
      split_field(_collect_b[n], {dev_buf.begin(), dev_buf.end()}, split_key, pc_type);
    }

    // Switch communicator
    comm_barrier();

    push_communicator(split_key);
    updateR();
    comm_barrier();

    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_PREAMBLE);

    // Load 'collected gauge field'
    logQuda(QUDA_DEBUG_VERBOSE, "Split grid loading gauge field...\n");
    if (!is_asqtad) {
      setupGaugeFields(collected_gauge, gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver,
                       gaugeExtended, thin_links_bkup, profile.profile);

    } else {
      setupGaugeFields(collected_milc_fatlink_field, gaugeFatPrecise, gaugeFatSloppy, gaugeFatPrecondition,
                       gaugeFatRefinement, gaugeFatEigensolver, gaugeFatExtended, fat_links_bkup, profile.profile);

      setupGaugeFields(collected_milc_longlink_field, gaugeLongPrecise, gaugeLongSloppy, gaugeLongPrecondition,
                       gaugeLongRefinement, gaugeLongEigensolver, gaugeLongExtended, long_links_bkup, profile.profile);
    }
    logQuda(QUDA_DEBUG_VERBOSE, "Split grid loaded gauge field...\n");

    // Load 'collected clover field'
    if (is_clover) {
      logQuda(QUDA_DEBUG_VERBOSE, "Split grid loading clover field...\n");
      setupCloverFields(collected_clover, cloverPrecise, cloverSloppy, cloverPrecondition, cloverRefinement,
                        cloverEigensolver, clov_bkup);
      logQuda(QUDA_DEBUG_VERBOSE, "Split grid loaded clover field...\n");
    }

    // Make a copy of the params we can mess with
    auto param_copy = *param;

    // Set solver input/output param location
    param_copy.input_location = cs_param_split.location;
    param_copy.output_location = cs_param_split.location;

    // Important: Don't use accessors for external formats any more
    // Since input fields are in Native order now
    param_copy.dirac_order = QUDA_INTERNAL_DIRAC_ORDER;

    // We need to set the cpu_prec in the param_copy, because the op() passed in
    // to us will try to create wrappers to the pointers we pass in. They expect
    // the input spinors to be on the host, and will use param_copy.cpu_prec to set
    // the precision. We want to avoid the situation, where the internal prec and the
    // cpu_prec are somehow different.
    param_copy.cpu_prec = _collect_b[0].Precision();

    // Do the solves
    std::vector<void *> x_raw(param->num_src_per_sub_partition);
    std::vector<void *> b_raw(param->num_src_per_sub_partition);
    for (auto i = 0u; i < x_raw.size(); i++) x_raw[i] = _collect_x[i].data();
    for (auto i = 0u; i < b_raw.size(); i++) b_raw[i] = _collect_b[i].data();
    op(x_raw, b_raw, param_copy, args...);

    auto split_rank = comm_rank();

    profileInvertMultiSrc.TPSTART(QUDA_PROFILE_EPILOGUE);
    push_communicator(default_comm_key);
    updateR();
    comm_barrier();

    // back to the default communicator, now join the param entries
    joinInvertParam(*param, param_copy, split_key, split_rank);

    // Join spinors: _h_x are aliases to host pointers in 'external order: QDP++, QDP-JIT, etc'
    for (int n = 0; n < param->num_src_per_sub_partition; n++) {
      // join fields
      join_field({dev_buf.begin(), dev_buf.end()}, _collect_x[n], split_key, pc_type);

      // export to desired location and layout
      for (int j = 0; j < num_sub_partition; j++) _h_x[n * num_sub_partition + j].copy(dev_buf[j]);
    }

    profileInvertMultiSrc.TPSTOP(QUDA_PROFILE_EPILOGUE);

    // Restore the gauge field
    if (!is_asqtad) {

      freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
      gaugePrecise = thin_links_bkup.precise;
      gaugeSloppy = thin_links_bkup.sloppy;
      gaugePrecondition = thin_links_bkup.precondition;
      gaugeRefinement = thin_links_bkup.refinement;
      gaugeEigensolver = thin_links_bkup.eigensolver;
      gaugeExtended = thin_links_bkup.extended;

    } else {
      // freeGaugeQuda();

      freeUniqueGaugeQuda(QUDA_ASQTAD_FAT_LINKS);
      gaugeFatPrecise = fat_links_bkup.precise;
      gaugeFatSloppy = fat_links_bkup.sloppy;
      gaugeFatPrecondition = fat_links_bkup.precondition;
      gaugeFatRefinement = fat_links_bkup.refinement;
      gaugeFatEigensolver = fat_links_bkup.eigensolver;
      gaugeFatExtended = fat_links_bkup.extended;

      freeUniqueGaugeQuda(QUDA_ASQTAD_LONG_LINKS);
      gaugeLongPrecise = long_links_bkup.precise;
      gaugeLongSloppy = long_links_bkup.sloppy;
      gaugeLongPrecondition = long_links_bkup.precondition;
      gaugeLongRefinement = long_links_bkup.refinement;
      gaugeLongEigensolver = long_links_bkup.eigensolver;
      gaugeLongExtended = long_links_bkup.extended;
    }

    if (is_clover) {
      freeCloverQuda();
      cloverPrecise = clov_bkup.precise;
      cloverSloppy = clov_bkup.sloppy;
      cloverPrecondition = clov_bkup.precondition;
      cloverRefinement = clov_bkup.refinement;
      cloverEigensolver = clov_bkup.eigensolver;
    }
  }

  profilerStop(__func__);
}

void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param)
{
  auto op = [](const std::vector<void *> &_x, const std::vector<void *> &_b, QudaInvertParam &param) {
    // check the gauge fields have been created
    GaugeField *gauge = checkGauge(&param);
    solve(_x, _b, param, *gauge);
  };
  callMultiSrcQuda(_hp_x, _hp_b, param, op);
}

void dslashMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity)
{
  auto op = [](const std::vector<void *> &_x, const std::vector<void *> &_b, QudaInvertParam &param, QudaParity parity) {
    for (auto i = 0u; i < _b.size(); i++) dslashQuda(_x[i], _b[i], &param, parity);
  };
  callMultiSrcQuda(_hp_x, _hp_b, param, op, parity);
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
  auto profile = pushProfile(profileMulti, param);
  profilerStart(__func__);

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
      errorQuda("For Staggered-type fermions, multi-shift solver only supports MATPC solution type");
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

  param->iter = 0;

  for (int i=0; i<param->num_offset-1; i++) {
    for (int j=i+1; j<param->num_offset; j++) {
      if (param->offset[i] > param->offset[j])
        errorQuda("Offsets must be ordered from smallest to largest");
    }
  }

  if (param->distance_pc_alpha0 != 0.0 && param->distance_pc_t0 >= 0) {
    errorQuda("Multi-shift solver does not support distance preconditioning");
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

  std::vector<ColorSpinorField> h_x;
  h_x.resize(param->num_offset);

  cpuParam.location = param->output_location;
  for(int i=0; i < param->num_offset; i++) {
    cpuParam.v = hp_x[i];
    h_x[i] = ColorSpinorField(cpuParam);
  }

  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param, QUDA_CUDA_FIELD_LOCATION);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  cudaParam.field = &h_b;
  ColorSpinorField b(cudaParam); // Creates b and downloads h_b to it

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

  profileMulti.TPSTART(QUDA_PROFILE_PREAMBLE);

  // Check source norms
  double nb = blas::norm2(b);
  if (nb==0.0) errorQuda("Source has zero norm");
  logQuda(QUDA_VERBOSE, "Source: %g\n", nb);

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
    MultiShiftCG cg_m(*m, *mSloppy, solverParam);
    cg_m(x, b, p, r2_old);
  }
  solverParam.updateInvertParam(*param);

  delete m;
  delete mSloppy;

  if (param->compute_true_res) {
    // check each shift has the desired tolerance and use sequential CG to refine
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField r(cudaParam);
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
        logQuda(QUDA_SUMMARIZE, "Refining shift %d: L2 residual %e / %e, heavy quark %e / %e (actual / requested)\n", i,
                param->true_res_offset[i], param->tol_offset[i], rsd_hq, tol_hq);

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
          MinResExt mre(*m, orthogonal, apply_mat, hermitian);
          mre(x[i], b, z, q);
        }

        SolverParam solverParam(refineparam);
        solverParam.iter = 0;
        solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
        solverParam.tol = (param->tol_offset[i] > 0.0 ? param->tol_offset[i] : iter_tol); // set L2 tolerance
        solverParam.tol_hq = param->tol_hq_offset[i];                                     // set heavy quark tolerance
        solverParam.delta = param->reliable_delta_refinement;

        {
          CG cg(*m, *mSloppy, *mSloppy, *mSloppy, solverParam);
          if (i == 0)
            cg(x[i], b, p[i], r2_old[i]);
          else
            cg(x[i], b);
        }

        solverParam.true_res_offset[i] = static_cast<double>(solverParam.true_res);
        solverParam.true_res_hq_offset[i] = static_cast<double>(solverParam.true_res_hq);
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

    logQuda(QUDA_VERBOSE, "Solution %d = %g\n", i, blas::norm2(x[i]));
    if (!param->make_resident_solution) h_x[i] = x[i];
  }

  profileMulti.TPSTART(QUDA_PROFILE_EPILOGUE);

  if (!param->make_resident_solution) solutionResident.clear();

  profileMulti.TPSTOP(QUDA_PROFILE_EPILOGUE);

  delete d;
  delete dSloppy;
  delete dPre;
  delete dRefine;

  profilerStop(__func__);
  popVerbosity();
}

void computeKSLinkQuda(void *fatlink, void *longlink, void *ulink, void *inlink, double *path_coeff, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileFatLink);
  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, fatlink, QUDA_GENERAL_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuFatLink(gParam); // create the host fatlink
  gParam.gauge = longlink;
  GaugeField cpuLongLink(gParam); // create the host longlink
  gParam.gauge = ulink;
  GaugeField cpuUnitarizedLink(gParam);
  gParam.link_type = param->type;
  gParam.gauge = inlink;
  GaugeField cpuInLink(gParam); // create the host sitelink

  // create the device fields
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField *cudaInLink = new GaugeField(gParam);

  cudaInLink->copy(cpuInLink);
  GaugeField *cudaInLinkEx = createExtendedGauge(*cudaInLink, R, profileFatLink);

  delete cudaInLink;

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(param->cuda_prec, true);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  if (longlink) {
    GaugeField longLink(gParam);
    longKSLink(longLink, *cudaInLinkEx, path_coeff);
    cpuLongLink.copy(longLink);
  }

  GaugeField fatLink(gParam);
  fatKSLink(fatLink, *cudaInLinkEx, path_coeff);
  if (fatlink) cpuFatLink.copy(fatLink);

  if (ulink) {
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    quda::setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                                     svd_abs_error);

    GaugeField unitarizedLink(gParam);

    *num_failures_h = 0;
    quda::unitarizeLinks(unitarizedLink, fatLink, num_failures_d); // unitarize on the gpu
    if (*num_failures_h > 0)
      errorQuda("Error in unitarization component of the hisq fattening: %d failures", *num_failures_h);

    // project onto SU(3) if using the Chroma convention
    if (param->staggered_phase_type == QUDA_STAGGERED_PHASE_CHROMA) {
      *num_failures_h = 0;
      const double tol = unitarizedLink.toleranceSU3();
      if (unitarizedLink.StaggeredPhaseApplied()) unitarizedLink.removeStaggeredPhase();
      projectSU3(unitarizedLink, tol, num_failures_d);
      if (!unitarizedLink.StaggeredPhaseApplied() && param->staggered_phase_applied)
        unitarizedLink.applyStaggeredPhase();
      if (*num_failures_h > 0) errorQuda("Error in the SU(3) unitarization: %d failures\n", *num_failures_h);
    }

    cpuUnitarizedLink.copy(unitarizedLink);
  }

  delete cudaInLinkEx;
}

void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileGaussianSmear);
  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, inlink, QUDA_ASQTAD_LONG_LINKS);
  gParam.gauge = twolink;
  GaugeField cpuTwoLink(gParam); // create the host twolink

  GaugeField *cudaInLinkEx = nullptr;

  if (inlink) {
    gParam.link_type = param->type;
    gParam.gauge     = inlink;
    GaugeField cpuInLink(gParam); // create the host sitelink

    // create the device fields
    gParam.reconstruct = param->reconstruct;
    gParam.setPrecision(param->cuda_prec, true);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField cudaInLink(gParam);

    cudaInLink.copy(cpuInLink);
    cudaInLinkEx = createExtendedGauge(cudaInLink, R, profileGaussianSmear);
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

  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = new GaugeField(gsParam);

  computeTwoLink(*gaugeSmeared, *cudaInLinkEx);
  gaugeSmeared->exchangeGhost();

  cpuTwoLink.copy(*gaugeSmeared);

  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  delete cudaInLinkEx;
}

int computeGaugeForceQuda(void* mom, void* siteLink,  int*** input_path_buf, int* path_length,
			  double* loop_coeff, int num_paths, int max_length, double eb3, QudaGaugeParam* qudaGaugeParam)
{
  auto profile = pushProfile(profileGaugeForce);
  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(*qudaGaugeParam, siteLink);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuSiteLink = !qudaGaugeParam->use_resident_gauge ? GaugeField(gParam) : GaugeField();

  if (qudaGaugeParam->use_resident_gauge && !gaugePrecise) errorQuda("No resident gauge field to use");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuSiteLink;
  gParam.reconstruct = qudaGaugeParam->reconstruct;
  gParam.setPrecision(qudaGaugeParam->cuda_prec, true);
  GaugeField cudaSiteLink = qudaGaugeParam->use_resident_gauge ? gaugePrecise->create_alias() : GaugeField(gParam);

  GaugeFieldParam gParamMom(*qudaGaugeParam, mom, QUDA_ASQTAD_MOM_LINKS);
  gParamMom.location = QUDA_CPU_FIELD_LOCATION;

  GaugeField cpuMom = !qudaGaugeParam->use_resident_mom ? GaugeField(gParamMom) : GaugeField();

  if (qudaGaugeParam->use_resident_mom && momResident.empty()) errorQuda("No resident momentum field to use");
  gParamMom.location = QUDA_CUDA_FIELD_LOCATION;
  gParamMom.create = qudaGaugeParam->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_COPY_FIELD_CREATE;
  gParamMom.field = &cpuMom;
  gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.setPrecision(qudaGaugeParam->cuda_prec, true);

  GaugeField cudaMom = qudaGaugeParam->use_resident_mom ? momResident.create_alias() : GaugeField(gParamMom);
  if (qudaGaugeParam->use_resident_mom && qudaGaugeParam->overwrite_mom) cudaMom.zero();

  GaugeField *cudaGauge = createExtendedGauge(cudaSiteLink, R, profileGaugeForce);
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
  if (!forceMonitor()) {
    gaugeForce(cudaMom, *cudaGauge, eb3, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);
  } else {
    // if we are monitoring the force, separate the force computation from the momentum update
    GaugeFieldParam gParam(cudaMom);
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    GaugeField force(gParam);
    gaugeForce(force, *cudaGauge, 1.0, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);
    updateMomentum(cudaMom, eb3, force, "gauge");
  }

  if (qudaGaugeParam->return_result_mom) cpuMom.copy(cudaMom);

  if (qudaGaugeParam->make_resident_gauge && !qudaGaugeParam->use_resident_gauge) {
    if (gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaSiteLink);
  }

  if (qudaGaugeParam->make_resident_mom && !qudaGaugeParam->use_resident_mom)
    std::exchange(momResident, cudaMom);
  else if (!qudaGaugeParam->make_resident_mom)
    momResident = GaugeField();

  if (qudaGaugeParam->make_resident_gauge) {
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaGauge;
  } else {
    delete cudaGauge;
  }

  return 0;
}

int computeGaugePathQuda(void *out, void *siteLink, int ***input_path_buf, int *path_length, double *loop_coeff,
                         int num_paths, int max_length, double eb3, QudaGaugeParam *qudaGaugeParam)
{
  auto profile = pushProfile(profileGaugePath);
  checkGaugeParam(qudaGaugeParam);

  GaugeFieldParam gParam(*qudaGaugeParam, siteLink);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuSiteLink = !qudaGaugeParam->use_resident_gauge ? GaugeField(gParam) : GaugeField();

  if (qudaGaugeParam->use_resident_gauge && !gaugePrecise) errorQuda("No resident gauge field to use");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuSiteLink;
  gParam.reconstruct = qudaGaugeParam->reconstruct;
  gParam.setPrecision(qudaGaugeParam->cuda_prec, true);
  GaugeField cudaSiteLink = qudaGaugeParam->use_resident_gauge ? gaugePrecise->create_alias() : GaugeField(gParam);

  GaugeFieldParam gParamOut(*qudaGaugeParam, out);
  gParamOut.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuOut = GaugeField(gParamOut);
  gParamOut.location = QUDA_CUDA_FIELD_LOCATION;
  gParamOut.create = qudaGaugeParam->overwrite_gauge ? QUDA_ZERO_FIELD_CREATE : QUDA_COPY_FIELD_CREATE;
  gParamOut.field = &cpuOut;
  gParamOut.reconstruct = QUDA_RECONSTRUCT_NO;
  gParamOut.setPrecision(qudaGaugeParam->cuda_prec, true);
  GaugeField cudaOut(gParamOut);

  GaugeField *cudaGauge = createExtendedGauge(cudaSiteLink, R, profileGaugePath);
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
  gaugePath(cudaOut, *cudaGauge, eb3, input_path_v, path_length_v, loop_coeff_v, num_paths, max_length);

  cpuOut.copy(cudaOut);

  if (qudaGaugeParam->make_resident_gauge && !qudaGaugeParam->use_resident_gauge) {
    if (gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaSiteLink);
  }

  if (qudaGaugeParam->make_resident_gauge) {
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaGauge;
  } else {
    delete cudaGauge;
  }

  return 0;
}

void momResidentQuda(void *mom, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileGaugeForce);
  checkGaugeParam(param);

  GaugeFieldParam gParamMom(*param, mom, QUDA_ASQTAD_MOM_LINKS);
  gParamMom.location = QUDA_CPU_FIELD_LOCATION;

  GaugeField cpuMom(gParamMom);

  if (param->make_resident_mom && !param->return_result_mom) {
    gParamMom.location = QUDA_CUDA_FIELD_LOCATION;
    gParamMom.create = QUDA_NULL_FIELD_CREATE;
    gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
    gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
    gParamMom.setPrecision(param->cuda_prec, true);
    gParamMom.create = QUDA_ZERO_FIELD_CREATE;
    momResident = GaugeField(gParamMom);
  } else if (param->return_result_mom && !param->make_resident_mom) {
    if (momResident.empty()) errorQuda("No resident momentum to return");
  } else {
    errorQuda("Unexpected combination make_resident_mom = %d return_result_mom = %d", param->make_resident_mom,
              param->return_result_mom);
  }

  if (param->make_resident_mom) {
    // we are downloading the momentum from the host
    momResident.copy(cpuMom);
  } else if (param->return_result_mom) {
    // we are uploading the momentum to the host
    cpuMom.copy(momResident);
    momResident = GaugeField();
  }
}

void createCloverQuda(QudaInvertParam* invertParam)
{
  auto profile = pushProfile(profileClover);
  if (!cloverPrecise) errorQuda("Clover field not allocated");

  QudaReconstructType recon = (gaugePrecise->Reconstruct() == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_12 : gaugePrecise->Reconstruct();
  // for clover we optimize to only send depth 1 halos in y/z/t (FIXME - make work for x, make robust in general)
  lat_dim_t R;
  for (int d=0; d<4; d++) R[d] = (d==0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  GaugeField *gauge
    = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, getProfile(), false, recon);

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
  tensorParam.setPrecision(tensorParam.Precision(), true);
  tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  GaugeField Fmunu(tensorParam);
  computeFmunu(Fmunu, *ex);
  computeClover(*cloverPrecise, Fmunu, invertParam->clover_coeff);

  // if the clover reconstruction is enabled then we just compute the trace log
  cloverInvert(*cloverPrecise, cloverPrecise->Reconstruct());

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

  GaugeField *cpuGauge = nullptr;
  if (gauge) cpuGauge = new GaugeField(gParam);

  gParam.setPrecision(gParam.Precision(), true);
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  auto *cudaGauge = new GaugeField(gParam);

  if (gauge) {
    cudaGauge->copy(*cpuGauge);
    delete cpuGauge;
  }

  return cudaGauge;
}

void saveGaugeFieldQuda(void *gauge, void *inGauge, QudaGaugeParam *param)
{
  auto *cudaGauge = reinterpret_cast<GaugeField *>(inGauge);

  GaugeFieldParam gParam(*param, gauge);
  gParam.geometry = cudaGauge->Geometry();

  GaugeField cpuGauge(gParam);
  cpuGauge.copy(*cudaGauge);
}

void destroyGaugeFieldQuda(void *gauge)
{
  auto *g = reinterpret_cast<GaugeField *>(gauge);
  delete g;
}

void computeStaggeredForceQuda(void *h_mom, double dt, double delta, void *, void **, QudaGaugeParam *gauge_param,
                               QudaInvertParam *inv_param)
{
  auto profile = pushProfile(profileStaggeredForce);

  GaugeFieldParam gParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);

  // create the host momentum field
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.reconstruct = gauge_param->reconstruct;
  GaugeField cpuMom(gParam);

  // create the device momentum field
  if (gauge_param->use_resident_mom && momResident.empty())
    errorQuda("Cannot use resident momentum field since none appears resident");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuMom;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.setPrecision(gParam.Precision(), true);
  GaugeField cudaMom = gauge_param->use_resident_mom ? momResident.create_alias() : GaugeField(gParam);

  // create temporary field for quark-field outer product
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  GaugeField cudaForce(gParam);
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

  // resident gauge field is required
  if (!gauge_param->use_resident_gauge || !gaugePrecise) errorQuda("Resident gauge field is required");
  if (!gaugePrecise->StaggeredPhaseApplied())
    errorQuda("Gauge field requires the staggered phase factors to be applied");

  // check if staggered phase is the desired one
  if (gauge_param->staggered_phase_type != gaugePrecise->StaggeredPhase()) {
    errorQuda("Requested staggered phase %d, but found %d\n",
              gauge_param->staggered_phase_type, gaugePrecise->StaggeredPhase());
  }

  const int nvector = inv_param->num_offset;
  std::vector<ColorSpinorField*> X(nvector);
  for (int i = 0; i < nvector; i++) X[i] = ColorSpinorField::Create(qParam);

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

#if 0
  if (inv_param->use_resident_solution) solutionResident.clear();
#endif
  delete dirac;

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
  updateMomentum(cudaMom, dt * delta, cudaForce, "staggered");

  // copy the momentum field back to the host
  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);

  if (gauge_param->make_resident_mom && !gauge_param->use_resident_mom)
    std::exchange(momResident, cudaMom);
  else if (!gauge_param->make_resident_mom)
    momResident = GaugeField();

  for (int i=0; i<nvector; i++) delete X[i];
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
  auto profile = pushProfile(profileHISQForce);
  checkGaugeParam(gParam);

  using namespace quda;
  using namespace quda::fermion_force;

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

  GaugeField stapleOprod(oParam);
  GaugeField oneLinkOprod(oParam);
  GaugeField naikOprod(oParam);

  double act_path_coeff[6] = {0, 1, level2_coeff[2], level2_coeff[3], level2_coeff[4], level2_coeff[5]};
  // You have to look at the MILC routine to understand the following
  // Basically, I have already absorbed the one-link coefficient

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
      GaugeField *oprod[2] = {&stapleOprod, &naikOprod};

      // loop over different quark fields
      for (int i = 0; i < num_terms; ++i) {

        // Wrap the MILC quark field
        qParam.v = fermion[i];
        ColorSpinorField cpuQuark(qParam); // create host quark field

        cudaQuark = cpuQuark;
        computeStaggeredOprod(oprod, cudaQuark, coeff[i], 3);
      }
    }

    { // naik terms
      oneLinkOprod.copy(stapleOprod);
      ax(level2_coeff[0], oneLinkOprod);
      GaugeField *oprod[2] = {&oneLinkOprod, &naikOprod};

      // loop over different quark fields
      for (int i = 0; i < num_naik_terms; ++i) {

        // Wrap the MILC quark field
        qParam.v = fermion[i + num_terms - num_naik_terms];
        ColorSpinorField cpuQuark(qParam); // create host quark field

        cudaQuark = cpuQuark;
        computeStaggeredOprod(oprod, cudaQuark, coeff[i + num_terms], 3);
      }
    }
  }

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

  GaugeField cudaInForce(oParam);
  copyExtendedGauge(cudaInForce, stapleOprod, QUDA_CUDA_FIELD_LOCATION);
  stapleOprod = GaugeField();

  GaugeField cudaOutForce(oParam);
  copyExtendedGauge(cudaOutForce, oneLinkOprod, QUDA_CUDA_FIELD_LOCATION);
  oneLinkOprod = GaugeField();

  // Create CPU momentum fields, prepare GPU momentum param
  GaugeFieldParam param(*gParam);
  param.location = QUDA_CPU_FIELD_LOCATION;
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  param.gauge = milc_momentum;
  GaugeField cpuMom = (!gParam->use_resident_mom) ? GaugeField(param) : GaugeField();

  param.location = QUDA_CUDA_FIELD_LOCATION;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.setPrecision(param.Precision(), true);
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
  wParam.link_type = QUDA_GENERAL_LINKS;
  wParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  wParam.gauge = (void *)w_link;
  GaugeField cpuWLink(wParam);

  GaugeFieldParam vParam(wParam);
  vParam.gauge = (void *)v_link;
  GaugeField cpuVLink(vParam);

  GaugeFieldParam uParam(vParam);
  uParam.gauge = (void *)u_link;
  GaugeField cpuULink(uParam);

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

  GaugeField cudaWLink(wParam);

  cudaWLink.copy(cpuWLink);

  cudaWLink.exchangeExtendedGhost(cudaWLink.R(), profileHISQForce);

  cudaInForce.exchangeExtendedGhost(R, profileHISQForce);
  cudaWLink.exchangeExtendedGhost(cudaWLink.R(), profileHISQForce);
  cudaOutForce.exchangeExtendedGhost(R, profileHISQForce);

  // Compute level two term
  hisqStaplesForce(cudaOutForce, cudaInForce, cudaWLink, act_path_coeff);

  // Load naik outer product
  copyExtendedGauge(cudaInForce, naikOprod, QUDA_CUDA_FIELD_LOCATION);
  cudaInForce.exchangeExtendedGhost(cudaWLink.R(), profileHISQForce);
  naikOprod = GaugeField();

  // Compute Naik three-link term contribution
  hisqLongLinkForce(cudaOutForce, cudaInForce, cudaWLink, act_path_coeff[1]);

  cudaOutForce.exchangeExtendedGhost(R, profileHISQForce);

  // Load the V field, which contains general matrices, to the device
  cudaWLink = GaugeField();

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
  GaugeField cudaVLink(vParam);

  cudaVLink.copy(cpuVLink);
  cudaVLink.exchangeExtendedGhost(cudaVLink.R(), profileHISQForce);

  *num_failures_h = 0;
  unitarizeForce(cudaInForce, cudaOutForce, cudaVLink, num_failures_d);

  if (*num_failures_h>0) errorQuda("Error in the unitarization component of the hisq fermion force: %d failures\n", *num_failures_h);

  // Load the U field, which contains U(3) matrices, to the device
  // TODO: in theory these should just be SU(3) matrices with MILC phases?
  cudaVLink = GaugeField();

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
  GaugeField cudaULink(uParam);

  cudaULink.copy(cpuULink);
  cudaULink.exchangeExtendedGhost(cudaULink.R(), profileHISQForce);

  // Compute Fat7-staple term
  cudaOutForce.zero();
  hisqStaplesForce(cudaOutForce, cudaInForce, cudaULink, fat7_coeff);

  cudaInForce = GaugeField();

  hisqCompleteForce(cudaOutForce, cudaULink);

  if (gParam->use_resident_mom && !momResident.Length()) errorQuda("No resident momentum field to use");
  GaugeField mom = gParam->use_resident_mom ? momResident.create_alias() : GaugeField(momParam);
  updateMomentum(mom, dt, cudaOutForce, "hisq");

  // Close the paths, make anti-hermitian, and store in compressed format
  if (gParam->return_result_mom) cpuMom.copy(mom);

  if (gParam->make_resident_mom && !gParam->use_resident_mom)
    std::exchange(momResident, mom);
  else if (!gParam->make_resident_mom)
    momResident = GaugeField();
}

void computeCloverForceQuda(void *h_mom, double dt, void **h_x, void **, double *coeff, double kappa2, double ck,
                            int nvector, double multiplicity, void *, QudaGaugeParam *gauge_param,
                            QudaInvertParam *inv_param)
{
  using namespace quda;
  auto profile = pushProfile(profileCloverForce, inv_param);

  checkGaugeParam(gauge_param);
  if (!gaugePrecise) errorQuda("No resident gauge field");
  if (!cloverPrecise) errorQuda("No resident clover field");

  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  // create the host momentum field
  GaugeField cpuMom = !gauge_param->use_resident_mom ? GaugeField(fParam) : GaugeField();

  // create the device momentum field
  fParam.location = QUDA_CUDA_FIELD_LOCATION;
  fParam.create = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_COPY_FIELD_CREATE;
  fParam.field = &cpuMom;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.setPrecision(gauge_param->cuda_prec, true);

  if (gauge_param->use_resident_mom && !momResident.Length()) errorQuda("No resident momentum field to use");
  GaugeField cudaMom = gauge_param->use_resident_mom ? momResident.create_alias() : GaugeField(fParam);
  if (gauge_param->use_resident_mom && gauge_param->overwrite_mom) cudaMom.zero();

  if (inv_param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
    errorQuda("Force computation only supports solution to MatPCDagMatPC");
  ColorSpinorParam qParam(nullptr, *inv_param, fParam.x, false, QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), x0(nvector);
  std::vector<double> force_coeff(nvector);
  std::vector<array<double, 2>> ferm_epsilon(nvector);

  QudaParity parity = inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);

    if (!inv_param->use_resident_solution) {
      ColorSpinorParam cpuParam(h_x[i], *inv_param, fParam.x, true, inv_param->input_location);
      ColorSpinorField cpuQuarkX(cpuParam);
      x[i][parity] = cpuQuarkX;
    } else {
      x[i][parity] = solutionResident[i];
    }

    force_coeff[i] = 2.0 * dt * coeff[i] * kappa2;
    ferm_epsilon[i] = {2.0 * ck * coeff[i] * dt, -kappa2 * 2.0 * ck * coeff[i] * dt};
  }

  if (inv_param->use_resident_solution && solutionResident.size() < (unsigned int)nvector)
    errorQuda("solutionResident.size() %lu does not match number of shifts %d", solutionResident.size(), nvector);

  // Make sure extendedGaugeResident has the correct R
  if (extendedGaugeResident) delete extendedGaugeResident;
  lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, getProfile());
  GaugeField &gaugeEx = *extendedGaugeResident;

  computeCloverForce(cudaMom, gaugeEx, *gaugePrecise, *cloverPrecise, x, x0, force_coeff, ferm_epsilon,
                     2.0 * ck * multiplicity * dt, false, *inv_param);

  // copy the outer product field back to the host
  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);
  if (gauge_param->make_resident_mom && gauge_param->use_resident_mom)
    std::exchange(momResident, cudaMom);
  else if (!gauge_param->make_resident_mom)
    momResident = GaugeField();
}

void computeTMCloverForceQuda(void *h_mom, void **h_x, void **h_x0, double *coeff, int nvector,
                              QudaGaugeParam *gauge_param, QudaInvertParam *inv_param, int detratio)
{
  using namespace quda;
  auto profile = pushProfile(profileTMCloverForce, inv_param);

  checkGaugeParam(gauge_param);
  if (!gaugePrecise) errorQuda("No resident gauge field");
  if (!cloverPrecise) errorQuda("No resident clover field");

  double kappa = inv_param->kappa;
  double k_csw_ov_8 = kappa * inv_param->clover_csw / 8.0;

  GaugeFieldParam gParamMom(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom = !gauge_param->use_resident_mom ? GaugeField(gParamMom) : GaugeField();

  // create the device momentum field
  gParamMom.location = QUDA_CUDA_FIELD_LOCATION;
  gParamMom.create = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_COPY_FIELD_CREATE;
  gParamMom.field = &cpuMom;
  gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.setPrecision(gauge_param->cuda_prec, true);

  if (gauge_param->use_resident_mom && !momResident.Length()) errorQuda("No resident momentum field to use");
  GaugeField gpuMom = gauge_param->use_resident_mom ? momResident.create_alias() : GaugeField(gParamMom);
  if (gauge_param->use_resident_mom && gauge_param->overwrite_mom) gpuMom.zero();

  if (inv_param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
    errorQuda("Force computation only supports solution to MatPCDagMatPC");
  ColorSpinorParam qParam(nullptr, *inv_param, gParamMom.x, false, QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(gauge_param->cuda_prec, gauge_param->cuda_prec, true);
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), x0(nvector);
  std::vector<double> force_coeff(nvector);
  std::vector<array<double, 2>> ferm_epsilon(nvector);

  QudaParity parity = inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    ColorSpinorParam cpuParam(h_x[i], *inv_param, gParamMom.x, true, inv_param->input_location);
    ColorSpinorField cpuQuarkX(cpuParam);
    x[i][parity] = cpuQuarkX; // in tmLQCD-parlance this is the odd part of X

    if (detratio && inv_param->twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) {
      x0[i] = ColorSpinorField(qParam);
      ColorSpinorParam cpuParam0(h_x0[i], *inv_param, gParamMom.x, true, inv_param->input_location);
      ColorSpinorField cpuQuarkX0(cpuParam0);
      x0[i][parity] = cpuQuarkX0;
    }

    force_coeff[i] = 1.0 * coeff[i];
    ferm_epsilon[i] = {k_csw_ov_8 * coeff[i], k_csw_ov_8 * coeff[i] / (kappa * kappa)};
  }

  // Make sure extendedGaugeResident has the correct R
  if (extendedGaugeResident) delete extendedGaugeResident;
  lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  extendedGaugeResident = createExtendedGauge(*gaugePrecise, R, profileTMCloverForce);
  GaugeField &gaugeEx = *extendedGaugeResident;

  computeCloverForce(gpuMom, gaugeEx, *gaugePrecise, *cloverPrecise, x, x0, force_coeff, ferm_epsilon,
                     k_csw_ov_8 * 32.0, detratio, *inv_param);

  if (gauge_param->return_result_mom) cpuMom.copy(gpuMom);
  if (gauge_param->make_resident_mom && gauge_param->use_resident_mom)
    std::exchange(momResident, gpuMom);
  else if (!gauge_param->make_resident_mom)
    momResident = GaugeField();
}

void updateGaugeFieldQuda(void *gauge, void *momentum, double dt, int conj_mom, int exact, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileGaugeUpdate);
  checkGaugeParam(param);

  // create the host fields
  GaugeFieldParam gParam(*param, gauge, QUDA_SU3_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  GaugeField cpuGauge = need_cpu ? GaugeField(gParam) : GaugeField();

  GaugeFieldParam gParamMom(*param, momentum, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom = !param->use_resident_mom ? GaugeField(gParamMom) : GaugeField();

  // create the device fields
  if (param->use_resident_mom && momResident.empty()) errorQuda("No resident mom field allocated");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuMom;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.setPrecision(gParam.Precision(), true);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.pad = 0;
  GaugeField cudaMom = param->use_resident_mom ? momResident.create_alias() : GaugeField(gParam);

  if (param->use_resident_gauge && !gaugePrecise) errorQuda("No resident gauge field allocated");
  gParam.link_type = QUDA_SU3_LINKS;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  gParam.field = &cpuGauge;
  GaugeField u_in = param->use_resident_gauge ? gaugePrecise->create_alias() : GaugeField(gParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField u_out(gParam);

  // perform the update
  updateGaugeField(u_out, dt, u_in, cudaMom, (bool)conj_mom, (bool)exact);

  // copy the gauge field back to the host
  if (param->return_result_gauge) cpuGauge.copy(u_out);

  if (param->make_resident_gauge) {
    if (gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, u_out);
  }

  if (param->make_resident_mom && !param->use_resident_mom)
    std::exchange(momResident, cudaMom);
  else if (!param->make_resident_mom)
    momResident = GaugeField();
}

void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param)
{
  auto profile = pushProfile(profileProject);
  checkGaugeParam(param);

  // create the gauge field
  GaugeFieldParam gParam(*param, gauge_h, QUDA_SU3_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  GaugeField cpuGauge = need_cpu ? GaugeField(gParam) : GaugeField();

  // create the device fields
  if (param->use_resident_gauge && !gaugePrecise) errorQuda("No resident gauge field to use");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuGauge;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  GaugeField cudaGauge = param->use_resident_gauge ? gaugePrecise->create_alias() : GaugeField(gParam);

  *num_failures_h = 0;

  // project onto SU(3)
  if (cudaGauge.StaggeredPhaseApplied()) cudaGauge.removeStaggeredPhase();
  projectSU3(cudaGauge, tol, num_failures_d);
  if (!cudaGauge.StaggeredPhaseApplied() && param->staggered_phase_applied) cudaGauge.applyStaggeredPhase();

  if (*num_failures_h > 0) errorQuda("Error in the SU(3) unitarization: %d failures\n", *num_failures_h);

  if (param->return_result_gauge) cpuGauge.copy(cudaGauge);

  if (param->make_resident_gauge && !param->use_resident_gauge) {
    if (gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaGauge);
  }
}

void staggeredPhaseQuda(void *gauge_h, QudaGaugeParam *param)
{
  auto profile = pushProfile(profilePhase);
  checkGaugeParam(param);

  // create the gauge field
  GaugeFieldParam gParam(*param, gauge_h, QUDA_GENERAL_LINKS);
  bool need_cpu = !param->use_resident_gauge || param->return_result_gauge;
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuGauge = need_cpu ? GaugeField(gParam) : GaugeField();

  // create the device fields
  if (param->use_resident_gauge && !gaugePrecise) errorQuda("No resident gauge field to use");
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.field = &cpuGauge;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  GaugeField cudaGauge = param->use_resident_gauge ? gaugePrecise->create_alias() : GaugeField(gParam);

  *num_failures_h = 0;

  // apply / remove phase as appropriate
  if (!cudaGauge.StaggeredPhaseApplied())
    cudaGauge.applyStaggeredPhase();
  else
    cudaGauge.removeStaggeredPhase();

  if (param->return_result_gauge) cpuGauge.copy(cudaGauge);

  if (param->make_resident_gauge && !param->use_resident_gauge) {
    if (gaugePrecise) freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaGauge);
  }
}

// evaluate the momentum action
double momActionQuda(void* momentum, QudaGaugeParam* param)
{
  auto profile = pushProfile(profileMomAction);
  checkGaugeParam(param);

  // create the momentum fields
  GaugeFieldParam gParam(*param, momentum, QUDA_ASQTAD_MOM_LINKS);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuMom = !param->use_resident_mom ? GaugeField(gParam) : GaugeField();

  // create the device fields
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.field = &cpuMom;
  gParam.create = QUDA_COPY_FIELD_CREATE;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.setPrecision(param->cuda_prec, true);

  if (param->use_resident_mom && momResident.empty()) errorQuda("No resident mom field allocated");
  GaugeField cudaMom = param->use_resident_mom ? momResident.create_alias() : GaugeField(gParam);

  // perform the update
  double action = computeMomAction(cudaMom);

  if (param->make_resident_mom && !param->use_resident_mom)
    std::exchange(momResident, cudaMom);
  else if (!param->make_resident_mom)
    momResident = GaugeField();

  return action;
}

void gaussGaugeQuda(unsigned long long seed, double sigma)
{
  auto profile = pushProfile(profileGauss);

  if (!gaugePrecise) errorQuda("Cannot generate Gauss GaugeField as there is no resident gauge field");
  quda::gaugeGauss(*gaugePrecise, seed, sigma);

  if (extendedGaugeResident) {
    extendedGaugeResident->copy(*gaugePrecise);
    extendedGaugeResident->exchangeExtendedGhost(R, profileGauss, redundant_comms);
  }
}

void gaussMomQuda(unsigned long long seed, double sigma)
{
  auto profile = pushProfile(profileGauss);
  if (momResident.empty()) errorQuda("Cannot generate Gauss GaugeField as there is no resident momentum field");
  quda::gaugeGauss(momResident, seed, sigma);
}

/*
 * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
 */
void plaqQuda(double plaq[3])
{
  auto profile = pushProfile(profilePlaq);

  if (!gaugePrecise) errorQuda("Cannot compute plaquette as there is no resident gauge field");

  GaugeField *data = extendedGaugeResident ? extendedGaugeResident : createExtendedGauge(*gaugePrecise, R, profilePlaq);
  extendedGaugeResident = data;

  double3 plaq3 = quda::plaquette(*data);
  plaq[0] = plaq3.x;
  plaq[1] = plaq3.y;
  plaq[2] = plaq3.z;
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
  auto profile = pushProfile(profileWuppertal);
  pushVerbosity(inv_param->verbosity);
  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  GaugeField *precise = nullptr;

  if (gaugeSmeared != nullptr) {
    logQuda(QUDA_VERBOSE, "Wuppertal smearing done with gaugeSmeared\n");
    GaugeFieldParam gParam(*gaugePrecise);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    precise = new GaugeField(gParam);
    copyExtendedGauge(*precise, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
    precise->exchangeGhost();
  } else {
    logQuda(QUDA_VERBOSE, "Wuppertal smearing done with gaugePrecise\n");
    precise = gaugePrecise;
  }

  ColorSpinorParam cpuParam(h_in, *inv_param, precise->X(), false, inv_param->input_location);
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);
  in = in_h;

  logQuda(QUDA_DEBUG_VERBOSE, "In CPU %e CUDA %e\n", blas::norm2(in_h), blas::norm2(in));

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
    ApplyLaplace(out, in, *precise, 3, a, b, in, parity, comm_dim, profileWuppertal);
    logQuda(QUDA_DEBUG_VERBOSE, "Step %d, vector norm %e\n", i, blas::norm2(out));
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField out_h(cpuParam);
  out_h = out;

  logQuda(QUDA_DEBUG_VERBOSE, "Out CPU %e CUDA %e\n", blas::norm2(out_h), blas::norm2(out));

  if (gaugeSmeared != nullptr) delete precise;

  popVerbosity();
}

void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param)
{
  if (smear_param->n_steps == 0) return;
  auto profile = pushProfile(profileGaussianSmear, smear_param);

  QudaInvertParam *inv_param = smear_param->inv_param;

  if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);
  checkInvertParam(inv_param);

  if (gaugeSmeared == nullptr || smear_param->compute_2link != 0) {

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
    gaugeSmeared = new GaugeField(gParam);

    GaugeField *two_link_ext = createExtendedGauge(*gaugePrecise, R, profileGauge); // aux field

    computeTwoLink(*gaugeSmeared, *two_link_ext);

    gaugeSmeared->exchangeGhost();

    delete two_link_ext;
  }

  if (!initialized) errorQuda("QUDA not initialized");

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

  // Copy host data to device
  in = in_h;

  const double ftmp    = -(smear_param->width*smear_param->width)/(4.0*smear_param->n_steps*4.0);  /* Extra 4 to compensate for stride 2 */
  // Scale up the source to prevent underflow
  profileGaussianSmear.TPSTART(QUDA_PROFILE_COMPUTE);

  const double msq = 1. / ftmp;
  const double a       = inv_param->laplace3D * 2.0 + msq;
  const QudaParity  parity   = QUDA_INVALID_PARITY;
  for (int i = 0; i < smear_param->n_steps; i++) {
    if (i > 0) std::swap(in, out);

    qsmear_op.Expose()->SmearOp(out, in, a, 0.0, smear_param->t0, parity);
    logQuda(QUDA_DEBUG_VERBOSE, "Step %d, vector norm %e\n", i, blas::norm2(out));
    blas::axpby(a * ftmp, in, -ftmp, out);
  }

  profileGaussianSmear.TPSTOP(QUDA_PROFILE_COMPUTE);

  // Copy device data to host.
  in_h = out;

  delete d;

  if (smear_param->delete_2link != 0) { freeUniqueGaugeQuda(QUDA_SMEARED_LINKS); }
}

void performGaugeSmearQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)
{
  auto profile = pushProfile(profileGaugeSmear);
  pushOutputPrefix("performGaugeSmearQuda: ");
  checkGaugeSmearParam(smear_param);

  if (gaugePrecise == nullptr) errorQuda("Precise gauge field must be loaded");
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileGaugeSmear);

  GaugeFieldParam gParam(*gaugeSmeared);
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  GaugeField tmp(gParam);

  int measurement_n = 0; // The nth measurement to take
  gaugeObservablesQuda(&obs_param[measurement_n]);
  logQuda(QUDA_SUMMARIZE, "step %03d plaquette (mean %.16e, spatial %.16e temporal %.16e) Q charge = %.16e\n", 0,
          obs_param[measurement_n].plaquette[0], obs_param[measurement_n].plaquette[1],
          obs_param[measurement_n].plaquette[2], obs_param[measurement_n].qcharge);

  // set default dir_ignore = 3 for APE and STOUT for compatibility
  int dir_ignore = smear_param->dir_ignore;
  if (dir_ignore < 0
      && (smear_param->smear_type == QUDA_GAUGE_SMEAR_APE || smear_param->smear_type == QUDA_GAUGE_SMEAR_STOUT)) {
    dir_ignore = 3;
  }

  for (unsigned int i = 0; i < smear_param->n_steps; i++) {
    switch (smear_param->smear_type) {
    case QUDA_GAUGE_SMEAR_APE: APEStep(*gaugeSmeared, tmp, smear_param->alpha, dir_ignore); break;
    case QUDA_GAUGE_SMEAR_STOUT: STOUTStep(*gaugeSmeared, tmp, smear_param->rho, dir_ignore); break;
    case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
      OvrImpSTOUTStep(*gaugeSmeared, tmp, smear_param->rho, smear_param->epsilon, dir_ignore);
      break;
    case QUDA_GAUGE_SMEAR_HYP:
      HYPStep(*gaugeSmeared, tmp, smear_param->alpha1, smear_param->alpha2, smear_param->alpha3, dir_ignore);
      break;
    default: errorQuda("Unknown gauge smear type %d", smear_param->smear_type);
    }

    if ((i + 1) % smear_param->meas_interval == 0) {
      measurement_n++;
      gaugeObservablesQuda(&obs_param[measurement_n]);
      logQuda(QUDA_SUMMARIZE, "step %03d plaquette (mean %.16e, spatial %.16e temporal %.16e) Q charge = %.16e\n",
              i + 1, obs_param[measurement_n].plaquette[0], obs_param[measurement_n].plaquette[1],
              obs_param[measurement_n].plaquette[2], obs_param[measurement_n].qcharge);
    }
  }

  popOutputPrefix();
}

void performWFlowQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)
{
  auto profile = pushProfile(profileWFlow);
  pushOutputPrefix("performWFlowQuda: ");
  checkGaugeSmearParam(smear_param);

  if (smear_param->restart) {
    if (gaugeSmeared == nullptr) errorQuda("gaugeSmeared must be loaded");
  } else {
    if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
    gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileWFlow);
  }

  GaugeFieldParam gParamEx(*gaugeSmeared);
  GaugeField gaugeAux(gParamEx);

  GaugeFieldParam gParam(*gaugePrecise);
  gParam.reconstruct = QUDA_RECONSTRUCT_NO; // temporary field is not on manifold so cannot use reconstruct
  GaugeField gaugeTemp(gParam);

  GaugeField &in = *gaugeSmeared;
  GaugeField &out = gaugeAux;

  int measurement_n = 0; // The nth measurement to take

  gaugeObservables(in, obs_param[measurement_n]);

  logQuda(QUDA_SUMMARIZE, "flow t, plaquette, E_tot, E_spatial, E_temporal, Q charge\n");
  logQuda(QUDA_SUMMARIZE, "%le %.16e %+.16e %+.16e %+.16e %+.16e\n", smear_param->t0, obs_param[0].plaquette[0],
          obs_param[0].energy[0], obs_param[0].energy[1], obs_param[0].energy[2], obs_param[0].qcharge);

  for (unsigned int i = 0; i < smear_param->n_steps; i++) {
    // Perform W1, W2, and Vt Wilson Flow steps as defined in
    // https://arxiv.org/abs/1006.4518v3
    if (i > 0) std::swap(in, out); // output from prior step becomes input for next step
    WFlowStep(out, gaugeTemp, in, smear_param->epsilon, smear_param->smear_type);

    if ((i + 1) % smear_param->meas_interval == 0) {
      measurement_n++; // increment measurements.
      gaugeObservables(out, obs_param[measurement_n]);
      logQuda(QUDA_SUMMARIZE, "%le %.16e %+.16e %+.16e %+.16e %+.16e\n", (smear_param->t0 + smear_param->epsilon * (i + 1)),
              obs_param[measurement_n].plaquette[0], obs_param[measurement_n].energy[0],
              obs_param[measurement_n].energy[1], obs_param[measurement_n].energy[2], obs_param[measurement_n].qcharge);
    }
  }
  // copy out to gaugeSmeared so that flowed gauge can be saved to host and WFlow can be restarted 
  copyExtendedGauge(*gaugeSmeared, out, QUDA_CUDA_FIELD_LOCATION);
  gaugeSmeared->exchangeExtendedGhost( gaugeSmeared->R() );

  popOutputPrefix();
}

// perform forward gradient flow on gauge and spinor field following the algorithm in arXiv:1302.5246 (Appendix D)
// the gauge flow steps are identical to Wilson Flow algorithm in arXiv:1006.4518 (Vt <-> W3)
void performGFlowQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaGaugeSmearParam *smear_param,
                      QudaGaugeObservableParam *obs_param)
{

  auto profile = pushProfile(profileGFlow);
  pushOutputPrefix("performGFlowQuda: ");
  checkGaugeSmearParam(smear_param);

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  if (smear_param->restart) {
    if (gaugeSmeared == nullptr) errorQuda("gaugeSmeared must be loaded");
  } else {
    if (gaugePrecise == nullptr) errorQuda("Gauge field must be loaded");
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
    gaugeSmeared = createExtendedGauge(*gaugePrecise, R, profileGFlow);
  }

  GaugeFieldParam gParamEx(*gaugeSmeared);
  GaugeField gaugeAux(gParamEx);

  GaugeFieldParam gParam(*gaugePrecise);
  gParam.reconstruct = QUDA_RECONSTRUCT_NO; // temporary field is not on manifold so cannot use reconstruct
  GaugeField gaugeTemp(gParam);

  GaugeField &gin = *gaugeSmeared;
  GaugeField &gout = gaugeAux;

  // helper gauge field for Laplace operator
  GaugeField precise;
  GaugeFieldParam gParam_helper(*gaugePrecise);
  gParam_helper.create = QUDA_NULL_FIELD_CREATE;
  precise = GaugeField(gParam_helper);

  // spinor fields
  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), false, inv_param->input_location);
  ColorSpinorField fin_h(cpuParam);

  ColorSpinorParam deviceParam(cpuParam, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField fin(deviceParam);
  fin = fin_h;

  deviceParam.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField fout(deviceParam);

  int parity = 0;

  // initialize a and b for Laplace operator
  double a = 1.;
  double b = -8.;

  int comm_dim[4] = {};
  // only switch on comms needed for directions with a derivative
  for (int i = 0; i < 4; i++) { comm_dim[i] = comm_dim_partitioned(i); }

  // auxilliary fermion fields [0], [1], [2] and [3]
  ColorSpinorField f_temp0(deviceParam);
  ColorSpinorField f_temp1(deviceParam);
  ColorSpinorField f_temp2(deviceParam);
  ColorSpinorField f_temp3(deviceParam);
  ColorSpinorField f_temp4(deviceParam);

  // set [3] = input spinor
  f_temp3 = fin;

  int measurement_n = 0; // The nth measurement to take

  gaugeObservables(gin, obs_param[measurement_n]);

  logQuda(QUDA_SUMMARIZE, "flow t, plaquette, norm(f_spinor)\n");
  logQuda(QUDA_SUMMARIZE, "%le %.16e %+.16e\n", smear_param->t0, obs_param[0].plaquette[0], blas::norm2(fin));

  // loop, iterations of gf
  for (unsigned int i = 0; i < smear_param->n_steps; i++) {

    if (i > 0) std::swap(gin, gout); // output from prior step becomes input for next step

    // init auxilliary fields [0], [1] and [2] as [3]
    f_temp0 = f_temp3;
    f_temp1 = f_temp3;
    f_temp2 = f_temp3;

    // STEP 1
    // [4] = Laplace [0]
    copyExtendedGauge(precise, gin, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp4, f_temp0, precise, 4, a, b, f_temp0, parity, comm_dim, profileGFlow);

    // [0] = [4] = Laplace [0] = Laplace [3]
    f_temp0 = f_temp4;

    // [1] <- epsilon/4 x [0] + [1] = [3] + epsilon /4 x Laplace [3]
    blas::axpy(smear_param->epsilon / 4., f_temp0, f_temp1);

    // apply step W1 of gauge field flow part
    GFlowStep(gout, gaugeTemp, gin, smear_param->epsilon, smear_param->smear_type, WFLOW_STEP_W1);

    // STEP 2
    // [3] <- [1]
    f_temp3 = f_temp1;

    // [4] <- Laplace [1]
    copyExtendedGauge(precise, gout, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp4, f_temp1, precise, 4, a, b, f_temp1, parity, comm_dim, profileGFlow);

    // [1] <- [4]
    f_temp1 = f_temp4;

    // [2] <- 8/9 x epsilon x [1] + [2]
    blas::axpy(smear_param->epsilon * 8. / 9., f_temp1, f_temp2);

    // [2] <- -2/9 x epsilon x [0] + [2]
    blas::axpy(-smear_param->epsilon * 2. / 9., f_temp0, f_temp2);

    // apply step W2 of gauge field flow part
    GFlowStep(gin, gaugeTemp, gout, smear_param->epsilon, smear_param->smear_type, WFLOW_STEP_W2);

    // STEP 3
    // [4] <- Laplace [2]
    copyExtendedGauge(precise, gin, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp4, f_temp2, precise, 4, a, b, f_temp2, parity, comm_dim, profileGFlow);

    // [2] <- [4] = Laplace [2]
    f_temp2 = f_temp4;

    // [3] <- 3/4 x epsilon x [2] + [3]
    blas::axpy(smear_param->epsilon * 3. / 4., f_temp2, f_temp3);

    // set output spinor = [3]
    fout = f_temp3;

    // apply step W3 (Vt) of gauge field flow part
    GFlowStep(gout, gaugeTemp, gin, smear_param->epsilon, smear_param->smear_type, WFLOW_STEP_VT);

    if ((i + 1) % smear_param->meas_interval == 0) {
      measurement_n++; // increment measurements.
      gaugeObservables(gout, obs_param[measurement_n]);
      logQuda(QUDA_SUMMARIZE, "%le %.16e %+.16e\n", (smear_param->t0 + smear_param->epsilon * (i + 1)),
              obs_param[measurement_n].plaquette[0], blas::norm2(fout));
    }
  } /* end of one iteration of GF application */

  // copy gout to gaugeSmeared so that flowed gauge can be saved to host and WFlow can be restarted
  copyExtendedGauge(*gaugeSmeared, gout, QUDA_CUDA_FIELD_LOCATION);
  gaugeSmeared->exchangeExtendedGhost(gaugeSmeared->R());

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField fout_h(cpuParam);
  fout_h = fout;

  popOutputPrefix();

} /* end of performGFlowQuda */

int computeGaugeFixingOVRQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                              const unsigned int verbose_interval, const double relax_boost, const double tolerance,
                              const unsigned int reunit_interval, const unsigned int stopWtheta, QudaGaugeParam *param)
{
  auto profile = pushProfile(GaugeFixOVRQuda);
  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, gauge);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuGauge(gParam);

  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.link_type = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  GaugeField cudaInGauge(gParam);

  cudaInGauge.copy(cpuGauge);

  GaugeField *cudaInGaugeEx = createExtendedGauge(cudaInGauge, R, GaugeFixOVRQuda);

  // perform the update
  gaugeFixingOVR(*cudaInGaugeEx, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval,
                 stopWtheta);

  copyExtendedGauge(cudaInGauge, *cudaInGaugeEx, QUDA_CUDA_FIELD_LOCATION);

  // copy the gauge field back to the host
  cpuGauge.copy(cudaInGauge);

  if (param->make_resident_gauge) {
    freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaInGauge);
    if (extendedGaugeResident) delete extendedGaugeResident;
    extendedGaugeResident = cudaInGaugeEx;
  } else {
    delete cudaInGaugeEx;
  }

  return 0;
}

int computeGaugeFixingFFTQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                              const unsigned int verbose_interval, const double alpha, const unsigned int autotune,
                              const double tolerance, const unsigned int stopWtheta, QudaGaugeParam *param)
{
  auto profile = pushProfile(GaugeFixFFTQuda);
  checkGaugeParam(param);

  GaugeFieldParam gParam(*param, gauge);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpuGauge(gParam);

  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.link_type = param->type;
  gParam.reconstruct = param->reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  GaugeField cudaInGauge(gParam);

  cudaInGauge.copy(cpuGauge);

  // perform the update
  gaugeFixingFFT(cudaInGauge, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

  // copy the gauge field back to the host
  cpuGauge.copy(cudaInGauge);

  if (param->make_resident_gauge) {
    freeUniqueGaugeQuda(QUDA_WILSON_LINKS);
    gaugePrecise = new GaugeField();
    std::exchange(*gaugePrecise, cudaInGauge);
  }

  return 0;
}

void contractFTQuda(void **prop_array_flavor_1, void **prop_array_flavor_2, void **result, const QudaContractType cType,
                    void *cs_param_ptr, const int src_colors, const int *X, const int *const source_position,
                    const int n_mom, const int *const mom_modes, const QudaFFTSymmType *const fft_type)
{
  auto profile = pushProfile(profileContractFT);

  // create ColorSpinorFields from void** and parameter
  auto cs_param = (ColorSpinorParam *)cs_param_ptr;
  const size_t nSpin = cs_param->nSpin;
  const size_t src_nColor = src_colors;
  cs_param->location = QUDA_CPU_FIELD_LOCATION;
  cs_param->create = QUDA_REFERENCE_FIELD_CREATE;

  // The number of complex contraction results expected in the output
  size_t num_out_results = nSpin * nSpin;

  // FIXME can we merge the two propagators if they are the same to save mem?
  // wrap CPU host side pointers
  std::vector<ColorSpinorField> h_prop1, h_prop2;
  h_prop1.reserve(nSpin * src_nColor);
  h_prop2.reserve(nSpin * src_nColor);
  for (size_t i = 0; i < nSpin * src_nColor; i++) {
    cs_param->v = prop_array_flavor_1[i];
    h_prop1.push_back(ColorSpinorField(*cs_param));
    cs_param->v = prop_array_flavor_2[i];
    h_prop2.push_back(ColorSpinorField(*cs_param));
  }

  // Create device spinor fields
  ColorSpinorParam cudaParam(*cs_param);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // not relevant for staggered
  cudaParam.setPrecision(cs_param->Precision(), cs_param->Precision(), true);

  std::vector<ColorSpinorField> d_prop1, d_prop2;
  d_prop1.reserve(nSpin * src_nColor);
  d_prop2.reserve(nSpin * src_nColor);
  for (size_t i = 0; i < nSpin * src_nColor; i++) {
    d_prop1.push_back(ColorSpinorField(cudaParam));
    d_prop2.push_back(ColorSpinorField(cudaParam));
  }

  // temporal or spatial correlator?
  size_t corr_dim = 0, local_decay_dim_slices = 0;
  if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z)
    corr_dim = 2;
  else if (cType == QUDA_CONTRACT_TYPE_DR_FT_T || cType == QUDA_CONTRACT_TYPE_STAGGERED_FT_T)
    corr_dim = 3;
  else
    errorQuda("Unsupported contraction type %d given", cType);

  // The number of slices in the decay dimension on this MPI rank.
  local_decay_dim_slices = X[corr_dim];

  // The number of slices in the decay dimension globally.
  size_t global_decay_dim_slices = local_decay_dim_slices * comm_dim(corr_dim);

  // Transfer data from host to device
  for (size_t i = 0; i < nSpin * src_nColor; i++) {
    d_prop1[i] = h_prop1[i];
    d_prop2[i] = h_prop2[i];
  }

  // Array for all decay slices and spins, is zeroed prior to kernel launch
  std::vector<Complex> result_global(global_decay_dim_slices * num_out_results);

  profileContractFT.TPSTART(QUDA_PROFILE_COMPUTE);
  for (int mom_idx = 0; mom_idx < n_mom; ++mom_idx) {

    for (size_t s1 = 0; s1 < nSpin; s1++) {
      for (size_t b1 = 0; b1 < nSpin; b1++) {
        for (size_t c1 = 0; c1 < src_nColor; c1++) {

          std::fill(result_global.begin(), result_global.end(), 0.0);
          contractSummedQuda(d_prop1[s1 * src_nColor + c1], d_prop2[b1 * src_nColor + c1], result_global, cType,
                             source_position, &mom_modes[4 * mom_idx], &fft_type[4 * mom_idx], s1, b1);

          comm_allreduce_sum(result_global);
          for (size_t t = 0; t < global_decay_dim_slices; t++) {
            for (size_t G_idx = 0; G_idx < num_out_results; G_idx++) {
              int index = 2 * (global_decay_dim_slices * num_out_results * mom_idx + num_out_results * t + G_idx);
              ((double *)*result)[index + 0] += result_global[num_out_results * t + G_idx].real();
              ((double *)*result)[index + 1] += result_global[num_out_results * t + G_idx].imag();
            }
          }
        }
      }
    }
  }
  profileContractFT.TPSTOP(QUDA_PROFILE_COMPUTE);
}

void contractQuda(const void *hp_x, const void *hp_y, void *h_result, const QudaContractType cType,
                  QudaInvertParam *param, const int *X)
{
  auto profile = pushProfile(profileContract);
  // DMH: Easiest way to construct ColorSpinorField? Do we require the user
  //     to declare and fill and invert_param, or can it just be hacked?.

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

  x[0] = h_x;
  y[0] = h_y;

  contractQuda(x[0], y[0], d_result, cType);

  profileContract.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(h_result, d_result, data_bytes, qudaMemcpyDeviceToHost);
  profileContract.TPSTOP(QUDA_PROFILE_D2H);

  pool_device_free(d_result);
}

void gaugeObservablesQuda(QudaGaugeObservableParam *param)
{
  auto profile = pushProfile(profileGaugeObs);
  checkGaugeObservableParam(param);

  if (!gaugePrecise) errorQuda("Cannot compute gauge observables as there is no resident gauge field");

  GaugeField *gauge = nullptr;
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

  gaugeObservables(*gauge, *param);
}

static void check_param(double _Complex *host_sinks, void **host_quark, int n_quark, int tile_quark, void **host_evec,
                        int n_evec, int tile_evec, QudaInvertParam *inv_param, const int X[4])
{
  if (host_sinks == nullptr) errorQuda("Invalid host_sink ptr");
  if (host_quark == nullptr) errorQuda("Invalid host_quark ptr");
  for (auto i = 0; i < n_quark; i++)
    if (host_quark[i] == nullptr) errorQuda("Invalid host_quark[%d] ptr", i);
  if (tile_quark < 1) errorQuda("Invalid tiling parameter %d (must be positive)", tile_quark);
  if (host_evec == nullptr) errorQuda("Invalid host_evec ptr");
  for (auto i = 0; i < n_evec; i++)
    if (host_evec[i] == nullptr) errorQuda("Invalid host_evec[%d] ptr", i);
  if (tile_evec < 1) errorQuda("Invalid tiling parameter %d (must be positive)", tile_evec);
  if (inv_param == nullptr) errorQuda("Invalid QudaInvertParam ptr");
  for (int i = 0; i < 4; i++)
    if (X[i] < 1 || X[i] > 512) errorQuda("Invalid lattice dimension %d", i);
}

void laphSinkProject(double _Complex *host_sinks, void **host_quark, int n_quark, int tile_quark, void **host_evec,
                     int n_evec, int tile_evec, QudaInvertParam *inv_param, const int X[4])
{
  auto profile = pushProfile(profileSinkProject, inv_param);

  // check parameters are valid
  check_param(host_sinks, host_quark, n_quark, tile_quark, host_evec, n_evec, tile_evec, inv_param, X);

  // Parameter object describing the sources and smeared quarks
  lat_dim_t x = {X[0], X[1], X[2], X[3]};
  ColorSpinorParam cpu_quark_param(host_quark, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> quark(n_quark);
  for (auto i = 0; i < n_quark; i++) {
    cpu_quark_param.v = host_quark[i];
    quark[i] = ColorSpinorField(cpu_quark_param);
  }

  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  // Switch to spin 1
  cpu_evec_param.nSpin = 1;
  cpu_evec_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> evec(n_evec);
  for (auto i = 0; i < n_evec; i++) {
    cpu_evec_param.v = host_evec[i];
    evec[i] = ColorSpinorField(cpu_evec_param);
  }

  // Create device vectors
  ColorSpinorParam quda_quark_param(cpu_quark_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  quda_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  std::vector<ColorSpinorField> quda_quark(tile_quark, quda_quark_param);

  // Create device vectors for evecs
  ColorSpinorParam quda_evec_param(cpu_evec_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  std::vector<ColorSpinorField> quda_evec(tile_evec, quda_evec_param);

  auto Lt = x[3] * comm_dim(3);
  std::vector<Complex> hostSink(n_quark * n_evec * Lt * 4);

  for (auto i = 0; i < n_quark; i += tile_quark) {                       // iterate over all quarks
    auto tile_i = std::min(tile_quark, n_quark - i);                     // handle remainder here
    for (auto tq = 0; tq < tile_i; tq++) quda_quark[tq] = quark[i + tq]; // download quarks

    for (auto j = 0; j < n_evec; j += tile_evec) {                       // iterate over all EV
      auto tile_j = std::min(tile_evec, n_evec - j);                     // handle remainder here
      for (auto te = 0; te < tile_j; te++) quda_evec[te] = evec[j + te]; // download evecs

      std::vector<Complex> tmp(tile_i * tile_j * x[3] * 4);

      // We now perform the projection onto the eigenspace. The data
      // is placed in host_sinks in  T, spin order
      evecProjectLaplace3D(tmp, {quda_quark.begin(), quda_quark.begin() + tile_i},
                           {quda_evec.begin(), quda_evec.begin() + tile_j});

      for (auto tq = 0; tq < tile_i; tq++) {
        for (auto te = 0; te < tile_j; te++) {
          for (auto t = 0; t < x[3]; t++) {
            auto t_global = X[3] * comm_coord(3) + t;
            for (auto s = 0; s < 4; s++) {
              hostSink[(((i + tq) * n_evec + (j + te)) * Lt + t_global) * 4 + s]
                = tmp[((tq * tile_j + te) * x[3] + t) * 4 + s];
            }
          }
        }
      }
    }
  }

  comm_allreduce_sum(hostSink);

  for (auto i = 0; i < n_quark * n_evec * Lt * 4; i++) { // iterate over all quarks
    reinterpret_cast<std::complex<double> *>(host_sinks)[i] = hostSink[i];
  }
}
