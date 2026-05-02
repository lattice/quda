#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>

#include <quda.h>
#include <quda_milc_interface.h>
#include <milc_interface_internal.hpp>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <unitarization_links.h>
#include <ks_improved_force.h>
#include <dslash_quda.h>
#include <invert_quda.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// code for NVTX taken from Jiri Kraus' blog post:
// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#ifdef INTERFACE_NVTX
#include "nvtx3/nvToolsExt.h"

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_colors;\
  nvtxEventAttributes_t eventAttrib = {}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

namespace quda
{
  void setDiracEigParam(DiracParam &, QudaInvertParam *, bool, bool);
}

static bool initialized = false;
#ifdef MULTI_GPU
static int commsGridDim[4];
#endif
static int localDim[4];

static bool invalidate_quda_gauge = true;
static bool create_quda_gauge = false;

static bool have_resident_gauge = false;

static bool invalidate_quda_mom = true;

static bool invalidate_quda_mg = true;

static void *df_preconditioner = nullptr;

static void *preserved_deflation_space[2] = {nullptr, nullptr};
static double preserved_evals_mass[2] = {-1.0, -1.0};

using namespace quda;
using namespace quda::fermion_force;


#define QUDAMILC_VERBOSE 1

template <bool start> void inline qudamilc_called(const char *func, QudaVerbosity verb)
{
  // add NVTX markup if enabled
  if (start) {
    PUSH_RANGE(func, 1);
  } else {
    POP_RANGE;
  }

  #ifdef QUDAMILC_VERBOSE
  if (verb >= QUDA_VERBOSE) {
    if (start) {
      printfQuda("QUDA_MILC_INTERFACE: %s (called) \n", func);
    } else {
      printfQuda("QUDA_MILC_INTERFACE: %s (return) \n", func);
    }
  }
#endif
}

template <bool start> void inline qudamilc_called(const char *func) { qudamilc_called<start>(func, getVerbosity()); }

void qudaSetMPICommHandle(void *mycomm) { setMPICommHandleQuda(mycomm); }

void qudaInit(QudaInitArgs_t input)
{
  // Calling qudamilc_called with QUDA_SUMMARIZE hand-baked in is intentional:
  // if the default verbosity is QUDA_VERBOSE or greater, the printfQuda
  // inside qudamilc_called will barf because qudaSetLayout hasn't been called yet.
  if (initialized) return;
  setVerbosityQuda(input.verbosity, "", stdout);
  qudamilc_called<true>(__func__, QUDA_SUMMARIZE);
  qudaSetLayout(input.layout);
  initialized = true;
  qudamilc_called<false>(__func__, QUDA_SUMMARIZE);
}

void qudaCleanUpDeflationSpace()
{
  qudamilc_called<true>(__func__);
  for (int p = 0; p < 2; p++) {
    if (preserved_deflation_space[p]) {
      deflation_space *space = reinterpret_cast<deflation_space *>(preserved_deflation_space[p]);
      logQuda(QUDA_VERBOSE, "Cleaning up parity %d deflation space of size %lu\n", p, space->evecs.size());
      space->evecs.clear();
      space->evals.clear();
      delete space;
      preserved_deflation_space[p] = nullptr;
      preserved_evals_mass[p] = -1.0;
    }
  }
  qudamilc_called<false>(__func__);
}

void qudaFinalize()
{
  qudamilc_called<true>(__func__, QUDA_VERBOSE);
  endQuda();
  qudamilc_called<false>(__func__, QUDA_VERBOSE);
}
#if defined(MULTI_GPU) && !defined(QMP_COMMS)
/**
 *  Implements a lexicographical mapping of node coordinates to ranks,
 *  with t varying fastest.
 */
static int rankFromCoords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = dims[i] * rank + coords[i];
  }
  return rank;
}
#endif

void qudaSetLayout(QudaLayout_t input)
{
  int local_dim[4];
  for(int dir=0; dir<4; ++dir){ local_dim[dir] = input.latsize[dir]; }
#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir){ local_dim[dir] /= input.machsize[dir]; }
#endif
  for(int dir=0; dir<4; ++dir){
    if(local_dim[dir]%2 != 0){
      printf("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

  for(int dir=0; dir<4; ++dir) localDim[dir] = local_dim[dir];

#ifdef MULTI_GPU
  for (int dir = 0; dir < 4; ++dir) commsGridDim[dir] = input.machsize[dir];
#ifdef QMP_COMMS
  initCommsGridQuda(4, commsGridDim, nullptr, nullptr);
#else
  initCommsGridQuda(4, commsGridDim, rankFromCoords, (void *)(commsGridDim));
#endif
  static int device = -1;
#else
  static int device = input.device;
#endif

  initQuda(device);
}

void *qudaAllocatePinned(size_t bytes) { return pool_pinned_malloc(bytes); }

void qudaFreePinned(void *ptr) { pool_pinned_free(ptr); }

void *qudaAllocateManaged(size_t bytes) { return managed_malloc(bytes); }

void qudaFreeManaged(void *ptr) { managed_free(ptr); }

void qudaHisqParamsInit(QudaHisqParams_t params)
{
  static bool initialized = false;

  if (initialized) return;
  qudamilc_called<true>(__func__);

  const bool reunit_allow_svd = (params.reunit_allow_svd) ? true : false;
  const bool reunit_svd_only  = (params.reunit_svd_only) ? true : false;
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;

  quda::fermion_force::setUnitarizeForceConstants(unitarize_eps,
      params.force_filter,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);

  setUnitarizeLinksConstants(unitarize_eps,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);

  initialized = true;
  qudamilc_called<false>(__func__);
  return;
}



static QudaGaugeParam newMILCGaugeParam(const int* dim, QudaPrecision prec, QudaLinkType link_type)
{
  QudaGaugeParam gParam = newQudaGaugeParam();
  for(int dir=0; dir<4; ++dir) gParam.X[dir] = dim[dir];
  gParam.cuda_prec_sloppy = gParam.cpu_prec = gParam.cuda_prec = prec;
  gParam.type = link_type;

  gParam.reconstruct_sloppy = gParam.reconstruct = ((link_type == QUDA_SU3_LINKS) ? QUDA_RECONSTRUCT_12 : QUDA_RECONSTRUCT_NO);
  gParam.gauge_order   = QUDA_MILC_GAUGE_ORDER;
  gParam.t_boundary    = QUDA_PERIODIC_T;
  gParam.gauge_fix     = QUDA_GAUGE_FIXED_NO;
  gParam.scale         = 1.0;
  gParam.anisotropy    = 1.0;
  gParam.tadpole_coeff = 1.0;
  gParam.scale         = 0;
  return gParam;
}

static  void invalidateGaugeQuda() {
  qudamilc_called<true>(__func__);
  freeGaugeQuda();
  invalidate_quda_gauge = true;
  have_resident_gauge = false;
  qudamilc_called<false>(__func__);
}

static void getReconstruct(QudaReconstructType &reconstruct, QudaReconstructType &reconstruct_sloppy)
{
  static bool recon_queried = false;
  static QudaReconstructType reconstruct_in = QUDA_RECONSTRUCT_INVALID;
  static QudaReconstructType reconstruct_sloppy_in = QUDA_RECONSTRUCT_INVALID;
  if (!recon_queried) {
    char *reconstruct_env = getenv("QUDA_MILC_HISQ_RECONSTRUCT");
    if (!reconstruct_env || strcmp(reconstruct_env, "18") == 0) {
      reconstruct_in = QUDA_RECONSTRUCT_NO;
    } else if (strcmp(reconstruct_env, "13") == 0) {
      reconstruct_in = QUDA_RECONSTRUCT_13;
    } else if (strcmp(reconstruct_env, "9") == 0) {
      reconstruct_in = QUDA_RECONSTRUCT_9;
    } else {
      errorQuda("QUDA_MILC_HISQ_RECONSTRUCT=%s not supported", reconstruct_env);
    }
    char *reconstruct_sloppy_env = getenv("QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY");
    if (!reconstruct_sloppy_env) { // if env is not set, default to using outer reconstruct type
      reconstruct_sloppy_in = reconstruct_in;
    } else if (strcmp(reconstruct_sloppy_env, "18") == 0) {
      reconstruct_sloppy_in = QUDA_RECONSTRUCT_NO;
    } else if (strcmp(reconstruct_sloppy_env, "13") == 0) {
      reconstruct_sloppy_in = QUDA_RECONSTRUCT_13;
    } else if (strcmp(reconstruct_sloppy_env, "9") == 0) {
      reconstruct_sloppy_in = QUDA_RECONSTRUCT_9;
    } else {
      errorQuda("QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=%s not supported", reconstruct_sloppy_env);
    }
    recon_queried = true;
  }
  reconstruct = reconstruct_in;
  reconstruct_sloppy = reconstruct_sloppy_in;
}

void qudaLoadKSLink(int prec, QudaFatLinkArgs_t, const double act_path_coeff[6], void *inlink, void *fatlink,
                    void *longlink)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam param = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  param.staggered_phase_applied = 1;
  param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;

  computeKSLinkQuda(fatlink, longlink, nullptr, inlink, const_cast<double*>(act_path_coeff), &param);

  // requires loadGaugeQuda to be called in subequent solver
  invalidateGaugeQuda();

  // this flags that we are using QUDA to create the HISQ links
  create_quda_gauge = true;
  qudamilc_called<false>(__func__);
}

void qudaLoadUnitarizedLink(int prec, QudaFatLinkArgs_t, const double act_path_coeff[6], void *inlink, void *fatlink,
                            void *ulink)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam param = newMILCGaugeParam(localDim,
					   (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
					   QUDA_GENERAL_LINKS);

  computeKSLinkQuda(fatlink, nullptr, ulink, inlink, const_cast<double*>(act_path_coeff), &param);

  // requires loadGaugeQuda to be called in subequent solver
  invalidateGaugeQuda();

  // this flags that we are using QUDA to create the HISQ links
  create_quda_gauge = true;
  qudamilc_called<false>(__func__);
}


void qudaHisqForce(int prec, int num_terms, int num_naik_terms, double dt, double** coeff, void** quark_field,
                   const double level2_coeff[6], const double fat7_coeff[6],
                   const void* const w_link, const void* const v_link, const void* const u_link,
                   void* const milc_momentum)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam gParam = newMILCGaugeParam(localDim, (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  // Use to specify the reconstruct for the HISQ force calculation
  getReconstruct(gParam.reconstruct, gParam.reconstruct_sloppy);

  if (!invalidate_quda_mom) {
    gParam.use_resident_mom = true;
    gParam.make_resident_mom = true;
    gParam.return_result_mom = false;
  } else {
    gParam.use_resident_mom = false;
    gParam.make_resident_mom = false;
    gParam.return_result_mom = true;
  }

  computeHISQForceQuda(milc_momentum, dt, level2_coeff, fat7_coeff,
                       w_link, v_link, u_link,
                       quark_field, num_terms, num_naik_terms, coeff,
                       &gParam);

  have_resident_gauge = false;
  qudamilc_called<false>(__func__);
  return;
}

void qudaAsqtadForce(int, const double[6], const void *const[4], const void *const[4], const void *const, void *const)
{
  errorQuda("This interface has been removed and is no longer supported");
}

void qudaComputeOprod(int, int, int, double **, double, void **, void *[3])
{
  errorQuda("This interface has been removed and is no longer supported");
}

void qudaUpdateUPhasedPipeline(int prec, double eps, QudaMILCSiteArg_t *arg, int phase_in, int want_gaugepipe)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);
  void *gauge = arg->site ? arg->site : arg->link;
  void *mom = arg->site ? arg->site : arg->mom;

  qudaGaugeParam.gauge_offset = arg->link_offset;
  qudaGaugeParam.mom_offset = arg->mom_offset;
  qudaGaugeParam.site_size = arg->size;
  qudaGaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;

  qudaGaugeParam.staggered_phase_applied = phase_in;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  if (phase_in) qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  if (want_gaugepipe) {
    qudaGaugeParam.make_resident_gauge = true;
    qudaGaugeParam.return_result_gauge = true;
    if (!have_resident_gauge) {
      qudaGaugeParam.use_resident_gauge = false;
      have_resident_gauge = true;
      if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("QUDA_MILC_INTERFACE: Using gauge pipeline \n"); }
    } else {
      qudaGaugeParam.use_resident_gauge = true;
    }
  }

  if (!invalidate_quda_mom) {
    qudaGaugeParam.use_resident_mom = true;
    qudaGaugeParam.make_resident_mom = true;
  } else {
    qudaGaugeParam.use_resident_mom = false;
    qudaGaugeParam.make_resident_mom = false;
  }

  updateGaugeFieldQuda(gauge, mom, eps, 0, 0, &qudaGaugeParam);
  qudamilc_called<false>(__func__);
  return;
}

void qudaUpdateUPhased(int prec, double eps, QudaMILCSiteArg_t *arg, int phase_in)
{
  qudaUpdateUPhasedPipeline(prec, eps, arg, phase_in, 0);
}

void qudaUpdateU(int prec, double eps, QudaMILCSiteArg_t *arg) { qudaUpdateUPhased(prec, eps, arg, 0); }

void qudaRephase(int prec, void *gauge, int flag, double i_mu)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  qudaGaugeParam.staggered_phase_applied = 1 - flag;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  qudaGaugeParam.i_mu = i_mu;
  qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;

  staggeredPhaseQuda(gauge, &qudaGaugeParam);
  qudamilc_called<false>(__func__);
  return;
}

void qudaUnitarizeSU3Phased(int prec, double tol, QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  void *gauge = arg->site ? arg->site : arg->link;
  qudaGaugeParam.gauge_offset = arg->link_offset;
  qudaGaugeParam.site_size = arg->size;
  qudaGaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  qudaGaugeParam.staggered_phase_applied = phase_in;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  // when we take care of phases in QUDA we need to respect MILC boundary conditions.
  if (phase_in) qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;

  if (!have_resident_gauge) {
    qudaGaugeParam.make_resident_gauge = false;
    qudaGaugeParam.use_resident_gauge = false;
  } else {
    qudaGaugeParam.use_resident_gauge = true;
    qudaGaugeParam.make_resident_gauge = true;
  }
  qudaGaugeParam.return_result_gauge = true;
  have_resident_gauge = false;

  projectSU3Quda(gauge, tol, &qudaGaugeParam);
  invalidateGaugeQuda();
  qudamilc_called<false>(__func__);
  return;
}

void qudaUnitarizeSU3(int prec, double tol, QudaMILCSiteArg_t *arg) { qudaUnitarizeSU3Phased(prec, tol, arg, 0); }

// download the momentum from MILC and place into the resident mom field
void qudaMomLoad(int prec, QudaMILCSiteArg_t *arg)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam param
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  void *mom = arg->site ? arg->site : arg->mom;
  param.mom_offset = arg->mom_offset;
  param.site_size = arg->size;
  param.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  param.make_resident_mom = 1;
  param.return_result_mom = 0;

  momResidentQuda(mom, &param);
  invalidate_quda_mom = false;

  qudamilc_called<false>(__func__);
}

// upload the momentum to MILC and invalidate the current resident momentum
void qudaMomSave(int prec, QudaMILCSiteArg_t *arg)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam param
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  void *mom = arg->site ? arg->site : arg->mom;
  param.mom_offset = arg->mom_offset;
  param.site_size = arg->size;
  param.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  param.make_resident_mom = 0;
  param.return_result_mom = 1;

  momResidentQuda(mom, &param);
  invalidate_quda_mom = true;

  qudamilc_called<false>(__func__);
}

double qudaMomAction(int prec, QudaMILCSiteArg_t *arg)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam param
    = newMILCGaugeParam(localDim, (prec == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);

  void *mom = arg->site ? arg->site : arg->mom;
  param.mom_offset = arg->mom_offset;
  param.site_size = arg->size;
  param.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  param.make_resident_mom = 0;

  if (!invalidate_quda_mom) {
    param.use_resident_mom = true;
    param.make_resident_mom = true;
    invalidate_quda_mom = false;
  } else { // no momentum residency
    param.use_resident_mom = false;
    param.make_resident_mom = false;
    invalidate_quda_mom = true;
  }

  double action = momActionQuda(mom, &param);

  qudamilc_called<false>(__func__);

  return action;
}

static inline int opp(int dir){
  return 7-dir;
}


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

void qudaGaugeForcePhased(int precision, int num_loop_types, double milc_loop_coeff[3], double eb3,
                          QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);

  int numPaths = 0;
  switch (num_loop_types) {
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
    errorQuda("Invalid num_loop_types = %d\n", num_loop_types);
  }

  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim,
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_SU3_LINKS);
  void *gauge = arg->site ? arg->site : arg->link;
  void *mom = arg->site ? arg->site : arg->mom;

  qudaGaugeParam.gauge_offset = arg->link_offset;
  qudaGaugeParam.mom_offset = arg->mom_offset;
  qudaGaugeParam.site_size = arg->size;
  qudaGaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  qudaGaugeParam.staggered_phase_applied = phase_in;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  if (phase_in) qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  if (phase_in) qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;

  if (!have_resident_gauge) {
    qudaGaugeParam.make_resident_gauge = true;
    qudaGaugeParam.use_resident_gauge = false;
    // have_resident_gauge = true;
  } else {
    qudaGaugeParam.make_resident_gauge = true;
    qudaGaugeParam.use_resident_gauge = true;
  }

  double *loop_coeff = static_cast<double*>(safe_malloc(numPaths*sizeof(double)));
  int *length = static_cast<int*>(safe_malloc(numPaths*sizeof(int)));

  if (num_loop_types >= 1) for(int i= 0; i< 6; ++i) {
      loop_coeff[i] = milc_loop_coeff[0];
      length[i] = 3;
    }
  if (num_loop_types >= 2) for(int i= 6; i<24; ++i) {
      loop_coeff[i] = milc_loop_coeff[1];
      length[i] = 5;
    }
  if (num_loop_types >= 3) for(int i=24; i<48; ++i) {
      loop_coeff[i] = milc_loop_coeff[2];
      length[i] = 5;
    }

  int** input_path_buf[4];
  for(int dir=0; dir<4; ++dir){
    input_path_buf[dir] = static_cast<int**>(safe_malloc(numPaths*sizeof(int*)));
    for(int i=0; i<numPaths; ++i){
      input_path_buf[dir][i] = static_cast<int*>(safe_malloc(length[i]*sizeof(int)));
    }
    createGaugeForcePaths(input_path_buf[dir], dir, num_loop_types);
  }

  if (!invalidate_quda_mom) {
    qudaGaugeParam.use_resident_mom = true;
    qudaGaugeParam.make_resident_mom = true;
    qudaGaugeParam.return_result_mom = false;

    // this means when we compute the momentum, we acummulate to the
    // preexisting resident momentum instead of overwriting it
    qudaGaugeParam.overwrite_mom = false;
  } else {
    qudaGaugeParam.use_resident_mom = false;
    qudaGaugeParam.make_resident_mom = false;
    qudaGaugeParam.return_result_mom = true;

    // this means we compute momentum into a fresh field, copy it back
    // and sum to current momentum in MILC.  This saves an initial
    // CPU->GPU download of the current momentum.
    qudaGaugeParam.overwrite_mom = false;
  }

  int max_length = 6;

  computeGaugeForceQuda(mom, gauge, input_path_buf, length,
			loop_coeff, numPaths, max_length, eb3, &qudaGaugeParam);

  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<numPaths; ++i) host_free(input_path_buf[dir][i]);
    host_free(input_path_buf[dir]);
  }

  host_free(length);
  host_free(loop_coeff);

  qudamilc_called<false>(__func__);
  return;
}

void qudaGaugeForce(int precision, int num_loop_types, double milc_loop_coeff[3], double eb3, QudaMILCSiteArg_t *arg)
{
  qudaGaugeForcePhased(precision, num_loop_types, milc_loop_coeff, eb3, arg, 0);
}

/**
 * @brief Reusable routine that creates a qudaGaugeParam for gauge-related observable measurements
 *
 * @param[in] precision MILC precision
 * @param[in] arg MILC Site arg structure
 * @param[in] phase_in Whether or not phases have been applied
 * @return A qudaGaugeParam that can be passed to QUDA interface functions
 */
QudaGaugeParam createGaugeParamForObservables(int precision, QudaMILCSiteArg_t *arg, int phase_in)
{
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, (precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_WILSON_LINKS);

  qudaGaugeParam.gauge_offset = arg->link_offset;
  qudaGaugeParam.mom_offset = arg->mom_offset;
  qudaGaugeParam.site_size = arg->size;
  qudaGaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;
  qudaGaugeParam.staggered_phase_applied = phase_in;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  // FIXME: phases and boundary conditions are "munged" together inside QUDA, so the unphase function
  // doesn't change the boundary condition flag. This setting guarantees that phases and boundary conditions
  // are consistently set under the hood --- but we still need an extra minus sign on the output.
  qudaGaugeParam.t_boundary = QUDA_PERIODIC_T;
  // if (phase_in) qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  if (phase_in) qudaGaugeParam.reconstruct_sloppy = qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;

  if (!have_resident_gauge) {
    qudaGaugeParam.make_resident_gauge = false;
    qudaGaugeParam.use_resident_gauge = false;
  } else {
    qudaGaugeParam.use_resident_gauge = true;
    qudaGaugeParam.make_resident_gauge = true;
  }

  return qudaGaugeParam;
}

void qudaGaugeLoopTracePhased(int precision, double *traces, int **input_path_buf, int *path_length, double *loop_coeff,
                              int num_paths, int max_length, double factor, QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam qudaGaugeParam = createGaugeParamForObservables(precision, arg, phase_in);
  void *gauge = arg->site ? arg->site : arg->link;

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_gauge_loop_trace = QUDA_BOOLEAN_TRUE;
  obsParam.traces = reinterpret_cast<double _Complex *>(traces);
  obsParam.input_path_buff = input_path_buf;
  obsParam.path_length = path_length;
  obsParam.loop_coeff = loop_coeff;
  obsParam.num_paths = num_paths;
  obsParam.max_length = max_length;
  obsParam.factor = factor;
  obsParam.remove_staggered_phase = phase_in ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  qudamilc_called<false>(__func__);
  return;
}

void qudaPlaquettePhased(int precision, double plaq[3], QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam qudaGaugeParam = createGaugeParamForObservables(precision, arg, phase_in);
  void *gauge = arg->site ? arg->site : arg->link;

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obsParam.remove_staggered_phase = phase_in ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  // Let MILC apply its own Nc normalization
  plaq[0] = obsParam.plaquette[0];
  plaq[1] = obsParam.plaquette[1];
  plaq[2] = obsParam.plaquette[2];

  qudamilc_called<false>(__func__);
  return;
}

void qudaPolyakovLoopPhased(int precision, double ploop[2], int dir, QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);

  if (dir != 3) errorQuda("Invalid direction %d, only the temporal Polyakov loop can be computed at this time", dir);

  QudaGaugeParam qudaGaugeParam = createGaugeParamForObservables(precision, arg, phase_in);
  void *gauge = arg->site ? arg->site : arg->link;

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
  obsParam.remove_staggered_phase = phase_in ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  // FIXME: see comment in createGaugeParamForObservables
  ploop[0] = -obsParam.ploop[0];
  ploop[1] = -obsParam.ploop[1];

  qudamilc_called<false>(__func__);
  return;
}

void qudaGaugeMeasurementsPhased(int precision, double plaq[3], double ploop[2], int dir, double *traces,
                                 int **input_path_buf, int *path_length, double *loop_coeff, int num_paths,
                                 int max_length, double factor, QudaMILCSiteArg_t *arg, int phase_in)
{
  qudamilc_called<true>(__func__);

  if (dir != 3) errorQuda("Invalid direction %d, only the temporal Polyakov loop can be computed at this time", dir);

  QudaGaugeParam qudaGaugeParam = createGaugeParamForObservables(precision, arg, phase_in);
  void *gauge = arg->site ? arg->site : arg->link;

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obsParam.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
  obsParam.compute_gauge_loop_trace = QUDA_BOOLEAN_TRUE;
  obsParam.traces = reinterpret_cast<double _Complex *>(traces);
  obsParam.input_path_buff = input_path_buf;
  obsParam.path_length = path_length;
  obsParam.loop_coeff = loop_coeff;
  obsParam.num_paths = num_paths;
  obsParam.max_length = max_length;
  obsParam.factor = factor;
  obsParam.remove_staggered_phase = phase_in ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  // Let MILC apply its Nc normalization
  plaq[0] = obsParam.plaquette[0];
  plaq[1] = obsParam.plaquette[1];
  plaq[2] = obsParam.plaquette[2];

  // FIXME: see comment in createGaugeParamForObservables
  ploop[0] = -obsParam.ploop[0];
  ploop[1] = -obsParam.ploop[1];

  qudamilc_called<false>(__func__);
  return;
}

// set the params for the single mass solver
static void setInvertParams(QudaPrecision cpu_prec, QudaPrecision cuda_prec, QudaPrecision cuda_prec_sloppy,
                            double mass, double target_residual, double target_residual_hq, int maxiter,
                            double reliable_delta, QudaParity parity, QudaVerbosity verbosity,
                            QudaInverterType inverter, QudaInvertParam *invertParam)
{
  invertParam->verbosity = verbosity;
  invertParam->mass = mass;
  invertParam->tol = target_residual;
  invertParam->tol_hq = target_residual_hq;

  invertParam->residual_type = static_cast<QudaResidualType_s>(0);
  invertParam->residual_type = (target_residual != 0) ?
    static_cast<QudaResidualType_s>(invertParam->residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    invertParam->residual_type;
  invertParam->residual_type = (target_residual_hq != 0) ?
    static_cast<QudaResidualType_s>(invertParam->residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    invertParam->residual_type;

  invertParam->heavy_quark_check = (invertParam->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 1 : 0);
  if (invertParam->heavy_quark_check) {
    invertParam->max_hq_res_increase = 5;       // this caps the number of consecutive hq residual increases
    invertParam->max_hq_res_restart_total = 10; // this caps the number of hq restarts in case of solver stalling
  }

  invertParam->use_sloppy_partial_accumulator = 0;
  invertParam->num_offset = 0;

  invertParam->inv_type = inverter;
  invertParam->maxiter = maxiter;
  invertParam->reliable_delta = reliable_delta;

  invertParam->mass_normalization = QUDA_MASS_NORMALIZATION;
  invertParam->cpu_prec = cpu_prec;
  invertParam->cuda_prec = cuda_prec;
  invertParam->cuda_prec_sloppy = invertParam->heavy_quark_check ? cuda_prec : cuda_prec_sloppy;
  invertParam->cuda_prec_precondition = cuda_prec_sloppy;

  invertParam->gcrNkrylov = 10;

  invertParam->solution_type = QUDA_MATPC_SOLUTION;
  invertParam->solve_type = QUDA_DIRECT_PC_SOLVE;
  invertParam->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // not used, but required by the code.
  invertParam->dirac_order = QUDA_DIRAC_ORDER;

  invertParam->dslash_type = QUDA_ASQTAD_DSLASH;
  invertParam->Ls = 1;
  invertParam->gflops = 0.0;

  invertParam->input_location = QUDA_CPU_FIELD_LOCATION;
  invertParam->output_location = QUDA_CPU_FIELD_LOCATION;

  if (parity == QUDA_EVEN_PARITY) { // even parity
    invertParam->matpc_type = QUDA_MATPC_EVEN_EVEN;
  } else if (parity == QUDA_ODD_PARITY) {
    invertParam->matpc_type = QUDA_MATPC_ODD_ODD;
  } else {
    errorQuda("Invalid parity\n");
  }

  invertParam->dagger = QUDA_DAG_NO;
  invertParam->use_init_guess = QUDA_USE_INIT_GUESS_YES;

  // for the preconditioner
  invertParam->inv_type_precondition = QUDA_CG_INVERTER;
  invertParam->tol_precondition = 1e-1;
  invertParam->maxiter_precondition = 2;
  invertParam->verbosity_precondition = QUDA_SILENT;

  invertParam->compute_action = 0;
}


// Set params for the multi-mass solver.
static void setInvertParams(QudaPrecision cpu_prec, QudaPrecision cuda_prec, QudaPrecision cuda_prec_sloppy,
                            int num_offset, const double offset[], const double target_residual_offset[],
                            const double target_residual_hq_offset[], int maxiter, double reliable_delta,
                            QudaParity parity, QudaVerbosity verbosity, QudaInverterType inverter,
                            QudaInvertParam *invertParam)
{
  const double null_mass = -1;

  setInvertParams(cpu_prec, cuda_prec, cuda_prec_sloppy, null_mass, target_residual_offset[0],
                  target_residual_hq_offset[0], maxiter, reliable_delta, parity, verbosity, inverter, invertParam);

  invertParam->num_offset = num_offset;
  for (int i = 0; i < num_offset; ++i) {
    invertParam->offset[i] = offset[i];
    invertParam->tol_offset[i] = target_residual_offset[i];
    invertParam->tol_hq_offset[i] = target_residual_hq_offset[i];
  }
}

static void setGaugeParams(QudaGaugeParam &fat_param, QudaGaugeParam &long_param, const void *const longlink,
                           const int dim[4], QudaPrecision cpu_prec, QudaPrecision cuda_prec,
                           QudaPrecision cuda_prec_sloppy, double tadpole, double naik_epsilon)
{
  for (int dir = 0; dir < 4; ++dir) fat_param.X[dir] = dim[dir];

  fat_param.cpu_prec = cpu_prec;
  fat_param.cuda_prec = cuda_prec;
  fat_param.cuda_prec_sloppy = cuda_prec_sloppy;
  fat_param.cuda_prec_precondition = cuda_prec_sloppy;
  fat_param.reconstruct = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
  fat_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  fat_param.anisotropy = 1.0;
  fat_param.t_boundary = QUDA_PERIODIC_T; // anti-periodic boundary conditions are built into the gauge field
  fat_param.gauge_order = QUDA_MILC_GAUGE_ORDER;

  if (longlink != nullptr) {
    // improved staggered parameters
    fat_param.type = QUDA_ASQTAD_FAT_LINKS;

    // now set the long link parameters needed
    long_param = fat_param;
    long_param.tadpole_coeff = tadpole;
    long_param.scale = -(1.0 + naik_epsilon) / (24.0 * long_param.tadpole_coeff * long_param.tadpole_coeff);
    long_param.type = QUDA_THREE_LINKS;
    getReconstruct(long_param.reconstruct, long_param.reconstruct_sloppy);
    long_param.reconstruct_precondition = long_param.reconstruct_sloppy;
  } else {
    // naive staggered parameters
    fat_param.type = QUDA_SU3_LINKS;
    fat_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  }

}

static void setEigensolverParams(QudaEigensolverArgs_t eig_args, QudaEigParam *qep)
{
  qep->block_size = eig_args.block_size;
  qep->n_conv = eig_args.n_conv;
  qep->n_ev_deflate = eig_args.n_ev_deflate;
  qep->n_ev = eig_args.n_ev;
  qep->n_kr = eig_args.n_kr;
  qep->tol = eig_args.tol;
  qep->max_restarts = eig_args.max_restarts;
  qep->poly_deg = eig_args.poly_deg;
  qep->a_min = eig_args.a_min;
  qep->a_max = eig_args.a_max;
  strcpy(qep->vec_infile, eig_args.vec_infile);
  strcpy(qep->vec_outfile, eig_args.vec_outfile);
  qep->preserve_evals = eig_args.preserve_evals;
  qep->batched_rotate = eig_args.batched_rotate;
  qep->save_prec = eig_args.save_prec;
  qep->partfile = eig_args.partfile;
  qep->io_parity_inflate = eig_args.io_parity_inflate;
  qep->use_norm_op = eig_args.use_norm_op;
  qep->use_pc = eig_args.use_pc;
  qep->eig_type = eig_args.eig_type;
  qep->spectrum = eig_args.spectrum;
  qep->qr_tol = eig_args.qr_tol;
  qep->require_convergence = eig_args.require_convergence;
  qep->check_interval = eig_args.check_interval;
  qep->use_dagger = eig_args.use_dagger;
  qep->compute_gamma5 = eig_args.compute_gamma5;
  qep->compute_svd = eig_args.compute_svd;
  qep->use_eigen_qr = eig_args.use_eigen_qr;
  qep->use_poly_acc = eig_args.use_poly_acc;
  qep->arpack_check = eig_args.arpack_check;
  qep->compute_evals_batch_size = eig_args.compute_evals_batch_size;
  qep->preserve_deflation = eig_args.preserve_deflation;
}

static void setColorSpinorParams(const int dim[4], QudaPrecision precision, ColorSpinorParam *param)
{
  param->nColor = 3;
  param->nSpin = 1;
  param->nDim = 4;

  for (int dir = 0; dir < 4; ++dir) param->x[dir] = dim[dir];
  param->x[0] /= 2;

  param->setPrecision(precision);
  param->pad = 0;
  param->siteSubset = QUDA_PARITY_SITE_SUBSET;
  param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
  param->pc_type = QUDA_4D_PC;
  param->create = QUDA_ZERO_FIELD_CREATE;
}

void setDeflationParam(QudaPrecision ritz_prec, QudaFieldLocation location_ritz, QudaMemoryType mem_type_ritz,
                       QudaExtLibType deflation_ext_lib, char vec_infile[], char vec_outfile[], QudaEigParam *df_param)
{
  df_param->import_vectors = strcmp(vec_infile,"") ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  df_param->cuda_prec_ritz = ritz_prec;
  df_param->location       = location_ritz;
  df_param->mem_type_ritz  = mem_type_ritz;

  df_param->run_verify     = QUDA_BOOLEAN_FALSE;

  df_param->nk = df_param->invert_param->n_ev;
  df_param->np = df_param->invert_param->n_ev * df_param->invert_param->deflation_grid;

  df_param->extlib_type = deflation_ext_lib;

  // set file i/o parameters
  strcpy(df_param->vec_infile, vec_infile);
  strcpy(df_param->vec_outfile, vec_outfile);
  df_param->io_parity_inflate = QUDA_BOOLEAN_TRUE;
}

static size_t getColorVectorOffset(QudaParity local_parity, bool even_odd_exchange, const int dim[4])
{
  size_t offset;
  int volume = dim[0]*dim[1]*dim[2]*dim[3];

  if(local_parity == QUDA_EVEN_PARITY){
    offset = even_odd_exchange ? volume*6/2 : 0;
  }else{
    offset = even_odd_exchange ? 0 : volume*6/2;
  }
  return offset;
}

void qudaMultishiftInvert(int external_precision, int quda_precision, int num_offsets, double *const offset,
                          QudaInvertArgs_t inv_args, const double target_residual[],
                          const double target_fermilab_residual[], const void *const fatlink,
                          const void *const longlink, void *source, void **solutionArray, double *const final_residual,
                          double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  if (target_residual[0] == 0) errorQuda("qudaMultishiftInvert: zeroth target residual cannot be zero\n");

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  static bool force_double_queried = false;
  static bool do_not_force_double = false;
  if (!force_double_queried) {
    char *donotusedouble_env = getenv("QUDA_MILC_OVERRIDE_DOUBLE_MULTISHIFT"); // disable forcing outer double precision
    if (donotusedouble_env && (!(strcmp(donotusedouble_env, "0") == 0))) {
      do_not_force_double = true;
      printfQuda("Disabling always using double as fine precision for MILC multishift\n");
    }
    force_double_queried = true;
  }

  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  bool use_mixed_precision = (((quda_precision == 2) && inv_args.mixed_precision)
                              || ((quda_precision == 1) && (inv_args.mixed_precision == 2))) ?
    true :
    false;

  QudaPrecision device_precision_sloppy;
  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  // override fine precision to double, switch to mixed as necessary
  if (!do_not_force_double && device_precision == QUDA_SINGLE_PRECISION) {
    // force outer double
    device_precision = QUDA_DOUBLE_PRECISION;
    if (device_precision_sloppy == QUDA_SINGLE_PRECISION) use_mixed_precision = true;
  }

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  const double reliable_delta = (use_mixed_precision ? 1e-1 : 0.0);
  setInvertParams(host_precision, device_precision, device_precision_sloppy, num_offsets, offset, target_residual,
                  target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
                  QUDA_CG_INVERTER, &invertParam);

  if (inv_args.mixed_precision == 1) {
    fat_param.cuda_prec_refinement_sloppy = QUDA_HALF_PRECISION;
    long_param.cuda_prec_refinement_sloppy = QUDA_HALF_PRECISION;
    long_param.reconstruct_refinement_sloppy = long_param.reconstruct_sloppy;
    invertParam.cuda_prec_refinement_sloppy = QUDA_HALF_PRECISION;
    invertParam.reliable_delta_refinement = 0.1;
  }

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) invalidateGaugeQuda();

  // set the solver
  if (invalidate_quda_gauge || !create_quda_gauge) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;

  void **sln_pointer = (void **)safe_malloc(num_offsets * sizeof(void *));
  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;
  void* src_pointer = static_cast<char*>(source) + quark_offset;

  for (int i = 0; i < num_offsets; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[i]) + quark_offset;

  invertMultiShiftQuda(sln_pointer, src_pointer, &invertParam);
  host_free(sln_pointer);

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for (int i = 0; i < num_offsets; ++i) {
    final_residual[i] = invertParam.true_res_offset[i];
    final_fermilab_residual[i] = invertParam.true_res_hq_offset[i];
  } // end loop over number of offsets

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaMultiShiftInvert

// Project the low modes off of source vector(s)
void qudaProject(int external_precision, void **source, void **solution, int nvec, int n_evec, QudaParity parity)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);
  logQuda(QUDA_VERBOSE, "Projecting %d low modes out of %d source vectors for parity %s\n", n_evec, nvec,
          parity == QUDA_EVEN_PARITY ? "EVEN" : "ODD");
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  // Multiple sweeps of projection to improve precision
  int nsweeps = 2;

  // Check inputs
  for (int i = 0; i < nvec; i++)
    if (!source[i] || !solution[i]) errorQuda("Source or solution vector %d is null!", i);

  // MILC sends pointers to full parity vectors, but QUDA uses single parity vectors
  // so for odd parity, need to use offset
  int vec_offset = getColorVectorOffset(parity, false, localDim) * host_precision;

  // Device-side deflation space
  if (parity != QUDA_EVEN_PARITY && parity != QUDA_ODD_PARITY) errorQuda("Invalid parity %d", parity);
  deflation_space *space = reinterpret_cast<deflation_space *>(preserved_deflation_space[parity]);
  if (!space) errorQuda("Failed to get %s parity deflation space!", parity == QUDA_EVEN_PARITY ? "EVEN" : "ODD");

  // Wrap host vectors
  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);
  csParam.location = QUDA_CPU_FIELD_LOCATION;
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  std::vector<ColorSpinorField> src_h(nvec);
  std::vector<ColorSpinorField> sol_h(nvec);
  for (int i = 0; i < nvec; i++) {
    csParam.v = static_cast<void *>(static_cast<char *>(source[i]) + vec_offset);
    src_h[i] = ColorSpinorField(csParam);
    csParam.v = static_cast<void *>(static_cast<char *>(solution[i]) + vec_offset);
    sol_h[i] = ColorSpinorField(csParam);
  }

  // Setup device side vectors
  ColorSpinorParam gpuParam(space->evecs[0]);
  gpuParam.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField> src(nvec);
  std::vector<ColorSpinorField> tmp(nvec);
  for (int i = 0; i < nvec; i++) {
    tmp[i] = ColorSpinorField(gpuParam);
    src[i] = ColorSpinorField(gpuParam);
    src[i] = src_h[i]; // Copy host sources to device sources
  }

  // Do nsweeps of projection on device
  for (int sweep = 0; sweep < nsweeps; sweep++) {

    for (int i = 0; i < nvec; i++) blas::zero(tmp[i]);

    // 1. Take block inner product: (V_i)^dag * src = s_i
    std::vector<Complex> s(n_evec * src.size());
    blas::block::cDotProduct(s, {space->evecs.begin(), space->evecs.begin() + n_evec}, {src.begin(), src.end()});

    // 2. Build projected component: Sum_i V_i * s_i = tmp
    blas::block::caxpy(s, {space->evecs.begin(), space->evecs.begin() + n_evec}, {tmp.begin(), tmp.end()});

    // 3. Subtract projection in place: src = src - tmp
    for (int i = 0; i < nvec; i++) blas::axpy(-1.0, tmp[i], src[i]);
    ;
  }

  // Copy solution back to host
  for (int i = 0; i < nvec; i++) sol_h[i] = src[i];

  qudamilc_called<false>(__func__, verbosity);
} // qudaProject

// Get pointers to QUDA's deflation space objects
// Useful for passing eigenvectors and eigenvalues back to MILC
void qudaGetDeflationSpace(void **evecs, double *evals, QudaParity parity, int Nvecs)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // Device-side deflation space
  if (parity != QUDA_EVEN_PARITY && parity != QUDA_ODD_PARITY) errorQuda("Invalid parity %d", parity);
  deflation_space *space = reinterpret_cast<deflation_space *>(preserved_deflation_space[parity]);
  if (!space) errorQuda("Failed to get %s parity deflation space!", parity == QUDA_EVEN_PARITY ? "EVEN" : "ODD");
  if (static_cast<size_t>(Nvecs) > space->evecs.size())
    errorQuda("Requested %d eigenvectors, but deflation space has only %lu", Nvecs, space->evecs.size());

  // Copy eigenvectors if requested
  if (evecs) {
    // Set up host fields
    ColorSpinorParam csParam(space->evecs[0]);
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    std::vector<ColorSpinorField> host_evecs(Nvecs);
    for (int i = 0; i < Nvecs; i++) {
      csParam.v = evecs[i];
      host_evecs[i] = ColorSpinorField(csParam);
      host_evecs[i] = space->evecs[i]; // Copy to host
    }
  }

  // Copy eigenvalues if requested
  if (evals)
    for (int i = 0; i < Nvecs; i++) evals[i] = space->evals[i].real();

  qudamilc_called<false>(__func__, verbosity);
} // qudaGetDeflationSpace

// Load single parity deflation space with eigenvectors generated from eigensolve, loaded from file,
// passed from MILC, or generated from other parity eigenvectors
void qudaLoadDeflationSpace(int external_precision, int quda_precision, const void *const fatlink,
                            const void *const longlink, double mass, QudaInvertArgs_t inv_args,
                            QudaEigensolverArgs_t eigargs, void **evecs, QudaMilcEigLoad load_type)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;
  switch (inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }
  QudaParity parity = inv_args.evenodd;
  if (parity != QUDA_EVEN_PARITY && parity != QUDA_ODD_PARITY) errorQuda("Invalid parity %d", parity);
  QudaParity other_parity = parity == QUDA_EVEN_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  double epsilon = device_precision == QUDA_DOUBLE_PRECISION ? __DBL_EPSILON__ : __FLT_EPSILON__;
  int n_evecs = eigargs.n_conv;

  // Load gauge fields if not done yet
  if (invalidate_quda_gauge || !create_quda_gauge) {
    QudaGaugeParam fat_param = newQudaGaugeParam();
    QudaGaugeParam long_param = newQudaGaugeParam();
    setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                   inv_args.tadpole, inv_args.naik_epsilon);
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  // Load deflation space
  if (load_type == QUDA_MILC_EIG_COMPUTE) {
    // Main deflation space is obtained by calling the deflatable inverter with dummy source
    // Incoming inv_args can have inv_args.max_iter=1
    logQuda(QUDA_VERBOSE, "Computing deflation space (or loading from file) for parity %d and mass %e\n", parity, mass);

    double final_residual, final_fermilab_residual;
    int num_iters = 0;

    ColorSpinorParam csParam;
    setColorSpinorParams(localDim, host_precision, &csParam);
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET; // qudaInvertDeflatable expects full-parity vectors
    csParam.x[0] *= 2;
    ColorSpinorField source(csParam);
    ColorSpinorField solution(csParam);
    source.Source(QUDA_POINT_SOURCE, inv_args.evenodd, 0, 0); // using dummy point source

    qudaInvertDeflatable(external_precision, quda_precision, mass, inv_args, eigargs, 1.0, 0.0, fatlink, longlink,
                         static_cast<void *>(source.data()), static_cast<void *>(solution.data()), &final_residual,
                         &final_fermilab_residual, &num_iters);

  } else if (load_type == QUDA_MILC_EIG_FROM_OTHER_PARITY) {
    logQuda(QUDA_VERBOSE, "Computing deflation space for parity %d from parity %d\n", parity, other_parity);
    double other_parity_mass = preserved_evals_mass[other_parity];

    // Get preserved other parity deflation space
    deflation_space *other_parity_space = reinterpret_cast<deflation_space *>(preserved_deflation_space[other_parity]);
    if (!other_parity_space)
      errorQuda("Failed to get %s parity deflation space!", parity == QUDA_EVEN_PARITY ? "ODD" : "EVEN");
    if (other_parity_space->evecs.size() < static_cast<size_t>(n_evecs))
      errorQuda("Other parity deflation space too small!");

    // Setup new deflation space
    ColorSpinorParam gpuParam(other_parity_space->evecs[0]);
    deflation_space *space = new deflation_space;
    space->svd = false;
    resize(space->evecs, n_evecs, gpuParam);
    space->evals.resize(n_evecs, 0.0);

    // Create Dirac operator
    QudaInvertParam invertParam = newQudaInvertParam();
    setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, 1.0, 0.0, inv_args.max_iter, 1e-1,
                    parity, verbosity, QUDA_CG_INVERTER, &invertParam);
    invertParam.cuda_prec_eigensolver = eigargs.prec_eigensolver;
    DiracParam diracEigParam;
    setDiracEigParam(diracEigParam, &invertParam, true, false);
    Dirac *dEig = Dirac::create(diracEigParam);

    // Temp vector on GPU
    gpuParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField temp(gpuParam);

    Complex n_unit(-1.0, 0.0);

    for (int i = 0; i < n_evecs; i++) {

      // Compute other parity eigenvector v_o = i*D_oe*v_e/\lambda_e
      dEig->Dslash(temp, other_parity_space->evecs[i], parity);
      auto norm2 = blas::norm2(temp);
      blas::ax(1.0 / sqrt(norm2), temp);
      space->evecs[i] = temp;

      // Compute eigenvalues, lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      dEig->M(temp, space->evecs[i]);
      auto eval = other_parity_space->evals[i];       // re-use eigenvalues by default
      if (fabs(mass - other_parity_mass) > epsilon) { // recompute eigenvalues if mass doesn't match
        auto vtAv = blas::cDotProduct(space->evecs[i], temp);
        auto v2 = blas::norm2(space->evecs[i]);
        eval = vtAv / sqrt(v2);
      }
      space->evals[i] = eval;

      // res^2 = |\lambda*v - A*v|
      auto res = blas::caxpbyNorm(eval, space->evecs[i], n_unit, temp);
      logQuda(QUDA_VERBOSE, "Eval[%04d] = (%+.16e,%+.16e), Residual = %+.16e\n", i, eval.real(), eval.imag(),
              sqrt(res[0]));
    }
    delete dEig;

    // Preserve deflation space
    preserved_deflation_space[parity] = space;
    preserved_evals_mass[parity] = mass;

  } else if (load_type == QUDA_MILC_EIG_LOAD) {

    logQuda(QUDA_VERBOSE, "Loading deflation space of size %d for parity %d and mass %e\n", n_evecs, parity, mass);

    if (!evecs) errorQuda("qudaLoadDeflationSpace called with load_type QUDA_MILC_EIG_LOAD but evecs is null!");

    QudaInvertParam invertParam = newQudaInvertParam();
    setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, 1.0, 0.0, inv_args.max_iter, 1e-1,
                    parity, verbosity, QUDA_CG_INVERTER, &invertParam);
    invertParam.cuda_prec_eigensolver = eigargs.prec_eigensolver;
    ColorSpinorParam csParam;
    setColorSpinorParams(localDim, host_precision, &csParam);
    ColorSpinorParam gpuParam(csParam, invertParam, QUDA_CUDA_FIELD_LOCATION);

    // Setup deflation space
    deflation_space *space = new deflation_space;
    space->svd = false;
    resize(space->evecs, n_evecs, gpuParam);
    space->evals.resize(n_evecs, 0.0);

    // Create Dirac operator
    DiracParam diracEigParam;
    setDiracEigParam(diracEigParam, &invertParam, true, false);
    Dirac *dEig = Dirac::create(diracEigParam);

    // Temp vector on GPU
    gpuParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField temp(gpuParam);

    Complex n_unit(-1.0, 0.0);

    // MILC sends pointer to full parity evecs, but QUDA uses single parity vectors
    // so for odd parity, need to use offset
    int evec_offset = getColorVectorOffset(parity, false, localDim) * host_precision;

    const lat_dim_t dims = {localDim[0], localDim[1], localDim[2], localDim[3]};
    for (int i = 0; i < n_evecs; i++) {

      // Copy each evec to host-side spinor and then to device-side deflation space
      void *evec_ptr = static_cast<char *>(evecs[i]) + evec_offset;
      ColorSpinorParam cpuParam(evec_ptr, invertParam, dims, true, QUDA_CPU_FIELD_LOCATION);
      ColorSpinorField in_evec(cpuParam);
      space->evecs[i] = in_evec;

      // Compute eigenvalue, lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      dEig->M(temp, space->evecs[i]);
      auto vtAv = blas::cDotProduct(space->evecs[i], temp);
      auto v2 = blas::norm2(space->evecs[i]);
      auto eval = vtAv / sqrt(v2);
      space->evals[i] = eval;

      // Compute residual, res^2 = |\lambda*v - A*v|
      auto res = blas::caxpbyNorm(eval, space->evecs[i], n_unit, temp);
      logQuda(QUDA_SUMMARIZE, "Eval[%04d] = (%+.16e,%+.16e), Residual = %+.16e\n", i, eval.real(), eval.imag(),
              sqrt(res[0]));
    }

    delete dEig;

    // Preserve deflation space
    preserved_deflation_space[parity] = space;
    preserved_evals_mass[parity] = mass;

  } else {
    errorQuda("Unrecognized load_type");
  }

  qudamilc_called<false>(__func__, verbosity);
} // qudaLoadDeflationSpace

// Wrapper function for qudaInvertDeflatable to maintain backward compatibility with old(er) MILC
void qudaInvert(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                double target_residual, double target_fermilab_residual, const void *const fatlink,
                const void *const longlink, void *source, void *solution, double *const final_residual,
                double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // If this function is called then QUDA is not doing deflation
  // Create dummy QudaEigensolverArgs_t that requests 0 eigenvalues
  QudaEigensolverArgs_t eig_args;
  eig_args.struct_size = sizeof(eig_args);
  eig_args.n_ev_deflate = 0;
  eig_args.vec_in_parity = QUDA_EVEN_PARITY;
  eig_args.prec_eigensolver = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  qudaInvertDeflatable(external_precision, quda_precision, mass, inv_args, eig_args, target_residual,
                       target_fermilab_residual, fatlink, longlink, source, solution, final_residual,
                       final_fermilab_residual, num_iters);

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvert

void qudaInvertDeflatable(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                          QudaEigensolverArgs_t eig_args, double target_residual, double target_fermilab_residual,
                          const void *const fatlink, const void *const longlink, void *source, void *solution,
                          double *const final_residual, double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // Pass this along to qudaInvertMsrcDeflatable as a single-source solve
  qudaInvertMsrcDeflatable(external_precision, quda_precision, mass, inv_args, eig_args, target_residual,
                           target_fermilab_residual, fatlink, longlink, &source, &solution, final_residual,
                           final_fermilab_residual, num_iters, 1);

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvertDeflatable

void qudaDslash(int external_precision, int quda_precision, QudaInvertArgs_t inv_args, const void *const fatlink,
                const void *const longlink, void* src, void* dst, int* num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = device_precision;
  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);
  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  QudaParity other_parity = local_parity == QUDA_EVEN_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;

  setInvertParams(host_precision, device_precision, device_precision_sloppy, 0.0, 0, 0, 0, 0.0, local_parity, verbosity,
                  QUDA_CG_INVERTER, &invertParam);
  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);
  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) invalidateGaugeQuda();

  if (invalidate_quda_gauge || !create_quda_gauge) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;
  int src_offset = getColorVectorOffset(other_parity, false, localDim);
  int dst_offset = getColorVectorOffset(local_parity, false, localDim);
  dslashQuda(static_cast<char *>(dst) + dst_offset * host_precision,
             static_cast<char *>(src) + src_offset * host_precision, &invertParam, local_parity);
  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaDslash

void qudaShift(int external_precision, int quda_precision, const void *const links, void *src, void *dst, int dir,
               int sym, int reloadGaugeField)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = device_precision;

  QudaGaugeParam gparam = newQudaGaugeParam();
  QudaGaugeParam dparam = newQudaGaugeParam();

  setGaugeParams(gparam, dparam, nullptr, localDim, host_precision, device_precision, device_precision_sloppy, 1.0, 0.0);
  gparam.type = QUDA_WILSON_LINKS;
  gparam.make_resident_gauge = true;
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParams(host_precision, device_precision, device_precision_sloppy, 0.0, 0, 0, 0, 0.0, QUDA_EVEN_PARITY,
                  verbosity, QUDA_CG_INVERTER, &invertParam);
  invertParam.solution_type = QUDA_MAT_SOLUTION;

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] *= 2;
  QudaDslashType saveDslash = invertParam.dslash_type;
  invertParam.dslash_type = QUDA_COVDEV_DSLASH;

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (reloadGaugeField || !canReuseResidentGauge(&invertParam)) {
    if (links == nullptr) {
      errorQuda("Can't offload a null gauge field\n");
      exit(1);
    }
    loadGaugeQuda(const_cast<void *>(links), &gparam);
    // Assume the caller resets reloadGaugeField
    // invalidate_quda_gauge = false;
  }
  invertParam.dslash_type = saveDslash;

  if ((sym < 1) || (sym > 3)) {
    errorQuda("Wrong shift. Select forward (1), backward (2) or symmetric (3).\n");
  } else {
    shiftQuda(dst, src, dir, sym, &invertParam);
  }

  qudamilc_called<false>(__func__, verbosity);
} // qudaShift

void qudaSpinTaste(int external_precision, int quda_precision, const void *const links, void *src, void *dst, int spin,
                   int taste, int reloadGaugeField)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = device_precision;

  QudaGaugeParam gparam = newQudaGaugeParam();
  QudaGaugeParam dparam = newQudaGaugeParam();

  setGaugeParams(gparam, dparam, nullptr, localDim, host_precision, device_precision, device_precision_sloppy, 1.0, 0.0);
  gparam.type = QUDA_WILSON_LINKS;
  gparam.make_resident_gauge = true;
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParams(host_precision, device_precision, device_precision_sloppy, 0.0, 0, 0, 0, 0.0, QUDA_EVEN_PARITY,
                  verbosity, QUDA_CG_INVERTER, &invertParam);
  invertParam.solution_type = QUDA_MAT_SOLUTION;

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] *= 2;
  QudaDslashType saveDslash = invertParam.dslash_type;
  invertParam.dslash_type = QUDA_COVDEV_DSLASH;

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (reloadGaugeField || !canReuseResidentGauge(&invertParam)) {
    if (links == nullptr) {
      errorQuda("Can't offload a null gauge field\n");
      exit(1);
    }
    loadGaugeQuda(const_cast<void *>(links), &gparam);
    // Assume the caller resets reloadGaugeField
  }
  invertParam.dslash_type = saveDslash;

  spinTasteQuda(dst, src, spin, taste, &invertParam);

  qudamilc_called<false>(__func__, verbosity);
} // qudaSpinTaste

// Wrapper function for qudaInvertMsrcDeflatable to maintain backward compatibility with old(er) MILC
void qudaInvertMsrc(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                    double target_residual, double target_fermilab_residual, const void *const fatlink,
                    const void *const longlink, void **sourceArray, void **solutionArray, double *const final_residual,
                    double *const final_fermilab_residual, int *num_iters, int num_src)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // If this function is called then QUDA is not doing deflation
  // Create dummy QudaEigensolverArgs_t that requests 0 eigenvalues
  QudaEigensolverArgs_t eig_args;
  eig_args.struct_size = sizeof(eig_args);
  eig_args.n_ev_deflate = 0;
  eig_args.vec_in_parity = QUDA_EVEN_PARITY;
  eig_args.prec_eigensolver = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  qudaInvertMsrcDeflatable(external_precision, quda_precision, mass, inv_args, eig_args, target_residual,
                           target_fermilab_residual, fatlink, longlink, sourceArray, solutionArray, final_residual,
                           final_fermilab_residual, num_iters, num_src);

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvertMsrc

void qudaInvertMsrcDeflatable(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                              QudaEigensolverArgs_t eig_args, double target_residual, double target_fermilab_residual,
                              const void *const fatlink, const void *const longlink, void **sourceArray,
                              void **solutionArray, double *const final_residual, double *const final_fermilab_residual,
                              int *num_iters, int num_src)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // parameters for the eigensolve/deflation
  QudaEigParam qep = newQudaEigParam();
  setEigensolverParams(eig_args, &qep);

  if (eig_args.struct_size != sizeof(eig_args))
    errorQuda("Unexpected QudaEigensolverArgs_t struct size %lu, expected %lu", eig_args.struct_size, sizeof(eig_args));

  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaInvert: requesting zero residual\n");
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  static bool force_double_queried = false;
  static bool do_not_force_double = false;
  if (!force_double_queried) {
    char *donotusedouble_env = getenv("QUDA_MILC_OVERRIDE_DOUBLE_MULTISHIFT"); // disable forcing outer double precision
    if (donotusedouble_env && (!(strcmp(donotusedouble_env, "0") == 0))) {
      do_not_force_double = true;
      printfQuda("Disabling always using double as fine precision for MILC multishift\n");
    }
    force_double_queried = true;
  }

  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  QudaPrecision device_precision_sloppy;
  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  // override fine precision to double, switch to mixed as necessary
  if (!do_not_force_double && device_precision == QUDA_SINGLE_PRECISION) {
    // force outer double
    device_precision = QUDA_DOUBLE_PRECISION;
  }

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);

  QudaParity local_parity = inv_args.evenodd;
  QudaParity other_parity = local_parity == QUDA_EVEN_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  if (local_parity != QUDA_EVEN_PARITY && local_parity != QUDA_ODD_PARITY) errorQuda("Invalid parity %d", local_parity);
  const double reliable_delta = 1e-1;

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, target_residual,
                  target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
                  QUDA_CG_INVERTER, &invertParam);
  invertParam.num_src = num_src;

  // Deflation parameters
  invertParam.eig_param = (qep.n_ev_deflate > 0) ? &qep : nullptr;
  invertParam.tol_restart = eig_args.tol_restart;
  invertParam.cuda_prec_eigensolver = eig_args.prec_eigensolver;

  // Deflation space
  if (invertParam.eig_param && qep.preserve_deflation) { // if want deflation and use preserved space
    qep.preserve_deflation_space = preserved_deflation_space[local_parity];
    if (!qep.preserve_deflation_space) { // if does not exist yet
      // Check if other parity space exists
      // If so, construct this parity deflation space from other parity deflation space
      // Else, this is skipped and the deflation space is constructed via eigensolve during the call to the inverter
      if (preserved_deflation_space[other_parity]) {
        qudaLoadDeflationSpace(external_precision, quda_precision, fatlink, longlink, mass, inv_args, eig_args, nullptr,
                               QUDA_MILC_EIG_FROM_OTHER_PARITY);
        // This parity deflation space should now exist
        qep.preserve_deflation_space = preserved_deflation_space[local_parity];
        if (!qep.preserve_deflation_space) errorQuda("Failed to load deflation space!");
      }
    }
    // Check that preserved eigenvalues are for this mass
    double epsilon = device_precision == QUDA_DOUBLE_PRECISION ? __DBL_EPSILON__ : __FLT_EPSILON__;
    if (fabs(mass - preserved_evals_mass[local_parity]) > epsilon) {
      logQuda(QUDA_VERBOSE, "Resetting eigenvalues to mass %e\n", invertParam.mass);
      qep.preserve_evals = QUDA_BOOLEAN_FALSE;
    }
  }

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) invalidateGaugeQuda();

  if (invalidate_quda_gauge || !create_quda_gauge) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;

  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;
  void **sln_pointer = (void **)safe_malloc(num_src * sizeof(void *));
  void **src_pointer = (void **)safe_malloc(num_src * sizeof(void *));

  for (int i = 0; i < num_src; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[i]) + quark_offset;
  for (int i = 0; i < num_src; ++i) src_pointer[i] = static_cast<char *>(sourceArray[i]) + quark_offset;

  invertMultiSrcQuda(sln_pointer, src_pointer, &invertParam);

  host_free(sln_pointer);
  host_free(src_pointer);

  // Preserve deflation space
  if (invertParam.eig_param && qep.preserve_deflation) {
    preserved_deflation_space[local_parity] = qep.preserve_deflation_space;
    preserved_evals_mass[local_parity] = mass;
  }

  // The conventions for num_iters, final_residual, and final_fermilab_residual are taken from the
  // convention in `generic_ks/d_congrad5_fn_milc.c` (commit 414fb31). Here, a block solve
  // is emulated as a series of sequential solves. Each individual solve overrides the
  // final tolerance and iteration counts from the previous solve. Therefore, num_iters
  // as well as the tolerances come from the last solve.

  // invertParam.iter is the total number of iterations for the block solver, which is ~=
  // to the number of iterations the last rhs would take.
  *num_iters = invertParam.iter;

  // MILC only cares about a single residual, which happens to be the last one as described above.
  *final_residual = invertParam.true_res[num_src - 1];
  *final_fermilab_residual = invertParam.true_res_hq[num_src - 1];

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvertMsrcDeflatable

void qudaEigCGInvert(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                     double target_residual, double target_fermilab_residual, const void *const fatlink,
                     const void *const longlink,
                     void *source,   // array of source vectors -> overwritten on exit
                     void *solution, // temporary
                     QudaEigArgs_t eig_args,
                     const int rhs_idx,       // current rhs
                     const int last_rhs_flag, // is this the last rhs to solve
                     double *const final_residual, double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaInvert: requesting zero residual\n");

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;

  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  double& target_res = target_residual;
  double& target_res_hq = target_fermilab_residual;
  const double reliable_delta = 1e-1;

  setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, target_res, target_res_hq,
                  inv_args.max_iter, reliable_delta, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);

  QudaEigParam  df_param = newQudaEigParam();
  df_param.invert_param = &invertParam;

  invertParam.n_ev = eig_args.nev;
  invertParam.max_search_dim     = eig_args.max_search_dim;
  invertParam.deflation_grid     = eig_args.deflation_grid;
  invertParam.tol_restart        = eig_args.tol_restart;
  invertParam.eigcg_max_restarts = eig_args.eigcg_max_restarts;
  invertParam.max_restart_num    = eig_args.max_restart_num;
  invertParam.inc_tol            = eig_args.inc_tol;
  invertParam.eigenval_tol       = eig_args.eigenval_tol;
  invertParam.rhs_idx            = rhs_idx;

  if ((inv_args.solver_type != QUDA_INC_EIGCG_INVERTER) && (inv_args.solver_type != QUDA_EIGCG_INVERTER))
    errorQuda("Incorrect inverter type.\n");
  invertParam.inv_type = inv_args.solver_type;

  if (inv_args.solver_type == QUDA_INC_EIGCG_INVERTER) invertParam.inv_type_precondition = QUDA_INVALID_INVERTER;

  setDeflationParam(eig_args.prec_ritz, eig_args.location_ritz, eig_args.mem_type_ritz, eig_args.deflation_ext_lib, eig_args.vec_infile, eig_args.vec_outfile, &df_param);

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) invalidateGaugeQuda();

  if ((invalidate_quda_gauge || !create_quda_gauge) && (rhs_idx == 0)) { // do this for the first RHS
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;

  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;

  if(rhs_idx == 0) df_preconditioner = newDeflationQuda(&df_param);

  invertParam.deflation_op = df_preconditioner;

  invertQuda(static_cast<char *>(solution) + quark_offset, static_cast<char *>(source) + quark_offset, &invertParam);

  if (last_rhs_flag) destroyDeflationQuda(df_preconditioner);

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res[0];
  *final_fermilab_residual = invertParam.true_res_hq[0];

  if (!create_quda_gauge && last_rhs_flag) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaEigCGInvert

void qudaContractFT(int external_precision, QudaContractArgs_t *cont_args, void *const quark1, void *const quark2,
                    double *corr)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  ColorSpinorParam csParam;
  { // set ColorSpinorParam block
    csParam.nColor = 3;
    csParam.nSpin = 1; // Support only staggered color fields for now
    for (int dir = 0; dir < 4; ++dir) csParam.x[dir] = localDim[dir];
    csParam.x[4] = 1;
    csParam.setPrecision(host_precision);
    csParam.pad = 0;
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless for staggered, but required by the code.
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.pc_type = QUDA_4D_PC; // must be set
  }

  int const n_mom = cont_args->n_mom;
  int *const mom_modes = cont_args->mom_modes;
  const QudaFFTSymmType *const fft_type = cont_args->fft_type;
  int const *source_position = cont_args->source_position;

  QudaContractType cType = QUDA_CONTRACT_TYPE_STAGGERED_FT_T;
  int const src_colors = 1;
  // Only one pair of color fields and one result, so only one element in the arrays
  void *prop_array_flavor_1[1] = {quark1};
  void *prop_array_flavor_2[1] = {quark2};
  void *result[1] = {corr};

  contractFTQuda(prop_array_flavor_1, prop_array_flavor_2, result, cType, &csParam, src_colors, localDim,
                 source_position, n_mom, mom_modes, fft_type);

  qudamilc_called<false>(__func__, verbosity);
} // qudaContractFT

void *qudaMultigridCreate(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                          const void *const fatlink, const void *const longlink, const char *const mg_param_file)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // Flip the sign of the mass to fix a consistency issue between MILC, QUDA full
  // parity dslash operator
  mass = -mass;

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = QUDA_SINGLE_PRECISION;

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);

  // Set some other smart defaults
  fat_param.type = QUDA_ASQTAD_FAT_LINKS;
  fat_param.cuda_prec_refinement_sloppy = fat_param.cuda_prec_sloppy;
  fat_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  long_param.type = QUDA_ASQTAD_LONG_LINKS;
  long_param.cuda_prec_refinement_sloppy = long_param.cuda_prec_sloppy;
  long_param.reconstruct_refinement_sloppy = long_param.reconstruct_sloppy;

  // Prepare a multigrid pack
  milcMultigridPack *mg_pack = new milcMultigridPack;

  // Set parameters incl. loading from the parameter file here.
  milcSetMultigridParam(mg_pack, host_precision, device_precision, device_precision_sloppy, mass, mg_param_file);

  fat_param.cuda_prec_precondition = mg_pack->preconditioner_precision;
  long_param.cuda_prec_precondition = mg_pack->preconditioner_precision;

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  // compounding hack: *num_iters == 1 is always true here
  // if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) invalidateGaugeQuda();
  invalidateGaugeQuda();

  if (invalidate_quda_gauge || !create_quda_gauge) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  mg_pack->mg_preconditioner = newMultigridQuda(&mg_pack->mg_param);
  mg_pack->last_mass = mass;

  invalidate_quda_mg = false;

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);

  return (void *)mg_pack;
}

void qudaInvertMG(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                  double target_residual, double target_fermilab_residual, const void *const fatlink,
                  const void *const longlink, void *mg_pack_ptr, int mg_rebuild_type, void *source, void *solution,
                  double *const final_residual, double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // Pass this along to qudaInvertMsrcMG as a single-source solve
  qudaInvertMsrcMG(external_precision, quda_precision, mass, inv_args, target_residual, target_fermilab_residual,
                   fatlink, longlink, mg_pack_ptr, mg_rebuild_type, &source, &solution, final_residual,
                   final_fermilab_residual, num_iters, 1);

  qudamilc_called<false>(__func__, verbosity);
}

void qudaInvertMsrcMG(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                      double target_residual, double target_fermilab_residual, const void *const fatlink,
                      const void *const longlink, void *mg_pack_ptr, int mg_rebuild_type, void **sourceArray,
                      void **solutionArray, double *const final_residual, double *const final_fermilab_residual,
                      int *num_iters, int num_src)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  // FIXME: Flip the sign of the mass to fix a consistency issue between
  // MILC, QUDA full parity dslash operator
  mass = -mass;

  milcMultigridPack *mg_pack = (milcMultigridPack *)(mg_pack_ptr);

  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaInvert: requesting zero residual\n");

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = QUDA_SINGLE_PRECISION; // required for MG

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, longlink, localDim, host_precision, device_precision, device_precision_sloppy,
                 inv_args.tadpole, inv_args.naik_epsilon);

  fat_param.cuda_prec_refinement_sloppy = fat_param.cuda_prec_sloppy;
  fat_param.cuda_prec_precondition = mg_pack->preconditioner_precision;
  fat_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  long_param.type = QUDA_ASQTAD_LONG_LINKS;
  long_param.cuda_prec_refinement_sloppy = long_param.cuda_prec_sloppy;
  long_param.cuda_prec_precondition = mg_pack->preconditioner_precision;
  long_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd; // ignored, just needed to set some defaults
  const double reliable_delta = 1e-4;

  setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, target_residual,
                  target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
                  QUDA_GCR_INVERTER, &invertParam);

  invertParam.inv_type = QUDA_GCR_INVERTER;
  invertParam.preconditioner = mg_pack->mg_preconditioner;
  invertParam.inv_type_precondition = QUDA_MG_INVERTER;
  invertParam.solution_type = QUDA_MAT_SOLUTION;
  invertParam.solve_type = QUDA_DIRECT_SOLVE;
  invertParam.verbosity_precondition = QUDA_VERBOSE;

  invertParam.cuda_prec_sloppy = QUDA_SINGLE_PRECISION; // req'd
  invertParam.cuda_prec_precondition = mg_pack->preconditioner_precision;
  invertParam.gcrNkrylov = 15;
  invertParam.pipeline = 16; // pipeline, get from file

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);

  // dirty hack to invalidate the cached gauge field without breaking interface compatability
  if (*num_iters == -1 || !canReuseResidentGauge(&invertParam)) {
    invalidateGaugeQuda();
    invalidate_quda_mg = true;
  }

  if (mass != mg_pack->last_mass) {
    mg_pack->mg_param.invert_param->mass = mass;
    mg_pack->last_mass = mass;
    invalidateGaugeQuda();
    invalidate_quda_mg = true;
  }

  if (invalidate_quda_gauge || !create_quda_gauge || invalidate_quda_mg) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;

    // FIXME: hack to reset gaugeFatPrecise (see interface_quda.cpp), etc.
    // Solution is to have a version of this that _only_
    // rebuilds the Dirac matrices, I believe.
    if (mg_rebuild_type == 1) {
      if (verbosity >= QUDA_VERBOSE) printfQuda("Performing a full MG solver update\n");
      mg_pack->mg_param.thin_update_only = QUDA_BOOLEAN_FALSE;
    } else {
      if (verbosity >= QUDA_VERBOSE) printfQuda("Performing a thin MG solver update\n");
      mg_pack->mg_param.thin_update_only = QUDA_BOOLEAN_TRUE;
    }
    updateMultigridQuda(mg_pack->mg_preconditioner, &mg_pack->mg_param);
    invalidate_quda_mg = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;

  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;

  // Perform the solve one batch at a time
  int batch_size = mg_pack->input_struct.block_solver_batch_size;
  if (batch_size == -1) batch_size = num_src;

  // Allocate space for the solution and source vectors
  void **sln_pointer = (void **)safe_malloc(batch_size * sizeof(void *));
  void **src_pointer = (void **)safe_malloc(batch_size * sizeof(void *));

  for (int b = 0; b < num_src; b += batch_size) {
    int batch_src = std::min(batch_size, num_src - b);
    invertParam.num_src = batch_src;

    for (int i = 0; i < batch_src; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[b + i]) + quark_offset;
    for (int i = 0; i < batch_src; ++i) src_pointer[i] = static_cast<char *>(sourceArray[b + i]) + quark_offset;

    // FIXME: due to sign convention woes passing in an initial
    // guess is currently broken. Needs a sign flip to fix.
    // MG is fast enough we won't worry...
    invertMultiSrcQuda(sln_pointer, src_pointer, &invertParam);
  }

  // FIXME: Flip sign on solution to correct for mass convention
  int cv_size = localDim[0] * localDim[1] * localDim[2] * localDim[3] * 3 * 2; // (dimension * Nc = 3 * cplx)
  for (int k = 0; k < num_src; ++k) {
    if (host_precision == QUDA_DOUBLE_PRECISION) {
      auto soln = (double *)(solutionArray[k]);
      for (long i = 0; i < cv_size; i++) { soln[i] = -soln[i]; }
    } else {
      auto soln = (float *)(solutionArray[k]);
      for (long i = 0; i < cv_size; i++) { soln[i] = -soln[i]; }
    }
  }

  host_free(sln_pointer);
  host_free(src_pointer);

  // The conventions for num_iters, final_residual, and final_fermilab_residual are taken from the
  // convention in `generic_ks/d_congrad5_fn_milc.c` (commit 414fb31). Here, a block solve
  // is emulated as a series of sequential solves. Each individual solve overrides the
  // final tolerance and iteration counts from the previous solve. Therefore, num_iters
  // as well as the tolerances come from the last solve.

  // invertParam.iter is the total number of iterations for the block solver, which is ~=
  // to the number of iterations the last rhs would take.
  *num_iters = invertParam.iter;

  // MILC only cares about a single residual, which happens to be the last one as described above.
  *final_residual = invertParam.true_res[num_src - 1];
  *final_fermilab_residual = invertParam.true_res_hq[num_src - 1];

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
}

void qudaMultigridDestroy(void *mg_pack_ptr)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  if (mg_pack_ptr != 0) {
    milcMultigridPack *mg_pack = (milcMultigridPack *)(mg_pack_ptr);
    destroyMultigridQuda(mg_pack->mg_preconditioner);
    delete mg_pack;
  }

  qudamilc_called<false>(__func__, verbosity);
}

static int clover_alloc = 0;

void* qudaCreateGaugeField(void* gauge, int geometry, int precision)
{
  qudamilc_called<true>(__func__);
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, qudaPrecision, (geometry == 1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS);
  qudamilc_called<false>(__func__);
  return createGaugeFieldQuda(gauge, geometry, &qudaGaugeParam);
}


void qudaSaveGaugeField(void* gauge, void* inGauge)
{
  qudamilc_called<true>(__func__);
  auto cudaGauge = reinterpret_cast<GaugeField *>(inGauge);
  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim, cudaGauge->Precision(), QUDA_GENERAL_LINKS);
  saveGaugeFieldQuda(gauge, inGauge, &qudaGaugeParam);
  qudamilc_called<false>(__func__);
}


void qudaDestroyGaugeField(void* gauge)
{
  qudamilc_called<true>(__func__);
  destroyGaugeFieldQuda(gauge);
  qudamilc_called<false>(__func__);
}


void setInvertParam(QudaInvertParam &invertParam, QudaInvertArgs_t &inv_args,
		    int external_precision, int quda_precision, double kappa, double reliable_delta);

void qudaCloverForce(void *mom, double dt, void **x, void **p, double *coeff, double kappa, double ck,
		     int nvec, double multiplicity, void *gauge, int precision, QudaInvertArgs_t inv_args)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam
    = newMILCGaugeParam(localDim, (precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, QUDA_GENERAL_LINKS);
  qudaGaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER; // refers to momentum gauge order

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, precision, precision, kappa, 0);
  invertParam.num_offset = nvec;
  for (int i=0; i<nvec; ++i) invertParam.offset[i] = 0.0; // not needed
  invertParam.clover_coeff = 0.0; // not needed

  // solution types
  invertParam.solution_type      = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam.solve_type         = QUDA_NORMOP_PC_SOLVE;
  invertParam.inv_type           = QUDA_CG_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;

  invertParam.verbosity = getVerbosity();
  invertParam.verbosity_precondition = QUDA_SILENT;
  invertParam.use_resident_solution = inv_args.use_resident_solution;

  computeCloverForceQuda(mom, dt, x, p, coeff, -kappa * kappa, ck, nvec, multiplicity, gauge, &qudaGaugeParam,
                         &invertParam);
  qudamilc_called<false>(__func__);
}

void setGaugeParams(QudaGaugeParam &qudaGaugeParam, const int dim[4], QudaInvertArgs_t &inv_args,
                    int external_precision, int quda_precision)
{

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;

  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  for (int dir = 0; dir < 4; ++dir) qudaGaugeParam.X[dir] = dim[dir];

  qudaGaugeParam.anisotropy = 1.0;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;
  qudaGaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;

  if(trivial_phase){
    qudaGaugeParam.t_boundary = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
    qudaGaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  }else{
    qudaGaugeParam.t_boundary = QUDA_PERIODIC_T;
    qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
    qudaGaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }

  qudaGaugeParam.cpu_prec = host_precision;
  qudaGaugeParam.cuda_prec = device_precision;
  qudaGaugeParam.cuda_prec_sloppy = device_precision_sloppy;
  qudaGaugeParam.cuda_prec_precondition = device_precision_sloppy;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
}

void setInvertParam(QudaInvertParam &invertParam, QudaInvertArgs_t &inv_args,
		    int external_precision, int quda_precision, double kappa, double reliable_delta) {

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;
  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  static const QudaVerbosity verbosity = getVerbosity();

  invertParam.dslash_type                   = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa                         = kappa;
  invertParam.dagger                        = QUDA_DAG_NO;
  invertParam.mass_normalization            = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov                    = 30;
  invertParam.reliable_delta                = reliable_delta;
  invertParam.maxiter                       = inv_args.max_iter;

  invertParam.cuda_prec_precondition        = device_precision_sloppy;
  invertParam.verbosity_precondition        = verbosity;
  invertParam.verbosity        = verbosity;
  invertParam.cpu_prec                      = host_precision;
  invertParam.cuda_prec                     = device_precision;
  invertParam.cuda_prec_sloppy              = device_precision_sloppy;
  invertParam.gamma_basis                   = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order                   = QUDA_DIRAC_ORDER;
  invertParam.clover_cpu_prec               = host_precision;
  invertParam.clover_cuda_prec              = device_precision;
  invertParam.clover_cuda_prec_sloppy       = device_precision_sloppy;
  invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
  invertParam.clover_order                  = QUDA_PACKED_CLOVER_ORDER;

  invertParam.compute_action = 0;
}


void qudaLoadGaugeField(int external_precision,
    int quda_precision,
    QudaInvertArgs_t inv_args,
    const void* milc_link) {
  qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam = newQudaGaugeParam();
  setGaugeParams(qudaGaugeParam, localDim, inv_args, external_precision, quda_precision);

  loadGaugeQuda(const_cast<void *>(milc_link), &qudaGaugeParam);
  qudamilc_called<false>(__func__);
} // qudaLoadGaugeField


void qudaFreeGaugeField() {
    qudamilc_called<true>(__func__);
  freeGaugeQuda();
    qudamilc_called<false>(__func__);
} // qudaFreeGaugeField

void qudaFreeTwoLink()
{
  qudamilc_called<true>(__func__);
  freeGaugeTwoLinkQuda();
  qudamilc_called<false>(__func__);
} // qudaFreeTwoLink

void qudaLoadCloverField(int external_precision, int quda_precision, QudaInvertArgs_t inv_args, void *milc_clover,
                         void *milc_clover_inv, QudaSolutionType solution_type, QudaSolveType solve_type, QudaInverterType inverter,
                         double clover_coeff, int compute_trlog, double *trlog)
{
  qudamilc_called<true>(__func__);
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, 0.0, 0.0);
  invertParam.solution_type = solution_type;
  invertParam.solve_type = solve_type;
  invertParam.inv_type = inverter;
  invertParam.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  invertParam.compute_clover_trlog = compute_trlog;
  invertParam.clover_coeff = clover_coeff;

  // Hacks to mollify checkInvertParams which is called from
  // loadCloverQuda. These "required" parameters are irrelevant here.
  // Better procedure: invertParam should be defined in
  // qudaCloverInvert and qudaEigCGCloverInvert and passed here
  // instead of redefining a partial version here
  invertParam.tol = 0.;
  invertParam.tol_hq = 0.;
  invertParam.residual_type = static_cast<QudaResidualType_s>(0);

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (clover_alloc == 0) {
      loadCloverQuda(milc_clover, milc_clover_inv, &invertParam);
      clover_alloc = 1;
    } else {
      errorQuda("Clover term already allocated");
    }
  }

  if (compute_trlog) {
    trlog[0] = invertParam.trlogA[0];
    trlog[1] = invertParam.trlogA[1];
  }
  qudamilc_called<false>(__func__);
} // qudaLoadCoverField

void qudaFreeCloverField() {
  qudamilc_called<true>(__func__);
  if (clover_alloc==1) {
    freeCloverQuda();
    clover_alloc = 0;
  } else {
    errorQuda("Trying to free non-allocated clover term");
  }
  qudamilc_called<false>(__func__);
} // qudaFreeCloverField


void qudaCloverInvert(int external_precision,
    int quda_precision,
    double kappa,
    double clover_coeff,
    QudaInvertArgs_t inv_args,
    double target_residual,
    double target_fermilab_residual,
    const void* link,
    void* clover, // could be stored in Milc format
    void* cloverInverse,
    void* source,
    void* solution,
    double* const final_residual,
    double* const final_fermilab_residual,
    int* num_iters)
{
  qudamilc_called<true>(__func__);
  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaCloverInvert: requesting zero residual\n");

  if (link) qudaLoadGaugeField(external_precision, quda_precision, inv_args, link);

  if (clover || cloverInverse) {
    qudaLoadCloverField(external_precision, quda_precision, inv_args, clover, cloverInverse, QUDA_MAT_SOLUTION,
                        QUDA_DIRECT_PC_SOLVE, QUDA_BICGSTAB_INVERTER, clover_coeff, 0, 0);
  }

  double reliable_delta = 1e-1;

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual != 0) ? static_cast<QudaResidualType_s> ( invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : invertParam.residual_type;
  invertParam.residual_type = (target_fermilab_residual != 0) ? static_cast<QudaResidualType_s> (invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : invertParam.residual_type;

  invertParam.tol =  target_residual;
  invertParam.tol_hq = target_fermilab_residual;
  invertParam.heavy_quark_check = (invertParam.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 1 : 0);
  invertParam.clover_coeff = clover_coeff;

  // solution types
  invertParam.solution_type      = QUDA_MAT_SOLUTION;
  invertParam.inv_type           = inv_args.solver_type == QUDA_CG_INVERTER ? QUDA_CG_INVERTER : QUDA_BICGSTAB_INVERTER;
  invertParam.solve_type         = invertParam.inv_type == QUDA_CG_INVERTER ? QUDA_NORMOP_PC_SOLVE : QUDA_DIRECT_PC_SOLVE;
  invertParam.matpc_type         = QUDA_MATPC_ODD_ODD;

  invertQuda(solution, source, &invertParam);

  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res[0];
  *final_fermilab_residual = invertParam.true_res_hq[0];

  if (clover || cloverInverse) qudaFreeCloverField();
  if (link) qudaFreeGaugeField();
  qudamilc_called<false>(__func__);
} // qudaCloverInvert

void qudaEigCGCloverInvert(int external_precision, int quda_precision, double kappa, double clover_coeff,
                           QudaInvertArgs_t inv_args, double target_residual, double target_fermilab_residual,
                           const void *link,
                           void *clover, // could be stored in Milc format
                           void *cloverInverse,
                           void *source,   // array of source vectors -> overwritten on exit!
                           void *solution, // temporary
                           QudaEigArgs_t eig_args,
                           const int rhs_idx,       // current rhs
                           const int last_rhs_flag, // is this the last rhs to solve?
                           double *const final_residual, double *const final_fermilab_residual, int *num_iters)
{
  qudamilc_called<true>(__func__);
  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaCloverInvert: requesting zero residual\n");

  if (link && (rhs_idx == 0)) qudaLoadGaugeField(external_precision, quda_precision, inv_args, link);

  if ( (clover || cloverInverse) && (rhs_idx == 0)) {
    qudaLoadCloverField(external_precision, quda_precision, inv_args, clover, cloverInverse, QUDA_MAT_SOLUTION,
                        QUDA_DIRECT_PC_SOLVE, QUDA_INC_EIGCG_INVERTER, clover_coeff, 0, 0);
  }

  double reliable_delta = 1e-1;

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual != 0) ? static_cast<QudaResidualType_s> ( invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : invertParam.residual_type;
  invertParam.residual_type = (target_fermilab_residual != 0) ? static_cast<QudaResidualType_s> (invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : invertParam.residual_type;

  invertParam.tol =  target_residual;
  invertParam.tol_hq = target_fermilab_residual;
  invertParam.heavy_quark_check = (invertParam.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 1 : 0);
  invertParam.clover_coeff = clover_coeff;

  // solution types
  invertParam.solution_type      = QUDA_MAT_SOLUTION;
  invertParam.matpc_type         = QUDA_MATPC_ODD_ODD;

//!
  QudaEigParam  df_param = newQudaEigParam();
  df_param.invert_param = &invertParam;

  invertParam.solve_type = QUDA_NORMOP_PC_SOLVE;
  invertParam.n_ev = eig_args.nev;
  invertParam.max_search_dim     = eig_args.max_search_dim;
  invertParam.deflation_grid     = eig_args.deflation_grid;
  invertParam.cuda_prec_ritz     = eig_args.prec_ritz;
  invertParam.tol_restart        = eig_args.tol_restart;
  invertParam.eigcg_max_restarts = eig_args.eigcg_max_restarts;
  invertParam.max_restart_num    = eig_args.max_restart_num;
  invertParam.inc_tol            = eig_args.inc_tol;
  invertParam.eigenval_tol       = eig_args.eigenval_tol;
  invertParam.rhs_idx            = rhs_idx;


  if((inv_args.solver_type != QUDA_INC_EIGCG_INVERTER) && (inv_args.solver_type != QUDA_EIGCG_INVERTER)) errorQuda("Incorrect inverter type.\n");
  invertParam.inv_type = inv_args.solver_type;

  if(inv_args.solver_type == QUDA_INC_EIGCG_INVERTER) invertParam.inv_type_precondition = QUDA_INVALID_INVERTER;

  setDeflationParam(eig_args.prec_ritz, eig_args.location_ritz, eig_args.mem_type_ritz, eig_args.deflation_ext_lib, eig_args.vec_infile, eig_args.vec_outfile, &df_param);

  if(rhs_idx == 0)  df_preconditioner = newDeflationQuda(&df_param);
  invertParam.deflation_op = df_preconditioner;

  invertQuda(solution, source, &invertParam);

  if (last_rhs_flag) destroyDeflationQuda(df_preconditioner);

  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res[0];
  *final_fermilab_residual = invertParam.true_res_hq[0];

  if ( (clover || cloverInverse) && last_rhs_flag) qudaFreeCloverField();
  if (link && last_rhs_flag) qudaFreeGaugeField();
  qudamilc_called<false>(__func__);
} // qudaEigCGCloverInvert

void qudaCloverMultishiftInvert(int external_precision, int quda_precision, int num_offsets, double *const offset,
                                double kappa, double clover_coeff, QudaInvertArgs_t inv_args,
                                const double *target_residual_offset, void *source, void **solutionArray,
                                double *const final_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  for (int i = 0; i < num_offsets; ++i) {
    if (target_residual_offset[i] == 0) errorQuda("qudaCloverMultishiftInvert: target residual cannot be zero\n");
  }

  // if doing a pure double-precision multi-shift solve don't use reliable updates
  const bool use_mixed_precision = (((quda_precision==2) && inv_args.mixed_precision) ||
                                     ((quda_precision==1) && (inv_args.mixed_precision==2)) ) ? true : false;
  double reliable_delta = (use_mixed_precision) ? 1e-2 : 0.0;
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  invertParam.num_offset = num_offsets;
  for(int i=0; i<num_offsets; ++i){
    invertParam.offset[i] = offset[i];
    invertParam.tol_offset[i] = target_residual_offset[i];
  }
  invertParam.tol = target_residual_offset[0];
  invertParam.clover_coeff = clover_coeff;

  // solution types
  invertParam.solution_type      = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam.solve_type         = QUDA_NORMOP_PC_SOLVE;
  invertParam.inv_type           = QUDA_CG_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;

  invertParam.verbosity = verbosity;
  invertParam.verbosity_precondition = QUDA_SILENT;

  invertParam.make_resident_solution = inv_args.make_resident_solution;
  invertParam.compute_true_res = 0;

  if (num_offsets==1 && offset[0] == 0) {
    // set the solver
    char *quda_solver = getenv("QUDA_MILC_CLOVER_SOLVER");

    // default is chronological CG
    if (!quda_solver || strcmp(quda_solver,"CHRONO_CG_SOLVER")==0) {
      // use CG with chronological forecasting
      invertParam.chrono_use_resident = 1;
      invertParam.chrono_make_resident = 1;
      invertParam.chrono_max_dim = 10;
    } else if (strcmp(quda_solver,"BICGSTAB_SOLVER")==0){
      // use two-step BiCGStab
      invertParam.inv_type = QUDA_BICGSTAB_INVERTER;
      invertParam.solve_type = QUDA_DIRECT_PC_SOLVE;
    } else if (strcmp(quda_solver,"CG_SOLVER")==0){
      // regular CG
      invertParam.chrono_use_resident = 0;
      invertParam.chrono_make_resident = 0;
    }

    invertQuda(solutionArray[0], source, &invertParam);
    *final_residual = invertParam.true_res[0];
  } else {
    invertMultiShiftQuda(solutionArray, source, &invertParam);
    for (int i=0; i<num_offsets; ++i) final_residual[i] = invertParam.true_res_offset[i];
  }

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;

  qudamilc_called<false>(__func__, verbosity);
} // qudaCloverMultishiftInvert

void qudaGaugeFixingOVR(int precision, unsigned int gauge_dir, int Nsteps, int verbose_interval, double relax_boost,
                        double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, QudaMILCSiteArg_t *arg)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim,
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_SU3_LINKS);
  void *gauge = arg->site ? arg->site : arg->link;

  qudaGaugeParam.gauge_offset = arg->link_offset;
  qudaGaugeParam.mom_offset = arg->mom_offset;
  qudaGaugeParam.site_size = arg->size;
  qudaGaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;

  computeGaugeFixingOVRQuda(gauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval,
                            stopWtheta, &qudaGaugeParam);

  qudamilc_called<false>(__func__, verbosity);
}

void qudaGaugeFixingFFT( int precision,
    unsigned int gauge_dir,
    int Nsteps,
    int verbose_interval,
    double alpha,
    unsigned int autotune,
    double tolerance,
    unsigned int stopWtheta,
    void* milc_sitelink
    )
{
  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim,
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);
  qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  //qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_12;

  computeGaugeFixingFFTQuda(milc_sitelink, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta,
                            &qudaGaugeParam);
}

void qudaTwoLinkGaussianSmear( int external_precision, int quda_precision, void * h_gauge, void * source, QudaTwoLinkQuarkSmearArgs_t qsmear_args )
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  QudaPrecision cpu_prec = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision cuda_prec = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = cuda_prec;

  // inverter setup ---------------------
  QudaInvertParam invertParam = newQudaInvertParam();

  double mass = 0.0;
  QudaParity parity = QUDA_EVEN_PARITY; // need to fix

  // dummies
  double target_residual = 1e-14;
  double target_residual_hq = 1e-14;
  int maxiter = 1;
  double reliable_delta = 0.1;
  QudaInverterType inverter = QUDA_CG_INVERTER;
  
  setInvertParams( cpu_prec, cuda_prec, cuda_prec_sloppy, mass, target_residual, target_residual_hq, maxiter, reliable_delta, parity, verbosity, inverter, &invertParam );
  
  invertParam.laplace3D = qsmear_args.laplaceDim;
  //---------------------------- inverter setup

  // gauge setup ---------------------------
  QudaGaugeParam gaugeParam = newQudaGaugeParam();

  int * dim = localDim;
  
  // dummies
  double tadpole = 0;
  double naik_epsilon = 0;
  
  setGaugeParams( gaugeParam, gaugeParam, nullptr, dim, cpu_prec, cuda_prec, cuda_prec_sloppy, tadpole, naik_epsilon );

  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO; // need to fix
  gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  
  //--------------------------- gauge setup

  // Load gauge field
  if (qsmear_args.compute_2link == 0)
    gaugeParam.use_resident_gauge = 1;
  else
    loadGaugeQuda(const_cast<void *>(h_gauge), &gaugeParam);

  // quark smearing parameters
  QudaQuarkSmearParam qsmearParam;
  qsmearParam.inv_param = &invertParam;
  qsmearParam.n_steps = qsmear_args.n_steps;
  qsmearParam.width = qsmear_args.width;
  qsmearParam.compute_2link = qsmear_args.compute_2link;
  qsmearParam.delete_2link = qsmear_args.delete_2link;
  qsmearParam.t0 = qsmear_args.t0;
  
  // run gaussian smearing
  performTwoLinkGaussianSmearNStep( source, &qsmearParam );

  qudamilc_called<false>(__func__, verbosity);  
} //qudaTwoLinkGaussianSmear
