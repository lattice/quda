#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <quda.h>
#include <quda_milc_interface.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <string.h>
#include <unitarization_links.h>
#include <ks_improved_force.h>
#include <dslash_quda.h>
#include <invert_quda.h>

#include <vector>
#include <fstream>

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

void qudaFinalize()
{
  qudamilc_called<true>(__func__);
  endQuda();
  qudamilc_called<false>(__func__);
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
  gParam.ga_pad        = 0;
  gParam.site_ga_pad   = 0;
  gParam.mom_ga_pad    = 0;
  gParam.llfat_ga_pad  = 0;
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

  qudaGaugeParam.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = qudaGaugeParam.X[1] * qudaGaugeParam.X[2] * qudaGaugeParam.X[3] / 2;
  int y_face_size = qudaGaugeParam.X[0] * qudaGaugeParam.X[2] * qudaGaugeParam.X[3] / 2;
  int z_face_size = qudaGaugeParam.X[0] * qudaGaugeParam.X[1] * qudaGaugeParam.X[3] / 2;
  int t_face_size = qudaGaugeParam.X[0] * qudaGaugeParam.X[1] * qudaGaugeParam.X[2] / 2;
  int pad_size = x_face_size > y_face_size ? x_face_size : y_face_size;
  pad_size = pad_size > z_face_size ? pad_size : z_face_size;
  pad_size = pad_size > t_face_size ? pad_size : t_face_size;
  qudaGaugeParam.ga_pad = pad_size;
#endif

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

static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
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
  fat_param.ga_pad = getLinkPadding(dim);

  if (longlink != nullptr) {
    // improved staggered parameters
    fat_param.type = QUDA_ASQTAD_FAT_LINKS;

    // now set the long link parameters needed
    long_param = fat_param;
    long_param.tadpole_coeff = tadpole;
    long_param.scale = -(1.0 + naik_epsilon) / (24.0 * long_param.tadpole_coeff * long_param.tadpole_coeff);
    long_param.type = QUDA_THREE_LINKS;
    long_param.ga_pad = 3*fat_param.ga_pad;
    getReconstruct(long_param.reconstruct, long_param.reconstruct_sloppy);
    long_param.reconstruct_precondition = long_param.reconstruct_sloppy;
  } else {
    // naive staggered parameters
    fat_param.type = QUDA_SU3_LINKS;
    fat_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  }

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

void qudaInvert(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                double target_residual, double target_fermilab_residual, const void *const fatlink,
                const void *const longlink, void *source, void *solution, double *const final_residual,
                double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

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

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  const double reliable_delta = 1e-1;

  setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, target_residual,
                  target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
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

  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;

  invertQuda(static_cast<char *>(solution) + quark_offset, static_cast<char *>(source) + quark_offset, &invertParam);

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvert


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

  dslashQuda(static_cast<char*>(dst) + dst_offset*host_precision,
	     static_cast<char*>(src) + src_offset*host_precision,
	     &invertParam, local_parity);

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaDslash

void qudaInvertMsrc(int external_precision, int quda_precision, double mass, QudaInvertArgs_t inv_args,
                    double target_residual, double target_fermilab_residual, const void *const fatlink,
                    const void *const longlink, void **sourceArray, void **solutionArray, double *const final_residual,
                    double *const final_fermilab_residual, int *num_iters, int num_src)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  if (target_fermilab_residual == 0 && target_residual == 0) errorQuda("qudaInvert: requesting zero residual\n");

  // static const QudaVerbosity verbosity = getVerbosity();
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
  const double reliable_delta = 1e-1;

  setInvertParams(host_precision, device_precision, device_precision_sloppy, mass, target_residual,
                  target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
                  QUDA_CG_INVERTER, &invertParam);
  invertParam.num_src = num_src;

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

  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;
  void **sln_pointer = (void **)safe_malloc(num_src * sizeof(void *));
  void **src_pointer = (void **)safe_malloc(num_src * sizeof(void *));

  for (int i = 0; i < num_src; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[i]) + quark_offset;
  for (int i = 0; i < num_src; ++i) src_pointer[i] = static_cast<char *>(sourceArray[i]) + quark_offset;

  invertMultiSrcQuda(sln_pointer, src_pointer, &invertParam, nullptr, nullptr);

  host_free(sln_pointer);
  host_free(src_pointer);

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  if (!create_quda_gauge) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaInvert

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
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  if (!create_quda_gauge && last_rhs_flag) invalidateGaugeQuda();

  qudamilc_called<false>(__func__, verbosity);
} // qudaEigCGInvert

// Structure used to handle loading from input file
struct mgInputStruct {

  int mg_levels;
  bool verify_results;
  QudaPrecision preconditioner_precision; // precision for near-nulls, coarse links
  QudaTransferType
    optimized_kd; // use the optimized KD operator (true), naive coarsened operator (false), or optimized dropped links (drop)
  bool setup_use_mma[QUDA_MAX_MG_LEVEL];  // accelerate setup using MMA routines
  bool dslash_use_mma[QUDA_MAX_MG_LEVEL]; // accelerate dslash using MMA routines
  bool allow_truncation;     // allow dropping the long links for small (less than three) aggregate directions
  bool dagger_approximation; // use the dagger approximation to Xinv, which is X^dagger

  /**
   * Setup:
   * There is no near-null vector generation on the first and last (coarsest) level.
   * - The second level is the KD preconditioned staggered/HISQ operator, which is not a coarsening of the fine operator
   * - By definition there is no coarsening of the coarsest level
   * For this reason most of these variables are ignored on the first and last level.
   * We do reuse `nvec` on the coarsest level to specify the size of coarsest-level deflation basis
   * For reference: geo_block_size[0] does get defined internally (1 1 1 1 for optimized, 2 2 2 2 for coarse KD)
   */
  int nvec[QUDA_MAX_MG_LEVEL];                   // ignored on first level, reused for deflation size on last level
  QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL]; // ignored on first and last level
  double setup_tol[QUDA_MAX_MG_LEVEL];           // ignored on first and last level
  double setup_maxiter[QUDA_MAX_MG_LEVEL];       // ignored on first and last level
  int setup_ca_basis_size[QUDA_MAX_MG_LEVEL];    // ignored on first and last level
  char mg_vec_infile[QUDA_MAX_MG_LEVEL][256];    // ignored on first and last level
  char mg_vec_outfile[QUDA_MAX_MG_LEVEL][256];   // ignored on first and last level
  int geo_block_size[QUDA_MAX_MG_LEVEL][4]; // ignored on first and last level (values on first level are prescribed)

  /**
   * Solve:
   * The coarse solver parameters are ignored on the first level because it is
   * the outer solver, and as such we reuse values specified in MILC (tolerance, max iterations)
   * Some of these are fixed (for now) and will be exposed in the future:
   * - Solve type (for now fixed to full operator, will eventually expose Schur operator)
   * - Solver (for now fixed to GCR, will eventually expose PCG for Schur operator)
   * The smoother types are ignored for the coarsest level because, by definition, there is no
   * still coarser operator to smooth
   */
  QudaSolveType coarse_solve_type[QUDA_MAX_MG_LEVEL]; // ignored on first and second level
  QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL];  // ignored on first level
  double coarse_solver_tol[QUDA_MAX_MG_LEVEL];        // ignored on first level
  int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL];       // ignored on first level
  int coarse_solver_ca_basis_size[QUDA_MAX_MG_LEVEL]; // only used last level
  QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL];  // all but last level
  int nu_pre[QUDA_MAX_MG_LEVEL];                      // all but last level
  int nu_post[QUDA_MAX_MG_LEVEL];                     // all but last level

  // Misc
  QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL]; // all levels

  // Coarsest level deflation
  int deflate_n_ev;
  int deflate_n_kr;
  int deflate_max_restarts;
  double deflate_tol;
  bool deflate_use_poly_acc;
  double deflate_a_min; // ignored if no polynomial acceleration
  int deflate_poly_deg; // ignored if no polynomial acceleration

  void setArrayDefaults()
  {
    // set dummy values so all elements are initialized
    // some of these values get immediately overriden in the
    // constructor, in some cases with identical values:
    // this is to separate "initializing" with "best practices"
    for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
      nvec[i] = 24;
      setup_inv[i] = QUDA_CGNR_INVERTER;
      setup_tol[i] = 1e-5;
      setup_maxiter[i] = 500;
      setup_ca_basis_size[i] = 4;
      mg_vec_infile[i][0] = 0;
      mg_vec_outfile[i][0] = 0;
      for (int d = 0; d < 4; d++) { geo_block_size[i][d] = 2; }

      setup_use_mma[i] = true;
      dslash_use_mma[i] = true;

      coarse_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
      coarse_solver[i] = QUDA_GCR_INVERTER;
      coarse_solver_tol[i] = 0.25;
      coarse_solver_maxiter[i] = 16;
      coarse_solver_ca_basis_size[i] = 16;
      smoother_type[i] = QUDA_CA_GCR_INVERTER;
      nu_pre[i] = 0;
      nu_post[i] = 2;

      mg_verbosity[i] = QUDA_SUMMARIZE;
    }
  }

  // set defaults
  mgInputStruct() :
    mg_levels(4),
    verify_results(true),
    preconditioner_precision(QUDA_HALF_PRECISION),
    optimized_kd(QUDA_TRANSFER_OPTIMIZED_KD),
    allow_truncation(false),
    dagger_approximation(false),
    deflate_n_ev(66),
    deflate_n_kr(128),
    deflate_max_restarts(50),
    deflate_tol(1e-5),
    deflate_use_poly_acc(false),
    deflate_a_min(1e-2),
    deflate_poly_deg(50)
  {
    /* initialize internal arrays */
    setArrayDefaults();

    /* required or best-practice values for typical solves */
    nvec[0] = 3;              // must be this
    geo_block_size[0][0] = 1; // must be this...
    geo_block_size[0][1] = 1; // "
    geo_block_size[0][2] = 1; // "
    geo_block_size[0][3] = 1; // "

    nvec[1] = 64;
    geo_block_size[1][0] = 2;
    geo_block_size[1][1] = 2;
    geo_block_size[1][2] = 2;
    geo_block_size[1][3] = 2;

    nvec[2] = 96;
    geo_block_size[2][0] = 2;
    geo_block_size[2][1] = 2;
    geo_block_size[2][2] = 2;
    geo_block_size[2][3] = 2;

    /* Setup */

    /* level 0 -> 1 is K-D, no customization */

    /* Level 1 (pseudo-fine) to 2 (intermediate) */
    setup_inv[1] = QUDA_CGNR_INVERTER;
    setup_tol[1] = 1e-5;
    setup_maxiter[1] = 500;
    setup_ca_basis_size[1] = 4;
    mg_vec_infile[1][0] = 0;
    mg_vec_outfile[1][0] = 0;

    /* Level 2 (intermediate) to 3 (coarsest) */
    setup_inv[2] = QUDA_CGNR_INVERTER;
    setup_tol[2] = 1e-5;
    setup_maxiter[2] = 500;
    setup_ca_basis_size[2] = 4;
    mg_vec_infile[2][0] = 0;
    mg_vec_outfile[2][0] = 0;

    /* Solve info */

    /* Level 0 only needs a smoother */
    smoother_type[0] = QUDA_CA_GCR_INVERTER;
    nu_pre[0] = 0;
    nu_post[0] = 4;

    /* Level 1 */
    coarse_solver[1] = QUDA_GCR_INVERTER;
    coarse_solver_tol[1] = 5e-2;
    coarse_solver_maxiter[1] = 4;
    coarse_solver_ca_basis_size[1] = 4; // generally unused b/c not coarsest level
    smoother_type[1] = QUDA_CA_GCR_INVERTER;
    nu_pre[1] = 0;
    nu_post[1] = 2;

    /* Level 2 */
    coarse_solve_type[2] = QUDA_DIRECT_PC_SOLVE;
    coarse_solver[2] = QUDA_GCR_INVERTER;
    coarse_solver_tol[2] = 0.25;
    coarse_solver_maxiter[2] = 4;
    coarse_solver_ca_basis_size[2] = 4; // generally unused b/c not coarsest level
    smoother_type[2] = QUDA_CA_GCR_INVERTER;
    nu_pre[2] = 0;
    nu_post[2] = 2;

    /* Level 3 */
    coarse_solve_type[3] = QUDA_DIRECT_PC_SOLVE;
    coarse_solver[3] = QUDA_CA_GCR_INVERTER; // use CGNR for non-deflated... sometimes
    coarse_solver_tol[3] = 0.25;
    coarse_solver_maxiter[3] = 16; // use larger for non-deflated
    coarse_solver_ca_basis_size[3] = 16; // ignored for non-CA solvers

    /* Misc */
    mg_verbosity[0] = QUDA_SUMMARIZE;
    mg_verbosity[1] = QUDA_SUMMARIZE;
    mg_verbosity[2] = QUDA_SUMMARIZE;
    mg_verbosity[3] = QUDA_SUMMARIZE;

    /* Deflation */
    nvec[3] = 0; // 64; // do not deflate
    mg_vec_infile[3][0] = 0;
    mg_vec_outfile[3][0] = 0;
    deflate_n_ev = 66;
    deflate_n_kr = 128;
    deflate_tol = 1e-3;
    deflate_max_restarts = 50;
    deflate_use_poly_acc = false;
    deflate_a_min = 1e-2;
    deflate_poly_deg = 20;
  }

  QudaInverterType getQudaInverterType(const char *name)
  {
    if (strcmp(name, "gcr") == 0) {
      return QUDA_GCR_INVERTER;
    } else if (strcmp(name, "cgnr") == 0) {
      return QUDA_CGNR_INVERTER;
    } else if (strcmp(name, "cgne") == 0) {
      return QUDA_CGNE_INVERTER;
    } else if (strcmp(name, "ca-cgnr") == 0) {
      return QUDA_CA_CGNR_INVERTER;
    } else if (strcmp(name, "ca-cgne") == 0) {
      return QUDA_CA_CGNE_INVERTER;
    } else if (strcmp(name, "bicgstab") == 0) {
      return QUDA_BICGSTAB_INVERTER;
    } else if (strcmp(name, "bicgstab-l") == 0) {
      return QUDA_BICGSTABL_INVERTER;
    } else if (strcmp(name, "ca-gcr") == 0) {
      return QUDA_CA_GCR_INVERTER;
    } else {
      return QUDA_INVALID_INVERTER;
    }
  }

  QudaPrecision getQudaPrecision(const char *name)
  {
    if (strcmp(name, "single") == 0) {
      return QUDA_SINGLE_PRECISION;
    } else if (strcmp(name, "half") == 0) {
      return QUDA_HALF_PRECISION;
    } else {
      return QUDA_INVALID_PRECISION;
    }
  }

  QudaSolveType getQudaSolveType(const char *name)
  {
    if (strcmp(name, "direct") == 0) {
      return QUDA_DIRECT_SOLVE;
    } else if (strcmp(name, "direct-pc") == 0) {
      return QUDA_DIRECT_PC_SOLVE;
    } else {
      return QUDA_INVALID_SOLVE;
    }
  }

  QudaTransferType getQudaTransferType(const char *name)
  {
    if (strcmp(name, "true") == 0) {
      return QUDA_TRANSFER_OPTIMIZED_KD;
    } else if (strcmp(name, "false") == 0) {
      return QUDA_TRANSFER_COARSE_KD;
    } else if (strcmp(name, "drop") == 0) {
      return QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG;
    } else {
      return QUDA_TRANSFER_INVALID;
    }
  }

  QudaVerbosity getQudaVerbosity(const char *name)
  {
    if (strcmp(name, "silent") == 0) {
      return QUDA_SILENT;
    } else if (strcmp(name, "summarize") == 0 || strcmp(name, "false") == 0) {
      // false == summary is for backwards compatibility
      return QUDA_SUMMARIZE;
    } else if (strcmp(name, "verbose") == 0 || strcmp(name, "true") == 0) {
      // true == verbose is for backwards compatibility
      return QUDA_VERBOSE;
    } else if (strcmp(name, "debug") == 0) {
      return QUDA_DEBUG_VERBOSE;
    } else {
      return QUDA_INVALID_VERBOSITY;
    }
  }

  // parse out a line
  bool update(std::vector<std::string> &input_line)
  {

    int error_code = 0; // no error
                        // 1 = wrong number of arguments

    if (strcmp(input_line[0].c_str(), "mg_levels") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        mg_levels = atoi(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "verify_results") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        verify_results = input_line[1][0] == 't' ? true : false;
      }

    } else if (strcmp(input_line[0].c_str(), "preconditioner_precision") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        preconditioner_precision = getQudaPrecision(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "optimized_kd") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        optimized_kd = getQudaTransferType(input_line[1].c_str());
      }
    } else if (strcmp(input_line[0].c_str(), "use_mma") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        if (input_line[1][0] == 't') {
          for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
            setup_use_mma[i] = true;
            dslash_use_mma[i] = true;
          }
        } else {
          for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
            setup_use_mma[i] = false;
            dslash_use_mma[i] = false;
          }
        }
      }
    } else if (strcmp(input_line[0].c_str(), "allow_truncation") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        allow_truncation = input_line[1][0] == 't' ? true : false;
      }
    } else if (strcmp(input_line[0].c_str(), "dagger_approximation") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        dagger_approximation = input_line[1][0] == 't' ? true : false;
      }
    } else if (strcmp(input_line[0].c_str(), "mg_verbosity") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        mg_verbosity[atoi(input_line[1].c_str())] = getQudaVerbosity(input_line[2].c_str());
      }

    } else /* Begin Setup */
      if (strcmp(input_line[0].c_str(), "nvec") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        nvec[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "geo_block_size") == 0) {
      if (input_line.size() < 6) {
        error_code = 1;
      } else {
        for (int d = 0; d < 4; d++) geo_block_size[atoi(input_line[1].c_str())][d] = atoi(input_line[2 + d].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "setup_inv") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        setup_inv[atoi(input_line[1].c_str())] = getQudaInverterType(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "setup_tol") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        setup_tol[atoi(input_line[1].c_str())] = atof(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "setup_maxiter") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        setup_maxiter[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "setup_ca_basis_size") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        setup_ca_basis_size[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "mg_vec_infile") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        strcpy(mg_vec_infile[atoi(input_line[1].c_str())], input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "mg_vec_outfile") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        strcpy(mg_vec_outfile[atoi(input_line[1].c_str())], input_line[2].c_str());
      }

    } else /* Begin Solvers */
      if (strcmp(input_line[0].c_str(), "coarse_solve_type") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        coarse_solve_type[atoi(input_line[1].c_str())] = getQudaSolveType(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "coarse_solver") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        coarse_solver[atoi(input_line[1].c_str())] = getQudaInverterType(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "coarse_solver_tol") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        coarse_solver_tol[atoi(input_line[1].c_str())] = atof(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "coarse_solver_maxiter") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        coarse_solver_maxiter[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "coarse_solver_ca_basis_size") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        coarse_solver_ca_basis_size[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "smoother_type") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        smoother_type[atoi(input_line[1].c_str())] = getQudaInverterType(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "nu_pre") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        nu_pre[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "nu_post") == 0) {
      if (input_line.size() < 3) {
        error_code = 1;
      } else {
        nu_post[atoi(input_line[1].c_str())] = atoi(input_line[2].c_str());
      }

    } else /* Begin Deflation */
      if (strcmp(input_line[0].c_str(), "deflate_n_ev") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_n_ev = atoi(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_n_kr") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_n_kr = atoi(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_max_restarts") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_max_restarts = atoi(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_tol") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_tol = atof(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_use_poly_acc") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_use_poly_acc = input_line[1][0] == 't' ? true : false;
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_a_min") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_a_min = atof(input_line[1].c_str());
      }

    } else if (strcmp(input_line[0].c_str(), "deflate_poly_deg") == 0) {
      if (input_line.size() < 2) {
        error_code = 1;
      } else {
        deflate_poly_deg = atoi(input_line[1].c_str());
      }

    } else {
      printf("Invalid option %s\n", input_line[0].c_str());
      return false;
    }

    if (error_code == 1) {
      printf("Input option %s has an invalid number of arguments\n", input_line[0].c_str());
      return false;
    }

    return true;
  }
};

// Internal structure that maintains `QudaMultigridParam`,
// `QudaInvertParam`, `QudaEigParam`s, and the traditional
// void* returned by `newMultigridQuda`.
// last_mass tracks rebuilds based on changing the mass.
struct milcMultigridPack {
  QudaMultigridParam mg_param;
  QudaInvertParam mg_inv_param;
  QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
  QudaPrecision preconditioner_precision;
  double last_mass;
  void *mg_preconditioner;
};

// Parameters defining the eigensolver
void milcSetMultigridEigParam(QudaEigParam &mg_eig_param, mgInputStruct &input_struct, int level)
{
  mg_eig_param.eig_type = QUDA_EIG_TR_LANCZOS;  // mg_eig_type[level];
  mg_eig_param.spectrum = QUDA_SPECTRUM_SR_EIG; // mg_eig_spectrum[level];
  if ((mg_eig_param.eig_type == QUDA_EIG_TR_LANCZOS || mg_eig_param.eig_type)
      && !(mg_eig_param.spectrum == QUDA_SPECTRUM_LR_EIG || mg_eig_param.spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.n_ev = input_struct.deflate_n_ev; // mg_eig_n_ev[level];
  mg_eig_param.n_kr = input_struct.deflate_n_kr; // mg_eig_n_kr[level];
  mg_eig_param.n_conv = input_struct.nvec[level];
  mg_eig_param.n_ev_deflate = -1;  // deflate everything that converged
  mg_eig_param.batched_rotate = 0; // mg_eig_batched_rotate[level];
  mg_eig_param.require_convergence
    = QUDA_BOOLEAN_TRUE; // mg_eig_require_convergence[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.tol = input_struct.deflate_tol;                   // mg_eig_tol[level];
  mg_eig_param.check_interval = 10;                              // mg_eig_check_interval[level];
  mg_eig_param.max_restarts = input_struct.deflate_max_restarts; // mg_eig_max_restarts[level];

  mg_eig_param.compute_svd = QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_norm_op = QUDA_BOOLEAN_TRUE; // mg_eig_use_normop[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_dagger = QUDA_BOOLEAN_FALSE; // mg_eig_use_dagger[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.use_poly_acc = input_struct.deflate_use_poly_acc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.poly_deg = input_struct.deflate_poly_deg; // mg_eig_poly_deg[level];
  mg_eig_param.a_min = input_struct.deflate_a_min;       // mg_eig_amin[level];
  mg_eig_param.a_max = 0.0;                              // compute estimate // mg_eig_amax[level];

  // set file i/o parameters
  // Give empty strings, Multigrid will handle IO.
  strcpy(mg_eig_param.vec_infile, "");
  strcpy(mg_eig_param.vec_outfile, "");
  mg_eig_param.io_parity_inflate = QUDA_BOOLEAN_FALSE; // do not inflate coarse vectors
  mg_eig_param.save_prec = QUDA_SINGLE_PRECISION;      // cannot save in fixed point

  strcpy(mg_eig_param.QUDA_logfile, "" /*eig_QUDA_logfile*/);
}

void milcSetMultigridParam(milcMultigridPack *mg_pack, QudaPrecision host_precision, QudaPrecision device_precision,
                           QudaPrecision device_precision_sloppy, double mass, const char *const mg_param_file)
{
  static const QudaVerbosity verbosity = getVerbosity();
  QudaMultigridParam &mg_param = mg_pack->mg_param;

  // Create an input struct
  mgInputStruct input_struct;

  // Load input struct on rank 0
  if (comm_rank() == 0) {
    std::ifstream input_file(mg_param_file, std::ios_base::in);

    if (!input_file.is_open()) { errorQuda("MILC interface MG input file %s does not exist!", mg_param_file); }

    // enter parameter loop
    char buffer[1024];
    std::vector<std::string> elements;
    while (!input_file.eof()) {

      elements.clear();

      // get line
      input_file.getline(buffer, 1024);

      // split on spaces, tabs
      char *pch = strtok(buffer, " \t");
      while (pch != nullptr) {
        elements.emplace_back(std::string(pch));
        pch = strtok(nullptr, " \t");
      }

      // skip empty lines, comments
      if (elements.size() == 0 || elements[0][0] == '#') continue;

      // debug: print back out
      if (verbosity == QUDA_VERBOSE) {
        for (auto elem : elements) { printf("%s ", elem.c_str()); }
        printf("\n");
      }

      input_struct.update(elements);
    }
  }

  comm_barrier();
  comm_broadcast((void *)&input_struct, sizeof(mgInputStruct));

  auto mg_levels = input_struct.mg_levels;

  // Prepare eigenvector params
  for (int i = 0; i < mg_levels; i++) {
    mg_pack->mg_eig_param[i] = newQudaEigParam();
    milcSetMultigridEigParam(mg_pack->mg_eig_param[i], input_struct, i);
  }

  mg_pack->mg_inv_param = newQudaInvertParam();
  mg_pack->mg_param = newQudaMultigridParam();
  mg_pack->last_mass = mass;

  mg_pack->mg_param.invert_param = &mg_pack->mg_inv_param;
  for (int i = 0; i < mg_levels; i++) { mg_pack->mg_param.eig_param[i] = &mg_pack->mg_eig_param[i]; }

  QudaInvertParam &inv_param = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

  inv_param.Ls = 1;

  inv_param.cpu_prec = host_precision;
  inv_param.cuda_prec = device_precision;
  inv_param.cuda_prec_sloppy = device_precision_sloppy;
  inv_param.cuda_prec_precondition = input_struct.preconditioner_precision;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = QUDA_ASQTAD_DSLASH; // dslash_type;

  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4.0 + inv_param.mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  // this gets ignored
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN; // matpc_type;

  // req'd for staggered/hisq
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  auto solve_type = QUDA_DIRECT_SOLVE;
  inv_param.solve_type = solve_type;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels; // set from file

  // whether or not we allow dropping a long link when an aggregation size is smaller than 3
  mg_param.allow_truncation = input_struct.allow_truncation ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // whether or not we use the dagger approximation
  mg_param.staggered_kd_dagger_approximation = input_struct.dagger_approximation ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  for (int i = 0; i < mg_param.n_level; i++) {

    if (i == 0) {
      for (int j = 0; j < 4; j++) {
        mg_param.geo_block_size[i][j]
          = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? 2 : 1; // Kahler-Dirac blocking
      }
    } else {
      for (int j = 0; j < 4; j++) { mg_param.geo_block_size[i][j] = input_struct.geo_block_size[i][j]; }
    }

    mg_param.setup_use_mma[i] = input_struct.setup_use_mma[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.dslash_use_mma[i] = input_struct.dslash_use_mma[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // mg_param.use_eig_solver[i] = QUDA_BOOLEAN_FALSE; //mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    if (i == mg_param.n_level - 1 && input_struct.nvec[i] > 0) {
      mg_param.use_eig_solver[i] = QUDA_BOOLEAN_TRUE;
    } else {
      mg_param.use_eig_solver[i] = QUDA_BOOLEAN_FALSE;
    }

    mg_param.verbosity[i] = input_struct.mg_verbosity[i];
    mg_param.setup_inv_type[i] = input_struct.setup_inv[i];
    mg_param.num_setup_iter[i] = 1; // num_setup_iter[i];
    mg_param.setup_tol[i] = input_struct.setup_tol[i];
    mg_param.setup_maxiter[i] = input_struct.setup_maxiter[i];

    // Basis to use for CA solver setup --- heuristic for CA-GCR is empirical
    if (is_ca_solver(input_struct.setup_inv[i])) {
      if (input_struct.setup_inv[i] == QUDA_CA_GCR_INVERTER && input_struct.setup_ca_basis_size[i] <= 8)
        mg_param.setup_ca_basis[i] = QUDA_POWER_BASIS;
      else
        mg_param.setup_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // setup_ca_basis[i];
    } else {
      mg_param.setup_ca_basis[i] = QUDA_POWER_BASIS; // setup_ca_basis[i];
    }

    // Basis size for CA solver setup
    mg_param.setup_ca_basis_size[i] = input_struct.setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = 0.0;  // setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = -1.0; // use power iterations // setup_ca_lambda_max[i];

    mg_param.spin_block_size[i] = 1;
    // change this to refresh fields when mass or links change
    mg_param.setup_maxiter_refresh[i] = 0; // setup_maxiter_refresh[i];
    mg_param.n_vec[i]
      = (i == 0) ? ((input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? 24 : 3) : input_struct.nvec[i];
    mg_param.n_block_ortho[i] = 2; // n_block_ortho[i];                          // number of times to Gram-Schmidt
    mg_param.precision_null[i] = input_struct.preconditioner_precision; // precision to store the null-space basis
    mg_param.smoother_halo_precision[i]
      = input_struct.preconditioner_precision; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = input_struct.nu_pre[i];
    mg_param.nu_post[i] = input_struct.nu_post[i];
    mg_param.mu_factor[i] = 1.; // mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // top level: coarse vs optimized KD, otherwise standard aggregation.
    if (i == 0) {
      mg_param.transfer_type[i] = input_struct.optimized_kd;
    } else {
      mg_param.transfer_type[i] = QUDA_TRANSFER_AGGREGATE;
    }

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = input_struct.coarse_solver[i];
    mg_param.coarse_solver_tol[i] = input_struct.coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = input_struct.coarse_solver_maxiter[i];

    // Basis size for CA coarse solvers
    if (input_struct.coarse_solver_ca_basis_size[i] > input_struct.coarse_solver_maxiter[i]) {
      mg_param.coarse_solver_ca_basis_size[i] = input_struct.coarse_solver_maxiter[i];
    } else {
      mg_param.coarse_solver_ca_basis_size[i] = input_struct.coarse_solver_ca_basis_size[i];
    }

    // Basis to use for CA basis coarse solvers --- heuristic for CA-GCR is empirical
    if (is_ca_solver(input_struct.coarse_solver[i])) {
      if (input_struct.coarse_solver[i] == QUDA_CA_GCR_INVERTER && mg_param.coarse_solver_ca_basis_size[i] <= 8)
        mg_param.coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
      else
        mg_param.coarse_solver_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // coarse_solver_ca_basis[i];
    } else {
      mg_param.coarse_solver_ca_basis[i] = QUDA_POWER_BASIS; // coarse_solver_ca_basis[i];
    }

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = 0.0;  // coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = -1.0; // use power iterations // coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = input_struct.smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = 1e-10; // smoother_tol[i];

    // Basis to use for CA basis smoothers --- heuristic for CA-GCR is empirical
    if (is_ca_solver(input_struct.smoother_type[i])) {
      if (input_struct.smoother_type[i] == QUDA_CA_GCR_INVERTER && mg_param.nu_pre[i] <= 8 && mg_param.nu_post[i] <= 8)
        mg_param.smoother_solver_ca_basis[i] = QUDA_POWER_BASIS;
      else
        mg_param.smoother_solver_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // smoother_solver_ca_basis[i];
    } else {
      mg_param.smoother_solver_ca_basis[i] = QUDA_POWER_BASIS; // smoother_solver_ca_basis[i];
    }

    // Minimum and maximum eigenvalue for Chebyshev CA basis smoothers
    mg_param.smoother_solver_ca_lambda_min[i] = 0.0;  // smoother_solver_ca_lambda_min[i];
    mg_param.smoother_solver_ca_lambda_max[i] = -1.0; // smoother_solver_ca_lambda_max[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    // from test routines: // smoother_solve_type[i];
    switch (i) {
    case 0: mg_param.smoother_solve_type[0] = QUDA_DIRECT_SOLVE; break;
    case 1:
      mg_param.smoother_solve_type[1]
        = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? QUDA_DIRECT_PC_SOLVE : QUDA_DIRECT_SOLVE;
      break;
    default: mg_param.smoother_solve_type[i] = input_struct.coarse_solve_type[i]; break;
    }

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = QUDA_INVALID_SCHWARZ; // schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i]
      = QUDA_BOOLEAN_TRUE; // (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = 1; // schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment

      // Always this for now
      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }

    } else if (i == 1) {

      // Always this for now.
      mg_param.coarse_grid_solution_type[i]
        = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;
    } else {

      if (input_struct.coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (input_struct.coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("unexpected solve type = %d\n", input_struct.coarse_solve_type[i]);
      }
    }

    mg_param.omega[i] = 0.85; // ignored // omega; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;       //  solver_location[i];
    mg_param.setup_location[i] = QUDA_CUDA_FIELD_LOCATION; // setup_location[i];
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_FALSE;

  // coarsening the spin on the first restriction is undefined for staggered fields.
  mg_param.spin_block_size[0] = 0;
  if (input_struct.optimized_kd == QUDA_TRANSFER_OPTIMIZED_KD
      || input_struct.optimized_kd == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
    mg_param.spin_block_size[1] = 0;

  mg_param.setup_type = QUDA_NULL_VECTOR_SETUP;     // setup_type;
  mg_param.pre_orthonormalize = QUDA_BOOLEAN_FALSE; // pre_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.post_orthonormalize = QUDA_BOOLEAN_TRUE; // post_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.compute_null_vector
    = QUDA_COMPUTE_NULL_VECTOR_YES; // generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = QUDA_BOOLEAN_TRUE; // generate_all_levels ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.run_verify = input_struct.verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_low_mode_check = QUDA_BOOLEAN_FALSE;     // low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_oblique_proj_check = QUDA_BOOLEAN_FALSE; // oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.preserve_deflation = QUDA_BOOLEAN_TRUE;      // FIXME, controversial, should update if mass changes?

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], input_struct.mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], input_struct.mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
  }

  mg_param.coarse_guess = QUDA_BOOLEAN_FALSE; // mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-6; // reliable_delta;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;

  inv_param.verbosity = input_struct.mg_verbosity[0];

  // We need to pass this back to the fat/long links for the outer-most level.
  mg_pack->preconditioner_precision = input_struct.preconditioner_precision;
}

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

  // FIXME: due to sign convention woes passing in an initial
  // guess is currently broken. Needs a sign flip to fix.
  // MG is fast enough we won't worry...

  invertQuda(static_cast<char *>(solution) + quark_offset, static_cast<char *>(source) + quark_offset, &invertParam);

  // FIXME: Flip sign on solution to correct for mass convention
  int cv_size = localDim[0] * localDim[1] * localDim[2] * localDim[3] * 3 * 2; // (dimension * Nc = 3 * cplx)
  if (host_precision == QUDA_DOUBLE_PRECISION) {
    auto soln = (double *)(solution);
    for (long i = 0; i < cv_size; i++) { soln[i] = -soln[i]; }
  } else {
    auto soln = (float *)(solution);
    for (long i = 0; i < cv_size; i++) { soln[i] = -soln[i]; }
  }

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

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
  cudaGaugeField* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);
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
  qudaGaugeParam.ga_pad = getLinkPadding(dim);
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
  freeGaugeSmearedQuda();
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
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

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
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

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
    *final_residual = invertParam.true_res;
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

  double timeinfo[3];
  computeGaugeFixingOVRQuda(gauge, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval,
                            stopWtheta, &qudaGaugeParam, timeinfo);

  printfQuda("Time H2D: %lf\n", timeinfo[0]);
  printfQuda("Time to Compute: %lf\n", timeinfo[1]);
  printfQuda("Time D2H: %lf\n", timeinfo[2]);
  printfQuda("Time all: %lf\n", timeinfo[0]+timeinfo[1]+timeinfo[2]);

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


  double timeinfo[3];
  computeGaugeFixingFFTQuda(milc_sitelink, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta, \
    &qudaGaugeParam, timeinfo);

  printfQuda("Time H2D: %lf\n", timeinfo[0]);
  printfQuda("Time to Compute: %lf\n", timeinfo[1]);
  printfQuda("Time D2H: %lf\n", timeinfo[2]);
  printfQuda("Time all: %lf\n", timeinfo[0]+timeinfo[1]+timeinfo[2]);
}
