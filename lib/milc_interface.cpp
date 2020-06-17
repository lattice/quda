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

#define MAX(a,b) ((a)>(b)?(a):(b))

#ifdef BUILD_MILC_INTERFACE

// code for NVTX taken from Jiri Kraus' blog post:
// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#ifdef INTERFACE_NVTX

#if QUDA_NVTX_VERSION == 3
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
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
static int gridDim[4];
static int localDim[4];

static bool invalidate_quda_gauge = true;
static bool create_quda_gauge = false;

static bool invalidate_quda_mom = true;

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
  if (initialized) return;
  setVerbosityQuda(input.verbosity, "", stdout);
  qudamilc_called<true>(__func__);
  qudaSetLayout(input.layout);
  initialized = true;
  qudamilc_called<false>(__func__);
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
  for(int dir=0; dir<4; ++dir)  gridDim[dir] = input.machsize[dir];
#ifdef QMP_COMMS
  initCommsGridQuda(4, gridDim, nullptr, nullptr);
#else
  initCommsGridQuda(4, gridDim, rankFromCoords, (void *)(gridDim));
#endif
  static int device = -1;
#else
  for(int dir=0; dir<4; ++dir)  gridDim[dir] = 1;
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

  if(initialized) return;
  qudamilc_called<true>(__func__);

#if defined(GPU_HISQ_FORCE) || defined(GPU_UNITARIZE)
  const bool reunit_allow_svd = (params.reunit_allow_svd) ? true : false;
  const bool reunit_svd_only  = (params.reunit_svd_only) ? true : false;
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
#endif

#ifdef GPU_HISQ_FORCE
  quda::fermion_force::setUnitarizeForceConstants(unitarize_eps,
      params.force_filter,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);
#endif

#ifdef GPU_UNITARIZE
  setUnitarizeLinksConstants(unitarize_eps,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);
#endif // UNITARIZE_GPU

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
  qudamilc_called<false>(__func__);
}

void qudaLoadKSLink(int prec, QudaFatLinkArgs_t fatlink_args,
    const double act_path_coeff[6], void* inlink, void* fatlink, void* longlink)
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



void qudaLoadUnitarizedLink(int prec, QudaFatLinkArgs_t fatlink_args,
			    const double act_path_coeff[6], void* inlink, void* fatlink, void* ulink)
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
  qudamilc_called<false>(__func__);
  return;
}


void qudaAsqtadForce(int prec, const double act_path_coeff[6],
                     const void* const one_link_src[4], const void* const naik_src[4],
                     const void* const link, void* const milc_momentum)
{
  errorQuda("This interface has been removed and is no longer supported");
}



void qudaComputeOprod(int prec, int num_terms, int num_naik_terms, double** coeff, double scale,
                      void** quark_field, void* oprod[3])
{
  errorQuda("This interface has been removed and is no longer supported");
}


void qudaUpdateU(int prec, double eps, QudaMILCSiteArg_t *arg)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);
  void *gauge = arg->site ? arg->site : arg->link;
  void *mom = arg->site ? arg->site : arg->mom;

  gaugeParam.gauge_offset = arg->link_offset;
  gaugeParam.mom_offset = arg->mom_offset;
  gaugeParam.site_size = arg->size;
  gaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;

  if (!invalidate_quda_mom) {
    gaugeParam.use_resident_mom = true;
    gaugeParam.make_resident_mom = true;
  } else {
    gaugeParam.use_resident_mom = false;
    gaugeParam.make_resident_mom = false;
  }

  updateGaugeFieldQuda(gauge, mom, eps, 0, 0, &gaugeParam);
  qudamilc_called<false>(__func__);
  return;
}

void qudaRephase(int prec, void *gauge, int flag, double i_mu)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
						QUDA_GENERAL_LINKS);

  gaugeParam.staggered_phase_applied = 1-flag;
  gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gaugeParam.i_mu = i_mu;
  gaugeParam.t_boundary    = QUDA_ANTI_PERIODIC_T;

  staggeredPhaseQuda(gauge, &gaugeParam);
  qudamilc_called<false>(__func__);
  return;
}

void qudaUnitarizeSU3(int prec, double tol, QudaMILCSiteArg_t *arg)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
						QUDA_GENERAL_LINKS);

  void *gauge = arg->site ? arg->site : arg->link;
  gaugeParam.gauge_offset = arg->link_offset;
  gaugeParam.site_size = arg->size;
  gaugeParam.gauge_order = arg->site ? QUDA_MILC_SITE_GAUGE_ORDER : QUDA_MILC_GAUGE_ORDER;

  projectSU3Quda(gauge, tol, &gaugeParam);
  qudamilc_called<false>(__func__);
  return;
}

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


void qudaGaugeForce( int precision,
		     int num_loop_types,
		     double milc_loop_coeff[3],
		     double eb3,
		     QudaMILCSiteArg_t *arg)
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


static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
}

// set the params for the single mass solver
static void setInvertParams(const int dim[4], QudaPrecision cpu_prec, QudaPrecision cuda_prec,
                            QudaPrecision cuda_prec_sloppy, double mass, double target_residual,
                            double target_residual_hq, int maxiter, double reliable_delta, QudaParity parity,
                            QudaVerbosity verbosity, QudaInverterType inverter, QudaInvertParam *invertParam)
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

  invertParam->solution_type = QUDA_MATPC_SOLUTION;
  invertParam->solve_type = QUDA_DIRECT_PC_SOLVE;
  invertParam->preserve_source = QUDA_PRESERVE_SOURCE_YES;
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
  invertParam->sp_pad = 0;
  invertParam->use_init_guess = QUDA_USE_INIT_GUESS_YES;

  // for the preconditioner
  invertParam->inv_type_precondition = QUDA_CG_INVERTER;
  invertParam->tol_precondition = 1e-1;
  invertParam->maxiter_precondition = 2;
  invertParam->verbosity_precondition = QUDA_SILENT;

  invertParam->compute_action = 0;
}


// Set params for the multi-mass solver.
static void setInvertParams(const int dim[4], QudaPrecision cpu_prec, QudaPrecision cuda_prec,
                            QudaPrecision cuda_prec_sloppy, int num_offset, const double offset[],
                            const double target_residual_offset[], const double target_residual_hq_offset[],
                            int maxiter, double reliable_delta, QudaParity parity, QudaVerbosity verbosity,
                            QudaInverterType inverter, QudaInvertParam *invertParam)
{
  const double null_mass = -1;

  setInvertParams(dim, cpu_prec, cuda_prec, cuda_prec_sloppy, null_mass, target_residual_offset[0],
                  target_residual_hq_offset[0], maxiter, reliable_delta, parity, verbosity, inverter, invertParam);

  invertParam->num_offset = num_offset;
  for (int i = 0; i < num_offset; ++i) {
    invertParam->offset[i] = offset[i];
    invertParam->tol_offset[i] = target_residual_offset[i];
    invertParam->tol_hq_offset[i] = target_residual_hq_offset[i];
  }
}

static void getReconstruct(QudaReconstructType &reconstruct, QudaReconstructType &reconstruct_sloppy)
{
  {
    char *reconstruct_env = getenv("QUDA_MILC_HISQ_RECONSTRUCT");
    if (!reconstruct_env || strcmp(reconstruct_env, "18") == 0) {
      reconstruct = QUDA_RECONSTRUCT_NO;
    } else if (strcmp(reconstruct_env, "13") == 0) {
      reconstruct = QUDA_RECONSTRUCT_13;
    } else if (strcmp(reconstruct_env, "9") == 0) {
      reconstruct = QUDA_RECONSTRUCT_9;
    } else {
      errorQuda("QUDA_MILC_HISQ_RECONSTRUCT=%s not supported", reconstruct_env);
    }
  }

  {
    char *reconstruct_sloppy_env = getenv("QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY");
    if (!reconstruct_sloppy_env) { // if env is not set, default to using outer reconstruct type
      reconstruct_sloppy = reconstruct;
    } else if (strcmp(reconstruct_sloppy_env, "18") == 0) {
      reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    } else if (strcmp(reconstruct_sloppy_env, "13") == 0) {
      reconstruct_sloppy = QUDA_RECONSTRUCT_13;
    } else if (strcmp(reconstruct_sloppy_env, "9") == 0) {
      reconstruct_sloppy = QUDA_RECONSTRUCT_9;
    } else {
      errorQuda("QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=%s not supported", reconstruct_sloppy_env);
    }
  }
}

static void setGaugeParams(QudaGaugeParam &fat_param, QudaGaugeParam &long_param, const void *const fatlink,
                           const void *const longlink, const int dim[4], QudaPrecision cpu_prec,
                           QudaPrecision cuda_prec, QudaPrecision cuda_prec_sloppy, double tadpole, double naik_epsilon)
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

  df_param->nk       = df_param->invert_param->nev;
  df_param->np       = df_param->invert_param->nev*df_param->invert_param->deflation_grid;

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
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const bool use_mixed_precision = (((quda_precision==2) && inv_args.mixed_precision) ||
                                     ((quda_precision==1) && (inv_args.mixed_precision==2)) ) ? true : false;
  QudaPrecision device_precision_sloppy;
  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  QudaGaugeParam fat_param = newQudaGaugeParam();
  QudaGaugeParam long_param = newQudaGaugeParam();
  setGaugeParams(fat_param, long_param, fatlink, longlink, localDim, host_precision, device_precision,
                 device_precision_sloppy, inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  const double reliable_delta = (use_mixed_precision ? 1e-1 : 0.0);
  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, num_offsets, offset,
                  target_residual, target_fermilab_residual, inv_args.max_iter, reliable_delta, local_parity, verbosity,
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
  if (*num_iters == -1) invalidateGaugeQuda();

  // set the solver
  if (invalidate_quda_gauge || !create_quda_gauge) {
    loadGaugeQuda(const_cast<void *>(fatlink), &fat_param);
    if (longlink != nullptr) loadGaugeQuda(const_cast<void *>(longlink), &long_param);
    invalidate_quda_gauge = false;
  }

  if (longlink == nullptr) invertParam.dslash_type = QUDA_STAGGERED_DSLASH;

  void** sln_pointer = (void**)malloc(num_offsets*sizeof(void*));
  int quark_offset = getColorVectorOffset(local_parity, false, localDim) * host_precision;
  void* src_pointer = static_cast<char*>(source) + quark_offset;

  for (int i = 0; i < num_offsets; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[i]) + quark_offset;

  invertMultiShiftQuda(sln_pointer, src_pointer, &invertParam);
  free(sln_pointer);

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
  setGaugeParams(fat_param, long_param, fatlink, longlink, localDim, host_precision, device_precision,
                 device_precision_sloppy, inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  const double reliable_delta = 1e-1;

  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, mass, target_residual,
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
  setGaugeParams(fat_param, long_param, fatlink, longlink, localDim, host_precision, device_precision,
                 device_precision_sloppy, inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  QudaParity other_parity = local_parity == QUDA_EVEN_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;

  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, 0.0, 0, 0, 0, 0.0, local_parity,
                  verbosity, QUDA_CG_INVERTER, &invertParam);

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
  setGaugeParams(fat_param, long_param, fatlink, longlink, localDim, host_precision, device_precision,
                 device_precision_sloppy, inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  const double reliable_delta = 1e-1;

  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, mass, target_residual,
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
  void** sln_pointer = (void**)malloc(num_src*sizeof(void*));
  void** src_pointer = (void**)malloc(num_src*sizeof(void*));

  for (int i = 0; i < num_src; ++i) sln_pointer[i] = static_cast<char *>(solutionArray[i]) + quark_offset;
  for (int i = 0; i < num_src; ++i) src_pointer[i] = static_cast<char *>(sourceArray[i]) + quark_offset;

  invertMultiSrcQuda(sln_pointer, src_pointer, &invertParam);

  free(sln_pointer);
  free(src_pointer);

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
  setGaugeParams(fat_param, long_param, fatlink, longlink, localDim, host_precision, device_precision,
                 device_precision_sloppy, inv_args.tadpole, inv_args.naik_epsilon);

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  double& target_res = target_residual;
  double& target_res_hq = target_fermilab_residual;
  const double reliable_delta = 1e-1;

  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, mass, target_res, target_res_hq,
                  inv_args.max_iter, reliable_delta, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);

  QudaEigParam  df_param = newQudaEigParam();
  df_param.invert_param = &invertParam;

  invertParam.nev                = eig_args.nev;
  invertParam.max_search_dim     = eig_args.max_search_dim;
  invertParam.deflation_grid     = eig_args.deflation_grid;
  invertParam.cuda_prec_ritz     = eig_args.prec_ritz;
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


static int clover_alloc = 0;

void* qudaCreateGaugeField(void* gauge, int geometry, int precision)
{
  qudamilc_called<true>(__func__);
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, qudaPrecision,
      (geometry==1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS);
  qudamilc_called<false>(__func__);
  return createGaugeFieldQuda(gauge, geometry, &gaugeParam);
}


void qudaSaveGaugeField(void* gauge, void* inGauge)
{
  qudamilc_called<true>(__func__);
  cudaGaugeField* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, cudaGauge->Precision(), QUDA_GENERAL_LINKS);
  saveGaugeFieldQuda(gauge, inGauge, &gaugeParam);
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
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim,
						(precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
						QUDA_GENERAL_LINKS);
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER; // refers to momentume gauge order

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

  computeCloverForceQuda(mom, dt, x, p, coeff, -kappa*kappa, ck, nvec, multiplicity,
			 gauge, &gaugeParam, &invertParam);
  qudamilc_called<false>(__func__);
}


void setGaugeParams(QudaGaugeParam &gaugeParam, const int dim[4], QudaInvertArgs_t &inv_args,
                    int external_precision, int quda_precision) {

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;

  switch(inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = dim[dir];

  gaugeParam.anisotropy               = 1.0;
  gaugeParam.type                     = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order              = QUDA_MILC_GAUGE_ORDER;

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;

  if(trivial_phase){
    gaugeParam.t_boundary               = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_12;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_12;
  }else{
    gaugeParam.t_boundary               = QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_NO;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_NO;
  }

  gaugeParam.cpu_prec                 = host_precision;
  gaugeParam.cuda_prec                = device_precision;
  gaugeParam.cuda_prec_sloppy         = device_precision_sloppy;
  gaugeParam.cuda_prec_precondition   = device_precision_sloppy;
  gaugeParam.gauge_fix                = QUDA_GAUGE_FIXED_NO;
  gaugeParam.ga_pad                   = getLinkPadding(dim);
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
  invertParam.preserve_source               = QUDA_PRESERVE_SOURCE_NO;
  invertParam.gamma_basis                   = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order                   = QUDA_DIRAC_ORDER;
  invertParam.sp_pad                        = 0;
  invertParam.cl_pad                        = 0;
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
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  setGaugeParams(gaugeParam, localDim,  inv_args, external_precision, quda_precision);

  loadGaugeQuda(const_cast<void*>(milc_link), &gaugeParam);
    qudamilc_called<false>(__func__);
} // qudaLoadGaugeField


void qudaFreeGaugeField() {
    qudamilc_called<true>(__func__);
  freeGaugeQuda();
    qudamilc_called<false>(__func__);
} // qudaFreeGaugeField

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
  invertParam.nev                = eig_args.nev;
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


void qudaCloverMultishiftInvert(int external_precision,
    int quda_precision,
    int num_offsets,
    double* const offset,
    double kappa,
    double clover_coeff,
    QudaInvertArgs_t inv_args,
    const double* target_residual_offset,
    const void* milc_link,
    void* milc_clover,
    void* milc_clover_inv,
    void* source,
    void** solutionArray,
    double* const final_residual,
    int* num_iters)
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
                        double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, void *milc_sitelink)
{
  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim,
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_SU3_LINKS);
  qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  //qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_12;

  double timeinfo[3];
  computeGaugeFixingOVRQuda(milc_sitelink, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta, \
    &qudaGaugeParam, timeinfo);

  printfQuda("Time H2D: %lf\n", timeinfo[0]);
  printfQuda("Time to Compute: %lf\n", timeinfo[1]);
  printfQuda("Time D2H: %lf\n", timeinfo[2]);
  printfQuda("Time all: %lf\n", timeinfo[0]+timeinfo[1]+timeinfo[2]);
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

#endif // BUILD_MILC_INTERFACE
