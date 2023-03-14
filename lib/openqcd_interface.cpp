#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>

#include <quda_openqcd_interface.h>
#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <invert_quda.h>
// #include "../../openQxD-devel/include/su3.h"
// #include "../../openQxD-devel/include/flags.h"
// #include "../../openQxD-devel/include/utils.h"
// #include "../../openQxD-devel/include/lattice.h"
// #include "../../openQxD-devel/include/global.h"

// #include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// code for NVTX taken from Jiri Kraus' blog post:
// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#ifdef INTERFACE_NVTX

#if QUDA_NVTX_VERSION == 3
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif

static const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
static const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                                                          \
  {                                                                                                                    \
    int color_id = cid;                                                                                                \
    color_id = color_id % num_colors;                                                                                  \
    nvtxEventAttributes_t eventAttrib = {0};                                                                           \
    eventAttrib.version = NVTX_VERSION;                                                                                \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                  \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                                                           \
    eventAttrib.color = colors[color_id];                                                                              \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                                 \
    eventAttrib.message.ascii = name;                                                                                  \
    nvtxRangePushEx(&eventAttrib);                                                                                     \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

static bool initialized = false;
static int commsGridDim[4];
static int localDim[4];

using namespace quda;

// #define QUDAMILC_VERBOSE 1
// template <bool start> void inline qudamilc_called(const char *func, QudaVerbosity verb)
// {
//   // add NVTX markup if enabled
//   if (start) {
//     PUSH_RANGE(func, 1);
//   } else {
//     POP_RANGE;
//   }

// #ifdef QUDAMILC_VERBOSE
//   if (verb >= QUDA_VERBOSE) {
//     if (start) {
//       printfQuda("QUDA_MILC_INTERFACE: %s (called) \n", func);
//     } else {
//       printfQuda("QUDA_MILC_INTERFACE: %s (return) \n", func);
//     }
//   }
// #endif
// }

// template <bool start> void inline qudamilc_called(const char *func) { qudamilc_called<start>(func, getVerbosity()); }


static int safe_mod(int x,int y)
{
   if (x>=0)
      return x%y;
   else
      return (y-(abs(x)%y))%y;
}


// fdata should point to 4 integers in order {NPROC0, NPROC1, NPROC2, NPROC3}
// coords is the 4D cartesian coordinate of a rank.
static int rankFromCoords(const int *coords, void *fdata) // TODO:
{ 
  int *NPROC = static_cast<int *>(fdata);
  // int *NPROC = BLK_NPROC + 4;
	
  int ib;
  int n0_OpenQxD;
  int n1_OpenQxD;
  int n2_OpenQxD;
  int n3_OpenQxD;
  // int NPROC0_OpenQxD;
  int NPROC1_OpenQxD;
  int NPROC2_OpenQxD;
  int NPROC3_OpenQxD;

  n0_OpenQxD=coords[3];
  n1_OpenQxD=coords[0];
  n2_OpenQxD=coords[1];
  n3_OpenQxD=coords[2];

  // NPROC0_OpenQxD=NPROC[3];
  NPROC1_OpenQxD=NPROC[0];
  NPROC2_OpenQxD=NPROC[1];
  NPROC3_OpenQxD=NPROC[2];

  
  ib=n0_OpenQxD;
  ib=ib*NPROC1_OpenQxD+n1_OpenQxD;
  ib=ib*NPROC2_OpenQxD+n2_OpenQxD;
  ib=ib*NPROC3_OpenQxD+n3_OpenQxD;
  printf("Coords are: %d,%d,%d,%d \n Rank is: %d \n\n",coords[0],coords[1],coords[2],coords[3],ib);
  return ib;
}

void openQCD_qudaSetLayout(openQCD_QudaLayout_t input)
{
  int local_dim[4];
  for (int dir = 0; dir < 4; ++dir) { local_dim[dir] = input.latsize[dir]; }
#ifdef MULTI_GPU
  for (int dir = 0; dir < 4; ++dir) { local_dim[dir] /= input.machsize[dir]; }
#endif
  for (int dir = 0; dir < 4; ++dir) {
    if (local_dim[dir] % 2 != 0) {
      printf("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }
  // TODO: do we need to track this here
  for (int dir = 0; dir < 4; ++dir) localDim[dir] = local_dim[dir];

#ifdef MULTI_GPU
  for (int dir = 0; dir < 4; ++dir) commsGridDim[dir] = input.machsize[dir];
// TODO: would we ever want to run with QMP COMMS?
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

void openQCD_qudaInit(openQCD_QudaInitArgs_t input)
{
  if (initialized) return;
  setVerbosityQuda(input.verbosity, "", stdout);
  // qudamilc_called<true>(__func__);
  openQCD_qudaSetLayout(input.layout);
  initialized = true;
  // qudamilc_called<false>(__func__);
  // geometry(); // Establish helper indexes from openQxD
}

void openQCD_qudaFinalize() { endQuda(); }

// not sure we want to use allocators, but in case we want to
#if 0
void *qudaAllocatePinned(size_t bytes) { return pool_pinned_malloc(bytes); }

void qudaFreePinned(void *ptr) { pool_pinned_free(ptr); }

void *qudaAllocateManaged(size_t bytes) { return managed_malloc(bytes); }

void qudaFreeManaged(void *ptr) { managed_free(ptr); }
#endif


static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
}

static QudaGaugeParam newOpenQCDGaugeParam(const int *dim, QudaPrecision prec)
{
  QudaGaugeParam gParam = newQudaGaugeParam();
  for (int dir = 0; dir < 4; ++dir) gParam.X[dir] = dim[dir];
  gParam.cuda_prec_sloppy = gParam.cpu_prec = gParam.cuda_prec = prec;
  gParam.type = QUDA_SU3_LINKS;

  gParam.reconstruct_sloppy = gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;
  gParam.t_boundary = QUDA_PERIODIC_T;
  gParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gParam.scale = 1.0;
  gParam.anisotropy = 1.0;
  gParam.tadpole_coeff = 1.0;
  gParam.scale = 0;
  gParam.ga_pad = getLinkPadding(dim);

  return gParam;
}

void openQCD_qudaPlaquette(int precision, double plaq[3], void *gauge)
{
  // qudamilc_called<true>(__func__);

  QudaGaugeParam qudaGaugeParam
    = newOpenQCDGaugeParam(localDim, (precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION); // FIXME:
  // reateGaugeParamForObservables(precision, arg, phase_in);

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obsParam.remove_staggered_phase = QUDA_BOOLEAN_FALSE; //
  // phase_in ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  // Let MILC apply its own Nc normalization
  plaq[0] = obsParam.plaquette[0];
  plaq[1] = obsParam.plaquette[1];
  plaq[2] = obsParam.plaquette[2];

  // qudamilc_called<false>(__func__);
  return;
}

// static int openQCD_index()
// {
//   // This function is the helper for ipt in QUDA


//   return ix;
// }

// int openQCD_ipt(int iy)
// {
//   // This function computes the ipt index from iy (lexicographical index)
//   int x0,x1,x2,x3;
//   int k,mu,ix,iy,iz,iw;
//   int bo[4],bs[4],ifc[8];



// }

// static int getLinkPadding(const int dim[4])
// {
//   int padding = MAX(dim[1] * dim[2] * dim[3] / 2, dim[0] * dim[2] * dim[3] / 2);
//   padding = MAX(padding, dim[0] * dim[1] * dim[3] / 2);
//   padding = MAX(padding, dim[0] * dim[1] * dim[2] / 2);
//   return padding;
// }

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




static void setColorSpinorParams(const int dim[4], QudaPrecision precision, ColorSpinorParam *param)
{
  param->nColor = 3;
  param->nSpin = 1; // TODO:
  param->nDim = 4; // TODO: check how to adapt this for openqxd

  for (int dir = 0; dir < 4; ++dir) param->x[dir] = dim[dir];
  param->x[0] /= 2;

  param->setPrecision(precision);
  param->pad = 0;
  param->siteSubset = QUDA_PARITY_SITE_SUBSET;   // TODO: check how to adapt this for openqxd
  param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;   // TODO: check how to adapt this for openqxd
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.  // TODO:
  param->create = QUDA_ZERO_FIELD_CREATE; // TODO: check how to adapt this for openqxd
}

#if 0
void openQCD_qudaInvert(int external_precision, int quda_precision, double mass, openQCD_QudaInvertArgs_t inv_args,
                double target_residual, double target_fermilab_residual, const void *const fatlink,
                const void *const longlink, void *source, void *solution, double *const final_residual,
                double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  // qudamilc_called<true>(__func__, verbosity);

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
  switch (inv_args.mixed_precision) {
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

void openQCD_qudaDslash(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args, const void *const fatlink,
                const void *const longlink, void *src, void *dst, int *num_iters)
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
#endif

// void* openQCD_qudaCreateGaugeField(void *gauge, int geometry, int precision)
// {
//   qudamilc_called<true>(__func__);
//   QudaPrecision qudaPrecision = (precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
//   QudaGaugeParam qudaGaugeParam
//     = newMILCGaugeParam(localDim, qudaPrecision, (geometry == 1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS);
//   qudamilc_called<false>(__func__);
//   return createGaugeFieldQuda(gauge, geometry, &qudaGaugeParam);
// }

// void qudaSaveGaugeField(void *gauge, void *inGauge)
// {
//   qudamilc_called<true>(__func__);
//   cudaGaugeField *cudaGauge = reinterpret_cast<cudaGaugeField *>(inGauge);
//   QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim, cudaGauge->Precision(), QUDA_GENERAL_LINKS);
//   saveGaugeFieldQuda(gauge, inGauge, &qudaGaugeParam);
//   qudamilc_called<false>(__func__);
// }

// void qudaDestroyGaugeField(void *gauge)
// {
//   qudamilc_called<true>(__func__);
//   destroyGaugeFieldQuda(gauge);
//   qudamilc_called<false>(__func__);
// }

// void setInvertParam(QudaInvertParam &invertParam, openQCD_QudaInvertArgs_t &inv_args, int external_precision,
//                     int quda_precision, double kappa, double reliable_delta);

void setGaugeParams(QudaGaugeParam &qudaGaugeParam, const int dim[4], openQCD_QudaInvertArgs_t &inv_args,
                    int external_precision, int quda_precision)
{

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;

  switch (inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  for (int dir = 0; dir < 4; ++dir) qudaGaugeParam.X[dir] = dim[dir];

  qudaGaugeParam.anisotropy = 1.0;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;
  qudaGaugeParam.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for (int dir = 0; dir < 3; ++dir) {
    if (inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if (inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;

  if (trivial_phase) {
    qudaGaugeParam.t_boundary = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
    qudaGaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  } else {
    qudaGaugeParam.t_boundary = QUDA_PERIODIC_T;
    qudaGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
    qudaGaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }

  qudaGaugeParam.cpu_prec = host_precision;
  qudaGaugeParam.cuda_prec = device_precision;
  qudaGaugeParam.cuda_prec_sloppy = device_precision_sloppy;
  qudaGaugeParam.cuda_prec_precondition = device_precision_sloppy;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  // qudaGaugeParam.ga_pad = getLinkPadding(dim);
}

void setInvertParam(QudaInvertParam &invertParam, openQCD_QudaInvertArgs_t &inv_args, int external_precision,
                    int quda_precision, double kappa, double reliable_delta)
{

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;
  switch (inv_args.mixed_precision) {
  case 2: device_precision_sloppy = QUDA_HALF_PRECISION; break;
  case 1: device_precision_sloppy = QUDA_SINGLE_PRECISION; break;
  default: device_precision_sloppy = device_precision;
  }

  static const QudaVerbosity verbosity = getVerbosity();

  invertParam.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa = kappa;
  invertParam.dagger = QUDA_DAG_NO;
  invertParam.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov = 30;
  invertParam.reliable_delta = reliable_delta;
  invertParam.maxiter = inv_args.max_iter;

  invertParam.cuda_prec_precondition = device_precision_sloppy;
  invertParam.verbosity_precondition = verbosity;
  invertParam.verbosity = verbosity;
  invertParam.cpu_prec = host_precision;
  invertParam.cuda_prec = device_precision;
  invertParam.cuda_prec_sloppy = device_precision_sloppy;
  invertParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order = QUDA_DIRAC_ORDER;
  invertParam.clover_cpu_prec = host_precision;
  invertParam.clover_cuda_prec = device_precision;
  invertParam.clover_cuda_prec_sloppy = device_precision_sloppy;
  invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
  invertParam.clover_order = QUDA_PACKED_CLOVER_ORDER;

  invertParam.compute_action = 0;
}

void openQCD_qudaLoadGaugeField(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args,
                                const void *milc_link)
{
  // qudamilc_called<true>(__func__);
  QudaGaugeParam qudaGaugeParam = newQudaGaugeParam();
  setGaugeParams(qudaGaugeParam, localDim, inv_args, external_precision, quda_precision);

  loadGaugeQuda(const_cast<void *>(milc_link), &qudaGaugeParam);
  // qudamilc_called<false>(__func__);
} // qudaLoadGaugeField

void openQCD_qudaFreeGaugeField()
{
  // qudamilc_called<true>(__func__);
  freeGaugeQuda();
  // qudamilc_called<false>(__func__);
} // qudaFreeGaugeField
