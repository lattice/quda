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

static openQCD_QudaInitArgs_t input;
static QudaInvertParam invertParam = newQudaInvertParam();
static openQCD_QudaState_t qudaState = {false, false, false};
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

/*******************************************
 * 
 * LAYOUT AND INIT
 * 
 *******************************************/

/**
 * @brief      Calculate the rank from coordinates.
 *
 * @param[in]  coords  coords is the 4D cartesian coordinate of a rank
 * @param[in]  fdata   should point to 4 integers in order {NPROC0, NPROC1,
 *                     NPROC2, NPROC3}
 *
 * @return     rank
 */
static int rankFromCoords(const int *coords, void *fdata) // TODO:
{
  int *NPROC = static_cast<int *>(fdata);
  int ib;

  ib = coords[3];
  ib = ib*NPROC[0] + coords[0];
  ib = ib*NPROC[1] + coords[1];
  ib = ib*NPROC[2] + coords[2];

  return ib;
}


/**
 * @brief      Set layout parameters.
 *
 * @param[in]  layout  The layout
 */
void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout)
{
  int local_dim[4];
  for (int dir = 0; dir < 4; ++dir) {
    local_dim[dir] = layout.latsize[dir];
  }

#ifdef MULTI_GPU
  for (int dir = 0; dir < 4; ++dir) {
    local_dim[dir] /= layout.machsize[dir];
  }
#endif
  for (int dir = 0; dir < 4; ++dir) {
    if (local_dim[dir] % 2 != 0) {
      printfQuda("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }
  // TODO: do we need to track this here
  for (int dir = 0; dir < 4; ++dir) {
    localDim[dir] = local_dim[dir];
  }

#ifdef MULTI_GPU
  for (int dir = 0; dir < 4; ++dir) {
    commsGridDim[dir] = layout.machsize[dir];
  }
// TODO: would we ever want to run with QMP COMMS?
#ifdef QMP_COMMS
  initCommsGridQuda(4, commsGridDim, nullptr, nullptr);
#else
  initCommsGridQuda(4, commsGridDim, rankFromCoords, (void *)(commsGridDim));
#endif

  static int device = -1;
#else
  static int device = layout.device;
#endif

  initQuda(device);
}


void openQCD_qudaInit(openQCD_QudaInitArgs_t in)
{
  if (qudaState.initialized) return;
  setVerbosityQuda(in.verbosity, "QUDA: ", in.logfile);
  openQCD_qudaSetLayout(in.layout);

  input = in;
  qudaState.initialized = true;
  // geometry_openQxD(); // TODO: in the future establish ipt and other helper indexes from openQxD
}

void openQCD_qudaFinalize() {
  endQuda();
}


static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1] * dim[2] * dim[3] / 2, dim[0] * dim[2] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[2] / 2);
  return padding;
}

/*******************************************
 * 
 * SETTINGS AND PARAMETERS
 * 
 *******************************************/

/**
 * @brief      OPENQCD GAUGE PARAMS
 *             
 * @param[in]  dim   dimensions
 * @param[in]  prec  precision
 *
 * @return     The quda gauge parameter.
 */
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


/* PARAMS FOR SPINOR FIELDS */
static void setColorSpinorParams(const int dim[4], QudaPrecision precision, ColorSpinorParam *param)
{
  param->nColor = 3;
  param->nSpin = 4; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  param->nDim = 4;  // TODO: check how to adapt this for openqxd

  for (int dir = 0; dir < 4; ++dir) param->x[dir] = dim[dir];
  // param->x[0] /= 2;  // for staggered sites only FIXME:?

  param->setPrecision(precision);
  param->pad = 0;
  param->siteSubset = QUDA_FULL_SITE_SUBSET; // FIXME: check how to adapt this for openqxd
  param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER; // FIXME: check how to adapt this for openqxd // EVEN-ODD is only about inner ordering in quda
  param->fieldOrder = QUDA_OPENQCD_FIELD_ORDER;       // FIXME:
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.  // // FIXME::
  param->create = QUDA_ZERO_FIELD_CREATE; // // FIXME:: check how to adapt this for openqxd ?? created -0 in weird places
}


/* PARAMS FOR DSLASH AND INVERSION */
static void setInvertParams(QudaPrecision cpu_prec, QudaPrecision cuda_prec, QudaPrecision cuda_prec_sloppy,
                            double mass, double target_residual, double target_residual_hq, int maxiter,
                            double reliable_delta, QudaParity parity, QudaVerbosity verbosity,
                            QudaInverterType inverter)
{
  invertParam.verbosity = verbosity;
  invertParam.mass = mass;
  invertParam.tol = target_residual;
  invertParam.tol_hq = target_residual_hq;

  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual != 0) ?
    static_cast<QudaResidualType_s>(invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    invertParam.residual_type;
  invertParam.residual_type = (target_residual_hq != 0) ?
    static_cast<QudaResidualType_s>(invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    invertParam.residual_type;

  invertParam.heavy_quark_check = (invertParam.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 1 : 0);
  if (invertParam.heavy_quark_check) {
    invertParam.max_hq_res_increase = 5;       // this caps the number of consecutive hq residual increases
    invertParam.max_hq_res_restart_total = 10; // this caps the number of hq restarts in case of solver stalling
  }

  invertParam.use_sloppy_partial_accumulator = 0;
  invertParam.num_offset = 0;

  invertParam.inv_type = inverter;
  invertParam.maxiter = maxiter;
  invertParam.reliable_delta = reliable_delta;

  invertParam.mass_normalization = QUDA_MASS_NORMALIZATION;
  invertParam.cpu_prec = cpu_prec;
  invertParam.cuda_prec = cuda_prec;
  invertParam.cuda_prec_sloppy = invertParam.heavy_quark_check ? cuda_prec : cuda_prec_sloppy;
  invertParam.cuda_prec_precondition = cuda_prec_sloppy;

  invertParam.gcrNkrylov = 10;

  invertParam.solution_type = QUDA_MATPC_SOLUTION;
  invertParam.solve_type = QUDA_DIRECT_PC_SOLVE;
  invertParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // not used, but required by the code.
  invertParam.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;

  invertParam.dslash_type = QUDA_WILSON_DSLASH; // FIXME: OR THIS; QUDA_ASQTAD_DSLASH;
  invertParam.Ls = 1;
  invertParam.gflops = 0.0;

  invertParam.input_location = QUDA_CPU_FIELD_LOCATION;
  invertParam.output_location = QUDA_CPU_FIELD_LOCATION;

  if (parity == QUDA_EVEN_PARITY) { // even parity
    invertParam.matpc_type = QUDA_MATPC_EVEN_EVEN;
  } else if (parity == QUDA_ODD_PARITY) {
    invertParam.matpc_type = QUDA_MATPC_ODD_ODD;
  } else {
    errorQuda("Invalid parity\n");
  }

  invertParam.dagger = QUDA_DAG_NO;
  invertParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;

  // for the preconditioner
  invertParam.inv_type_precondition = QUDA_CG_INVERTER;
  invertParam.tol_precondition = 1e-1;
  invertParam.maxiter_precondition = 2;
  invertParam.verbosity_precondition = QUDA_SILENT;

  invertParam.compute_action = 0;
}





/*******************************************
 * 
 * FUNCTIONS
 * 
 *******************************************/


double openQCD_qudaPlaquette(void)
{
  double plaq[3];

  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot calculate plaquette. Call openQCD_gaugeload() first.");
    return 0.0;
  }

  /*QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obsParam.remove_staggered_phase = QUDA_BOOLEAN_FALSE;
  gaugeObservablesQuda(&obsParam);

  // Note different Nc normalization!
  plaq[0] = obsParam.plaquette[0];
  plaq[1] = obsParam.plaquette[1];
  plaq[2] = obsParam.plaquette[2];*/

  plaqQuda(plaq);

/*  plaq[1] *= 3.0;
  plaq[2] *= 3.0;
  plaq[0] *= 3.0;*/

  // Note different Nc normalization wrt openQCD!
  return 3.0*plaq[0];
}


void openQCD_gaugeload(int precision, void *gauge)
{
  void *buffer;

  QudaGaugeParam qudaGaugeParam
    = newOpenQCDGaugeParam(localDim, (precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);

  buffer = malloc(4*input.volume*input.sizeof_su3_dble);
  input.reorder_gauge_openqcd_to_quda(gauge, buffer);
  loadGaugeQuda(buffer, &qudaGaugeParam);
  free(buffer);

  qudaState.gauge_loaded = true;

  return;
}


void openQCD_gaugesave(int precision, void *gauge)
{
  void *buffer;

  QudaGaugeParam qudaGaugeParam
    = newOpenQCDGaugeParam(localDim, (precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);

  buffer = malloc(4*input.volume*input.sizeof_su3_dble);
  saveGaugeQuda(buffer, &qudaGaugeParam);
  input.reorder_gauge_quda_to_openqcd(buffer, gauge);
  free(buffer);

  return;
}

void openQCD_qudaFreeGaugeField(void)
{
  freeGaugeQuda();
  qudaState.gauge_loaded = false;
  return;
}


/* 
 * SPINOR FIELDS 
 */


/* 
 * SPINOR AND GAUGE FIELDS 
 */


void openQCD_qudaSetDslashOptions(double kappa, double mu)
{
  static const QudaVerbosity verbosity = getVerbosity();

  invertParam.input_location = QUDA_CPU_FIELD_LOCATION;
  invertParam.output_location = QUDA_CPU_FIELD_LOCATION;
  invertParam.dslash_type = QUDA_WILSON_DSLASH;
  invertParam.inv_type = QUDA_CG_INVERTER; /* just set some */
  invertParam.kappa = kappa;
  invertParam.dagger = QUDA_DAG_NO;
  invertParam.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  invertParam.Ls = 1;       /**< Extent of the 5th dimension (for domain wall) */
  invertParam.mu = mu;    /**< Twisted mass parameter */
  /*invertParam.tm_rho = ?;*/  /**< Hasenbusch mass shift applied like twisted mass to diagonal (but not inverse) */
  /*invertParam.epsilon = ?;*/ /**< Twisted mass parameter */
  /*invertParam.twist_flavor = ??;*/  /**< Twisted mass flavor */
  invertParam.laplace3D = -1; /**< omit this direction from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D) */

  invertParam.cpu_prec = QUDA_DOUBLE_PRECISION;                /**< The precision used by the input fermion fields */
  invertParam.cuda_prec = QUDA_DOUBLE_PRECISION;               /**< The precision used by the QUDA solver */

  invertParam.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;       /**< The order of the input and output fermion fields */
  invertParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;    /**< Gamma basis of the input and output host fields */

  invertParam.verbosity = verbosity;               /**< The verbosity setting to use in the solver */
  invertParam.compute_action = 0;

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, invertParam.cpu_prec, &csParam);

  qudaState.dslash_setup = true;
}


void openQCD_qudaDslash(void *src, void *dst)
{
  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot apply Dslash. Call openQCD_gaugeload() first.");
    return;
  }

  if (!qudaState.dslash_setup) {
    errorQuda("Dslash parameters are not set, cannot apply Dslash!");
    return;
  }

  dslashQuda(static_cast<char *>(dst), static_cast<char *>(src), &invertParam, QUDA_EVEN_PARITY);

  return;
}


void openQCD_colorspinorloadsave(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args, void *src,
                        void *dst, void *gauge)
{
  static const QudaVerbosity verbosity = getVerbosity();

  QudaGaugeParam qudaGaugeParam
    = newOpenQCDGaugeParam(localDim, (quda_precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);

  loadGaugeQuda(gauge, &qudaGaugeParam);

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = device_precision;

  QudaInvertParam invertParam = newQudaInvertParam();

  QudaParity local_parity = inv_args.evenodd;
  QudaParity other_parity = local_parity == QUDA_EVEN_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;

  /* For reference:
  setInvertParams(QudaPrecision cpu_prec, QudaPrecision cuda_prec, QudaPrecision cuda_prec_sloppy,
                            double mass, double target_residual, double target_residual_hq, int maxiter,
                            double reliable_delta, QudaParity parity, QudaVerbosity verbosity,
                            QudaInverterType inverter, QudaInvertParam *invertParam) */
  /*setInvertParams(host_precision, device_precision, device_precision_sloppy, 0.0, 0, 0, 0, 0.0, local_parity, verbosity,
                  QUDA_CG_INVERTER, &invertParam);*/

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);

  dslashQudaTest(static_cast<char *>(dst), static_cast<char *>(src), &invertParam, local_parity);

  return;
} // openQCD_colorspinorloadsave

#if 0
void openQCD_qudaInvert(int external_precision, int quda_precision, double mass, openQCD_QudaInvertArgs_t inv_args,
                        double target_residual, double target_fermilab_residual, const void *const fatlink,
                        const void *const longlink, void *source, void *solution, double *const final_residual,
                        double *const final_fermilab_residual, int *num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();

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

} // qudaInvert
#endif

// void* openQCD_qudaCreateGaugeField(void *gauge, int geometry, int precision)
// {
//   qudamilc_called<true>(__func__);
//   QudaPrecision qudaPrecision = (precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
//   QudaGaugeParam qudaGaugeParam
//     = newMILCGaugeParam(localDim, qudaPrecision, (geometry == 1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS); // TODO:
//     change MILC to openQCD
//   qudamilc_called<false>(__func__);
//   return createGaugeFieldQuda(gauge, geometry, &qudaGaugeParam);
// }

// void qudaSaveGaugeField(void *gauge, void *inGauge)
// {
//   qudamilc_called<true>(__func__);
//   cudaGaugeField *cudaGauge = reinterpret_cast<cudaGaugeField *>(inGauge);
//   QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim, cudaGauge->Precision(), QUDA_GENERAL_LINKS); // TODO:
//   change MILC to openQCD saveGaugeFieldQuda(gauge, inGauge, &qudaGaugeParam); qudamilc_called<false>(__func__);
// }

// void qudaDestroyGaugeField(void *gauge)
// {
//   qudamilc_called<true>(__func__);
//   destroyGaugeFieldQuda(gauge);
//   qudamilc_called<false>(__func__);
// }

// void setInvertParam(QudaInvertParam &invertParam, openQCD_QudaInvertArgs_t &inv_args, int external_precision,
//                     int quda_precision, double kappa, double reliable_delta);

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

  invertParam.dslash_type = QUDA_WILSON_DSLASH;
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
  invertParam.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;
  invertParam.clover_cpu_prec = host_precision;
  invertParam.clover_cuda_prec = device_precision;
  invertParam.clover_cuda_prec_sloppy = device_precision_sloppy;
  invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
  invertParam.clover_order = QUDA_PACKED_CLOVER_ORDER;

  invertParam.compute_action = 0;
}

// TODO: OpenQCDMultigridPack functions a la MILC (cf. milc_interface.cpp)
