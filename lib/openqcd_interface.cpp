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
static QudaInvertParam invertParam;
static openQCD_QudaState_t qudaState = {false, false, false, false};

using namespace quda;


/**
 * @brief      Returns the local lattice dimensions as lat_dim_t
 *
 * @return     The local dimensions.
 */
static lat_dim_t get_local_dims(int *fill = nullptr)
{
  lat_dim_t X;

  for (int i=0; i<4; i++) {
    if (fill) {
      fill[i] = input.layout.L[i];
    } else {
      X[i] = input.layout.L[i];
    }
  }

  return X;
}


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
  for (int dir = 0; dir < 4; ++dir) {
    if (layout.N[dir] % 2 != 0) {
      errorQuda("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

#ifdef MULTI_GPU
// TODO: would we ever want to run with QMP COMMS?
#ifdef QMP_COMMS
  initCommsGridQuda(4, layout.nproc, nullptr, nullptr);
#else
  initCommsGridQuda(4, layout.nproc, rankFromCoords, (void *)(layout.nproc));
#endif
  static int device = -1; // enable a default allocation of devices to processes 
#else
  static int device = layout.device;
#endif

  initQuda(device);
}


static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1] * dim[2] * dim[3] / 2, dim[0] * dim[2] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[2] / 2);
  return padding;
}


/**
 * @brief      Initialize invert param struct
 *             
 * @return     The quda invert parameter struct.
 */
static QudaInvertParam newOpenQCDParam(void)
{
  static const QudaVerbosity verbosity = getVerbosity();

  QudaInvertParam param = newQudaInvertParam();

  param.verbosity = verbosity;

  param.cpu_prec = QUDA_DOUBLE_PRECISION; /* The precision used by the input fermion fields */
  param.cuda_prec = QUDA_DOUBLE_PRECISION; /* The precision used by the QUDA solver */

  /**
   * The order of the input and output fermion fields. Imposes fieldOrder =
   * QUDA_OPENQCD_FIELD_ORDER in color_spinor_field.h and
   * QUDA_OPENQCD_FIELD_ORDER makes quda to instantiate OpenQCDDiracOrder.
   */
  param.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;

  /* Gamma basis of the input and output host fields */
  param.gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;

  return param;
}


/**
 * @brief      Initialize gauge param struct
 *
 * @param[in]  prec  precision
 *
 * @return     The quda gauge parameter struct.
 */
static QudaGaugeParam newOpenQCDGaugeParam(QudaPrecision prec)
{
  QudaGaugeParam param = newQudaGaugeParam();

  get_local_dims(param.X);
  param.cuda_prec_sloppy = param.cpu_prec = param.cuda_prec = prec;
  param.type = QUDA_SU3_LINKS;

  param.reconstruct_sloppy = param.reconstruct = QUDA_RECONSTRUCT_NO;

  /**
   * This make quda to instantiate OpenQCDOrder
   */
  param.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;

  /**
   * Seems to have no effect ...
   */
  param.t_boundary = QUDA_PERIODIC_T;

  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.scale = 1.0;
  param.anisotropy = 1.0; // 1.0 means not anisotropic
  //param.tadpole_coeff = 1.0;
  //param.scale = 0;
  param.ga_pad = getLinkPadding(param.X); /* Why this? */

  return param;
}


/**
 * @brief      Initialize clover param struct
 *
 * @param[in]  kappa   hopping parameter
 * @param[in]  su3csw  The su 3 csw
 *
 * @return     The quda gauge parameter struct.
 */
static QudaInvertParam newOpenQCDCloverParam(double kappa, double su3csw)
{
  QudaInvertParam param = newOpenQCDParam();

  param.clover_location = QUDA_CPU_FIELD_LOCATION;
  param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
  param.clover_order = QUDA_FLOAT8_CLOVER_ORDER; /*QUDA_OPENQCD_CLOVER_ORDER; */

  param.compute_clover = true;
  param.kappa = kappa;
  param.clover_csw = su3csw;
  param.clover_coeff = 0.0;
  param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;

  return param;
}


void openQCD_qudaInit(openQCD_QudaInitArgs_t in)
{
  if (qudaState.initialized) return;
  input = in;

  setVerbosityQuda(input.verbosity, "QUDA: ", input.logfile);
  openQCD_qudaSetLayout(input.layout);
  qudaState.initialized = true;
}

void openQCD_qudaFinalize() {
  qudaState.initialized = false;
  endQuda();
}


double openQCD_qudaPlaquette(void)
{
  double plaq[3];

  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot calculate plaquette. Call openQCD_qudaGaugeLoad() first.");
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


void openQCD_qudaGaugeLoad(void *gauge)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(QUDA_DOUBLE_PRECISION);

  void* buffer = malloc(4*input.volume*input.sizeof_su3_dble);
  input.reorder_gauge_openqcd_to_quda(gauge, buffer);
  input.gauge = gauge;
  loadGaugeQuda(buffer, &param);
  free(buffer);

  qudaState.gauge_loaded = true;
}


void openQCD_qudaGaugeSave(void *gauge)
{
  QudaGaugeParam qudaGaugeParam = newOpenQCDGaugeParam(QUDA_DOUBLE_PRECISION);

  void* buffer = malloc(4*input.volume*input.sizeof_su3_dble);
  saveGaugeQuda(buffer, &qudaGaugeParam);
  input.reorder_gauge_quda_to_openqcd(buffer, gauge);
  free(buffer);
}

void openQCD_qudaGaugeFree(void)
{
  freeGaugeQuda();
  qudaState.gauge_loaded = false;
}


void openQCD_qudaCloverLoad(void *clover)
{
  /*QudaInvertParam qudaCloverParam = newOpenQCDCloverParam();
  loadCloverQuda(clover, NULL, &qudaCloverParam);*/
  errorQuda("openQCD_qudaCloverLoad() is not implemented yet.");
  qudaState.clover_loaded = true;
}

void openQCD_qudaCloverCreate(double su3csw)
{
  if (!qudaState.dslash_setup) {
    errorQuda("Need to call openQCD_qudaSetDwOptions() first!");
  }

  QudaInvertParam param = newOpenQCDCloverParam(invertParam.kappa, su3csw);

  /* Set to Wilson Dirac operator with Clover term */
  invertParam.dslash_type = QUDA_CLOVER_WILSON_DSLASH;

  /**
   * Leaving both h_clover = h_clovinv = NULL allocates the clover field on the
   * GPU and finally calls @createCloverQuda to calculate the clover field.
   */
  loadCloverQuda(NULL, NULL, &param);
  qudaState.clover_loaded = true;
}

void openQCD_qudaCloverFree(void)
{
  freeCloverQuda();
  qudaState.clover_loaded = false;
}


void openQCD_qudaSetDwOptions(double kappa, double mu)
{
  if (mu != 0.0) {
    errorQuda("twisted mass not implemented yet.");
  }

  invertParam = newOpenQCDParam();

  /* Set to Wilson Dirac operator */
  invertParam.dslash_type = QUDA_WILSON_DSLASH;

  /* Hopping parameter */
  invertParam.kappa = kappa;

  invertParam.inv_type = QUDA_CG_INVERTER; /* just set some, needed? */

  /* What is the difference? only works with QUDA_MASS_NORMALIZATION */
  invertParam.mass_normalization = QUDA_MASS_NORMALIZATION;

  /* Extent of the 5th dimension (for domain wall) */
  invertParam.Ls = 1;

  /* Twisted mass parameter */
  invertParam.mu = mu;

  qudaState.dslash_setup = true;
}


/**
 * @brief      Calculates the norm of a spinor.
 *
 * @param[in]  h_in  input spinor of type spinor_dble[NSPIN]
 *
 * @return     norm
 */
double openQCD_qudaNorm(void *h_in)
{
  QudaInvertParam param = newOpenQCDParam();

  ColorSpinorParam cpuParam(h_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  in = in_h;

  return blas::norm2(in);
}


/**
 * @brief      Applies Dirac matrix to spinor.
 *
 *             openQCD_out = gamma[dir] * openQCD_in
 *
 * @param[in]  dir          Dirac index, 0 <= dir <= 5, notice that dir is in
 *                          openQCD convention, ie. (0: t, 1: x, 2: y, 3: z, 4: 5, 5: 5)
 * @param[in]  openQCD_in   of type spinor_dble[NSPIN]
 * @param[out] openQCD_out  of type spinor_dble[NSPIN]
 */
void openQCD_qudaGamma(const int dir, void *openQCD_in, void *openQCD_out)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  // transfer the CPU field to GPU
  in = in_h;

  // creates a zero-field on the GPU
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  // gamma_i run within QUDA using QUDA fields
  switch (dir) {
  case 0: // t direction
    gamma3(out, in);
    break;
  case 1: // x direction
    gamma0(out, in);
    break;
  case 2: // y direction
    gamma1(out, in);
    break;
  case 3: // z direction
    gamma2(out, in);
    break;
  case 4:
  case 5:
    gamma5(out, in);
    break;
  default:
    errorQuda("Unknown gamma: %d\n", dir);
  }

  // creates a field on the CPU
  cpuParam.v = openQCD_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  // transfer the GPU field back to CPU
  out_h = out;
}


void openQCD_qudaDw(void *src, void *dst, QudaDagType dagger)
{
  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot apply Dslash. Call openQCD_qudaGaugeLoad() first.");
    return;
  }

  if (!qudaState.dslash_setup) {
    errorQuda("Dslash parameters are not set, cannot apply Dslash!");
    return;
  }

  invertParam.dagger = dagger;

  /* both fields reside on the CPU */
  invertParam.input_location = QUDA_CPU_FIELD_LOCATION;
  invertParam.output_location = QUDA_CPU_FIELD_LOCATION;

  MatQuda(static_cast<char *>(dst), static_cast<char *>(src), &invertParam);
}
