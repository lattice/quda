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


static openQCD_QudaState_t qudaState = {false, false, false, false, {}, {}};

using namespace quda;

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

template <bool start> void inline qudaopenqcd_called(const char *func, QudaVerbosity verb)
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
      printfQuda("QUDA_OPENQCD_INTERFACE: %s (called) \n", func);
    } else {
      printfQuda("QUDA_OPENQCD_INTERFACE: %s (return) \n", func);
    }
  }
#endif
}

template <bool start> void inline qudaopenqcd_called(const char *func) { qudaopenqcd_called<start>(func, getVerbosity()); }


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
      fill[i] = qudaState.layout.L[i];
    } else {
      X[i] = qudaState.layout.L[i];
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
 * Set set the local dimensions and machine topology for QUDA to use
 *
 * @param layout Struct defining local dimensions and machine topology
 */
void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout)
{
  int mynproc[4];
  for (int dir = 0; dir < 4; ++dir) {
    if (layout.N[dir] % 2 != 0) {
      errorQuda("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
    mynproc[dir] = layout.nproc[dir];
  }
  if(layout.cstar > 1) {
    mynproc[1] *= -1; /* y direction */
  }
  if(layout.cstar > 2) {
    mynproc[2] *= -1; /* z direction */
  }
}

#ifdef MULTI_GPU
// TODO: would we ever want to run with QMP COMMS?
#ifdef QMP_COMMS
  initCommsGridQuda(4, layout.nproc, nullptr, nullptr);
#else
  initCommsGridQuda(4, mynproc, rankFromCoords, (void *)(layout.nproc));
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
 * @brief      Creates a new quda parameter struct
 *
 * @return     The quda parameter struct.
 */
static QudaInvertParam newOpenQCDParam(void)
{
  static const QudaVerbosity verbosity = getVerbosity();

  QudaInvertParam param = newQudaInvertParam();

  param.verbosity = verbosity;

  param.cpu_prec = QUDA_DOUBLE_PRECISION;  // The precision used by the input fermion fields
  param.cuda_prec = QUDA_DOUBLE_PRECISION; // The precision used by the QUDA solver

  /* AA: This breaks GCR */
  // /* TH added for MG support */
  // param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION; // The precision used by the QUDA solver
  // param.cuda_prec_precondition = QUDA_HALF_PRECISION; // The precision used by the QUDA solver

  /**
   * The order of the input and output fermion fields. Imposes fieldOrder =
   * QUDA_OPENQCD_FIELD_ORDER in color_spinor_field.h and
   * QUDA_OPENQCD_FIELD_ORDER makes quda to instantiate OpenQCDDiracOrder.
   */

  param.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;

  // Gamma basis of the input and output host fields
  param.gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;

  return param;
}


/**
 * @brief      Initialize quda gauge param struct
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

  // This make quda to instantiate OpenQCDOrder
  param.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;

  // Seems to have no effect ...
  param.t_boundary = QUDA_PERIODIC_T;

  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.scale = 1.0;
  param.anisotropy = 1.0; // 1.0 means not anisotropic
  param.ga_pad = getLinkPadding(param.X); // Why this?

  return param;
}


void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout)
{
  if (qudaState.initialized) return;
  qudaState.init = init;
  qudaState.layout = layout;

  setVerbosityQuda(qudaState.init.verbosity, "QUDA: ", qudaState.init.logfile);
  qudaopenqcd_called<true>(__func__);
  openQCD_qudaSetLayout(qudaState.layout);
  qudaopenqcd_called<false>(__func__);
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

  plaqQuda(plaq);

  // Note different Nc normalization wrt openQCD!
  return 3.0*plaq[0];
}


void openQCD_qudaGaugeLoad(void *gauge, QudaPrecision prec)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec);

  /* Matthias Wagner: optimize that */
  void* buffer = pool_pinned_malloc(4*qudaState.init.volume*18*prec);
  qudaState.init.reorder_gauge_openqcd_to_quda(gauge, buffer);
  loadGaugeQuda(buffer, &param);
  pool_pinned_free(buffer);

  qudaState.gauge_loaded = true;
}


void openQCD_qudaGaugeSave(void *gauge, QudaPrecision prec)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec);

  void* buffer = pool_pinned_malloc(4*qudaState.init.volume*18*prec);
  saveGaugeQuda(buffer, &param);
  qudaState.init.reorder_gauge_quda_to_openqcd(buffer, gauge);
  pool_pinned_free(buffer);
}


void openQCD_qudaGaugeFree(void)
{
  freeGaugeQuda();
  qudaState.gauge_loaded = false;
}


void openQCD_qudaCloverLoad(void *clover, double kappa, double csw)
{
  QudaInvertParam param = newOpenQCDParam();
  param.clover_order = QUDA_OPENQCD_CLOVER_ORDER;
  param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;

  param.kappa = kappa;
  param.clover_csw = csw;
  param.clover_coeff = 0.0;

  loadCloverQuda(clover, NULL, &param);
  qudaState.clover_loaded = true;
}


void openQCD_qudaCloverFree(void)
{
  freeCloverQuda();
  qudaState.clover_loaded = false;
}


/**
 * @brief      Creates a new quda Dirac parameter struct
 *
 * @param[in]  p     OpenQCD Dirac parameter struct
 *
 * @return     The quda Dirac parameter struct.
 */
static QudaInvertParam newOpenQCDDiracParam(openQCD_QudaDiracParam_t p)
{
  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot setup Dirac operator / Clover term. Call openQCD_qudaGaugeLoad() first.");
  }

  QudaInvertParam param = newOpenQCDParam();

  param.dslash_type = QUDA_WILSON_DSLASH;
  param.kappa = p.kappa;
  param.mu = p.mu;
  param.dagger = p.dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

  if (p.su3csw != 0.0) {
    param.clover_location = QUDA_CUDA_FIELD_LOCATION; // seems to have no effect?
    param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    param.clover_order = QUDA_FLOAT8_CLOVER_ORDER; // what implication has this?

    param.compute_clover = true;
    param.clover_csw = p.su3csw;
    param.clover_coeff = 0.0;

    // Set to Wilson Dirac operator with Clover term
    param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;

    if (!qudaState.clover_loaded) {
      /**
       * Leaving both h_clover = h_clovinv = NULL allocates the clover field on
       * the GPU and finally calls @createCloverQuda to calculate the clover
       * field.
       */
      loadCloverQuda(NULL, NULL, &param); // Create the clover field
      qudaState.clover_loaded = true;
    }
  }

  param.inv_type = QUDA_CG_INVERTER; // just set some, needed?

  // What is the difference? only works with QUDA_MASS_NORMALIZATION
  param.mass_normalization = QUDA_MASS_NORMALIZATION;

  // Extent of the 5th dimension (for domain wall)
  param.Ls = 1;

  return param;
}


/**
 * @brief      Creates a new quda solver parameter struct
 *
 * @param[in]  p     OpenQCD Dirac parameter struct
 *
 * @return     The quda solver parameter struct.
 */
static QudaInvertParam newOpenQCDSolverParam(openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDDiracParam(p);

  param.compute_true_res = true;

  param.solution_type = QUDA_MAT_SOLUTION;
  param.solve_type = QUDA_DIRECT_SOLVE;
  param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  param.inv_type_precondition = QUDA_INVALID_INVERTER; // disables any preconditioning

  return param;
}


void openQCD_back_and_forth(void *h_in, void *h_out)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(h_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  // transfer the CPU field to GPU
  in = in_h;

  // creates a field on the CPU
  cpuParam.v = h_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  // creates a zero-field on the GPU
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  out = in;

  // transfer the GPU field back to CPU
  out_h = out;
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

double openQCD_qudaNorm_NoLoads(void *d_in)
{
  return blas::norm2(*reinterpret_cast<ColorSpinorField*>(d_in));
}


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
    /* UKQCD uses a different convention for Gamma matrices:
     * gamma5_ukqcd = gammax gammay gammaz gammat,
     * gamma5_openqcd = gammat gammax gammay gammaz,
     * and thus
     * gamma5_openqcd = -1 * U gamma5_ukqcd U^dagger,
     * with U the transformation matrix from OpenQCD to UKQCD. */
    blas::ax(-1.0, out);
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


void* openQCD_qudaH2D(void *openQCD_field)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField *in = new ColorSpinorField(cudaParam);

  *in = in_h; // transfer the CPU field to GPU

  return in;
}


void openQCD_qudaSpinorFree(void** quda_field)
{
  delete reinterpret_cast<ColorSpinorField*>(*quda_field);
  *quda_field = nullptr;
}

void openQCD_qudaD2H(void *quda_field, void *openQCD_field)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField out_h(cpuParam);

  ColorSpinorField* in = reinterpret_cast<ColorSpinorField*>(quda_field);
  ColorSpinorField out(*in);

  out_h = out; // transfer the GPU field to CPU
}


void openQCD_qudaDw_NoLoads(void *src, void *dst, openQCD_QudaDiracParam_t p)
{
}


void openQCD_qudaDw(void *src, void *dst, openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDDiracParam(p);

  // both fields reside on the CPU
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;

  MatQuda(static_cast<char *>(dst), static_cast<char *>(src), &param);
  /* AA: QUDA applies - Dw */
  /* blas::ax(-1.0, dst); */
}


double openQCD_qudaGCR(void *source, void *solution,
  openQCD_QudaDiracParam_t dirac_param, openQCD_QudaGCRParam_t gcr_param)
{
  QudaInvertParam param = newOpenQCDSolverParam(dirac_param);

  // both fields reside on the CPU
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;

  param.inv_type = QUDA_GCR_INVERTER;
  param.tol = gcr_param.tol;
  param.maxiter = gcr_param.nmx;
  param.gcrNkrylov = gcr_param.nkv;
  param.reliable_delta = gcr_param.reliable_delta;

  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), &param);

  printfQuda("true_res    = %e\n", param.true_res);
  printfQuda("true_res_hq = %.2e\n", param.true_res_hq);
  printfQuda("iter        = %d\n",   param.iter);
  printfQuda("gflops      = %.2e\n", param.gflops);
  printfQuda("secs        = %.2e\n", param.secs);
  /* this is not properly set */
  /* printfQuda("Nsteps      = %d\n",   param.Nsteps); */

  return param.true_res;
}

double openQCD_qudaMultigrid(void *source, void *solution, openQCD_QudaDiracParam_t dirac_param)
{
  QudaInvertParam invert_param = newOpenQCDSolverParam(dirac_param);
  QudaInvertParam invert_param_mg = newOpenQCDSolverParam(dirac_param);
  QudaMultigridParam multigrid_param = newQudaMultigridParam();

  //param.verbosity = QUDA_VERBOSE;
  invert_param.reliable_delta = 1e-5;
  invert_param.gcrNkrylov = 20;
  invert_param.maxiter = 2000;
  invert_param.tol = 1e-12;
  invert_param.inv_type = QUDA_GCR_INVERTER;
  invert_param.solution_type = QUDA_MAT_SOLUTION;
  invert_param.solve_type = QUDA_DIRECT_SOLVE;
  invert_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  invert_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  invert_param.inv_type_precondition = QUDA_MG_INVERTER;
  invert_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION; // The precision used by the QUDA solver
  invert_param.cuda_prec_precondition = QUDA_HALF_PRECISION; // The precision used by the QUDA solver

  invert_param_mg.reliable_delta = 1e-5;
  invert_param_mg.gcrNkrylov = 20;
  invert_param_mg.maxiter = 2000;
  invert_param_mg.tol = 1e-12;
  invert_param_mg.inv_type = QUDA_GCR_INVERTER;
  invert_param_mg.solution_type = QUDA_MAT_SOLUTION;
  invert_param_mg.solve_type = QUDA_DIRECT_SOLVE;
  invert_param_mg.matpc_type = QUDA_MATPC_EVEN_EVEN;
  invert_param_mg.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  invert_param_mg.inv_type_precondition = QUDA_MG_INVERTER;
  invert_param_mg.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invert_param_mg.dirac_order = QUDA_DIRAC_ORDER;

  // set the params, hard code the solver
  // parameters copied from recommended settings from Wiki
  multigrid_param.n_level = 2;
  multigrid_param.generate_all_levels = QUDA_BOOLEAN_TRUE;
  multigrid_param.run_verify = QUDA_BOOLEAN_FALSE;
  multigrid_param.invert_param = &invert_param_mg;
  multigrid_param.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;

  // try setting minimal parameters - leave rest to default
  // level 0 fine
  multigrid_param.geo_block_size[0][0] = 4; // xytz
  multigrid_param.geo_block_size[0][1] = 4;
  multigrid_param.geo_block_size[0][2] = 4;
  multigrid_param.geo_block_size[0][3] = 4;
  multigrid_param.n_vec[0] = 24;
  multigrid_param.spin_block_size[0] = 2;
  multigrid_param.precision_null[0] = QUDA_HALF_PRECISION; 
  multigrid_param.smoother[0] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother_tol[0] = 0.25;
  multigrid_param.location[0] = QUDA_CUDA_FIELD_LOCATION;
  multigrid_param.nu_pre[0] = 0;
  multigrid_param.nu_post[0] = 8;
  multigrid_param.omega[0] = 0.8;
  multigrid_param.smoother_solve_type[0] = QUDA_DIRECT_PC_SOLVE;
  multigrid_param.cycle_type[0] = QUDA_MG_CYCLE_RECURSIVE;
  multigrid_param.coarse_solver[0] = QUDA_GCR_INVERTER;
  multigrid_param.coarse_solver_tol[0] = 0.25;
  multigrid_param.coarse_solver_maxiter[0] = 50;
  multigrid_param.coarse_grid_solution_type[0] = QUDA_MAT_SOLUTION;

  // level 1 coarse
  // no smoother required for innermost
  // so no blocks
  multigrid_param.precision_null[1] = QUDA_HALF_PRECISION;
  multigrid_param.coarse_solver[1] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother[1] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother_tol[1] = 0.25;
  multigrid_param.spin_block_size[1] = 1;
  multigrid_param.coarse_solver_tol[1] = 0.25;
  multigrid_param.coarse_solver_maxiter[1] = 50;
  multigrid_param.coarse_grid_solution_type[1] = QUDA_MATPC_SOLUTION;
  multigrid_param.smoother_solve_type[1] = QUDA_DIRECT_PC_SOLVE;
  multigrid_param.cycle_type[1] = QUDA_MG_CYCLE_RECURSIVE;
  multigrid_param.location[1] = QUDA_CUDA_FIELD_LOCATION;
  multigrid_param.nu_pre[1] = 0;
  multigrid_param.nu_post[1] = 8;
  multigrid_param.omega[1] = 0.8;

  PUSH_RANGE("newMultigridQuda",4);
  void *mgprec = newMultigridQuda(&multigrid_param);
  invert_param.preconditioner = mgprec;
  POP_RANGE;

  PUSH_RANGE("invertQUDA",5);
  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), &invert_param);
  POP_RANGE;

  destroyMultigridQuda(mgprec);

  printfQuda("true_res    = %e\n", invert_param.true_res);
  printfQuda("true_res_hq = %.2e\n", invert_param.true_res_hq);
  printfQuda("iter        = %d\n",   invert_param.iter);
  printfQuda("gflops      = %.2e\n", invert_param.gflops);
  printfQuda("secs        = %.2e\n", invert_param.secs);

  return invert_param.true_res;
}
