#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>

#include <blas_quda.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>

#include <util_quda.h>
#include <random_quda.h>

#include <misc.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
// #include <staggered_dslash_reference.h>
#include <llfat_reference.h>
#include <gauge_field.h>
#include <unitarization_links.h>
// #include <blas_reference.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define mySpinorSiteSize 6

void **ghost_fatlink, **ghost_longlink;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

int X[4];

static int n_naiks = 1;

// Unitarization coefficients
static double unitarize_eps = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only = false;
static double svd_rel_error = 1e-4;
static double svd_abs_error = 1e-4;
static double max_allowed_error = 1e-11;

// For loading the gauge fields
int argc_copy;
char **argv_copy;

//!< Profiler for invertQuda
static quda::TimeProfile profileInvert("invertQuda");

static void set_params(QudaGaugeParam *gaugeParam, QudaInvertParam *inv_param, int X1, int X2, int X3, int X4,
                       QudaPrecision cpu_prec, QudaPrecision prec, QudaPrecision prec_sloppy,
                       QudaPrecision prec_refinement_sloppy, QudaReconstructType link_recon,
                       QudaReconstructType link_recon_sloppy, double mass, double tol, double tadpole_coeff)
{

  gaugeParam->X[0] = xdim;
  gaugeParam->X[1] = ydim;
  gaugeParam->X[2] = zdim;
  gaugeParam->X[3] = tdim;

  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = prec;
  gaugeParam->reconstruct = link_recon;
  gaugeParam->cuda_prec_sloppy = prec_sloppy;
  gaugeParam->cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  gaugeParam->reconstruct_sloppy = link_recon_sloppy;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam->anisotropy = 1.0;

  // For asqtad:
  // gaugeParam->tadpole_coeff = tadpole_coeff;
  // gaugeParam->scale = dslash_type != QUDA_ASQTAD_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gaugeParam->tadpole_coeff = 1.0;
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam->scale = -1.0 / 24.0;
    if (eps_naik != 0) { gaugeParam->scale *= (1.0 + eps_naik); }
  } else {
    gaugeParam->scale = 1.0;
  }
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam->t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam->staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->type = QUDA_WILSON_LINKS;

  gaugeParam->ga_pad = 0;

#ifdef MULTI_GPU
  int x_face_size = gaugeParam->X[1] * gaugeParam->X[2] * gaugeParam->X[3] / 2;
  int y_face_size = gaugeParam->X[0] * gaugeParam->X[2] * gaugeParam->X[3] / 2;
  int z_face_size = gaugeParam->X[0] * gaugeParam->X[1] * gaugeParam->X[3] / 2;
  int t_face_size = gaugeParam->X[0] * gaugeParam->X[1] * gaugeParam->X[2] / 2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam->ga_pad = pad_size;
#endif

  // Solver params

  inv_param->verbosity = QUDA_VERBOSE;
  inv_param->mass = mass;


  // outer solver parameters
  inv_param->inv_type = inv_type;
  inv_param->tol = tol;
  inv_param->tol_restart = 1e-3; // now theoretical background for this parameter...
  inv_param->maxiter = niter;
  inv_param->reliable_delta = reliable_delta;
  inv_param->use_alternative_reliable = alternative_reliable;
  inv_param->use_sloppy_partial_accumulator = false;
  inv_param->solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param->pipeline = pipeline;

  inv_param->Ls = Nsrc;

  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param->residual_type = static_cast<QudaResidualType_s>(0);
  inv_param->residual_type = (tol != 0) ?
    static_cast<QudaResidualType_s>(inv_param->residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    inv_param->residual_type;
  inv_param->residual_type = (tol_hq != 0) ?
    static_cast<QudaResidualType_s>(inv_param->residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    inv_param->residual_type;
  inv_param->heavy_quark_check = (inv_param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 5 : 0);

  inv_param->tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param->Nsteps = 2;

  // Specify Krylov sub-size for GCR, BICGSTAB(L), basis size for CA-CG, CA-GCR
  inv_param->gcrNkrylov = gcrNkrylov;

  // Specify basis for CA-CG, lambda min/max for Chebyshev basis
  //   lambda_max < lambda_max -> use power iters to generate
  inv_param->ca_basis = ca_basis;
  inv_param->ca_lambda_min = ca_lambda_min;
  inv_param->ca_lambda_max = ca_lambda_max;

  inv_param->solution_type = solution_type;
  inv_param->solve_type = solve_type;
  inv_param->matpc_type = matpc_type;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec;
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param->dirac_order = QUDA_DIRAC_ORDER;

  inv_param->dslash_type = dslash_type;

  inv_param->input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param->output_location = QUDA_CPU_FIELD_LOCATION;

  // domain decomposition preconditioner parameters
  inv_param->inv_type_precondition = QUDA_SD_INVERTER;
  inv_param->tol_precondition = 1e-1;
  inv_param->maxiter_precondition = 10;
  inv_param->verbosity_precondition = QUDA_SILENT;
  inv_param->cuda_prec_precondition = inv_param->cuda_prec_sloppy;

  int tmpint = MAX(X[1] * X[2] * X[3], X[0] * X[2] * X[3]);
  tmpint = MAX(tmpint, X[0] * X[1] * X[3]);
  tmpint = MAX(tmpint, X[0] * X[1] * X[2]);

  inv_param->sp_pad = tmpint;
}

// Wrap everything for the GPU construction of fat/long links here
void computeHISQLinksGPU(void **fatlink, void **longlink, void **fatlink_eps, void **longlink_eps, void **inlink,
                         void *qudaGaugeParamPtr, double **act_path_coeffs, double eps_naik)
{

  QudaGaugeParam gauge_param = *(reinterpret_cast<QudaGaugeParam *>(qudaGaugeParamPtr));
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // inlink in different format
  void *inlink_milc = pinned_malloc(4 * V * gaugeSiteSize * gSize);
  reorderQDPtoMILC(inlink_milc, inlink, V, gaugeSiteSize, gauge_param.cpu_prec, gauge_param.cpu_prec);

  // Paths for step 1:
  void *vlink_milc = pinned_malloc(4 * V * gaugeSiteSize * gSize); // V links
  void *wlink_milc = pinned_malloc(4 * V * gaugeSiteSize * gSize); // W links

  // Paths for step 2:
  void *fatlink_milc = pinned_malloc(4 * V * gaugeSiteSize * gSize);  // final fat ("X") links
  void *longlink_milc = pinned_malloc(4 * V * gaugeSiteSize * gSize); // final long links

  // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
  computeKSLinkQuda(vlink_milc, nullptr, wlink_milc, inlink_milc, act_path_coeffs[0], &gauge_param);

  // Create X and long links, 2nd path table set
  computeKSLinkQuda(fatlink_milc, longlink_milc, nullptr, wlink_milc, act_path_coeffs[1], &gauge_param);

  // Copy back
  reorderMILCtoQDP(fatlink, fatlink_milc, V, gaugeSiteSize, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderMILCtoQDP(longlink, longlink_milc, V, gaugeSiteSize, gauge_param.cpu_prec, gauge_param.cpu_prec);

  // Clean up GPU compute links
  host_free(inlink_milc);
  host_free(vlink_milc);
  host_free(wlink_milc);
  host_free(fatlink_milc);
  host_free(longlink_milc);
}

namespace quda
{
  // these are functions from interface quda
  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve);
  void massRescale(cudaColorSpinorField &b, QudaInvertParam &param);

} // namespace quda
quda::cudaGaugeField *checkGauge(QudaInvertParam *param);

using namespace quda;
void psibarpsiQuda(QudaInvertParam *param, QudaEigParam * eig_param, quda::RNG *rng)
{

  // profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  // if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  // checkInvertParam(param, hp_x, hp_b);

  // check the gauge fields have been created
  // cudaGaugeField *cudaGauge = checkGauge(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution
    = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE)
    || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type == QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || (param->solve_type == QUDA_DIRECT_PC_SOLVE);
  bool norm_error_solve = (param->solve_type == QUDA_NORMERR_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  quda::Dirac *d = nullptr;
  quda::Dirac *dSloppy = nullptr;
  quda::Dirac *dPre = nullptr;

  // create the dirac operator
  quda::createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  // rofileInvert.TPSTART(QUDA_PROFILE_H2D);

  // ColorSpinorField *b = nullptr;
  // ColorSpinorField *x = nullptr;
  ColorSpinorField *in = nullptr;
  ColorSpinorField *out = nullptr;
  const int *X = cudaGauge->X();

  // download source
  ColorSpinorParam cpuParam(nullptr, *param, X, pc_solution, param->input_location);
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;

  auto bb = std::make_unique<cudaColorSpinorField>(cudaParam);
  auto b = bb.get();
  auto xx = std::make_unique<cudaColorSpinorField>(cudaParam);
  auto x = xx.get();

  spinorNoise(*b, *rng, QUDA_NOISE_GAUSS);
  blas::zero(*x);

  // profileInvert.TPSTART(QUDA_PROFILE_PREAMBLE);

  double nb = blas::norm2(*b);
  if (nb == 0.0) errorQuda("Source has zero norm");

  if (getVerbosity() >= QUDA_VERBOSE) {
    // double nh_b = blas::norm2(*h_b);
    printfQuda("Source: CUDA copy = %g\n", nb);
    // if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) {
    //   double nh_x = blas::norm2(*h_x);
    //   double nx = blas::norm2(*x);
    //   printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
    // }
  }

  // rescale the source and solution vectors to help prevent the onset of underflow
  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    blas::ax(1.0 / sqrt(nb), *b);
    blas::ax(1.0 / sqrt(nb), *x);
  }

  massRescale(*static_cast<cudaColorSpinorField *>(b), *param);

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

  if (pc_solution && !pc_solve) { errorQuda("Preconditioned (PC) solution_type requires a PC solve_type"); }

  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }

  if (!mat_solution && norm_error_solve) { errorQuda("Normal-error solve requires Mat solution"); }

  if (param->inv_type_precondition == QUDA_MG_INVERTER && (!direct_solve || !mat_solution)) {
    errorQuda("Multigrid preconditioning only supported for direct solves");
  }

  if (param->chrono_use_resident && (norm_error_solve)) {
    errorQuda("Chronological forcasting only presently supported for M^dagger M solver");
  }

  // profileInvert.TPSTOP(QUDA_PROFILE_PREAMBLE);

  if (mat_solution && !direct_solve && !norm_error_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    solverParam.deflate = true;
    solverParam.eig_param = *eig_param;

    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    blas::copy(*in, *out);
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    solverParam.deflate = true;
    solverParam.eig_param = *eig_param;
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  } else if (!norm_error_solve) {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*param);
    solverParam.deflate = true;
    solverParam.eig_param = *eig_param;

    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    solverParam.updateInvertParam(*param);
    delete solve;
  } else { // norm_error_solve
    DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    cudaColorSpinorField tmp(*out);
    SolverParam solverParam(*param);
    solverParam.deflate = true;
    solverParam.eig_param = *eig_param;
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
    (*solve)(tmp, *in);    // y = (M M^\dag) b
    dirac.Mdag(*out, tmp); // x = M^dag y
    solverParam.updateInvertParam(*param);
    delete solve;
  }

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nx = blas::norm2(*x);
    printfQuda("Solution = %g\n", nx);
  }

  // profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);

  dirac.reconstruct(*x, *b, param->solution_type);

  if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION) {
    // rescale the solution
    blas::ax(sqrt(nb), *x);
  }
  // profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  // profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);

  Complex action = blas::cDotProduct(*b, *x);
  action /= static_cast<double>(X[0] * X[1] * X[2] * X[3]);

  printfQuda("PsiBarPsi (%g,%g)\n", action.real(), action.imag());

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nx = blas::norm2(*x);
    printfQuda("Reconstructed: CUDA solution = %g\n", nx);
  }
  // profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  // profileInvert.TPSTART(QUDA_PROFILE_FREE);

  delete d;
  delete dSloppy;
  delete dPre;

  // profileInvert.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

// Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param)
{
  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  if ((eig_type == QUDA_EIG_TR_LANCZOS || eig_type == QUDA_EIG_IR_LANCZOS)
      && !(eig_spectrum == QUDA_SPECTRUM_LR_EIG || eig_spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to Lanczos type solver");
  }

  // The solver will exit when nConv extremal eigenpairs have converged
  if (eig_nConv < 0) {
    eig_param.nConv = eig_nEv;
    eig_nConv = eig_nEv;
  } else {
    eig_param.nConv = eig_nConv;
  }

  eig_param.nEv = eig_nEv;
  eig_param.nKr = eig_nKr;
  eig_param.tol = eig_tol;
  eig_param.require_convergence = eig_require_convergence ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  // eig_param.cuda_prec_ritz = cuda_prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.use_dagger = eig_use_dagger ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  if (eig_compute_svd) {
    eig_param.use_dagger = QUDA_BOOLEAN_NO;
    eig_param.use_norm_op = QUDA_BOOLEAN_YES;
  }

  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
}


int invert_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  set_params(&gaugeParam, &inv_param, xdim, ydim, zdim, tdim, cpu_prec, prec, prec_sloppy, prec_refinement_sloppy,
             link_recon, link_recon_sloppy, mass, tol, tadpole_factor);

  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);


  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gaugeParam.X);
  dw_setDims(gaugeParam.X, Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *qdp_inlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *milc_fatlink = nullptr;
  void *milc_longlink = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V * gaugeSiteSize * gSize);
    qdp_fatlink[dir] = malloc(V * gaugeSiteSize * gSize);
    qdp_longlink[dir] = malloc(V * gaugeSiteSize * gSize);
  }
  milc_fatlink = malloc(4 * V * gaugeSiteSize * gSize);
  milc_longlink = malloc(4 * V * gaugeSiteSize * gSize);

  // load a field WITHOUT PHASES
  if (strcmp(latfile, "")) {
    read_gauge_field(latfile, qdp_inlink, gaugeParam.cpu_prec, gaugeParam.X, argc_copy, argv_copy);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gaugeParam, QUDA_STAGGERED_DSLASH, gaugeParam.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_inlink, 1, gaugeParam.cpu_prec, &gaugeParam);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gaugeParam.cpu_prec, &gaugeParam,
                                     QUDA_STAGGERED_DSLASH);
    }
    // createSiteLinkCPU(inlink, gaugeParam.cpu_prec, 0); // 0 for no phases
  }

  // #ifdef GPU_GAUGE_TOOLS
  //   gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  //   printfQuda("gaugePrecise: %lu\n", (unsigned long)gaugePrecise);
  //   double plaq[3];
  //   loadGaugeQuda(qdp_inlink, &gaugeParam);
  //   plaqQuda(plaq);
  //   gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  //   if (dslash_type != QUDA_LAPLACE_DSLASH) {
  //     plaq[0] = -plaq[0]; // correction because we've already put phases on the fields
  //     plaq[1] = -plaq[1];
  //     plaq[2] = -plaq[2];
  //   }

  //   printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  // #endif

  compute_fatlong = true;
  if (compute_fatlong) {

    ///////////////////////////
    // Set path coefficients //
    ///////////////////////////

    // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
    // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

    double u1 = 1.0 / tadpole_factor;
    double u2 = u1 * u1;
    double u4 = u2 * u2;
    double u6 = u4 * u2;

    // First path: create V, W links
    double act_path_coeff_1[6] = {
      (1.0 / 8.0),                             /* one link */
      u2 * (0.0),                              /* Naik */
      u2 * (-1.0 / 8.0) * 0.5,                 /* simple staple */
      u4 * (1.0 / 8.0) * 0.25 * 0.5,           /* displace link in two directions */
      u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), /* displace link in three directions */
      u4 * (0.0)                               /* Lepage term */
    };

    // Second path: create X, long links
    double act_path_coeff_2[6] = {
      ((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)), // one link
                                                        // One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik
      (-1.0 / 24.0),                                    // Naik
      (-1.0 / 8.0) * 0.5,                               // simple staple
      (1.0 / 8.0) * 0.25 * 0.5,                         // displace link in two directions
      (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),               // displace link in three directions
      (-2.0 / 16.0)                                     // Lepage term, correct O(a^2) 2x ASQTAD
    };

    // Paths for epsilon corrections. Not used if n_naiks = 1.
    double act_path_coeff_3[6] = {
      (1.0 / 8.0),   // one link b/c of Naik
      (-1.0 / 24.0), // Naik
      0.0,           // simple staple
      0.0,           // displace link in two directions
      0.0,           // displace link in three directions
      0.0            // Lepage term
    };

    double *act_paths[3] = {act_path_coeff_1, act_path_coeff_2, act_path_coeff_3};

    ////////////////////////////////////
    // Set unitarization coefficients //
    ////////////////////////////////////

    setUnitarizeLinksConstants(unitarize_eps, max_allowed_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                               svd_abs_error);

    ///////////////////////////////////////////////////////////////////////
    // Create some temporary space if we want to test the epsilon fields //
    ///////////////////////////////////////////////////////////////////////

    void *qdp_fatlink_naik_temp[4];
    void *qdp_longlink_naik_temp[4];
    if (n_naiks == 2) {
      for (int dir = 0; dir < 4; dir++) {
        qdp_fatlink_naik_temp[dir] = malloc(V * gaugeSiteSize * gSize);
        qdp_longlink_naik_temp[dir] = malloc(V * gaugeSiteSize * gSize);
      }
    }

    //////////////////////////
    // Create the GPU links //
    //////////////////////////

    // Skip eps field for now

    // Note: GPU link creation only works for single and double precision
    computeHISQLinksGPU(qdp_fatlink, qdp_longlink, (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
                        (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr, qdp_inlink, &gaugeParam, act_paths, eps_naik);

    if (n_naiks == 2) {
      // Override the naik fields into the fat/long link fields
      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir], qdp_fatlink_naik_temp[dir], V * gaugeSiteSize * gSize);
        memcpy(qdp_longlink[dir], qdp_longlink_naik_temp[dir], V * gaugeSiteSize * gSize);
        free(qdp_fatlink_naik_temp[dir]);
        qdp_fatlink_naik_temp[dir] = nullptr;
        free(qdp_longlink_naik_temp[dir]);
        qdp_longlink_naik_temp[dir] = nullptr;
      }
    }

  } // fat long

  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gaugeSiteSize, gaugeParam.cpu_prec, gaugeParam.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gaugeSiteSize, gaugeParam.cpu_prec, gaugeParam.cpu_prec);

  // parameters for color spinor fields
  ColorSpinorParam csParam;
  csParam.nColor = 3;
  csParam.nSpin = 1;
  csParam.nDim = 5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gaugeParam.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  csParam.x[4] = Nsrc;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;


#ifdef MULTI_GPU
  int tmp_value = MAX(ydim * zdim * tdim / 2, xdim * zdim * tdim / 2);
  tmp_value = MAX(tmp_value, xdim * ydim * tdim / 2);
  tmp_value = MAX(tmp_value, xdim * ydim * zdim / 2);
  int fat_pad = tmp_value;
  int link_pad = 3 * tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gaugeParam.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ? QUDA_SU3_LINKS :
                                                                                                   QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink, gaugeParam);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void **)cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gaugeParam);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void **)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif

  // load fat links
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.ga_pad = fat_pad;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  gaugeParam.cuda_prec_precondition = gaugeParam.cuda_prec_sloppy;
  gaugeParam.reconstruct_precondition = gaugeParam.reconstruct_sloppy;
  loadGaugeQuda(milc_fatlink, &gaugeParam);

  // load long links
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  gaugeParam.ga_pad = link_pad;
  gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  gaugeParam.reconstruct
    = (link_recon == QUDA_RECONSTRUCT_12 || link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_13 : link_recon;
  gaugeParam.reconstruct_sloppy = (link_recon_sloppy == QUDA_RECONSTRUCT_12 || link_recon_sloppy == QUDA_RECONSTRUCT_8) ?
    QUDA_RECONSTRUCT_13 :
    link_recon_sloppy;
  gaugeParam.cuda_prec_precondition = gaugeParam.cuda_prec_sloppy;
  gaugeParam.reconstruct_precondition = gaugeParam.reconstruct_sloppy;
  loadGaugeQuda(milc_longlink, &gaugeParam);

  double time0 = -((double)clock()); // Start the timer

  // double nrm2 = 0;
  // double src2 = 0;
  int ret = 0;
  unsigned long long seed = 12345;
  auto rng = std::make_unique<quda::RNG>(gaugeParam, seed);
  rng->Init();

  // int len = 0;
  // if (solution_type == QUDA_MAT_SOLUTION || solution_type == QUDA_MATDAG_MAT_SOLUTION) {
  //   len = V * Nsrc;
  // } else {
  //   len = Vh * Nsrc;
  // }

  switch (test_type) {
  case 0: // full parity solution
  case 1: // solving prec system, reconstructing
  case 2:

    psibarpsiQuda(&inv_param, &eig_param, rng.get());
    // pinvertQuda(out->V(), in->V(), &inv_param);
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;

    break;

  case 3: // even
  case 4:

    // invertQuda(out->V(), in->V(), &inv_param);

    time0 += clock();
    time0 /= CLOCKS_PER_SEC;

    break;

  default: errorQuda("Unsupported test type %d given", test_type);
    
  } // switch


  // Clean up gauge fields, at least
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) {
      free(qdp_inlink[dir]);
      qdp_inlink[dir] = nullptr;
    }
    if (qdp_fatlink[dir] != nullptr) {
      free(qdp_fatlink[dir]);
      qdp_fatlink[dir] = nullptr;
    }
    if (qdp_longlink[dir] != nullptr) {
      free(qdp_longlink[dir]);
      qdp_longlink[dir] = nullptr;
    }
  }
  if (milc_fatlink != nullptr) {
    free(milc_fatlink);
    milc_fatlink = nullptr;
  }
  if (milc_longlink != nullptr) {
    free(milc_longlink);
    milc_longlink = nullptr;
  }

#ifdef MULTI_GPU
  if (cpuFat != nullptr) {
    delete cpuFat;
    cpuFat = nullptr;
  }
  if (cpuLong != nullptr) {
    delete cpuLong;
    cpuLong = nullptr;
  }
#endif


  rng->Release();
  endQuda();
  return ret;
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
             get_staggered_test_type(test_type), xdim, ydim, zdim, tdim);

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
  printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
  printfQuda(" - size of Krylov space %d\n", eig_nKr);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
  if (eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {
    printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
               eig_use_normop ? "true" : "false");
  }
  if (eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  return;
}

void usage_extra(char **argv)
{
  printfQuda("Extra options:\n");
  printfQuda("    --test <0/1/2/3/4/5/6>                      # Test method\n");
  printfQuda("                                                0: Full parity inverter\n");
  printfQuda(
    "                                                1: Even even spinor CG inverter, reconstruct to full parity\n");
  printfQuda(
    "                                                2: Odd odd spinor CG inverter, reconstruct to full parity\n");
  printfQuda("                                                3: Even even spinor CG inverter\n");
  printfQuda("                                                4: Odd odd spinor CG inverter\n");
  printfQuda("                                                5: Even even spinor multishift CG inverter\n");
  printfQuda("                                                6: Odd odd spinor multishift CG inverter\n");
  printfQuda("    --cpu_prec <double/single/half>             # Set CPU precision\n");

  return;
}

// main
int main(int argc, char **argv)
{
  // Set a default
  solve_type = QUDA_INVALID_SOLVE;
  // command line options
  auto app = make_app();
  // app->get_formatter()->column_width(40);
  add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }


  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  dslash_type = QUDA_ASQTAD_DSLASH;
  solve_type = QUDA_DIRECT_PC_SOLVE;
  matpc_type = QUDA_MATPC_EVEN_EVEN;
  solution_type = QUDA_MAT_SOLUTION;

  if (prec_sloppy == QUDA_INVALID_PRECISION) { prec_sloppy = prec; }

  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) { prec_refinement_sloppy = prec_sloppy; }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) { link_recon_sloppy = link_recon; }

  if (inv_type != QUDA_CG_INVERTER && (test_type == 5 || test_type == 6)) {
    errorQuda("Preconditioning is currently not supported in multi-shift solver solvers");
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }

  display_test_info();

  printfQuda("dslash_type = %d\n", dslash_type);

  argc_copy = argc;
  argv_copy = argv;

  int ret = invert_test();

  // finalize the communications layer
  finalizeComms();

  return ret;
}
