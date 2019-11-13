#include "test_params.h"
#include <comm_quda.h>

// parameters parsed from the command line

#ifdef MULTI_GPU
int device = -1;
#else
int device = 0;
#endif

int rank_order;
std::array<int, 4> gridsize_from_cmdline = {1, 1, 1, 1};
auto &grid_x = gridsize_from_cmdline[0];
auto &grid_y = gridsize_from_cmdline[1];
auto &grid_z = gridsize_from_cmdline[2];
auto &grid_t = gridsize_from_cmdline[3];

std::array<int, 4> dim_partitioned = {0, 0, 0, 0};
QudaReconstructType link_recon = QUDA_RECONSTRUCT_NO;
QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_INVALID;
QudaReconstructType link_recon_precondition = QUDA_RECONSTRUCT_INVALID;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision prec_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_refinement_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_precondition = QUDA_INVALID_PRECISION;
QudaPrecision prec_null = QUDA_INVALID_PRECISION;
QudaPrecision prec_ritz = QUDA_INVALID_PRECISION;
QudaVerbosity verbosity = QUDA_SUMMARIZE;
std::array<int, 4> dim = {24, 24, 24, 24};
int &xdim = dim[0];
int &ydim = dim[1];
int &zdim = dim[2];
int &tdim = dim[3];
int Lsdim = 16;
QudaDagType dagger = QUDA_DAG_NO;
QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
int laplace3D = 4;
char latfile[256] = "";
bool unit_gauge = false;
double gaussian_sigma = 0.2;
char gauge_outfile[256] = "";
int Nsrc = 1;
int Msrc = 1;
int niter = 100;
int gcrNkrylov = 10;
QudaCABasis ca_basis = QUDA_POWER_BASIS;
double ca_lambda_min = 0.0;
double ca_lambda_max = -1.0;
int pipeline = 0;
int solution_accumulator_pipeline = 0;
int test_type = 0;
quda::mgarray<int> nvec = {};
quda::mgarray<char[256]> mg_vec_infile;
quda::mgarray<char[256]> mg_vec_outfile;
QudaInverterType inv_type;
bool inv_deflate = false;
QudaInverterType precon_type = QUDA_INVALID_INVERTER;
int multishift = 0;
bool verify_results = true;
bool low_mode_check = false;
bool oblique_proj_check = false;
double mass = 0.1;
double kappa = -1.0;
double mu = 0.1;
double epsilon = 0.01;
double m5 = -1.5;
double b5 = 1.5;
double c5 = 0.5;
double anisotropy = 1.0;
double tadpole_factor = 1.0;
double eps_naik = 0.0;
double clover_coeff = 0.1;
bool compute_clover = false;
bool compute_fatlong = false;
double tol = 1e-7;
double tol_hq = 0.;
double reliable_delta = 0.1;
bool alternative_reliable = false;
QudaTwistFlavorType twist_flavor = QUDA_TWIST_SINGLET;
QudaMassNormalization normalization = QUDA_KAPPA_NORMALIZATION;
QudaMatPCType matpc_type = QUDA_MATPC_EVEN_EVEN;
QudaSolveType solve_type = QUDA_NORMOP_PC_SOLVE;
QudaSolutionType solution_type = QUDA_MAT_SOLUTION;

int mg_levels = 2;

quda::mgarray<QudaFieldLocation> solver_location = {};
quda::mgarray<QudaFieldLocation> setup_location = {};

quda::mgarray<int> nu_pre = {};
quda::mgarray<int> nu_post = {};
quda::mgarray<int> n_block_ortho = {};
quda::mgarray<double> mu_factor = {};
quda::mgarray<QudaVerbosity> mg_verbosity = {};
quda::mgarray<QudaInverterType> setup_inv = {};
quda::mgarray<QudaSolveType> coarse_solve_type = {};
quda::mgarray<QudaSolveType> smoother_solve_type = {};
quda::mgarray<int> num_setup_iter = {};
quda::mgarray<double> setup_tol = {};
quda::mgarray<int> setup_maxiter = {};
quda::mgarray<int> setup_maxiter_refresh = {};
quda::mgarray<QudaCABasis> setup_ca_basis = {};
quda::mgarray<int> setup_ca_basis_size = {};
quda::mgarray<double> setup_ca_lambda_min = {};
quda::mgarray<double> setup_ca_lambda_max = {};
QudaSetupType setup_type = QUDA_NULL_VECTOR_SETUP;
bool pre_orthonormalize = false;
bool post_orthonormalize = true;
double omega = 0.85;
quda::mgarray<QudaInverterType> coarse_solver = {};
quda::mgarray<double> coarse_solver_tol = {};
quda::mgarray<QudaInverterType> smoother_type = {};
QudaPrecision smoother_halo_prec = QUDA_INVALID_PRECISION;
quda::mgarray<double> smoother_tol = {};
quda::mgarray<int> coarse_solver_maxiter = {};
quda::mgarray<QudaCABasis> coarse_solver_ca_basis = {};
quda::mgarray<int> coarse_solver_ca_basis_size = {};
quda::mgarray<double> coarse_solver_ca_lambda_min = {};
quda::mgarray<double> coarse_solver_ca_lambda_max = {};
bool generate_nullspace = true;
bool generate_all_levels = true;
quda::mgarray<QudaSchwarzType> schwarz_type = {};
quda::mgarray<int> schwarz_cycle = {};

// we only actually support 4 here currently
quda::mgarray<std::array<int, 4>> geo_block_size = {};
int nev = 8;
int max_search_dim = 64;
int deflation_grid = 16;
double tol_restart = 5e+3 * tol;

int eigcg_max_restarts = 3;
int max_restart_num = 3;
double inc_tol = 1e-2;
double eigenval_tol = 1e-1;

QudaExtLibType solver_ext_lib = QUDA_EIGEN_EXTLIB;
QudaExtLibType deflation_ext_lib = QUDA_EIGEN_EXTLIB;
QudaFieldLocation location_ritz = QUDA_CUDA_FIELD_LOCATION;
QudaMemoryType mem_type_ritz = QUDA_MEMORY_DEVICE;

// Parameters for the stand alone eigensolver
int eig_nEv = 16;
int eig_nKr = 32;
int eig_nConv = -1; // If unchanged, will be set to nEv
int eig_batched_rotate = 0; // If unchanged, will be set to maximum
bool eig_require_convergence = true;
int eig_check_interval = 10;
int eig_max_restarts = 1000;
double eig_tol = 1e-6;
bool eig_use_poly_acc = true;
int eig_poly_deg = 100;
double eig_amin = 0.1;
double eig_amax = 4.0;
bool eig_use_normop = true;
bool eig_use_dagger = false;
bool eig_compute_svd = false;
QudaEigSpectrumType eig_spectrum = QUDA_SPECTRUM_LR_EIG;
QudaEigType eig_type = QUDA_EIG_TR_LANCZOS;
bool eig_arpack_check = false;
char eig_arpack_logfile[256] = "arpack_logfile.log";
char eig_QUDA_logfile[256] = "QUDA_logfile.log";
char eig_vec_infile[256] = "";
char eig_vec_outfile[256] = "";

// Parameters for the MG eigensolver.
// The coarsest grid params are for deflation,
// all others are for PR vectors.
quda::mgarray<bool> mg_eig = {};
quda::mgarray<int> mg_eig_nEv = {};
quda::mgarray<int> mg_eig_nKr = {};
quda::mgarray<int> mg_eig_batched_rotate = {};
quda::mgarray<bool> mg_eig_require_convergence = {};
quda::mgarray<int> mg_eig_check_interval = {};
quda::mgarray<int> mg_eig_max_restarts = {};
quda::mgarray<double> mg_eig_tol = {};
quda::mgarray<bool> mg_eig_use_poly_acc = {};
quda::mgarray<int> mg_eig_poly_deg = {};
quda::mgarray<double> mg_eig_amin = {};
quda::mgarray<double> mg_eig_amax = {};
quda::mgarray<bool> mg_eig_use_normop = {};
quda::mgarray<bool> mg_eig_use_dagger = {};
quda::mgarray<QudaEigSpectrumType> mg_eig_spectrum = {};
quda::mgarray<QudaEigType> mg_eig_type = {};
bool mg_eig_coarse_guess = false;
bool mg_eig_preserve_deflation = false;

double heatbath_beta_value = 6.2;
int heatbath_warmup_steps = 10;
int heatbath_num_steps = 10;
int heatbath_num_heatbath_per_step = 5;
int heatbath_num_overrelax_per_step = 5;
bool heatbath_coldstart = false;

QudaContractType contract_type = QUDA_CONTRACT_TYPE_OPEN;

namespace
{
  CLI::TransformPairs<QudaCABasis> ca_basis_map {{"power", QUDA_POWER_BASIS}, {"chebyshev", QUDA_CHEBYSHEV_BASIS}};

  CLI::TransformPairs<QudaContractType> contract_type_map {{"open", QUDA_CONTRACT_TYPE_OPEN},
                                                           {"dr", QUDA_CONTRACT_TYPE_DR}};

  CLI::TransformPairs<QudaDslashType> dslash_type_map {{"wilson", QUDA_WILSON_DSLASH},
                                                       {"clover", QUDA_CLOVER_WILSON_DSLASH},
                                                       {"twisted-mass", QUDA_TWISTED_MASS_DSLASH},
                                                       {"twisted-clover", QUDA_TWISTED_CLOVER_DSLASH},
                                                       {"clover-hasenbusch-twist", QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH},
                                                       {"staggered", QUDA_STAGGERED_DSLASH},
                                                       {"asqtad", QUDA_ASQTAD_DSLASH},
                                                       {"domain-wall", QUDA_DOMAIN_WALL_DSLASH},
                                                       {"domain-wall-4d", QUDA_DOMAIN_WALL_4D_DSLASH},
                                                       {"mobius", QUDA_MOBIUS_DWF_DSLASH},
                                                       {"laplace", QUDA_LAPLACE_DSLASH}};

  CLI::TransformPairs<QudaTwistFlavorType> twist_flavor_type_map {{"singlet", QUDA_TWIST_SINGLET},
                                                                  {"deg-doublet", QUDA_TWIST_DEG_DOUBLET},
                                                                  {"nondeg-doublet", QUDA_TWIST_NONDEG_DOUBLET},
                                                                  {"no", QUDA_TWIST_NO}};

  CLI::TransformPairs<QudaInverterType> inverter_type_map {{"cg", QUDA_CG_INVERTER},
                                                           {"bicgstab", QUDA_BICGSTAB_INVERTER},
                                                           {"gcr", QUDA_GCR_INVERTER},
                                                           {"pcg", QUDA_PCG_INVERTER},
                                                           {"mpcg", QUDA_MPCG_INVERTER},
                                                           {"mpbicgstab", QUDA_MPBICGSTAB_INVERTER},
                                                           {"mr", QUDA_MR_INVERTER},
                                                           {"sd", QUDA_SD_INVERTER},
                                                           {"eigcg", QUDA_EIGCG_INVERTER},
                                                           {"inc-eigcg", QUDA_INC_EIGCG_INVERTER},
                                                           {"gmresdr", QUDA_GMRESDR_INVERTER},
                                                           {"gmresdr-proj", QUDA_GMRESDR_PROJ_INVERTER},
                                                           {"gmresdr-sh", QUDA_GMRESDR_SH_INVERTER},
                                                           {"fgmresdr", QUDA_FGMRESDR_INVERTER},
                                                           {"mg", QUDA_MG_INVERTER},
                                                           {"bicgstab-l", QUDA_BICGSTABL_INVERTER},
                                                           {"cgne", QUDA_CGNE_INVERTER},
                                                           {"cgnr", QUDA_CGNR_INVERTER},
                                                           {"cg3", QUDA_CG3_INVERTER},
                                                           {"cg3ne", QUDA_CG3NE_INVERTER},
                                                           {"cg3nr", QUDA_CG3NR_INVERTER},
                                                           {"ca-cg", QUDA_CA_CG_INVERTER},
                                                           {"ca-cgne", QUDA_CA_CGNE_INVERTER},
                                                           {"ca-cgnr", QUDA_CA_CGNR_INVERTER},
                                                           {"ca-gcr", QUDA_CA_GCR_INVERTER}};

  CLI::TransformPairs<QudaPrecision> precision_map {{"double", QUDA_DOUBLE_PRECISION},
                                                    {"single", QUDA_SINGLE_PRECISION},
                                                    {"half", QUDA_HALF_PRECISION},
                                                    {"quarter", QUDA_QUARTER_PRECISION}};

  CLI::TransformPairs<QudaSolutionType> solution_type_map {{"mat", QUDA_MAT_SOLUTION},
                                                           {"mat-dag-mat", QUDA_MATDAG_MAT_SOLUTION},
                                                           {"mat-pc", QUDA_MATPC_SOLUTION},
                                                           {"mat-pc-dag", QUDA_MATPC_DAG_SOLUTION},
                                                           {"mat-pc-dag-mat-pc", QUDA_MATPCDAG_MATPC_SOLUTION}};

  CLI::TransformPairs<QudaEigType> eig_type_map {
    {"trlm", QUDA_EIG_TR_LANCZOS}, {"irlm", QUDA_EIG_IR_LANCZOS}, {"iram", QUDA_EIG_IR_ARNOLDI}};

  CLI::TransformPairs<QudaSolveType> solve_type_map {
    {"direct", QUDA_DIRECT_SOLVE},       {"direct-pc", QUDA_DIRECT_PC_SOLVE}, {"normop", QUDA_NORMOP_SOLVE},
    {"normop-pc", QUDA_NORMOP_PC_SOLVE}, {"normerr", QUDA_NORMERR_SOLVE},     {"normerr-pc", QUDA_NORMERR_PC_SOLVE}};

  CLI::TransformPairs<QudaEigSpectrumType> seig_pectrum_map {
    {"SR", QUDA_SPECTRUM_SR_EIG}, {"LR", QUDA_SPECTRUM_LR_EIG}, {"SM", QUDA_SPECTRUM_SM_EIG},
    {"LM", QUDA_SPECTRUM_LM_EIG}, {"SI", QUDA_SPECTRUM_SI_EIG}, {"LI", QUDA_SPECTRUM_LI_EIG}};

  CLI::TransformPairs<QudaFieldLocation> field_location_map {{"cpu", QUDA_CPU_FIELD_LOCATION},
                                                             {"host", QUDA_CPU_FIELD_LOCATION},
                                                             {"gpu", QUDA_CUDA_FIELD_LOCATION},
                                                             {"device", QUDA_CUDA_FIELD_LOCATION}};

  CLI::TransformPairs<QudaVerbosity> verbosity_map {
    {"silent", QUDA_SILENT}, {"summarize", QUDA_SUMMARIZE}, {"verbose", QUDA_VERBOSE}, {"debug", QUDA_DEBUG_VERBOSE}};

  CLI::TransformPairs<QudaMassNormalization> mass_normalization_map {{"kappa", QUDA_KAPPA_NORMALIZATION},
                                                                     {"mass", QUDA_MASS_NORMALIZATION},
                                                                     {"asym-mass", QUDA_ASYMMETRIC_MASS_NORMALIZATION}};

  CLI::TransformPairs<QudaMatPCType> matpc_type_map {{"even-even", QUDA_MATPC_EVEN_EVEN},
                                                     {"odd-odd", QUDA_MATPC_ODD_ODD},
                                                     {"even-even-asym", QUDA_MATPC_EVEN_EVEN_ASYMMETRIC},
                                                     {"odd-odd-asym", QUDA_MATPC_ODD_ODD_ASYMMETRIC}};

  CLI::TransformPairs<QudaReconstructType> reconstruct_type_map {{"18", QUDA_RECONSTRUCT_NO},
                                                                 {"13", QUDA_RECONSTRUCT_13},
                                                                 {"12", QUDA_RECONSTRUCT_12},
                                                                 {"9", QUDA_RECONSTRUCT_9},
                                                                 {"8", QUDA_RECONSTRUCT_8}};

  CLI::TransformPairs<QudaEigSpectrumType> eig_spectrum_map {
    {"SR", QUDA_SPECTRUM_SR_EIG}, {"LR", QUDA_SPECTRUM_LR_EIG}, {"SM", QUDA_SPECTRUM_SM_EIG},
    {"LM", QUDA_SPECTRUM_LM_EIG}, {"SI", QUDA_SPECTRUM_SI_EIG}, {"LI", QUDA_SPECTRUM_LI_EIG}};

  CLI::TransformPairs<QudaSetupType> setup_type_map {{"test", QUDA_TEST_VECTOR_SETUP}, {"null", QUDA_TEST_VECTOR_SETUP}};

  CLI::TransformPairs<QudaExtLibType> extlib_map {{"eigen", QUDA_EIGEN_EXTLIB}, {"magma", QUDA_MAGMA_EXTLIB}};

} // namespace

std::shared_ptr<QUDAApp> make_app(std::string app_description, std::string app_name)
{
  auto quda_app = std::make_shared<QUDAApp>(app_description, app_name);
  quda_app->option_defaults()->always_capture_default();

  quda_app->add_option("--alternative-reliable", alternative_reliable, "use alternative reliable updates");
  quda_app->add_option("--anisotropy", anisotropy, "Temporal anisotropy factor (default 1.0)");

  quda_app->add_option("--ca-basis-type", ca_basis, "The basis to use for CA-CG (default power)")
    ->transform(CLI::QUDACheckedTransformer(ca_basis_map));
  quda_app->add_option(
    "--cheby-basis-eig-max",
    ca_lambda_max, "Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG (default is to guess with power iterations)");
  quda_app->add_option("--cheby-basis-eig-min", ca_lambda_min,
                       "Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG (default 0)");
  quda_app->add_option("--clover-coeff", clover_coeff, "Clover coefficient")->capture_default_str();
  quda_app->add_option("--compute-clover", compute_clover,
                       "Compute the clover field or use random numbers (default false)");
  quda_app->add_option("--compute-fat-long", compute_fatlong,
                       "Compute the fat/long field or use random numbers (default false)");

  quda_app
    ->add_option("--contraction-type", contract_type,
                 "Whether to leave spin elemental open, or use a gamma basis and contract on spin (default open)")
    ->transform(CLI::QUDACheckedTransformer(contract_type_map));
  ;
  quda_app->add_flag("--dagger", dagger, "Set the dagger to 1 (default 0)");
  quda_app->add_option("--device", device, "Set the CUDA device to use (default 0, single GPU only)")
    ->check(CLI::Range(0, 16));

  quda_app->add_option("--dslash-type", dslash_type, "Set the dslash type")
    ->transform(CLI::QUDACheckedTransformer(dslash_type_map));

  quda_app->add_option("--epsilon", epsilon, "Twisted-Mass flavor twist of Dirac operator (default 0.01)");
  quda_app->add_option("--epsilon-naik", eps_naik, "Epsilon factor on Naik term (default 0.0, suggested non-zero -0.1)");

  quda_app
    ->add_option("--flavor", twist_flavor,
                 "Set the twisted mass flavor type (singlet (default), deg-doublet, nondeg-doublet)")
    ->transform(CLI::QUDACheckedTransformer(twist_flavor_type_map));
  ;
  quda_app->add_option("--gaussian-sigma", gaussian_sigma,
                       "Width of the Gaussian noise used for random gauge field contruction (default 0.2)");

  quda_app->add_option("--heatbath-beta", heatbath_beta_value, "Beta value used in heatbath test (default 6.2)");
  quda_app->add_option("--heatbath-coldstart", heatbath_coldstart,
                       "Whether to use a cold or hot start in heatbath test (default false)");
  quda_app->add_option("--heatbath-num-hb-per-step", heatbath_num_heatbath_per_step,
                       "Number of heatbath hits per heatbath step (default 5)");
  quda_app->add_option("--heatbath-num-or-per-step", heatbath_num_overrelax_per_step,
                       "Number of overrelaxation hits per heatbath step (default 5)");
  quda_app->add_option("--heatbath-num-steps", heatbath_num_steps,
                       "Number of measurement steps in heatbath test (default 10)");
  quda_app->add_option("--heatbath-warmup-steps", heatbath_warmup_steps,
                       "Number of warmup steps in heatbath test (default 10)");

  quda_app->add_option("--inv-type", inv_type, "The type of solver to use (default cg)")
    ->transform(CLI::QUDACheckedTransformer(inverter_type_map));
  quda_app->add_option("--inv-deflate", inv_deflate, "Deflate the inverter using the eigensolver");
  quda_app->add_option("--kappa", kappa, "Kappa of Dirac operator (default 0.12195122... [equiv to mass])");
  quda_app->add_option(
    "--laplace3D", laplace3D,
    "Restrict laplace operator to omit the t dimension (n=3), or include all dims (n=4) (default 4)");
  quda_app->add_option("--load-gauge", latfile, "Load gauge field \" file \" for the test (requires QIO)");
  quda_app->add_option("--Lsdim", Lsdim, "Set Ls dimension size(default 16)");
  quda_app->add_option("--mass", mass, "Mass of Dirac operator (default 0.1)");

  quda_app->add_option("--mass-normalization", normalization, "Mass normalization (kappa (default) / mass / asym-mass)")
    ->transform(CLI::QUDACheckedTransformer(mass_normalization_map));

  quda_app
    ->add_option("--matpc", matpc_type, "Matrix preconditioning type (even-even, odd-odd, even-even-asym, odd-odd-asym)")
    ->transform(CLI::QUDACheckedTransformer(matpc_type_map));
  quda_app->add_option("--msrc", Msrc,
                       "Used for testing non-square block blas routines where nsrc defines the other dimension");
  quda_app->add_option("--mu", mu, "Twisted-Mass chiral twist of Dirac operator (default 0.1)");
  quda_app->add_option("--m5", m5, "Mass of shift of five-dimensional Dirac operators (default -1.5)");
  quda_app->add_option("--b5", b5, "Mobius b5 parameter (default 1.5)");
  quda_app->add_option("--c5", c5, "Mobius c5 parameter (default 0.5)");
  quda_app->add_option("--multishift", multishift, "Whether to do a multi-shift solver test or not (default false)");
  quda_app->add_option("--ngcrkrylov", gcrNkrylov,
                       "The number of inner iterations to use for GCR, BiCGstab-l, CA-CG (default 10)");
  quda_app->add_option("--niter", niter, "The number of iterations to perform (default 10)");
  quda_app->add_option("--nsrc", Nsrc,
                       "How many spinors to apply the dslash to simultaneusly (experimental for staggered only)");

  quda_app->add_option("--pipeline", pipeline,
                       "The pipeline length for fused operations in GCR, BiCGstab-l (default 0, no pipelining)");

  // // precision options

  CLI::QUDACheckedTransformer prec_transform(precision_map);
  quda_app->add_option("--prec", prec, "Precision in GPU")->transform(prec_transform);
  quda_app->add_option("--prec-precondition", prec_precondition, "Preconditioner precision in GPU")->transform(prec_transform);
  ;
  quda_app->add_option("--prec-refine", prec_refinement_sloppy, "Sloppy precision for refinement in GPU")
    ->transform(prec_transform);
  ;
  quda_app->add_option("--prec-ritz", prec_ritz, "Eigenvector precision in GPU")->transform(prec_transform);
  ;
  quda_app->add_option("--prec-sloppy", prec_sloppy, "Sloppy precision in GPU")->transform(prec_transform);
  ;
  quda_app->add_option("--prec-null", prec_null, "Precison TODO")->transform(prec_transform);
  ;

  quda_app->add_option("--precon-type", precon_type, "The type of solver to use (default none (=unspecified)).")
    ->transform(CLI::QUDACheckedTransformer(inverter_type_map));

  CLI::TransformPairs<int> rank_order_map {{"col", 0}, {"row", 1}};
  quda_app
    ->add_option("--rank-order", rank_order,
                 "Set the [t][z][y][x] rank order as either column major (t fastest, default) or row major (x fastest)")
    ->transform(CLI::QUDACheckedTransformer(rank_order_map));

  quda_app->add_option("--recon", link_recon, "Link reconstruction type")
    ->transform(CLI::QUDACheckedTransformer(reconstruct_type_map));
  quda_app->add_option("--recon-precondition", link_recon_precondition, "Preconditioner link reconstruction type")
    ->transform(CLI::QUDACheckedTransformer(reconstruct_type_map));
  quda_app->add_option("--recon-sloppy", link_recon_sloppy, "Sloppy link reconstruction type")
    ->transform(CLI::QUDACheckedTransformer(reconstruct_type_map));

  quda_app->add_option("--reliable-delta", reliable_delta, "Set reliable update delta factor");
  quda_app->add_option("--save-gauge", gauge_outfile,
                       "Save gauge field \" file \" for the test (requires QIO, heatbath test only)");

  quda_app->add_option("--solution-pipeline", solution_accumulator_pipeline,
                       "The pipeline length for fused solution accumulation (default 0, no pipelining)");

  quda_app
    ->add_option(
      "--solution-type", solution_type,
      "The solution we desire (mat (default), mat-dag-mat, mat-pc, mat-pc-dag-mat-pc (default for multi-shift))")
    ->transform(CLI::QUDACheckedTransformer(solution_type_map));

  quda_app
    ->add_option("--solve-type", solve_type,
                 "The type of solve to do (direct, direct-pc, normop, normop-pc, normerr, normerr-pc)")
    ->transform(CLI::QUDACheckedTransformer(solve_type_map));
  quda_app
    ->add_option("--solver-ext-lib-type", solver_ext_lib, "Set external library for the solvers  (default Eigen library)")
    ->transform(CLI::QUDACheckedTransformer(extlib_map));

  quda_app->add_option("--tadpole-coeff", tadpole_factor,
                       "Tadpole coefficient for HISQ fermions (default 1.0, recommended [Plaq]^1/4)");

  quda_app->add_option("--tol", tol, "Set L2 residual tolerance");
  quda_app->add_option("--tolhq", tol_hq, "Set heavy-quark residual tolerance");
  quda_app->add_option(
    "--unit-gauge", unit_gauge,
    "Generate a unit valued gauge field in the tests. If false, a random gauge is generated (default false)");

  quda_app->add_option("--verbosity", verbosity, "The the verbosity on the top level of QUDA( default summarize)")
    ->transform(CLI::QUDACheckedTransformer(verbosity_map));
  quda_app->add_option("--verify", verify_results, "Verify the GPU results using CPU results (default true)");

  // lattice dimensions
  auto dimopt = quda_app->add_option("--dim", dim, "Set space-time dimension (X Y Z T)")->check(CLI::Range(1, 512));
  auto sdimopt = quda_app
                   ->add_option(
                     "--sdim",
                     [](CLI::results_t res) {
                       return CLI::detail::lexical_cast(res[0], xdim) && CLI::detail::lexical_cast(res[0], ydim)
                         && CLI::detail::lexical_cast(res[0], zdim);
                     },
                     "Set space dimension(X/Y/Z) size")
                   ->type_name("INT")
                   ->check(CLI::Range(1, 512));

  quda_app->add_option("--xdim", xdim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  quda_app->add_option("--ydim", ydim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  quda_app->add_option("--zdim", zdim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  quda_app->add_option("--tdim", tdim, "Set T dimension size(default 24)")->check(CLI::Range(1, 512))->excludes(dimopt);

  // multi-gpu partitioning

  quda_app->add_option(
    "--partition",
    [](CLI::results_t res) {
      int p;
      auto retval = CLI::detail::lexical_cast(res[0], p);
      for (int j = 0; j < 4; j++) {
        if (p & (1 << j)) {
          commDimPartitionedSet(j);
          dim_partitioned[j] = 1;
        }
      }
      return retval;
    },
    "Set the communication topology (X=1, Y=2, Z=4, T=8, and combinations of these)");

  auto gridsizeopt
    = quda_app
        ->add_option("--gridsize", gridsize_from_cmdline, "Set the grid size in all four dimension (default 1 1 1 1)")
        ->expected(4);
  quda_app->add_option("--xgridsize", grid_x, "Set grid size in X dimension (default 1)")->excludes(gridsizeopt);
  quda_app->add_option("--ygridsize", grid_y, "Set grid size in Y dimension (default 1)")->excludes(gridsizeopt);
  quda_app->add_option("--zgridsize", grid_z, "Set grid size in Z dimension (default 1)")->excludes(gridsizeopt);
  quda_app->add_option("--tgridsize", grid_t, "Set grid size in T dimension (default 1)")->excludes(gridsizeopt);

  return quda_app;
}

void add_eigen_option_group(std::shared_ptr<QUDAApp> quda_app)
{

  // Ooption group for Eigensolver related options
  auto opgroup = quda_app->add_option_group("Eigensolver", "Options controlling eigensolver");

  opgroup->add_option("--eig-amax", eig_amax, "The maximum in the polynomial acceleration")->check(CLI::PositiveNumber);
  opgroup->add_option("--eig-amin", eig_amin, "The minimum in the polynomial acceleration")->check(CLI::PositiveNumber);

  opgroup->add_option("--eig-ARPACK-logfile", eig_arpack_logfile, "The filename storing the log from arpack");
  opgroup->add_option("--eig-QUDA-logfile", eig_QUDA_logfile,
                      "The filename storing the stdout from the QUDA eigensolver");
  opgroup->add_option("--eig-arpack-check", eig_arpack_check,
                      "Cross check the device data against ARPACK (requires ARPACK, default false)");
  opgroup->add_option(
    "--eig-check-interval", eig_check_interval,
    "Perform a convergence check every nth restart/iteration in the IRAM,IRLM/lanczos,arnoldi eigensolver");
  opgroup->add_option("--eig-compute-svd", eig_compute_svd,
                      "Solve the MdagM problem, use to compute SVD of M (default false)");
  opgroup->add_option("--eig-max-restarts", eig_max_restarts, "Perform n iterations of the restart in the eigensolver");
  opgroup->add_option("--eig-nConv", eig_nConv, "The number of converged eigenvalues requested");
  opgroup->add_option("--eig-nEv", eig_nEv, "The size of eigenvector search space in the eigensolver");
  opgroup->add_option("--eig-nKr", eig_nKr, "The size of the Krylov subspace to use in the eigensolver");
  opgroup->add_option("--eig-batched-rotate", eig_batched_rotate,
                      "The maximum number of extra eigenvectors the solver may allocate to perform a Ritz rotation.");
  opgroup->add_option("--eig-poly-deg", eig_poly_deg, "TODO");
  opgroup->add_option(
    "--eig-require-convergence",
    eig_require_convergence, "If true, the solver will error out if convergence is not attained. If false, a warning will be given (default true)");
  opgroup->add_option("--eig-save-vec", eig_vec_outfile, "Save eigenvectors to <file> (requires QIO)");
  opgroup->add_option("--eig-load-vec", eig_vec_infile, "Load eigenvectors to <file> (requires QIO)")
    ->check(CLI::ExistingFile);

  opgroup
    ->add_option("--eig-spectrum", eig_spectrum,
                 "The spectrum part to be calulated. S=smallest L=largest R=real M=modulus I=imaginary")
    ->transform(CLI::QUDACheckedTransformer(eig_spectrum_map));
  opgroup->add_option("--eig-tol", eig_tol, "The tolerance to use in the eigensolver");

  opgroup->add_option("--eig-type", eig_type, "The type of eigensolver to use (default trlm)")
    ->transform(CLI::QUDACheckedTransformer(eig_type_map));

  opgroup->add_option("--eig-use-dagger", eig_use_dagger,
                      "Solve the Mdag  problem instead of M (MMdag if eig-use-normop == true) (default false)");
  opgroup->add_option("--eig-use-normop", eig_use_normop,
                      "Solve the MdagM problem instead of M (MMdag if eig-use-dagger == true) (default false)");
  opgroup->add_option("--eig-use-poly-acc", eig_use_poly_acc, "Use Chebyshev polynomial acceleration in the eigensolver");
}

void add_deflation_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  auto opgroup = quda_app->add_option_group("Deflation", "Options controlling deflation");

  opgroup
    ->add_option("--df-deflation-grid", deflation_grid,
                 "Set maximum number of cycles needed to compute eigenvectors(default 1)")
    ->check(CLI::PositiveNumber);
  opgroup
    ->add_option(
      "--df-eigcg-max-restarts",
      eigcg_max_restarts, "Set how many iterative refinement cycles will be solved with eigCG within a single physical right hand site solve (default 4)")
    ->check(CLI::PositiveNumber);
  ;
  opgroup->add_option("--df-ext-lib-type", deflation_ext_lib,
                      "Set external library for the deflation methods  (default Eigen library)");
  opgroup->add_option("--df-location-ritz", location_ritz,
                      "Set memory location for the ritz vectors  (default cuda memory location)");
  opgroup->add_option("--df-max-restart-num", max_restart_num,
                      "Set maximum number of the initCG restarts in the deflation stage (default 3)");
  opgroup->add_option("--df-max-search-dim", max_search_dim, "Set the size of eigenvector search space (default 64)");
  opgroup->add_option("--df-mem-type-ritz", mem_type_ritz,
                      "Set memory type for the ritz vectors  (default device memory type)");
  opgroup->add_option("--df-nev", nev, "Set number of eigenvectors computed within a single solve cycle (default 8)");
  opgroup->add_option("--df-tol-eigenval", eigenval_tol, "Set maximum eigenvalue residual norm (default 1e-1)");
  opgroup->add_option("--df-tol-inc", inc_tol,
                      "Set tolerance for the subsequent restarts in the initCG solver  (default 1e-2)");
  opgroup->add_option("--df-tol-restart", tol_restart,
                      "Set tolerance for the first restart in the initCG solver(default 5e-5)");
}

void add_multigrid_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  auto opgroup = quda_app->add_option_group("MultiGrid", "Options controlling deflation");

  // MWTODO: clean this up - code duplication

  auto solve_type_transform = CLI::QUDACheckedTransformer(solve_type_map);

  // TODO
  quda_app->add_mgoption(
    opgroup, "--mg-block-size", geo_block_size, CLI::Validator(),
    "Set the geometric block size for the each multigrid levels transfer operator (default 4 4 4 4)");
  quda_app->add_mgoption(opgroup, "--mg-coarse-solve-type", coarse_solve_type, solve_type_transform,
                         "The type of solve to do on each level (direct, direct-pc) (default = solve_type)");

  auto solver_trans = CLI::QUDACheckedTransformer(inverter_type_map);
  quda_app->add_mgoption(opgroup, "--mg-coarse-solver", coarse_solver, solver_trans,
                         "The solver to wrap the V cycle on each level (default gcr, only for levels 1+)");

  quda_app->add_mgoption(opgroup, "--mg-coarse-solver-ca-basis-size", coarse_solver_ca_basis_size, CLI::PositiveNumber,
                         "The basis size to use for CA-CG setup of multigrid (default 4)");

  quda_app->add_mgoption(opgroup, "--mg-coarse-solver-ca-basis-type", coarse_solver_ca_basis,
                         CLI::QUDACheckedTransformer(ca_basis_map),
                         "The basis to use for CA-CG setup of multigrid(default power)");
  quda_app->add_mgoption(opgroup, "--mg-coarse-solver-cheby-basis-eig-max", coarse_solver_ca_lambda_max,
                         CLI::PositiveNumber,
                         "Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG in setup of multigrid "
                         "(default is to guess with power iterations)");
  quda_app->add_mgoption(
    opgroup, "--mg-coarse-solver-cheby-basis-eig-min", coarse_solver_ca_lambda_min, CLI::PositiveNumber,
    "Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default 0)");
  quda_app->add_mgoption(opgroup, "--mg-coarse-solver-maxiter", coarse_solver_maxiter, CLI::PositiveNumber,
                         "The coarse solver maxiter for each level (default 100)");
  quda_app->add_mgoption(opgroup, "--mg-coarse-solver-tol", coarse_solver_tol, CLI::PositiveNumber,
                         "The coarse solver tolerance for each level (default 0.25, only for levels 1+)");
  quda_app->add_mgoption(opgroup, "--mg-eig", mg_eig, CLI::Validator(),
                         "Use the eigensolver on this level (default false)");
  quda_app->add_mgoption(opgroup, "--mg-eig-amax", mg_eig_amax, CLI::PositiveNumber,
                         "The maximum in the polynomial acceleration (default 4.0)");
  quda_app->add_mgoption(opgroup, "--mg-eig-amin", mg_eig_amin, CLI::PositiveNumber,
                         "The minimum in the polynomial acceleration (default 0.1)");
  quda_app->add_mgoption(
    opgroup, "--mg-eig-check-interval", mg_eig_check_interval, CLI::Validator(),
    "Perform a convergence check every nth restart/iteration (only used in Implicit Restart types)");
  quda_app->add_option("--mg-eig-coarse-guess", mg_eig_coarse_guess,
                       "If deflating on the coarse grid, optionally use an initial guess (default = false)");
  quda_app->add_option("--mg-eig-preserve-deflation", mg_eig_preserve_deflation,
                       "If the multigrid operator is updated, preserve generated deflation space (default = false)");
  quda_app->add_mgoption(opgroup, "--mg-eig-max-restarts", mg_eig_max_restarts, CLI::PositiveNumber,
                         "Perform a maximun of n restarts in eigensolver (default 100)");
  quda_app->add_mgoption(opgroup, "--mg-eig-nEv", mg_eig_nEv, CLI::Validator(),
                         "The size of eigenvector search space in the eigensolver");
  quda_app->add_mgoption(opgroup, "--mg-eig-nKr", mg_eig_nKr, CLI::Validator(),
                         "The size of the Krylov subspace to use in the eigensolver");
  quda_app->add_mgoption(
    opgroup, "--mg-eig-batched-rotate", mg_eig_batched_rotate, CLI::Validator(),
    "The maximum number of extra eigenvectors the solver may allocate to perform a Ritz rotation.");
  quda_app->add_mgoption(opgroup, "--mg-eig-poly-deg", mg_eig_poly_deg, CLI::PositiveNumber,
                         "Set the degree of the Chebyshev polynomial (default 100)");
  quda_app->add_mgoption(
    opgroup, "--mg-eig-require-convergence", mg_eig_require_convergence,
    CLI::Validator(), "If true, the solver will error out if convergence is not attained. If false, a warning will be given (default true)");

  quda_app->add_mgoption(
    opgroup, "--mg-eig-spectrum", mg_eig_spectrum, CLI::QUDACheckedTransformer(eig_spectrum_map),
    "The spectrum part to be calulated. S=smallest L=largest R=real M=modulus I=imaginary (default SR)");
  quda_app->add_mgoption(opgroup, "--mg-eig-tol", mg_eig_tol, CLI::PositiveNumber,
                         "The tolerance to use in the eigensolver (default 1e-6)");

  quda_app->add_mgoption(opgroup, "--mg-eig-type", mg_eig_type, CLI::QUDACheckedTransformer(eig_type_map),
                         "The type of eigensolver to use (default trlm)");
  quda_app->add_mgoption(opgroup, "--mg-eig-use-dagger", mg_eig_use_dagger, CLI::Validator(),
                         "Solve the MMdag problem instead of M (MMdag if eig-use-normop == true) (default false)");
  quda_app->add_mgoption(opgroup, "--mg-eig-use-normop", mg_eig_use_normop, CLI::Validator(),
                         "Solve the MdagM problem instead of M (MMdag if eig-use-dagger == true) (default false)");
  quda_app->add_mgoption(opgroup, "--mg-eig-use-poly-acc", mg_eig_use_poly_acc, CLI::Validator(),
                         "Use Chebyshev polynomial acceleration in the eigensolver (default true)");
  opgroup->add_option(
    "--mg-generate-all-levels",
    generate_all_levels, "true=generate null-space on all levels, false=generate on level 0 and create other levels from that (default true)");
  opgroup->add_option("--mg-generate-nullspace", generate_nullspace,
                      "Generate the null-space vector dynamically (default true, if set false and mg-load-vec isn't "
                      "set, creates free-field null vectors)");
  opgroup->add_option("--mg-levels", mg_levels, "The number of multigrid levels to do (default 2)");

  // TODO
  quda_app->add_mgoption(opgroup, "--mg-load-vec", mg_vec_infile, CLI::Validator(),
                         "Load the vectors <file> for the multigrid_test (requires QIO)");
  quda_app->add_mgoption(opgroup, "--mg-save-vec", mg_vec_outfile, CLI::Validator(),
                         "Save the generated null-space vectors <file> from the multigrid_test (requires QIO)");

  opgroup->add_option(
    "--mg-low-mode-check", low_mode_check,
    "Measure how well the null vector subspace overlaps with the low eigenmode subspace (default false)");
  quda_app->add_mgoption(opgroup, "--mg-mu-factor", mu_factor, CLI::Validator(),
                         "Set the multiplicative factor for the twisted mass mu parameter on each level (default 1)");
  quda_app->add_mgoption(opgroup, "--mg-n-block-ortho", n_block_ortho, CLI::PositiveNumber,
                         "The number of times to run Gram-Schmidt during block orthonormalization (default 1)");
  quda_app->add_mgoption(opgroup, "--mg-nu-post", nu_post, CLI::PositiveNumber,
                         "The number of post-smoother applications to do at a given multigrid level (default 2)");
  quda_app->add_mgoption(opgroup, "--mg-nu-pre", nu_pre, CLI::PositiveNumber,
                         "The number of pre-smoother applications to do at a given multigrid level (default 2)");
  quda_app->add_mgoption(opgroup, "--mg-nvec", nvec, CLI::PositiveNumber,
                         "Number of null-space vectors to define the multigrid transfer operator on a given level");
  opgroup->add_option("--mg-oblique-proj-check", oblique_proj_check,
                      "Measure how well the null vector subspace adjusts the low eigenmode subspace (default false)");
  opgroup->add_option("--mg-omega", omega,
                      "The over/under relaxation factor for the smoother of multigrid (default 0.85)");
  opgroup->add_option("--mg-post-orth", post_orthonormalize,
                      "If orthonormalize the vector after inverting in the setup of multigrid (default true)");
  opgroup->add_option("--mg-pre-orth", pre_orthonormalize,
                      "If orthonormalize the vector before inverting in the setup of multigrid (default false)");

  quda_app->add_mgoption(opgroup, "--mg-schwarz-cycle", schwarz_cycle, CLI::PositiveNumber,
                         "The number of Schwarz cycles to apply per smoother application (default=1)");
  quda_app->add_mgoption(
    opgroup, "--mg-schwarz-type", schwarz_type, CLI::Validator(),
    "Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver) (default false)");
  quda_app->add_mgoption(opgroup, "--mg-setup-ca-basis-size", setup_ca_basis_size, CLI::PositiveNumber,
                         "The basis size to use for CA-CG setup of multigrid (default 4)");
  quda_app->add_mgoption(opgroup, "--mg-setup-ca-basis-type", setup_ca_basis, CLI::QUDACheckedTransformer(ca_basis_map),
                         "The basis to use for CA-CG setup of multigrid(default power)");
  quda_app->add_mgoption(opgroup, "--mg-setup-cheby-basis-eig-max", setup_ca_lambda_max, CLI::PositiveNumber,
                         "Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG in setup of multigrid "
                         "(default is to guess with power iterations)");
  quda_app->add_mgoption(
    opgroup, "--mg-setup-cheby-basis-eig-min", setup_ca_lambda_min, CLI::PositiveNumber,
    "Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default 0)");
  quda_app->add_mgoption(opgroup, "--mg-setup-inv", setup_inv, solver_trans,
                         "The inverter to use for the setup of multigrid (default bicgstab)");
  quda_app->add_mgoption(opgroup, "--mg-setup-iters", num_setup_iter, CLI::PositiveNumber,
                         "The number of setup iterations to use for the multigrid (default 1)");

  quda_app->add_mgoption(opgroup, "--mg-setup-location", setup_location, CLI::QUDACheckedTransformer(field_location_map),
                         "The location where the multigrid setup will be computed (default cuda)");
  quda_app->add_mgoption(
    opgroup, "--mg-setup-maxiter", setup_maxiter, CLI::Validator(),
    "The maximum number of solver iterations to use when relaxing on a null space vector (default 500)");
  quda_app->add_mgoption(
    opgroup, "--mg-setup-maxiter-refresh", setup_maxiter_refresh, CLI::Validator(),
    "The maximum number of solver iterations to use when refreshing the pre-existing null space vectors (default 100)");
  quda_app->add_mgoption(opgroup, "--mg-setup-tol", setup_tol, CLI::Validator(),
                         "The tolerance to use for the setup of multigrid (default 5e-6)");

  opgroup->add_option("--mg-setup-type", setup_type, "The type of setup to use for the multigrid (default null)")
    ->transform(CLI::QUDACheckedTransformer(setup_type_map));
  quda_app->add_mgoption(opgroup, "--mg-smoother", smoother_type, solver_trans,
                         "The smoother to use for multigrid (default mr)");

  CLI::QUDACheckedTransformer prec_transform(precision_map);
  opgroup
    ->add_option("--mg-smoother-halo-prec", smoother_halo_prec,
                 "The smoother halo precision (applies to all levels - defaults to null_precision)")
    ->transform(prec_transform);

  quda_app->add_mgoption(opgroup, "--mg-smoother-solve-type", smoother_solve_type, solve_type_transform,
                         "The type of solve to do in smoother (direct, direct-pc (default) )");
  quda_app->add_mgoption(opgroup, "--mg-smoother-tol", smoother_tol, CLI::Validator(),
                         "The smoother tolerance to use for each multigrid (default 0.25)");
  quda_app->add_mgoption(opgroup, "--mg-solve-location", solver_location, CLI::QUDACheckedTransformer(field_location_map),
                         "The location where the multigrid solver will run (default cuda)");

  quda_app->add_mgoption(opgroup, "--mg-verbosity", mg_verbosity, CLI::QUDACheckedTransformer(verbosity_map),
                         "The verbosity to use on each level of the multigrid (default summarize)");
}
