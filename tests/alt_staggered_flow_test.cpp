#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <timer.h>
#include <util_quda.h>
#include <host_utils.h>
#include <gauge_tools.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <unitarization_links.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include <misc.h>

#include <comm_quda.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
using namespace quda;
// Smearing variables
double gauge_smear_rho = 0.1;
double gauge_smear_epsilon = 0.1;
double gauge_smear_alpha = 0.6;
double gauge_smear_alpha1 = 0.75;
double gauge_smear_alpha2 = 0.6;
double gauge_smear_alpha3 = 0.3;
int gauge_smear_steps = 50;
int gauge_n_save = 3;
int hier_threshold = 6;
QudaGaugeSmearType gauge_smear_type = QUDA_GAUGE_SMEAR_STOUT;
int gauge_smear_dir_ignore = -1;
int measurement_interval = 5;
bool su_project = true;
bool has_naik = false;
int n_naiks = 1;

static double unitarize_eps = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only = false;
static double svd_rel_error = 1e-4;
static double svd_abs_error = 1e-4;
static double max_allowed_error = 1e-11;
void *milc_sitelink = nullptr;

// storage for CPU reference fat and long links w/zero Naik
void *fat_reflink[4] = {nullptr, nullptr, nullptr, nullptr};
void *long_reflink[4] = {nullptr, nullptr, nullptr, nullptr};

// storage for CPU reference fat and long links w/non-zero Naik
void *fat_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
void *long_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};

// Paths for step 1:
void *vlink = nullptr;
void *wlink = nullptr;

// Paths for step 2:
void *fatlink = nullptr;
void *longlink = nullptr;

// Place to accumulate Naiks
void *fatlink_eps = nullptr;
void *longlink_eps = nullptr;

void *qdp_sitelink[4] = {nullptr, nullptr, nullptr, nullptr};
void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
void *qdp_fatlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
void *qdp_longlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};


void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);

  // Specific test
  printfQuda("\n%s smearing\n", get_gauge_smear_str(gauge_smear_type));
  switch (gauge_smear_type) {
  case QUDA_GAUGE_SMEAR_APE: printfQuda(" - alpha %f\n", gauge_smear_alpha); break;
  case QUDA_GAUGE_SMEAR_STOUT: printfQuda(" - rho %f\n", gauge_smear_rho); break;
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
    printfQuda(" - rho %f\n", gauge_smear_rho);
    printfQuda(" - epsilon %f\n", gauge_smear_epsilon);
    break;
  case QUDA_GAUGE_SMEAR_HYP:
    printfQuda(" - alpha1 %f\n", gauge_smear_alpha1);
    printfQuda(" - alpha2 %f\n", gauge_smear_alpha2);
    printfQuda(" - alpha3 %f\n", gauge_smear_alpha3);
    break;
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: printfQuda(" - epsilon %f\n", gauge_smear_epsilon); break;
  default: errorQuda("Undefined test type %d given", test_type);
  }
  printfQuda(" - smearing steps %d\n", gauge_smear_steps);
  printfQuda(" - smearing ignore direction %d\n", gauge_smear_dir_ignore);
  printfQuda(" - Measurement interval %d\n", measurement_interval);
  
  printfQuda(" - has_naik %s\n", has_naik ? "true" : "false");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

void add_hisq_option_group(std::shared_ptr<QUDAApp> quda_app)
{
    //   // Option group for SU(3) related options
  auto opgroup = quda_app->add_option_group("HISQ", "Options controlling HISQ parameters");
  opgroup
    ->add_option(
      "--has_naik",
      has_naik, "has naik (for charm)");

}

int main(int argc, char **argv)
{
    
  auto app = make_app();
  add_su3_option_group(app);
  add_eigen_option_group(app);
  add_hisq_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  setStaggeredGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);
  gauge_param.reconstruct = link_recon;
  // All user inputs are now defined
  display_test_info();
    if (has_naik) {
    eps_naik = -0.03; // semi-arbitrary
    n_naiks = 2;
    } else {
    eps_naik = 0.0;
    n_naiks = 1;
    }
  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();
printfQuda("HII1\n");

for (int i = 0; i < 4; i++) qdp_sitelink[i] = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);

    // Note: this could be replaced with loading a gauge field
    createSiteLinkCPU(qdp_sitelink, gauge_param.cpu_prec, SiteLinkType::SITELINK_PHASE_NO);
    
    for (int i = 0; i < 4; i++) {
      qdp_fatlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      if (n_naiks > 1) {
        qdp_fatlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
        qdp_longlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      }
    }

    milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);

    reorderQDPtoMILC(milc_sitelink, qdp_sitelink, V, gauge_site_size, gauge_param.cuda_prec, gauge_param.cpu_prec);

    // Paths for step 1:
    vlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // V links
    wlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // W links

    // Paths for step 2:
    fatlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);  // final fat ("X") links
    longlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // final long links

    // Place to accumulate Naiks
    if (n_naiks > 1) {
      fatlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);  // epsilon fat links
      longlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // epsilon long naiks
    }

  double act_path[6];
  set_act_path(act_path, 0);
  computeKSLinkQuda(vlink, nullptr, wlink, milc_sitelink, act_path, &gauge_param);
    
  if (n_naiks > 1) {
    // Create Naiks, 3rd path table set
    set_act_path(act_path, 2);
    computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_path, &gauge_param);

    // Rescale+copy Naiks into Naik field
    cpu_axy(gauge_param.cuda_prec, eps_naik, fatlink, fatlink_eps, V * 4 * gauge_site_size);
    cpu_axy(gauge_param.cuda_prec, eps_naik, longlink, longlink_eps, V * 4 * gauge_site_size);
  } else {
        memset(fatlink, 0, V * 4 * gauge_site_size * gauge_param.cuda_prec);
        memset(longlink, 0, V * 4 * gauge_site_size * gauge_param.cuda_prec);
      }

      // Create X and long links, 2nd path table set
    set_act_path(act_path, 1);
      computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_path, &gauge_param);

      if (n_naiks > 1) {
        // Add into Naik field
        cpu_xpy(gauge_param.cuda_prec, fatlink, fatlink_eps, V * 4 * gauge_site_size);
        cpu_xpy(gauge_param.cuda_prec, longlink, longlink_eps, V * 4 * gauge_site_size);
      }


  loadFatLongGaugeQuda(fatlink, longlink, gauge_param);
    
  quda::host_timer_t host_timer, host_safe_timer, host_hier_timer, host_fwd_timer;

  QudaGaugeObservableParam *obs_param = new QudaGaugeObservableParam[gauge_smear_steps / measurement_interval + 1];
  for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
    obs_param[i] = newQudaGaugeObservableParam();
    obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
    obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
    obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  }

  QudaFermMeasurements ferm_meas = newQudaFermMeasurements();

  // We here set all the problem parameters for all possible smearing types.
  QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
  smear_param.smear_type = gauge_smear_type;
  smear_param.n_steps = gauge_smear_steps;
  smear_param.adj_n_save = gauge_n_save;
  smear_param.hier_threshold = hier_threshold;
  smear_param.meas_interval = measurement_interval;
  smear_param.alpha = gauge_smear_alpha;
  smear_param.rho = gauge_smear_rho;
  smear_param.epsilon = gauge_smear_epsilon;
  smear_param.alpha1 = gauge_smear_alpha1;
  smear_param.alpha2 = gauge_smear_alpha2;
  smear_param.alpha3 = gauge_smear_alpha3;
  smear_param.dir_ignore = gauge_smear_dir_ignore;


  quda::ColorSpinorField check,check_safe,check_hier,check_fwd;

  QudaInvertParam invParam = newQudaInvertParam();
    QudaMultigridParam mg_param;
    QudaInvertParam mg_invParam;
    QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
    QudaEigParam eig_param;
    bool use_split_grid = false;
    bool use_multi_src = false;
  setStaggeredInvertParam(invParam);
  if (!inv_deflate)
  invParam.eig_param = nullptr;

  setStaggeredInvertParam(invParam);

  // reset lambda_max if we're doing a testing loop to ensure correct lambma_max
  if (enable_testing) invParam.ca_lambda_max = -1.0;

  logQuda(QUDA_SUMMARIZE, "Solution = %s, Solve = %s, Solver = %s, Sloppy precision = %s\n",
          get_solution_str(invParam.solution_type), get_solve_str(invParam.solve_type),
          get_solver_str(invParam.inv_type), get_prec_str(invParam.cuda_prec_sloppy));

  // params related to split grid.
  for (int i = 0; i < 4; i++) invParam.split_grid[i] = grid_partition[i];
  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  use_split_grid = num_sub_partition > 1;
  use_multi_src = use_split_grid || (Nsrc_tile > 1);

  // Setup the multigrid preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    if (use_split_grid) { errorQuda("Split grid does not work with MG yet."); }
    mg_preconditioner = newMultigridQuda(&mg_param);
    invParam.preconditioner = mg_preconditioner;

    printfQuda("MG Setup Done: %g secs, %g Gflops\n", mg_param.invert_param->secs,
               mg_param.invert_param->gflops / mg_param.invert_param->secs);
    if (mg_param.invert_param->energy > 0) {
      printfQuda("Energy = %g J, Mean power = %g W, mean temp = %g C, mean clock = %f\n", mg_param.invert_param->energy,
                 mg_param.invert_param->power, mg_param.invert_param->temp, mg_param.invert_param->clock);
    }
  }
    //multishift: same linear system, different masses (charm + strange ie)
    //SET UP INV PARAM END
  if (Nsrc > QUDA_MAX_MULTI_SRC)
    errorQuda("Nsrc = %d which is great than QUDA_MAX_MULTI_SRC = %d\n", Nsrc, QUDA_MAX_MULTI_SRC);

  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &invParam, &gauge_param);
    
  std::vector<quda::ColorSpinorField> in_raw(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> in(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> out(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> out_flowed(Nsrc,cs_param);


  std::vector<void *> in_raw_ptr(Nsrc);
  std::vector<void *> in_ptr(Nsrc);
  std::vector<void *> out_ptr(Nsrc);
  std::vector<void *> out_flowed_ptr(Nsrc);

    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    }
    
    
    ferm_meas.meas_n = 5;
    std::vector<std::vector<std::complex<double>>> ppb;
    void* ptr_ppb = &ppb;
    void** data_ppb = &ptr_ppb;
    ferm_meas.ppb = data_ppb;

    printfQuda("At start ppb has %i elements\n",ppb.size());
    //simulates what user might do from external library

// Prepare rng, fill host spinors with random numbers
  //-----------------------------------------------------------------------------------

  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  // Create a temporary spinor just to seed the rng
  quda::ColorSpinorField tmp(cs_param);
  quda::RNG rng(tmp, 1234);
  tmp = quda::ColorSpinorField();

  for (int n = 0; n < Nsrc; n++) {
    // Populate the host spinor with random numbers.
    quda::spinorNoise(in_raw[n], rng, QUDA_NOISE_GAUSS);
    in_raw_ptr[n] = in_raw[n].data();
    in_ptr[n] = in[n].data();
    out_ptr[n] = out[n].data();
    out_flowed_ptr[n] = out_flowed[n].data();
  }

  performAdjGFlowHier(in_ptr.data(),in_raw_ptr.data(), &invParam, &smear_param, &ferm_meas, Nsrc);

  printfQuda("At end ppb has %i elements\n",ppb.size());

  in_raw = {};
  in = {};
  out = {};
  out_flowed = {};

  freeGaugeQuda();
  endQuda();

  finalizeComms();
  return 0;
}
