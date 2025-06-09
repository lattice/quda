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

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

void add_su3_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  CLI::TransformPairs<QudaGaugeSmearType> gauge_smear_type_map {{"ape", QUDA_GAUGE_SMEAR_APE},
                                                                {"stout", QUDA_GAUGE_SMEAR_STOUT},
                                                                {"ovrimp-stout", QUDA_GAUGE_SMEAR_OVRIMP_STOUT},
                                                                {"hyp", QUDA_GAUGE_SMEAR_HYP},
                                                                {"wilson", QUDA_GAUGE_SMEAR_WILSON_FLOW},
                                                                {"symanzik", QUDA_GAUGE_SMEAR_SYMANZIK_FLOW}};

  // Option group for SU(3) related options
  auto opgroup = quda_app->add_option_group("SU(3)", "Options controlling SU(3) tests");

  opgroup
    ->add_option(
      "--su3-smear-type",
      gauge_smear_type, "The type of action to use in the smearing. Options: APE, Stout, Over Improved Stout, HYP, Wilson Flow, Symanzik Flow (default stout)")
    ->transform(CLI::QUDACheckedTransformer(gauge_smear_type_map));
  ;
  opgroup->add_option("--su3-smear-alpha", gauge_smear_alpha, "alpha coefficient for APE smearing (default 0.6)");

  opgroup->add_option("--su3-smear-rho", gauge_smear_rho,
                      "rho coefficient for Stout and Over-Improved Stout smearing (default 0.1)");

  opgroup->add_option("--su3-smear-epsilon", gauge_smear_epsilon,
                      "epsilon coefficient for Over-Improved Stout smearing or Wilson flow (default 0.1)");

  opgroup->add_option("--su3-smear-alpha1", gauge_smear_alpha1, "alpha1 coefficient for HYP smearing (default 0.75)");
  opgroup->add_option("--su3-smear-alpha2", gauge_smear_alpha2, "alpha2 coefficient for HYP smearing (default 0.6)");
  opgroup->add_option("--su3-smear-alpha3", gauge_smear_alpha3, "alpha3 coefficient for HYP smearing (default 0.3)");

  opgroup->add_option(
    "--su3-smear-dir-ignore", gauge_smear_dir_ignore,
    "Direction to be ignored by the smearing, negative value means decided by --su3-smear-type (default -1)");

  opgroup->add_option("--su3-smear-steps", gauge_smear_steps, "The number of smearing steps to perform (default 50)");
    
  opgroup->add_option("--su3-adj-gauge-nsave", gauge_n_save, "The number of gauge steps to save for hierarchical adj grad flow");
    
  opgroup->add_option("--su3-hier-threshold", hier_threshold, "Minimum threshold for hierarchical adj grad flow");

  opgroup->add_option("--su3-measurement-interval", measurement_interval,
                      "Measure the field energy and/or topological charge every Nth step (default 5) ");

  opgroup->add_option("--su3-project", su_project,
                      "Project smeared gauge onto su3 manifold at measurement interval (default true)");
}

GaugeField cpuFatQDP = {};
GaugeField cpuLongQDP = {};
GaugeField cpuFatMILC = {};
GaugeField cpuLongMILC = {};

int main(int argc, char **argv)
{
    
  auto app = make_app();
  add_su3_option_group(app);
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  // add_testing_option_group(app);

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

  // All user inputs are now defined
  display_test_info();

  // void *gauge[4], *new_gauge[4];

  // for (int dir = 0; dir < 4; dir++) {
  //   gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  //   new_gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  // }

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();

  // // Load the gauge field to the device
  // constructHostGaugeField(gauge, gauge_param, argc, argv);
  // loadGaugeQuda((void *)gauge, &gauge_param);
  // saveGaugeQuda(new_gauge, &gauge_param);
  // start the timer
  quda::host_timer_t host_timer, host_safe_timer, host_hier_timer, host_fwd_timer;
printfQuda("HIIII\n");
  // The commented out section is all geared towards gauge observables, so unlikely to be needed for now
  // // Prepare various perf info
  // long long flops_plaquette = 6ll * 597 * V;
  // long long flops_ploop = 198ll * V + 6 * V / gauge_param.X[3];

  // // Prepare a gauge observable struct
  // QudaGaugeObservableParam param = newQudaGaugeObservableParam();

  // The user may specify which measurements they wish to perform/omit
  // using the QudaGaugeObservableParam struct, and whether or not to
  // perform suN projection at each measurement step. We recommend that
  // users perform suN projection.
  // A unique observable param struct is constructed for each measurement.

  // Gauge Smearing Routines
  //---------------------------------------------------------------------------
  // Stout smearing should be equivalent to APE smearing
  // on D dimensional lattices for rho = alpha/2*(D-1).
  // Typical values for
  // APE: alpha=0.6
  // Stout: rho=0.1
  // Over Improved Stout: rho=0.08, epsilon=-0.25
  //
  // Typically, the user will use smearing for Q charge data only, so
  // we hardcode to compute Q only and not the plaquette. Users may
  // of course set these as they wish.  SU(N) projection su_project=true is recommended.
  QudaGaugeObservableParam *obs_param = new QudaGaugeObservableParam[gauge_smear_steps / measurement_interval + 1];
  for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
    obs_param[i] = newQudaGaugeObservableParam();
    obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
    obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
    obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  }

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
  //if !inv_deflate
  invParam.eig_param = nullptr;

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuParam(gauge_param);
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuIn = GaugeField(cpuParam);
  cpuFatQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuFatMILC = GaugeField(cpuParam);

  cpuParam.link_type = QUDA_ASQTAD_LONG_LINKS;
  cpuParam.nFace = 3;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuLongQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuLongMILC = GaugeField(cpuParam);

  void *qdp_inlink[4] = {cpuIn.data(0), cpuIn.data(1), cpuIn.data(2), cpuIn.data(3)};
  void *qdp_fatlink[4] = {cpuFatQDP.data(0), cpuFatQDP.data(1), cpuFatQDP.data(2), cpuFatQDP.data(3)};
  void *qdp_longlink[4] = {cpuLongQDP.data(0), cpuLongQDP.data(1), cpuLongQDP.data(2), cpuLongQDP.data(3)};
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, 0, nullptr, true);

  // Reorder gauge fields to MILC order
  cpuFatMILC = cpuFatQDP;
  cpuLongMILC = cpuLongQDP;

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  freeGaugeQuda();

  void *fatlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  void *longlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  void *milc_sitelink;
  milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  void *longlink_ptr = longlink;
    
  double act_path[6];
  set_act_path(act_path, 0);

  GaugeFieldParam InParam(gauge_param);
  InParam.order = QUDA_MILC_GAUGE_ORDER;
  InParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  InParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuInNew = GaugeField(InParam);

  InParam.gauge = longlink;
  InParam.create = QUDA_REFERENCE_FIELD_CREATE;
  GaugeField ULink = GaugeField(InParam);

  InParam.gauge = fatlink;
  InParam.create = QUDA_REFERENCE_FIELD_CREATE;
  GaugeField VLink = GaugeField(InParam);
  //   int *num_failures_d = 0;
  // projectSU3(VLink, 1e-1, num_failures_d);
  // gaugeGauss(VLink,123,);
  // gaugeGauss(ULink,123,1e-1);
    
  computeKSLinkQuda(fatlink, longlink, nullptr, cpuInNew.data(), act_path, &gauge_param);

  // computeKSLinkQuda(VLink.data(), nullptr, ULink.data(), cpuInNew.data(), act_path, &gauge_param);

  loadFatLongGaugeQuda(cpuFatMILC.data(), cpuLongMILC.data(), gauge_param);

  // now copy back to QDP aliases, since these are used for the reference dslash
  cpuFatQDP = cpuFatMILC;
  cpuLongQDP = cpuLongMILC;
  // ensure QDP alias has exchanged ghosts
  cpuFatQDP.exchangeGhost();
  cpuLongQDP.exchangeGhost();

    //SET UP INV PARAM END
    
  // invParam.inv_type = inv_type;
  // invParam.solution_type = solution_type;
  // invParam.solve_type = solve_type;
  // invParam.cuda_prec_sloppy = cuda_prec_sloppy;
  // // multishift = ::testing::get<4>(param);
  // invParam.solution_accumulator_pipeline = solution_accumulator_pipeline;

  // // schwarz parameters
  // auto schwarz_param = ::testing::get<6>(param);
  // invParam.schwarz_type = precon_schwarz_type;
  // invParam.inv_type_precondition = inv_type_precondition;
  // invParam.cuda_prec_precondition = cuda_prec_precondition;

  // invParam.residual_type = ::testing::get<7>(param);

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
  std::vector<quda::ColorSpinorField> in_raw(Nsrc);
  std::vector<quda::ColorSpinorField> in(Nsrc);
  std::vector<quda::ColorSpinorField> out(Nsrc);
  std::vector<quda::ColorSpinorField> out_flowed(Nsrc);
  std::vector<quda::ColorSpinorField> out_multishift(Nsrc * multishift);
  std::vector<quda::ColorSpinorField> out_multishift_flowed(Nsrc * multishift);

    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    }
    
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &invParam, &gauge_param);
    //simulates what user might do from external library
  std::vector<std::vector<void *>> _hp_multi_x(Nsrc, std::vector<void *>(multishift));
  std::vector<std::vector<void *>> _hp_multi_x_flowed(Nsrc, std::vector<void *>(multishift));
    
  // Set up Masses
  std::vector<double> masses(multishift);

  if (multishift > 1) {
    if (use_split_grid)
      errorQuda("Multishift currently doesn't support split grid.\n");

    invParam.num_offset = multishift;

    // Consistency check for masses, tols, tols_hq size if we're setting custom values
    if (multishift_shifts.size() != 0)
      errorQuda("Multishift shifts are not supported for Wilson-type fermions");
    if (multishift_masses.size() != 0 && multishift_masses.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift mass count %d does not agree with number of masses passed in %lu\n", multishift, multishift_masses.size());
    if (multishift_tols.size() != 0 && multishift_tols.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols.size());
    if (multishift_tols_hq.size() != 0 && multishift_tols_hq.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift hq tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols_hq.size());

    // Copy offsets and tolerances into invParam; allocate and copy data pointers
    for (int i = 0; i < multishift; i++) {
      masses[i] = (multishift_masses.size() == 0 ? (mass + i * i * 0.01) : multishift_masses[i]);
      invParam.offset[i] = 4 * masses[i] * masses[i];
      invParam.tol_offset[i] = (multishift_tols.size() == 0 ? invParam.tol : multishift_tols[i]);
      invParam.tol_hq_offset[i] = (multishift_tols_hq.size() == 0 ? invParam.tol_hq : multishift_tols_hq[i]);

      // Allocate memory and set pointers
      for (int n = 0; n < Nsrc; n++) {
        out_multishift[n * multishift + i] = quda::ColorSpinorField(cs_param);
        _hp_multi_x[n][i] = out_multishift[n * multishift + i].data();
      }

      logQuda(QUDA_VERBOSE, "Multishift mass %d = %e ; tolerance %e ; hq tolerance %e\n", i, masses[i], invParam.tol_offset[i], invParam.tol_hq_offset[i]);
    }
  }

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
    in_raw[n] = quda::ColorSpinorField(cs_param);
    in[n] = quda::ColorSpinorField(cs_param);
    quda::spinorNoise(in_raw[n], rng, QUDA_NOISE_UNIFORM);
    performAdjGFlowHier(in[n].data(),in_raw[n].data(), &invParam, &smear_param, &gauge_param);
    out[n] = quda::ColorSpinorField(cs_param);
    out_flowed[n] = quda::ColorSpinorField(cs_param);
  }

  // Prepare rng, fill host spinors with random numbers END
  //-----------------------------------------------------------------------------------

  // QUDA invert test
  //----------------------------------------------------------------------------

  if (!use_multi_src || multishift > 1) {

    for (int n = 0; n < Nsrc; n++) {
      void *aw_hp_multi[] = {_hp_multi_x[n].data()};
      void *aw_hp_multi_f[] = {_hp_multi_x_flowed[n].data()};
        
      void *aw_out[] = {out[n].data()};
      void *aw_out_f[] = {out_flowed[n].data()};
        
      // If deflating, preserve the deflation space between solves
      if (inv_deflate) eig_param.preserve_deflation = n < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      // Perform QUDA inversions
      if (multishift > 1) {
        invertMultiShiftQuda(_hp_multi_x[n].data(), in[n].data(), &invParam);
        //QUESTION: is there _hp_multi_x_flowed[n] or or things to be indexed?
        performGFlowQuda(aw_hp_multi_f,aw_hp_multi, &invParam, &smear_param, obs_param, 1);
      } else {
        invertQuda(out[n].data(), in[n].data(), &invParam);
        performGFlowQuda(aw_out_f,aw_out, &invParam, &smear_param, obs_param, 1);
      }

      // move residuals to n^th location for verification after solves have finished
      invParam.true_res[n] = invParam.true_res[0];
      invParam.true_res_hq[n] = invParam.true_res_hq[0];

      time[n] = invParam.secs;
      gflops[n] = invParam.gflops / invParam.secs;
      iter[n] = invParam.iter;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n", invParam.iter, invParam.secs,
                 invParam.gflops / invParam.secs);
      if (invParam.energy > 0) {
        printfQuda("Energy = %g J, Mean power = %g W, mean temp = %g C, mean clock = %f\n\n", invParam.energy,
                   invParam.power, invParam.temp, invParam.clock);
      }
    }
  } else {

    invParam.num_src = Nsrc_tile;
    invParam.num_src_per_sub_partition = Nsrc_tile / num_sub_partition;
    // Host arrays for solutions, sources, and check
    std::vector<void *> _hp_x(Nsrc_tile);
    std::vector<void *> _hp_b(Nsrc_tile);
    std::vector<void *> _hp_x_flowed(Nsrc_tile);

    // void *aw_hp_multi[] = {_hp_multi_x[n].data()};
    // void *aw_hp_multi_f[] = {_hp_multi_x_flowed[n].data()};

    for (int j = 0; j < Nsrc; j += Nsrc_tile) {
      for (int i = 0; i < Nsrc_tile; i++) {
        _hp_x[i] = out[j + i].data();
        _hp_b[i] = in[j + i].data();
      }

      if (inv_deflate) eig_param.preserve_deflation = j < Nsrc - Nsrc_tile ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      invertMultiSrcQuda(_hp_x.data(), _hp_b.data(), &invParam);
      performGFlowQuda(_hp_x_flowed.data(),_hp_x.data(),&invParam, &smear_param, obs_param,1);

      // move residuals to (i+j)^th location for verification after solves have finished
      for (int i = 0; i < Nsrc_tile; i++) {
        invParam.true_res[j + i] = invParam.true_res[i];
        invParam.true_res_hq[j + i] = invParam.true_res_hq[i];
      }

      printfQuda("Done: %d sub-partitions - %i total iter / %g secs = %g Gflops, %g secs per source\n", num_sub_partition,
                 invParam.iter, invParam.secs, invParam.gflops / invParam.secs, invParam.secs / Nsrc_tile);
      if (invParam.energy > 0) {
        printfQuda("Energy = %g J (%g J per source), Mean power = %g W, mean temp = %g C, mean clock = %f\n\n",
                   invParam.energy, invParam.energy / Nsrc_tile, invParam.power, invParam.temp, invParam.clock);
      }
    }
  }

  // Free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // Compute timings
  if (!use_multi_src) performanceStats(time, gflops, iter);


    
    
  check = quda::ColorSpinorField(cs_param);
  //Add noise to spinor

  spinorNoise(check, rng, QUDA_NOISE_GAUSS);


  check_safe = quda::ColorSpinorField(cs_param);
  check_hier = quda::ColorSpinorField(cs_param);
  check_fwd = quda::ColorSpinorField(cs_param);

  void *check_arr[] = {check.data()};
  void *check_fwdarr[] = {check_fwd.data()};

  printf("Inspecting the very first element of the random fermion we will use:\n");
  check.PrintVector(0,0,0);
  printf("Inspecting the very first element of the 3 un-evolved fermions (should be zero):\n");
  printf("Hierarchical method:\n");
  check_hier.PrintVector(0,0,0);
  printf("Safe method:\n");
  check_safe.PrintVector(0,0,0);
  printf("Forward method:\n");
  check_fwd.PrintVector(0,0,0);
     
  host_timer.start(); // start the timer


     
    // Perform two adjoint flow algorithms, these methods dont alter the final value for the gauge so we excecute them first
    host_hier_timer.start();
    performAdjGFlowHier(check_hier.data(),check.data(), &invParam, &smear_param, &gauge_param);
    host_hier_timer.stop();
    host_safe_timer.start();
    performAdjGFlowSafe(check_safe.data(),check.data() , &invParam, &smear_param);
    host_safe_timer.stop();
    // Perform forward flow algorithm
    host_fwd_timer.start();
    performGFlowQuda(check_fwdarr,check_arr, &invParam, &smear_param, obs_param, 1);
    host_fwd_timer.stop();
      
    printfQuda("Time elapsed for adjoint hierarchical fermion/gauge smearing = %g secs\n", host_hier_timer.last());  
    printfQuda("Time elapsed for adjoint safe fermion/gauge smearing = %g secs\n", host_safe_timer.last());  
    printfQuda("Time elapsed for forward fermion/gauge smearing = %g secs\n", host_fwd_timer.last());   



  host_timer.stop(); // stop the timer
   
  printfQuda("Total time for collective fermion/gauge smearing = %g secs\n", host_timer.last());
  printf("Now, inspecting the very first element of the 3 evolved fermions:\n");
  printf("Hierarchical method:\n");
  check_hier.PrintVector(0,0,0);
  printf("Safe method:\n");
  check_safe.PrintVector(0,0,0);
  printf("Forward method:\n");
  check_fwd.PrintVector(0,0,0);

  // for (int dir = 0; dir < 4; dir++) {
  //   host_free(qdp_inlink[dir]);
  //   host_free(qdp_fatlink[dir]);
  //   host_free(qdp_longlink[dir]);
  // }

  freeGaugeQuda();
  endQuda();

  finalizeComms();
  return 0;
}
