#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <timer.h>
#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <misc.h>

#include <comm_quda.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Smearing variables
double gauge_smear_rho = 0.1;
double gauge_smear_epsilon = 0.1;
double gauge_smear_alpha = 0.6;
int gauge_smear_steps = 50;
QudaGaugeSmearType gauge_smear_type = QUDA_GAUGE_SMEAR_STOUT;
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
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: printfQuda(" - epsilon %f\n", gauge_smear_epsilon); break;
  default: errorQuda("Undefined test type %d given", test_type);
  }
  printfQuda(" - smearing steps %d\n", gauge_smear_steps);
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
                                                                {"wilson", QUDA_GAUGE_SMEAR_WILSON_FLOW},
                                                                {"symanzik", QUDA_GAUGE_SMEAR_SYMANZIK_FLOW}};

  // Option group for SU(3) related options
  auto opgroup = quda_app->add_option_group("SU(3)", "Options controlling SU(3) tests");

  opgroup
    ->add_option(
      "--su3-smear-type",
      gauge_smear_type, "The type of action to use in the smearing. Options: APE, Stout, Over Improved Stout, Wilson Flow, Symanzik Flow (default stout)")
    ->transform(CLI::QUDACheckedTransformer(gauge_smear_type_map));
  ;
  opgroup->add_option("--su3-smear-alpha", gauge_smear_alpha, "alpha coefficient for APE smearing (default 0.6)");

  opgroup->add_option("--su3-smear-rho", gauge_smear_rho,
                      "rho coefficient for Stout and Over-Improved Stout smearing (default 0.1)");

  opgroup->add_option("--su3-smear-epsilon", gauge_smear_epsilon,
                      "epsilon coefficient for Over-Improved Stout smearing or Wilson flow (default 0.1)");

  opgroup->add_option("--su3-smear-steps", gauge_smear_steps, "The number of smearing steps to perform (default 50)");

  opgroup->add_option("--su3-measurement-interval", measurement_interval,
                      "Measure the field energy and/or topological charge every Nth step (default 5) ");

  opgroup->add_option("--su3-project", su_project,
                      "Project smeared gauge onto su3 manifold at measurement interval (default true)");
}

int main(int argc, char **argv)
{

  auto app = make_app();
  add_su3_option_group(app);

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

  setWilsonGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);

  // All user inputs are now defined
  display_test_info();

  void *gauge[4], *new_gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    new_gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();

  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);
  saveGaugeQuda(new_gauge, &gauge_param);

  // Prepare various perf info
  long long flops_plaquette = 6ll * 597 * V;
  long long flops_ploop = 198ll * V + 6 * V / gauge_param.X[3];

  // Prepare a gauge observable struct
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();

  // start the timer
  quda::host_timer_t host_timer;

  // We call gaugeObservablesQuda multiple times to time each bit individually

  // Compute the plaquette
  param.compute_plaquette = QUDA_BOOLEAN_TRUE;

  // Tuning call
  gaugeObservablesQuda(&param);

  host_timer.start();
  for (int i = 0; i < niter; i++) gaugeObservablesQuda(&param);
  host_timer.stop();
  double secs_plaquette = host_timer.last() / niter;
  double perf_plaquette = flops_plaquette / (secs_plaquette * 1024 * 1024 * 1024);
  printfQuda(
    "Computed plaquette gauge precise is %.16e (spatial = %.16e, temporal = %.16e), done in %g seconds, %g GFLOPS\n",
    param.plaquette[0], param.plaquette[1], param.plaquette[2], secs_plaquette, perf_plaquette);
  param.compute_plaquette = QUDA_BOOLEAN_FALSE;

  // Compute the temporal Polyakov loop
  param.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;

  // Tuning call
  gaugeObservablesQuda(&param);

  host_timer.start();
  for (int i = 0; i < niter; i++) gaugeObservablesQuda(&param);
  host_timer.stop();
  double secs_ploop = host_timer.last() / niter;
  double perf_ploop = flops_ploop / (secs_ploop * 1024 * 1024 * 1024);
  printfQuda("Computed Polyakov loop gauge precise is %.16e +/- I %.16e , done in %g seconds, %g GFLOPS\n",
             param.ploop[0], param.ploop[1], secs_ploop, perf_ploop);
  param.compute_polyakov_loop = QUDA_BOOLEAN_FALSE;

  // Topological charge and gauge energy
  double q_charge_check = 0.0;
  // Size of floating point data
  size_t data_size = prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
  size_t array_size = V * data_size;
  void *qDensity = pinned_malloc(array_size);

  // start the timer
  host_timer.start();

  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param.qcharge_density = qDensity;

  gaugeObservablesQuda(&param);

  // stop the timer
  host_timer.stop();
  printfQuda("Computed Etot, Es, Et, Q is\n%.16e %.16e, %.16e %.16e\nDone in %g secs\n", param.energy[0],
             param.energy[1], param.energy[2], param.qcharge, host_timer.last());

  // Ensure host array sums to return value
  if (prec == QUDA_DOUBLE_PRECISION) {
    for (int i = 0; i < V; i++) q_charge_check += ((double *)qDensity)[i];
  } else {
    for (int i = 0; i < V; i++) q_charge_check += ((float *)qDensity)[i];
  }

  // release memory
  host_free(qDensity);

  // Q charge Reduction and normalisation
  quda::comm_allreduce_sum(q_charge_check);

  printfQuda("GPU value %e and host density sum %e. Q charge deviation: %e\n", param.qcharge, q_charge_check,
             param.qcharge - q_charge_check);

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
  smear_param.meas_interval = measurement_interval;
  smear_param.alpha = gauge_smear_alpha;
  smear_param.rho = gauge_smear_rho;
  smear_param.epsilon = gauge_smear_epsilon;

  host_timer.start(); // start the timer
  switch (smear_param.smear_type) {
  case QUDA_GAUGE_SMEAR_APE:
  case QUDA_GAUGE_SMEAR_STOUT:
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT: {
    performGaugeSmearQuda(&smear_param, obs_param);
    break;
  }

    // Here we use a typical use case which is different from simple smearing in that
    // the user will want to compute the plaquette values to compute the gauge energy.
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: {
    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    }
    performWFlowQuda(&smear_param, obs_param);
    break;
  }
  default: errorQuda("Undefined gauge smear type %d given", smear_param.smear_type);
  }

  host_timer.stop(); // stop the timer
  printfQuda("Total time for gauge smearing = %g secs\n", host_timer.last());

  if (verify_results) check_gauge(gauge, new_gauge, 1e-3, gauge_param.cpu_prec);

  for (int dir = 0; dir < 4; dir++) {
    host_free(gauge[dir]);
    host_free(new_gauge[dir]);
  }

  freeGaugeQuda();
  endQuda();

  finalizeComms();
  return 0;
}
