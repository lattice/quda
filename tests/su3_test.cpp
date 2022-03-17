#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <misc.h>

#include <comm_quda.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

double stout_smear_rho = 0.1;
double stout_smear_epsilon = -0.25;
double ape_smear_rho = 0.6;
int smear_steps = 50;
double wflow_epsilon = 0.01;
int wflow_steps = 100;
QudaWFlowType wflow_type = QUDA_WFLOW_TYPE_WILSON;
bool su_project = true;
int measurement_interval = 5;

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);
  switch (test_type) {
  case 0:
    printfQuda("\nAPE smearing\n");
    printfQuda(" - rho %f\n", ape_smear_rho);
    printfQuda(" - smearing steps %d\n", smear_steps);
    printfQuda(" - Measurement interval %d\n", measurement_interval);
    break;
  case 1:
    printfQuda("\nStout smearing\n");
    printfQuda(" - rho %f\n", stout_smear_rho);
    printfQuda(" - smearing steps %d\n", smear_steps);
    printfQuda(" - Measurement interval %d\n", measurement_interval);
    break;
  case 2:
    printfQuda("\nOver-Improved Stout smearing\n");
    printfQuda(" - rho %f\n", stout_smear_rho);
    printfQuda(" - epsilon %f\n", stout_smear_epsilon);
    printfQuda(" - smearing steps %d\n", smear_steps);
    printfQuda(" - Measurement interval %d\n", measurement_interval);
    break;
  case 3:
    printfQuda("\nWilson Flow\n");
    printfQuda(" - epsilon %f\n", wflow_epsilon);
    printfQuda(" - Wilson flow steps %d\n", wflow_steps);
    printfQuda(" - Wilson flow type %s\n", wflow_type == QUDA_WFLOW_TYPE_WILSON ? "Wilson" : "Symanzik");
    printfQuda(" - Measurement interval %d\n", measurement_interval);
    break;
  default: errorQuda("Undefined test type %d given", test_type);
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

void add_su3_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  CLI::TransformPairs<QudaWFlowType> wflow_type_map {{"wilson", QUDA_WFLOW_TYPE_WILSON},
                                                     {"symanzik", QUDA_WFLOW_TYPE_SYMANZIK}};

  // Option group for SU(3) related options
  auto opgroup = quda_app->add_option_group("SU(3)", "Options controlling SU(3) tests");
  opgroup->add_option("--su3-ape-rho", ape_smear_rho, "rho coefficient for APE smearing (default 0.6)");

  opgroup->add_option("--su3-stout-rho", stout_smear_rho,
                      "rho coefficient for Stout and Over-Improved Stout smearing (default 0.08)");

  opgroup->add_option("--su3-stout-epsilon", stout_smear_epsilon,
                      "epsilon coefficient for Over-Improved Stout smearing (default -0.25)");

  opgroup->add_option("--su3-smear-steps", smear_steps, "The number of smearing steps to perform (default 50)");

  opgroup->add_option("--su3-wflow-epsilon", wflow_epsilon, "The step size in the Runge-Kutta integrator (default 0.01)");

  opgroup->add_option("--su3-wflow-steps", wflow_steps,
                      "The number of steps in the Runge-Kutta integrator (default 100)");

  opgroup->add_option("--su3-wflow-type", wflow_type, "The type of action to use in the wilson flow (default wilson)")
    ->transform(CLI::QUDACheckedTransformer(wflow_type_map));
  ;

  opgroup->add_option("--su3-measurement-interval", measurement_interval,
                      "Measure the field energy and topological charge every Nth step (default 5) ");

  opgroup->add_option("--su3-project", su_project, "Project smeared gauge onto su3 manifold at measurement interval (default true)");
}

int main(int argc, char **argv)
{

  auto app = make_app();
  add_su3_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"APE", 0}, {"Stout", 1}, {"Over-Improved Stout", 2}, {"Wilson Flow", 3}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));

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

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette gauge precise is %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0], plaq[1],
             plaq[2]);

#ifdef GPU_GAUGE_TOOLS

  // All user inputs now defined
  display_test_info();

  // Topological charge and gauge energy
  double q_charge_check = 0.0;
  // Size of floating point data
  size_t data_size = prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
  size_t array_size = V * data_size;
  void *qDensity = safe_malloc(array_size);
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param.qcharge_density = qDensity;

  gaugeObservablesQuda(&param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Computed Etot, Es, Et, Q is\n%.16e %.16e, %.16e %.16e\nDone in %g secs\n", param.energy[0],
             param.energy[1], param.energy[2], param.qcharge, time0);

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
  // users perfrom suN projection.
  // A unique observable param struct is constructed for each measurement.
  
  // Gauge Smearing Routines
  //---------------------------------------------------------------------------
  // Stout smearing should be equivalent to APE smearing
  // on D dimensional lattices for rho = alpha/2*(D-1).
  // Typical APE values are aplha=0.6, rho=0.1 for Stout.  
  switch (test_type) {
  case 0: {
    // APE
    QudaGaugeObservableParam *ape_obs_param = new QudaGaugeObservableParam[wflow_steps/measurement_interval+1];
    for (int i=0; i<smear_steps/measurement_interval+1; i++) {
      ape_obs_param[i] = newQudaGaugeObservableParam();
      ape_obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
      ape_obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      ape_obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }
    
    time0 = -((double)clock()); // start the timer
    performAPEnStep(smear_steps, ape_smear_rho, measurement_interval, ape_obs_param);  
    time0 += clock();           // stop the timer
    time0 /= CLOCKS_PER_SEC;
    printfQuda("Total time for APE = %g secs\n", time0);
    break;
  }
  case 1: {
    // STOUT
    QudaGaugeObservableParam *stout_obs_param = new QudaGaugeObservableParam[wflow_steps/measurement_interval+1];
    for (int i=0; i<smear_steps/measurement_interval+1; i++) {
      stout_obs_param[i] = newQudaGaugeObservableParam();
      stout_obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
      stout_obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      stout_obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }
    
    time0 = -((double)clock()); // start the timer
    performSTOUTnStep(smear_steps, stout_smear_rho, measurement_interval, stout_obs_param);
    time0 += clock();           // stop the timer
    time0 /= CLOCKS_PER_SEC;
    printfQuda("Total time for STOUT = %g secs\n", time0);
    break;
  }

    // Topological charge routines
    //---------------------------------------------------------------------------
  case 2: {
    // Over-Improved STOUT
    QudaGaugeObservableParam *oi_stout_obs_param = new QudaGaugeObservableParam[wflow_steps/measurement_interval+1];
    for (int i=0; i<smear_steps/measurement_interval+1; i++) {
      oi_stout_obs_param[i] = newQudaGaugeObservableParam();
      oi_stout_obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
      oi_stout_obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      oi_stout_obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }

    time0 = -((double)clock()); // start the timer
    performOvrImpSTOUTnStep(smear_steps, stout_smear_rho, stout_smear_epsilon, measurement_interval, oi_stout_obs_param);
    time0 += clock();           // stop the timer
    time0 /= CLOCKS_PER_SEC;
    printfQuda("Total time for Over Improved STOUT = %g secs\n", time0);
    break;
  }
  case 3: {
    // Wilson Flow
    QudaGaugeObservableParam *wf_obs_param = new QudaGaugeObservableParam[wflow_steps/measurement_interval+1];
    for (int i=0; i<wflow_steps/measurement_interval+1; i++) {
      wf_obs_param[i] = newQudaGaugeObservableParam();
      wf_obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
      wf_obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      wf_obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }
    
    time0 = -((double)clock()); // start the timer
    performWFlownStep(wflow_steps, wflow_epsilon, measurement_interval, wflow_type, wf_obs_param);
    time0 += clock();           // stop the timer
    time0 /= CLOCKS_PER_SEC;
    printfQuda("Total time for Wilson Flow = %g secs\n", time0);
    break;
  }
  default: errorQuda("Undefined test type %d given", test_type);
  }

#else
  printfQuda("Skipping other gauge tests since gauge tools have not been compiled\n");
#endif

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
