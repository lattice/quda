// C++ headers
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <util_quda.h>
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <blas_quda.h>

// HMC params
int hmc_start = 0;
int hmc_updates = 200;
int hmc_therm_updates = 100;
int hmc_checkpoint = 5;
int hmc_traj_steps = 100;
double hmc_traj_length = 1.0;
bool hmc_coldstart = false;
QudaGaugeActionType hmc_gauge_action = QUDA_GAUGE_ACTION_TYPE_WILSON;
double hmc_beta = 6.2;

void setHMCParam(QudaHMCParam &hmc_param)
{
  hmc_param.start = hmc_start;
  hmc_param.updates = hmc_updates;
  hmc_param.therm_updates = hmc_therm_updates;
  hmc_param.traj_steps = hmc_traj_steps;
  hmc_param.traj_length = hmc_traj_length;
  hmc_param.coldstart = hmc_coldstart ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  hmc_param.beta = hmc_beta;  
}

CLI::TransformPairs<QudaGaugeActionType> gauge_action_type_map {{"wilson", QUDA_GAUGE_ACTION_TYPE_WILSON},
                                                                  {"symanzik", QUDA_GAUGE_ACTION_TYPE_SYMANZIK},
								  {"luscher-weisz", QUDA_GAUGE_ACTION_TYPE_LUSCHER_WEISZ}};

void add_hmc_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  CLI::QUDACheckedTransformer gauge_action_type_transform(gauge_action_type_map);
  
  // Option group for hmc related options
  auto opgroup = quda_app->add_option_group("hmc", "Options controlling hmc routines");
  opgroup->add_option("--hmc-beta", hmc_beta, "Beta value used in hmc test (default 6.2)");
  opgroup->add_option("--hmc-coldstart", hmc_coldstart,
                       "Whether to use a cold or hot start in hmc (default false)");
  opgroup->add_option("--hmc-traj-length", hmc_traj_length,
                       "The length of the trajectory in MD time (default 1.0)");
  opgroup->add_option("--hmc-traj-steps", hmc_traj_steps,
		      "The number of steps in the integration trajectory (default 25)");
  opgroup->add_option("--hmc-therm-updates", hmc_therm_updates,
		      "The number of trajectorys to traverse before measurement (default 100)");
  opgroup->add_option("--hmc-updates", hmc_updates,
		      "the number of updates to perfrom after thermalisation (default 100)");
  opgroup->add_option("--hmc-checkpoint", hmc_checkpoint,
		      "Number of measurement steps in hmc before checkpointing (default 5)");
  opgroup->add_option("--hmc-gauge-action", hmc_gauge_action, "The gauge action type to use: wilson, symanzik, luscher-weisz (default wilson)")
    ->transform(CLI::QUDACheckedTransformer(gauge_action_type_map));
}


void display_info()
{
  printfQuda("running the following measurement:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{
  auto app = make_app();
  add_hmc_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Initialize the QUDA library
  initQuda(device_ordinal);

  // Set verbosity
  setVerbosity(verbosity);

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Set the dimensions
  setDims(gauge_param.X);  
      
  // All user inputs now defined
  display_info();
  //----------------------------------------------------------------------------

  // Leapfrog HMC start
  //--------------------------------------------------------------------------
  QudaHMCParam hmc_param = newQudaHMCParam();
  setHMCParam(hmc_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  hmc_param.invert_param = &inv_param;
  hmc_param.gauge_param = &gauge_param;
  
  // Allocate space on the host for a gauge field, fill with a random gauge, a unit gauge
  // or a user supplied gauge.
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);  
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    // By passing null pointers to this function, QUDA infers that you want QUDA
    // compute the clover terms and inverses for you from the gauge field. If you
    // Want to pass your own clover fields, study the example given in the `invert_test.cpp`
    // file in quda/tests.
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }
  
  // Plaquette measurement
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_plaquette = QUDA_BOOLEAN_TRUE;

  // Run the QUDA computation
  gaugeObservablesQuda(&param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  
  printfQuda("Computed plaquette is %.16e (spatial = %.16e, temporal = %.16e)\n", param.plaquette[0], param.plaquette[1], param.plaquette[2]);
  //--------------------------------------------------------------------------  
  
  // Vector construct
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  auto in = quda::ColorSpinorField::Create(cs_param);
  auto out = quda::ColorSpinorField::Create(cs_param);
  auto check = quda::ColorSpinorField::Create(cs_param);
    
  // Start the timer
  time0 = -((double)clock());

  // RNG for populating the host spinor with random numbers.
  auto *rng = new quda::RNG(*check, 1234);
  
  // Run the QUDA computation. The the metropolis step is performed in the function.
  for(int i=0; i<hmc_param.therm_updates + hmc_param.therm_updates; i++) {

    // Momentum refresh
    constructRandomSpinorSource(in->V(), 4, 3, inv_param.cpu_prec, inv_param.solution_type, gauge_param.X, 4, *rng);

    // Reversibility check
    //if((i+1)%hmc_reversibility_check == 0) {
    if((i+1)%10 == 0 && 0) {

      // Save current momentum
      check = in;
      
      // Save current gauge 
      saveGaugeQuda((void *)gauge, &gauge_param);
    }

    
    // HMC step
    int accept = performLeapfrogStep(out->V(), in->V(), &hmc_param, i);

    // Checkpoint
    if((i+1)%hmc_checkpoint == 0) {
      
    }
    
    // Reversibility check
    //if((i+1)%hmc_reversibility_check == 0) {
    if((i+1)%10 == 0 && 0) {

      // Apply HMC from current config and check against old momenta
      // and gauge field.
      hmc_param.traj_length *= -1.0;
      performLeapfrogStep(out->V(), in->V(), &hmc_param, i);
      hmc_param.traj_length *= -1.0;
      
      void *gauge_reversed[4];
      for (int dir = 0; dir < 4; dir++) gauge_reversed[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      saveGaugeQuda((void *)gauge_reversed, &gauge_param);

      // Momentum check
      quda::blas::axpy(-1.0, *out, *check);
      printfQuda("Momentum diff = %.16e\n", sqrt(quda::blas::norm2(*check)));

      // Gauge check
      check_gauge(gauge, gauge_reversed, 1e-6, QUDA_DOUBLE_PRECISION);
      for (int dir = 0; dir < 4; dir++) host_free(gauge_reversed[dir]);
    }
  }
  
  // Stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("Total time for leapfrog  = %g secs\n", time0);
  
  // Save if output string is specified
  //if (strcmp(gauge_outfile,"") != 0) {
  //saveHostGaugeField(gauge, gauge_param, QUDA_WILSON_LINKS);
  //}
  //--------------------------------------------------------------------------
  
  // Clean up
  for (int dir = 0; dir < 4; dir++) host_free(gauge[dir]);
  delete in;
  delete out;
  delete check;
  delete rng;
  
  // Finalize the QUDA library
  endQuda();  
  finalizeComms();
  
  return 0;
}
