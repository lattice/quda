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
  constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
  inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
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
    constructRandomSpinorSource(in->V(), 4, 3, inv_param.cpu_prec, inv_param.solution_type, gauge_param.X, *rng);

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
  if (strcmp(gauge_outfile,"") != 0) {
    //saveHostGaugeField(gauge, gauge_param, QUDA_WILSON_LINKS);
  }
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
