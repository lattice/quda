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

void display_info()
{
  printfQuda("running the following measurement:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);
  printfQuda("\nWilson Flow\n");
  printfQuda(" - epsilon %f\n", wflow_epsilon);
  printfQuda(" - Wilson flow steps %d\n", wflow_steps);
  printfQuda(" - Wilson flow type %s\n", wflow_type == QUDA_WFLOW_TYPE_WILSON ? "Wilson" : "Symanzik");
  printfQuda(" - Measurement interval %d\n", measurement_interval);

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
  
  // Allocate space on the host
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // All user inputs now defined
  display_info();
  //----------------------------------------------------------------------------

  
  // Plaquette measurement
  //--------------------------------------------------------------------------
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
  
  // Leapfrog
  //--------------------------------------------------------------------------
  QudaHMCParam hmc_param = newQudaHMCParam();
  setHMCParam(hmc_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  hmc_param.invert_param = &inv_param;
  hmc_param.gauge_param = &gauge_param;
  
  // Vector construct
  quda::ColorSpinorParam cs_param;
  constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  auto in = quda::ColorSpinorField::Create(cs_param);
  auto out = quda::ColorSpinorField::Create(cs_param);
  auto check = quda::ColorSpinorField::Create(cs_param);
  
  // Start the timer
  time0 = -((double)clock());

  // Run the QUDA computation
  for(int i=0; i<hmc_param.updates; i++) {
    performLeapfrogStep(out->V(), in->V(), &hmc_param, i);
  }

  // Stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("Total time for leapfrog  = %g secs\n", time0);
  //--------------------------------------------------------------------------

  // If the --save-gauge flag was passed, this will save the Wilson flowed gauge field
  saveHostGaugeField(gauge, gauge_param, QUDA_WILSON_LINKS);
  
  // Clean up
  freeGaugeQuda();
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);
  delete in;
  delete out;
  delete check;
  
  // Finalize the QUDA library
  endQuda();  
  finalizeComms();
  
  return 0;
}
