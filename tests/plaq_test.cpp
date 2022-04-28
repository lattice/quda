#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

int main(int argc, char **argv)
{
  auto app = make_app();
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

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette gauge precise is %16.15e (spatial = %16.15e, temporal = %16.15e)\n", plaq[0], plaq[1],
             plaq[2]);

  freeGaugeQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) { host_free(gauge[dir]); }

  endQuda();
  finalizeComms();
  return 0;
}
