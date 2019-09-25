#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include "misc.h"

#include <qio_field.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;

void setGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif
}

int main(int argc, char **argv)
{

  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  setGaugeParam(gauge_param);
  setDims(gauge_param.X);
  size_t gSize = gauge_param.cpu_prec;

  void *gauge[4];

  for (int dir = 0; dir < 4; dir++) { gauge[dir] = malloc(V * gaugeSiteSize * gSize); }

  initQuda(device);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();

  bool load_gauge = strcmp(latfile, "");
  // load in the command line supplied gauge field
  if (load_gauge) {
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  }

  loadGaugeQuda(gauge, &gauge_param);

  if (!load_gauge) gaussGaugeQuda(1234, gaussian_sigma);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette gauge precise is %16.15e (spatial = %16.15e, temporal = %16.15e)\n", plaq[0], plaq[1],
             plaq[2]);

  freeGaugeQuda();
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) { free(gauge[dir]); }

  finalizeComms();
  return 0;
}


