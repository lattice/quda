#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h> // convenient quark field container

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type),
	     get_recon_str(link_recon),
	     get_recon_str(link_recon_sloppy),
	     xdim, ydim, zdim, tdim, Lsdim,
	     get_dslash_str(dslash_type),
	     get_mass_normalization_str(normalization));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));
}


int main(int argc, char **argv)
{
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %s not supported\n", get_dslash_str(dslash_type));
    exit(0);
  }

  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  // offsets used only by multi-shift solver
  inv_param.num_offset = 12;
  inv_param.num_src = Nsrc;
  double offset[12] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  // initialize the QUDA library
  initQuda(device);

  // Set some dimension parameters for the host routines
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  setSpinorSiteSize(24);

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);

    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
  }

  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  // Vector construct START
  //-----------------------------------------------------------------------------------
  quda::ColorSpinorField *check;
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  check = quda::ColorSpinorField::Create(cs_param);

  // Host arrays for solutions, sources, and check
  void **outMulti = (void **)malloc(inv_param.num_src * sizeof(void *));
  void **inMulti = (void **)malloc(inv_param.num_src * sizeof(void *));
  // QUDA host arrays
  std::vector<quda::ColorSpinorField *> qudaOutMulti(inv_param.num_src);
  std::vector<quda::ColorSpinorField *> qudaInMulti(inv_param.num_src);

  for (int i = 0; i < inv_param.num_src; i++) {
    // Allocate memory and set pointers
    qudaOutMulti[i] = quda::ColorSpinorField::Create(cs_param);
    outMulti[i] = qudaOutMulti[i]->V();

    qudaInMulti[i] = quda::ColorSpinorField::Create(cs_param);
    inMulti[i] = qudaInMulti[i]->V();
    // Populate the host spinor with random numbers.
    constructRandomSpinorSource(qudaInMulti[i]->V(), 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);
  }
  // Vector construct END
  //-----------------------------------------------------------------------------------

  // QUDA invert test BEGIN
  //----------------------------------------------------------------------------
  // start the timer
  double time0 = -((double)clock());

  // Perform the inversion
  invertMultiSrcQuda(outMulti, inMulti, &inv_param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs,
             inv_param.gflops / inv_param.secs, time0);

  // Perform host side verification of inversion if requested
  if (verify_results) {
    for (int i = 0; i < inv_param.num_src; i++) {
      verifyInversion(outMulti[i], inMulti[i], check->V(), gauge_param, inv_param, gauge, clover, clover_inv);
    }
  }
  // QUDA invert test COMPLETE
  //----------------------------------------------------------------------------

  rng->Release();
  delete rng;

  // Clean up memory allocations
  delete check;
  free(outMulti);
  free(inMulti);
  for (int i = 0; i < inv_param.num_src; i++) {
    delete qudaOutMulti[i];
    delete qudaInMulti[i];
  }

  freeGaugeQuda();
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  return 0;
}
