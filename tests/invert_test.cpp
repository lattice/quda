#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

#include <util_quda.h>
#include <random_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
             get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type),
             get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim, tdim, Lsdim,
             get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  if (inv_deflate) {
    printfQuda("\n   Eigensolver parameters\n");
    printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
    printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
    printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
    printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
    printfQuda(" - size of Krylov space %d\n", eig_nKr);
    printfQuda(" - solver tolerance %e\n", eig_tol);
    printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
    if (eig_compute_svd) {
      printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
      printfQuda(" - ***********************************************************\n");
      printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
      printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
      printfQuda(" - ***********************************************************\n");
    } else {
      printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
                 eig_use_normop ? "true" : "false");
    }
    if (eig_use_poly_acc) {
      printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
      printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
      if (eig_amax <= 0)
        printfQuda(" - Chebyshev polynomial maximum will be computed\n");
      else
        printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
    }
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));

  return;
}

int main(int argc, char **argv)
{
  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // Move these
  mg_verbosity[0] = QUDA_SILENT; // set default preconditioner verbosity
  if (multishift) solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; // set a correct default for the multi-shift solver

  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set some default values for precisions if none are passed through the command line
  setQudaDefaultPrecs();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // Check for deflation
  QudaEigParam eig_param = newQudaEigParam();
  if (inv_deflate) {
    inv_param.eig_param = &eig_param;
    setEigParam(eig_param);
  } else {
    inv_param.eig_param = nullptr;
  }

  // Initialize the QUDA library
  initQuda(device);
  display_test_info();

  // Set some dimension parameters
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }
  setSpinorSiteSize(24);

  // Allocate host side memory for the gauge field.
  void *gauge[4];
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  void *clover = 0, *clover_inv = 0;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    constructHostCloverField(clover, clover_inv, inv_param);
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
  }

  // Allocate host side memory for the spinor fields
  void *spinorOut = NULL, **spinorOutMulti = NULL;
  void *spinorIn = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  void *spinorCheck = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  if (multishift) {
    spinorOutMulti = (void**)malloc(inv_param.num_offset*sizeof(void *));
    for (int i=0; i<inv_param.num_offset; i++) {
      spinorOutMulti[i] = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
    }
  } else {
    spinorOut = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  }

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  for (int i = 0; i < Nsrc; i++) {

    // Populate the host spinor with random numbers.
    constructRandomSpinorSource(spinorIn, 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);
    // If deflating preserve the deflation space between solves
    eig_param.preserve_deflation = i < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    // Perform QUDA inversions
    if (multishift) {
      invertMultiShiftQuda(spinorOutMulti, spinorIn, &inv_param);
    } else {
      invertQuda(spinorOut, spinorIn, &inv_param);
    }

    time[i] = inv_param.secs;
    gflops[i] = inv_param.gflops / inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);
  }

  rng->Release();
  delete rng;

  // Compute performance statistics
  if (Nsrc > 1) performanceStats(time, gflops);
  delete[] time;
  delete[] gflops;

  // Perform host side verification of inversion if requested
  if (verify_results) {
    verifyInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge, clover, clover_inv);
  }

  // Clean up device memory allocationa
  free(spinorIn);
  free(spinorCheck);
  if (multishift) {
    for (int i = 0; i < inv_param.num_offset; i++) free(spinorOutMulti[i]);
    free(spinorOutMulti);
  } else {
    free(spinorOut);
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
