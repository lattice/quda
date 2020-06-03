#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <host_utils.h>
#include <command_line_params.h>
#include "misc.h"

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  if (eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", eig_block_size);
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
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator. We encapsualte the
  // inv_param structure inside the eig_param structure
  // to avoid any confusion
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setInvertParam(eig_inv_param);
  // Specific changes to the invert param for the eigensolver
  eig_inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  eig_inv_param.solve_type
    = (eig_inv_param.solution_type == QUDA_MAT_SOLUTION ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE);
  QudaEigParam eig_param = newQudaEigParam();
  // Place Invert param inside Eig param
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);

  // Initialize the QUDA library
  initQuda(device);
  display_test_info();

  // Set some dimension parameters
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, eig_inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // Set spinor site size (wilson types only in this file)
  int sss = 24;
  setSpinorSiteSize(sss);

  // Allocate host side memory for the gauge field.
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  void *clover = 0, *clover_inv = 0;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, eig_inv_param);
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &eig_inv_param);
  }

  // QUDA eigensolver test BEGIN
  //----------------------------------------------------------------------------
  // Host side arrays to store the eigenpairs computed by QUDA
  void **host_evecs = (void **)malloc(eig_nConv * sizeof(void *));
  for (int i = 0; i < eig_nConv; i++) {
    host_evecs[i] = (void *)malloc(V * eig_inv_param.Ls * sss * eig_inv_param.cpu_prec);
  }
  double _Complex *host_evals = (double _Complex *)malloc(eig_param.nEv * sizeof(double _Complex));

  // This function returns the host_evecs and host_evals pointers, populated with the
  // requested data, at the requested prec. All the information needed to perfom the
  // solve is in the eig_param container. If eig_param.arpack_check == true and
  // precision is double, the routine will use ARPACK rather than the GPU.
  double time = -((double)clock());
  if (eig_param.arpack_check && !(eig_inv_param.cpu_prec == QUDA_DOUBLE_PRECISION)) {
    errorQuda("ARPACK check only available in double precision");
  }

  eigensolveQuda(host_evecs, host_evals, &eig_param);
  time += (double)clock();
  printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", time / CLOCKS_PER_SEC);
  // QUDA eigensolver test COMPLETE
  //----------------------------------------------------------------------------

  // Clean up memory allocations
  for (int i = 0; i < eig_nConv; i++) free(host_evecs[i]);
  free(host_evecs);
  free(host_evals);

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
