#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
  // command line options
  auto app = make_app();
  add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // QUDA parameters begin here.
  //------------------------------------------------------------------------------
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

  // All user inputs now defined
  display_test_info();

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, eig_inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // set spinor site size
  int sss = 24;
  if (dslash_type == QUDA_LAPLACE_DSLASH) sss = 6;
  setSpinorSiteSize(sss);

  void *gauge[4], *clover = 0, *clover_inv = 0;

  for (int dir = 0; dir < 4; dir++) { gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size); }

  if (strcmp(latfile, "")) { // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      // unit SU(3) field
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      // random SU(3) field
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are rands in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = eig_inv_param.clover_cpu_prec;
    clover = malloc(V * clover_site_size * cSize);
    clover_inv = malloc(V * clover_site_size * cSize);
    if (!compute_clover) construct_clover_field(clover, norm, diag, eig_inv_param.clover_cpu_prec);

    eig_inv_param.compute_clover = compute_clover;
    if (compute_clover) eig_inv_param.return_clover = 1;
    eig_inv_param.compute_clover_inverse = 1;
    eig_inv_param.return_clover_inverse = 1;
  }

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void *)gauge, &gauge_param);

  // this line ensure that if we need to construct the clover inverse
  // (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    loadCloverQuda(clover, clover_inv, &eig_inv_param);
  }

  // QUDA eigensolver test
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

  // Deallocate host memory
  for (int i = 0; i < eig_nConv; i++) free(host_evecs[i]);
  free(host_evecs);
  free(host_evals);

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) { freeCloverQuda(); }

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);

  return 0;
}
